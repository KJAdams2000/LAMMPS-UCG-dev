/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Carolyn Phillips (U Mich), reservoir energy tally
                         Aidan Thompson (SNL) GJF formulation
                         Charles Sievers & Niels Gronbech-Jensen (UC Davis)
                             updated GJF formulation and included
                             statistically correct 2GJ velocity
------------------------------------------------------------------------- */

#include "fix_ucgld_langevin.h"

#include "atom.h"
#include "atom_vec_ellipsoid.h"
#include "comm.h"
#include "compute.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "input.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "random_mars.h"
#include "respa.h"
#include "update.h"
#include "variable.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

enum { NOBIAS, BIAS };
enum { CONSTANT, EQUAL, ATOM };

static constexpr double SINERTIA = 0.4;    // moment of inertia prefactor for sphere
static constexpr double EINERTIA = 0.2;    // moment of inertia prefactor for ellipsoid

/* ---------------------------------------------------------------------- */

Fix_UCGLD_Langevin::Fix_UCGLD_Langevin(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), gfactor1(nullptr), gfactor2(nullptr), ratio(nullptr),
    flangevin(nullptr), tforce(nullptr), franprev(nullptr), lv(nullptr),
    id_temp(nullptr), random(nullptr)
{
  if (narg < 7) error->all(FLERR, "Illegal fix langevin command");

  dynamic_group_allow = 1;
  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  ecouple_flag = 1;
  nevery = 1;

  if (utils::strmatch(arg[3], "^v_")) {
    error->all(FLERR, "lambda dynamic variable is not supported with variable temperature");
  } else {
    t_start = utils::numeric(FLERR, arg[3], false, lmp);
    t_target = t_start;
    tstyle = CONSTANT;
  }

  t_stop = utils::numeric(FLERR, arg[4], false, lmp);
  t_period = utils::numeric(FLERR, arg[5], false, lmp);
  seed = utils::inumeric(FLERR, arg[6], false, lmp);

  if (t_period <= 0.0) error->all(FLERR, "Fix langevin period must be > 0.0");
  if (seed <= 0) error->all(FLERR, "Illegal fix langevin command");

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp, seed + comm->me);

  // allocate per-type arrays for force prefactors

  gfactor1 = new double[atom->ntypes + 1];
  gfactor2 = new double[atom->ntypes + 1];
  ratio = new double[atom->ntypes + 1];

  // optional args

  for (int i = 1; i <= atom->ntypes; i++) ratio[i] = 1.0;

  // set temperature = nullptr, user can override via fix_modify if wants bias

  id_temp = nullptr;
  temperature = nullptr;

  energy = 0.0;

  // flangevin is unallocated until first call to setup()
  // compute_scalar checks for this and returns 0.0
  // if flangevin_allocated is not set

  flangevin = nullptr;
  flangevin_allocated = 0;
  franprev = nullptr;
  lv = nullptr;
  tforce = nullptr;
  maxatom1 = maxatom2 = 0;

  // setup atom-based array for franprev
  // register with Atom class
  // no need to set peratom_flag, b/c data is for internal use only

}

/* ---------------------------------------------------------------------- */

Fix_UCGLD_Langevin::~Fix_UCGLD_Langevin()
{
  if (copymode) return;

  delete random;
  delete[] gfactor1;
  delete[] gfactor2;
  delete[] ratio;
  delete[] id_temp;
  memory->destroy(flangevin);
  memory->destroy(tforce);
}

/* ---------------------------------------------------------------------- */

int Fix_UCGLD_Langevin::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void Fix_UCGLD_Langevin::init()
{
  if (id_temp) {
    temperature = modify->get_compute_by_id(id_temp);
    if (!temperature) {
      error->all(FLERR, "Temperature compute ID {} for fix {} does not exist", id_temp, style);
    } else {
      if (temperature->tempflag == 0)
        error->all(FLERR, "Compute ID {} for fix {} does not compute temperature", id_temp, style);
    }
  }

  // set force prefactors


  for (int i = 1; i <= atom->ntypes; i++) {
    gfactor1[i] = -atom->ucgml[i] / t_period / force->ftm2v;
    gfactor2[i] = sqrt(atom->ucgml[i]) / force->ftm2v;
    gfactor2[i] *= sqrt(24.0 * force->boltz / t_period / update->dt / force->mvv2e); 
      /// uniform distribution instead of Gaussian, for the sampling of random forces
    gfactor1[i] *= 1.0 / ratio[i];
    gfactor2[i] *= 1.0 / sqrt(ratio[i]);
  }
  

  if (temperature && temperature->tempbias)
    tbiasflag = BIAS;
  else
    tbiasflag = NOBIAS;

  if (utils::strmatch(update->integrate_style, "^respa")) {
    nlevels_respa = (static_cast<Respa *>(update->integrate))->nlevels;
  }

}

/* ---------------------------------------------------------------------- */

void Fix_UCGLD_Langevin::setup(int vflag)
{
  if (utils::strmatch(update->integrate_style, "^verlet"))
    post_force(vflag);
  else {
    auto respa = static_cast<Respa *>(update->integrate);
    respa->copy_flevel_f(nlevels_respa - 1);
    post_force_respa(vflag, nlevels_respa - 1, 0);
    respa->copy_f_flevel(nlevels_respa - 1);
  }
}

/* ---------------------------------------------------------------------- */
// clang-format off

void Fix_UCGLD_Langevin::post_force(int /*vflag*/)
{

  // enumerate all 2^6 possibilities for template parameters
  // this avoids testing them inside inner loop:
  // TSTYLEATOM, GJF, TALLY, BIAS, RMASS, ZERO

  if (tbiasflag == BIAS)
      post_force_templated<1>();
  else
      post_force_templated<0>();
}

/* ---------------------------------------------------------------------- */

void Fix_UCGLD_Langevin::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ----------------------------------------------------------------------
   modify forces using one of the many Langevin styles
------------------------------------------------------------------------- */

template<int Tp_BIAS>
void Fix_UCGLD_Langevin::post_force_templated()
{
  double gamma1,gamma2;

  double *lmdv = atom->ucgvl;
  double *lmdf = atom->ucgforce;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  // apply damping and thermostat to atoms in group

  // for Tp_TSTYLEATOM:
  //   use per-atom per-coord target temperature
  // for Tp_GJF:
  //   use Gronbech-Jensen/Farago algorithm
  //   else use regular algorithm
  // for Tp_TALLY:
  //   store drag plus random forces in flangevin[nlocal][3]
  // for Tp_BIAS:
  //   calculate temperature since some computes require temp
  //   computed on current nlocal atoms to remove bias
  //   test v = 0 since some computes mask non-participating atoms via v = 0
  //   and added force has extra term not multiplied by v = 0
  // for Tp_RMASS:
  //   use per-atom masses
  //   else use per-type masses
  // for Tp_ZERO:
  //   sum random force over all atoms in group
  //   subtract sum/count from each atom in group

  double fdrag,fran,fsum,fsumall;
  bigint count;
  double fswap;

  double boltz = force->boltz;
  double dt = update->dt;
  double mvv2e = force->mvv2e;
  double ftm2v = force->ftm2v;

  compute_target();

  // reallocate flangevin if necessary

  if (Tp_BIAS) temperature->compute_scalar();

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {

      gamma1 = gfactor1[type[i]];
      gamma2 = gfactor2[type[i]] * tsqrt;
      

      fran = gamma2*(random->uniform()-0.5);

      if (Tp_BIAS) {
        // temperature->remove_bias(i,lmdv[i]);
        fdrag = gamma1*lmdv[i];

        if (lmdv[i] == 0.0) fran = 0.0;

        // temperature->restore_bias(i,lmdv[i]);
      } else {
        fdrag = gamma1*lmdv[i];
      }

      lmdf[i] += fdrag + fran;

    }
  }
}

/* ----------------------------------------------------------------------
   compute lambda temperature
------------------------------------------------------------------------- */

void Fix_UCGLD_Langevin::end_of_step()
{
  double lmd_ek = 0.0;
  for (int i = 0; i < atom->nlocal; i++) {
    if (atom->mask[i] & groupbit) {
      lmd_ek += 0.5 * atom->ucgml[i] * atom->ucgvl[i] * atom->ucgvl[i] * force->mvv2e;
    }
  }
  lambda_temp = lmd_ek / (0.5 * force->boltz * atom->nlocal);
}

/* ----------------------------------------------------------------------
   set current t_target and t_sqrt
------------------------------------------------------------------------- */

void Fix_UCGLD_Langevin::compute_target()
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double delta = update->ntimestep - update->beginstep;
  if (delta != 0.0) delta /= update->endstep - update->beginstep;

  // if variable temp, evaluate variable, wrap with clear/add
  // reallocate tforce array if necessary

  if (tstyle == CONSTANT) {
    t_target = t_start + delta * (t_stop-t_start);
    tsqrt = sqrt(t_target);
  } else {
    modify->clearstep_compute();
    if (tstyle == EQUAL) {
      t_target = input->variable->compute_equal(tvar);
      if (t_target < 0.0)
        error->one(FLERR, "Fix langevin variable returned negative temperature");
      tsqrt = sqrt(t_target);
    } else {
      if (atom->nmax > maxatom2) {
        maxatom2 = atom->nmax;
        memory->destroy(tforce);
        memory->create(tforce,maxatom2,"langevin:tforce");
      }
      input->variable->compute_atom(tvar,igroup,tforce,1,0);
      for (int i = 0; i < nlocal; i++)
        if (mask[i] & groupbit)
            if (tforce[i] < 0.0)
              error->one(FLERR, "Fix langevin variable returned negative temperature");
    }
    modify->addstep_compute(update->ntimestep + 1);
  }
}


// clang-format on
/* ---------------------------------------------------------------------- */

void Fix_UCGLD_Langevin::reset_target(double t_new)
{
  t_target = t_start = t_stop = t_new;
}

/* ---------------------------------------------------------------------- */

void Fix_UCGLD_Langevin::reset_dt()
{
  if (atom->mass) {
    for (int i = 1; i <= atom->ntypes; i++) {
      gfactor2[i] = sqrt(atom->mass[i]) / force->ftm2v;
      gfactor2[i] *= sqrt(24.0 * force->boltz / t_period / update->dt / force->mvv2e);
      gfactor2[i] *= 1.0 / sqrt(ratio[i]);
    }
  }

}

/* ---------------------------------------------------------------------- */

int Fix_UCGLD_Langevin::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0], "temp") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "fix_modify", error);
    delete[] id_temp;
    id_temp = utils::strdup(arg[1]);
    temperature = modify->get_compute_by_id(id_temp);
    if (!temperature)
      error->all(FLERR, "Could not find fix_modify temperature compute ID: {}", id_temp);

    if (temperature->tempflag == 0)
      error->all(FLERR, "Fix_modify temperature compute {} does not compute temperature", id_temp);
    if (temperature->igroup != igroup && comm->me == 0)
      error->warning(FLERR, "Group for fix_modify temp != fix group: {} vs {}",
                     group->names[igroup], group->names[temperature->igroup]);
    return 2;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */
/// output current lambda temperature
/// used by compute_scalar() and thermo output
double Fix_UCGLD_Langevin::compute_scalar()
{
  return lambda_temp;
}

/* ----------------------------------------------------------------------
   extract thermostat properties
------------------------------------------------------------------------- */

void *Fix_UCGLD_Langevin::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str, "t_target") == 0) { return &t_target; }
  return nullptr;
}

/* ----------------------------------------------------------------------
   memory usage of tally array
------------------------------------------------------------------------- */

double Fix_UCGLD_Langevin::memory_usage()
{
  double bytes = 0.0;
  if (tforce) bytes += (double) atom->nmax * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   allocate atom-based array for franprev
------------------------------------------------------------------------- */

void Fix_UCGLD_Langevin::grow_arrays(int nmax)
{
  memory->grow(franprev, nmax, 3, "fix_langevin:franprev");
  memory->grow(lv, nmax, 3, "fix_langevin:lv");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based array
------------------------------------------------------------------------- */

void Fix_UCGLD_Langevin::copy_arrays(int i, int j, int /*delflag*/)
{
  franprev[j][0] = franprev[i][0];
  franprev[j][1] = franprev[i][1];
  franprev[j][2] = franprev[i][2];
  lv[j][0] = lv[i][0];
  lv[j][1] = lv[i][1];
  lv[j][2] = lv[i][2];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based array for exchange with another proc
------------------------------------------------------------------------- */

int Fix_UCGLD_Langevin::pack_exchange(int i, double *buf)
{
  int n = 0;
  buf[n++] = franprev[i][0];
  buf[n++] = franprev[i][1];
  buf[n++] = franprev[i][2];
  buf[n++] = lv[i][0];
  buf[n++] = lv[i][1];
  buf[n++] = lv[i][2];
  return n;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based array from exchange with another proc
------------------------------------------------------------------------- */

int Fix_UCGLD_Langevin::unpack_exchange(int nlocal, double *buf)
{
  int n = 0;
  franprev[nlocal][0] = buf[n++];
  franprev[nlocal][1] = buf[n++];
  franprev[nlocal][2] = buf[n++];
  lv[nlocal][0] = buf[n++];
  lv[nlocal][1] = buf[n++];
  lv[nlocal][2] = buf[n++];
  return n;
}
