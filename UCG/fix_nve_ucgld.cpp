#include "fix_nve_ucgld.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "respa.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace FixConst;

FixNVE_UCGLD::FixNVE_UCGLD(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
    
    auto plain_style = utils::strip_style_suffix(style, lmp);
    if (utils::strmatch(style, "^nve/ucgld") && narg > 3)
        error->all(FLERR, 3, "Unsupported additional arguments for fix {}", style);

    dynamic_group_allow = 1;
    time_integrate = 1;
}

FixNVE_UCGLD::~FixNVE_UCGLD() {
    // Destructor implementation if needed
    // reserved later for check
}

int FixNVE_UCGLD::setmask() {
    int mask = 0;
    mask |= INITIAL_INTEGRATE;
    mask |= FINAL_INTEGRATE;
    mask |= INITIAL_INTEGRATE_RESPA;
    mask |= FINAL_INTEGRATE_RESPA;
    return mask;
}

void FixNVE_UCGLD::init() {
    dtv = update->dt; // also used for lambda velocities
    dtf = 0.5 * update->dt * force->ftm2v; // also used for lambda forces

    if (utils::strmatch(update->integrate_style,"^respa"))
    step_respa = (dynamic_cast<Respa *>(update->integrate))->step;
}

void FixNVE_UCGLD::initial_integrate(int vflag) {
    double dtfm, dtflm;
    
    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;
    double *rmass = atom->rmass;
    double *mass = atom->mass;

    double *lmdm = atom->ucgml; // pseudo-mass for lambda variables
    double *lmd = atom->ucgl;
    double *lmdv = atom->ucgvl;
    double *lmdf = atom->ucgforce;

    int *type = atom->type;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    if (igroup == atom->firstgroup) nlocal = atom->nfirst;

    if (rmass) { // if per-atom rmass is defined, for example, for ellipsoids and rigid body particles
        for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
                // Apply Verlet integration for this atom
                dtfm = dtf / rmass[i];
                v[i][0] += dtfm * f[i][0];
                v[i][1] += dtfm * f[i][1];
                v[i][2] += dtfm * f[i][2];
                x[i][0] += dtv * v[i][0];
                x[i][1] += dtv * v[i][1];
                x[i][2] += dtv * v[i][2];

                // Apply Lambda Verlet integration for this atom
                dtflm = dtf / lmdm[i];
                lmdv[i] += dtflm * lmdf[i];
                lmd[i] += dtv * lmdv[i];
            }
        }
    } else { // for per-type mass
        for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
                // Apply Verlet integration for this atom
                dtfm = dtf / mass[type[i]];
                v[i][0] += dtfm * f[i][0];
                v[i][1] += dtfm * f[i][1];
                v[i][2] += dtfm * f[i][2];
                x[i][0] += dtv * v[i][0];
                x[i][1] += dtv * v[i][1];
                x[i][2] += dtv * v[i][2];

                // Apply Lambda Verlet integration for this atom
                dtflm = dtf / lmdm[i];
                lmdv[i] += dtflm * lmdf[i];
                lmd[i] += dtv * lmdv[i];
            }
        }
    }
}


void FixNVE_UCGLD::final_integrate() {
    double dtfm, dtflm;
    
    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;
    double *rmass = atom->rmass;
    double *mass = atom->mass;

    double *lmdm = atom->ucgml; // pseudo-mass for lambda variables
    double *lmd = atom->ucgl;
    double *lmdv = atom->ucgvl;
    double *lmdf = atom->ucgforce;

    int *type = atom->type;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    if (igroup == atom->firstgroup) nlocal = atom->nfirst;

    if (rmass) { // if per-atom rmass is defined, for example, for ellipsoids and rigid body particles
        for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
                // Apply Verlet integration for this atom
                dtfm = dtf / rmass[i];
                v[i][0] += dtfm * f[i][0];
                v[i][1] += dtfm * f[i][1];
                v[i][2] += dtfm * f[i][2];

                // Apply Lambda Verlet integration for this atom
                dtflm = dtf / lmdm[i];
                lmdv[i] += dtflm * lmdf[i];
            }
        }
    } else { // for per-type mass
        for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
                // Apply Verlet integration for this atom
                dtfm = dtf / mass[type[i]];
                v[i][0] += dtfm * f[i][0];
                v[i][1] += dtfm * f[i][1];
                v[i][2] += dtfm * f[i][2];

                // Apply Lambda Verlet integration for this atom
                dtflm = dtf / lmdm[i];
                lmdv[i] += dtflm * lmdf[i];
            }
        }
    }
}

void FixNVE_UCGLD::initial_integrate_respa(int vflag, int ilevel, int /*iloop*/)
{
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  // innermost level - NVE update of v and x
  // all other levels - NVE update of v

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVE_UCGLD::final_integrate_respa(int ilevel, int /*iloop*/)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVE_UCGLD::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}