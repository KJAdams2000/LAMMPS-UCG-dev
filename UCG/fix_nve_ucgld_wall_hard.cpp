#include "fix_nve_ucgld_wall_hard.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "respa.h"
#include "update.h"

using namespace LAMMPS_NS;
using namespace FixConst;

FixNVE_UCGLD_Wall_Hard::FixNVE_UCGLD_Wall_Hard(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg) {
    
    auto plain_style = utils::strip_style_suffix(style, lmp);
    if (utils::strmatch(style, "^nve/ucgld/wall/hard") && narg > 5)
        error->all(FLERR, 3, "Unsupported additional arguments for fix {}", style);

    dynamic_group_allow = 1;
    time_integrate = 1;
    bias_potential_flag = 0;
    int iarg = 3; // start after "nve/ucgld/wall/hard"
    barrier = 0.1; // default barrier height
    while (iarg < narg) {
        if (utils::strmatch(arg[iarg], "bias_potential")) {
            bias_potential_flag = 1;
            iarg++;
            if (iarg < narg)
                barrier = utils::numeric(FLERR, arg[iarg], false, lmp);
            iarg++;
        } else {
            error->all(FLERR, "Unknown argument for fix {}", style);
        }
    }    
}

FixNVE_UCGLD_Wall_Hard::~FixNVE_UCGLD_Wall_Hard() {
    // Destructor implementation if needed
    // reserved later for check
}

int FixNVE_UCGLD_Wall_Hard::setmask() {
    int mask = 0;
    mask |= INITIAL_INTEGRATE;
    mask |= FINAL_INTEGRATE;
    mask |= INITIAL_INTEGRATE_RESPA;
    mask |= FINAL_INTEGRATE_RESPA;
    if (bias_potential_flag) {
        mask |= POST_FORCE; // to apply bias force after main force calculations
    }
    return mask;
}

void FixNVE_UCGLD_Wall_Hard::init() {
    dtv = update->dt; // also used for lambda velocities
    dtf = 0.5 * update->dt * force->ftm2v; // also used for lambda forces

    if (utils::strmatch(update->integrate_style,"^respa"))
    step_respa = (dynamic_cast<Respa *>(update->integrate))->step;
}

void FixNVE_UCGLD_Wall_Hard::initial_integrate(int vflag) {
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
    int *ucgstate = atom->ucgstate;

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

                // set lambda state based on the value of lmd
                if (lmd[i] < 0.5) {
                    atom->ucgstate[i] = 0; 
                }
                else {
                    atom->ucgstate[i] = 1; 
                }
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

                // set lambda state based on the value of lmd
                if (lmd[i] < 0.5) {
                    ucgstate[i] = 0; 
                }
                else {
                    ucgstate[i] = 1; 
                }
                
            }
        }
    }
}


void FixNVE_UCGLD_Wall_Hard::final_integrate() {
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

                if (lmd[i] < 0.0) {
                    lmd[i] = -lmd[i]; // Reflect lambda if lambda is negative
                    lmdv[i] = -lmdv[i]; // Reflect velocity if lambda is negative
                } else if (lmd[i] > 1.0) {
                    lmd[i] = 2.0 - lmd[i]; // Reflect lambda if it exceeds 1.0
                    lmdv[i] = -lmdv[i]; // Reflect velocity if lambda exceeds 1.0
                }
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

                if (lmd[i] < 0.0) {
                    lmd[i] = -lmd[i]; // Reflect lambda if lambda is negative
                    lmdv[i] = -lmdv[i]; // Reflect velocity if lambda is negative
                } else if (lmd[i] > 1.0) {
                    lmd[i] = 2.0 - lmd[i]; // Reflect lambda if it exceeds 1.0
                    lmdv[i] = -lmdv[i]; // Reflect velocity if lambda exceeds 1.0
                }
            }
        }
    }
}

void FixNVE_UCGLD_Wall_Hard::initial_integrate_respa(int vflag, int ilevel, int /*iloop*/)
{
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  // innermost level - NVE update of v and x
  // all other levels - NVE update of v

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVE_UCGLD_Wall_Hard::final_integrate_respa(int ilevel, int /*iloop*/)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixNVE_UCGLD_Wall_Hard::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

double FixNVE_UCGLD_Wall_Hard::bias_force(double lmd, double H = 0.1) {
    // potential: (798*x**10-x**2+0.1)*10*H
    double x = lmd - 0.5;
    return (-7980 * x * x * x * x * x * x * x * x * x + 2 * x) 
           * 10 * H; // bias force based on the potential
}

void FixNVE_UCGLD_Wall_Hard::post_force(int vflag)
{
    // This method can be used to apply additional forces or modifications after the main force calculations
    // For example, applying a bias force based on the lambda variable
    double *lmdf = atom->ucgforce;
    double *lmd = atom->ucgl;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    if (igroup == atom->firstgroup) nlocal = atom->nfirst;

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            lmdf[i] += bias_force(lmd[i], barrier);
        }
    }
}
