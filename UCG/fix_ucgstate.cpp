/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Weizhi Xue (University of Chicago)

   Updates UCG state variables based on Posterior Probabilities.
------------------------------------------------------------------------- */

#include "fix_ucgstate.h"
#include "comm.h"
#include "modify.h"
#include "memory.h"
#include "atom.h"
#include "error.h"
#include "force.h"
#include "respa.h"
#include "update.h"
#include "random_mars.h"

using namespace LAMMPS_NS;
using namespace FixConst;

FixUCGState::FixUCGState(LAMMPS *lmp, int narg, char **arg): 
    Fix(lmp, narg, arg), random(nullptr) 
    {
    /* Usage:
    fix fix_id group_id ucgstate [ld]
    where ld is optional, if specified, then this fix will not update ucgstate and ucgl post force, 
    because that will be done by the ld integrator.
    */
    auto plain_style = utils::strip_style_suffix(style, lmp);
    ld_flag = 0;
    mc_flag = 0;
    mc_rate = 0.01;
    if (utils::strmatch(plain_style, "^ucgstate$") && narg > 6)
        error->all(FLERR, 3, "Too many arguments for fix {}", style);
    if (!atom->ucg_flag) {
        error->all(FLERR, 1, "fix ucgstate requires ucg atom style");
    }
    if (narg > 3) {
        if (utils::strmatch(arg[3], "ld")) ld_flag = 1;
        // if using "fix 1 all ucgstate ld", then ld_flag = 1, ucgstate is NOT updated post force (will be updated by integrator)
        // Then, this fix only handles the posterior probabilities.
        else if (utils::strmatch(arg[3], "mc"))
        {
            mc_flag = 1;
            if (narg == 4) error->all(FLERR, 1, "fix ucgstate mc requires seed and rate information");
            if (narg == 5) error->all(FLERR, 1, "fix ucgstate mc requires rate information");
            else {
                mc_seed = utils::inumeric(FLERR, arg[4], false, lmp);
                random = new RanMars(lmp, mc_seed + comm->me);
                mc_rate = utils::numeric(FLERR, arg[5], false, lmp);
            }
        }
        else error->all(FLERR, 1, "Unknown argument for fix {}: {}", style, arg[3]);
    }
    
    dynamic_group_allow = 1;
    time_integrate = 0; // This fix does not perform time integration.
    exp_softmax_scores = nullptr;
}

FixUCGState::~FixUCGState() {
    if (copymode) return;
    memory->destroy(exp_softmax_scores);
    delete random;
}

int FixUCGState::setmask() {
    int mask = 0;
    mask |= POST_FORCE;
    mask |= POST_FORCE_RESPA;
    mask |= MIN_POST_FORCE;
    return mask;
}

void FixUCGState::post_force(int vflag) {

    double *ucgl = atom->ucgl;
    int *ucgstate = atom->ucgstate;
    double *ucgp = atom->ucgp;
    int *num_ucgstates = atom->num_ucgstates;
    double **softmax_scores = atom->ucgsoftmaxscores;
    int nlocal = atom->nlocal;
    double softmax_denom;
    double mc_factor, mc_rand;

    for (int i = 0; i < nlocal; i++) {
        if (num_ucgstates[i] == 1) {
            if (!ld_flag) {ucgstate[i] = 0;}
            ucgp[i] = 1.0;
        }
        else {
            softmax_denom = 0.0;
            for (int si = 0; si < num_ucgstates[i]; si++){
                exp_softmax_scores[i][si] = std::exp(std::min(softmax_scores[i][si], 700.0));
                softmax_denom += exp_softmax_scores[i][si];
            }
            ucgp[i] = std::min(1.0-1e-6, std::max(1e-6, exp_softmax_scores[i][1] / softmax_denom));
            
            if (!ld_flag) {
                if (mc_flag) {
                    if (ucgstate[i] == 0) mc_factor = ucgp[i] / (1.0 - ucgp[i]);
                    else mc_factor = (1.0 - ucgp[i]) / ucgp[i];
                    mc_factor = std::min(mc_factor, 1.0) * mc_rate;
                    mc_rand = random->uniform();
                    if (mc_rand < mc_factor) {
                        ucgstate[i] = 0; // switch to state 0
                    } else {
                        ucgstate[i] = 1; // stay in state 1
                    }
                }
                else {
                    ucgstate[i] = static_cast<int>(std::round(ucgp[i]));
                }
            }
        }

       if (!ld_flag) { ucgl[i] = ucgp[i]; }
    }
}

void FixUCGState::post_force_respa(int vflag, int ilevel, int /*iloop*/) {
    post_force(vflag);
}

void FixUCGState::min_post_force(int vflag) {
    post_force(vflag);
}

void FixUCGState::setup(int vflag) {
    // Will need kBT information from thermostat fixes
    // Therefore, this fix needs to be set up AFTER thermostat fixes.
    double *pT = nullptr;
    int pdim;
    
    for (int ifix = 0; ifix < modify->nfix; ifix ++) {
        pT = (double*) modify->fix[ifix]->extract("t_target", pdim);
        if(pT) { T = (*pT); break; }
    }
    if (pT == nullptr) {
        error->all(FLERR, "FixUCGState requires a thermostat fix BEFORE ITSELF to set the target temperature T.");
    }
    kT = force->boltz * T;
    // utils::logmesg(lmp, "FixUCGState: kT = {}\n", kT);

    memory->grow(exp_softmax_scores, atom->nmax, atom->max_ucgstates, "fix_ucgstate:exp_softmax_scores");
    for (int i = 0; i < atom->nmax; i++) {
        for (int j = 0; j < atom->max_ucgstates; j++) {
            exp_softmax_scores[i][j] = 1.0; // initialize to 1.0
        }
    }

    // By calling post_force here, we can ensure that the posterior probabilities 
    // are initialized correctly at step 0 (i.e., when calling Verlet::setup)
    // This can give an evaluation of the posterior probabilities at step 0,
    // which can enable UCG state assignment with rerun.
    post_force(vflag);

}