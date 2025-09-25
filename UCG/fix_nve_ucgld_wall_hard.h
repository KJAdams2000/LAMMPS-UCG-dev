/*

Compare to fix nve or fix nve/spin to write this script.

will need:

fix->initial_integrate() to do first step of verlet
fix->final_integrate() to do second step of verlet
also initial_integrate_respa() and final_integrate_respa() for respa

fix->pre_force() to clear out atom->ucgforce and atom->ucgforce_long

*/

#ifdef FIX_CLASS
FixStyle(nve/ucgld/wall/hard, FixNVE_UCGLD_Wall_Hard);
#else 

#ifndef LMP_FIX_NVE_UCGLD_WALL_HARD_H
#define LMP_FIX_NVE_UCGLD_WALL_HARD_H

#include "fix.h"

namespace LAMMPS_NS {
    class FixNVE_UCGLD_Wall_Hard : public Fix {
    public:
        FixNVE_UCGLD_Wall_Hard(class LAMMPS *, int, char **);
        virtual ~FixNVE_UCGLD_Wall_Hard();
        virtual int setmask();
        virtual void init();
        // virtual void pre_force(int);
        virtual void initial_integrate(int);
        virtual void final_integrate();
        virtual void initial_integrate_respa(int, int, int);
        virtual void final_integrate_respa(int, int);
        virtual void reset_dt();

        virtual void post_force(int vflag);

    protected:
        double dtv, dtf;
        double *step_respa;
        int mass_require;
        int bias_potential_flag = 0;

        double bias_force(double, double);
        double barrier;
    };
}

#endif

#endif 
