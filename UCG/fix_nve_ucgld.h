/*

Compare to fix nve or fix nve/spin to write this script.

will need:

fix->initial_integrate() to do first step of verlet
fix->final_integrate() to do second step of verlet
also initial_integrate_respa() and final_integrate_respa() for respa

fix->pre_force() to clear out atom->ucgforce and atom->ucgforce_long

*/

#ifdef FIX_CLASS
FixStyle(nve/ucgld, FixNVE_UCGLD);
#else 

#ifndef LMP_FIX_NVE_UCGLD_H
#define LMP_FIX_NVE_UCGLD_H

#include "fix.h"

namespace LAMMPS_NS {
    class FixNVE_UCGLD : public Fix {
    public:
        FixNVE_UCGLD(class LAMMPS *, int, char **);
        virtual ~FixNVE_UCGLD();
        virtual int setmask();
        virtual void init();
        // virtual void pre_force(int);
        virtual void initial_integrate(int);
        virtual void final_integrate();
        virtual void initial_integrate_respa(int, int, int);
        virtual void final_integrate_respa(int, int);
        virtual void reset_dt();

    protected:
        double dtv, dtf;
        double *step_respa;
        int mass_require;
    };
}

#endif

#endif 