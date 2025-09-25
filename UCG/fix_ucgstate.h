#ifdef FIX_CLASS

FixStyle(ucgstate,FixUCGState);

#else

#ifndef LMP_FIX_UCGSTATE_H
#define LMP_FIX_UCGSTATE_H

#include "fix.h"

namespace LAMMPS_NS {
    class FixUCGState: public Fix {
    public:
        FixUCGState(class LAMMPS *, int, char **);
        ~FixUCGState();
        int setmask() override;
        // void pre_force(int) override;
        void post_force(int) override;
        // int state_update(int index);
        void post_force_respa(int, int, int) override;
        void min_post_force(int) override;
        void setup(int) override;
    private:
        double kT, T;
        int ld_flag, mc_flag, mc_seed; double mc_rate;
        double **exp_softmax_scores;
        class RanMars *random;
    };
}

#endif
#endif