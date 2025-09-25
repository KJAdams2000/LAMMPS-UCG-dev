
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

   This code modified the RLE-UCG theory to accommodate the Bethe approxi-
   mation, instead of the mean-field approximation used originally.

   The Bethe approximation implemented here is only good for two-state 
   systems.

   This pair style requests a HALF neighbor list. This pair style DOES NOT
   use the density as CV to compute the prior single point probabilities.

   As a consequence, no CV backforce is computed in this pair style,
   and the execution only require one loop over the half neighbor list.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(table_ucg_bethe, PairTable_UCG_Bethe)

#else

#ifndef LMP_PAIR_TABLE_UCG_BETHE_H
#define LMP_PAIR_TABLE_UCG_BETHE_H

#include "pair.h"
// #include "pair_table.h"

namespace LAMMPS_NS {

class PairTable_UCG_Bethe : public Pair {
      // In principle, we can also inherit from PairTable 
      // But all member functions may need to be overridden
      // which is useless.
      // So we directly copy and modify.
    public:  

        PairTable_UCG_Bethe(class LAMMPS *);
        virtual ~PairTable_UCG_Bethe();

        virtual void compute(int, int);

        void settings(int, char **); 
        // reads settings from the input pair_coeff command.

        void coeff(int, char **);
        void init_style();
        // Initialization command specific to this pair style.
        // We will check the atom_style here.
        // Atom style should be "ucg" with BOTH ucgstate annd ucgl entries.
        // Refer to src/SPIN/pair_spin.cpp

        double init_one(int, int);
        
        // same as PairTable, 
        void write_restart(FILE *);
        void read_restart(FILE *);
        void write_restart_settings(FILE *);
        void read_restart_settings(FILE *);
        double single(int, int, int, int, double, double, double, double &);
        void *extract(const char *, int &);

        enum{LOOKUP, LINEAR, SPLINE, BITMAP};
        enum{CHEMICAL_POTENTIAL, CHEMICAL_POTENTIAL_NOISE, UCGL, UCGP};
        enum{MF, BETHE};
        

    protected:

        // Normal pair table parameters and functions, directly copied from PairTable

        int tabstyle, tablength;
        struct Table {
            int ninput, rflag, fpflag, match, ntablebits;
            int nshiftbits, nmask;
            double rlo, rhi, fplo, fphi, cut;
            double *rfile, *efile, *ffile;
            double *e2file, *f2file;
            double innersq, delta, invdelta, deltasq6;
            double *rsq, *drsq, *e, *de, *f, *df, *e2, *f2;
        };
        int ntables;
        Table *tables;

        int **tabindex;

        virtual void allocate();
        void read_table(Table *, char *, char *);
        void param_extract(Table *, char *);
        void bcast_table(Table *);
        void spline_table(Table *);
        virtual void compute_table(Table *);
        void null_table(Table *);
        void free_table(Table *);
        static void spline(double *, double *, int, double, double, double *);
        static double splint(double *, double *, double *, int, double);

        // Communication functions.
        // TODO: may be changed to accomodate the new UCG AtomVec.
        // int pack_reverse_comm(int, int, double *);
        // void unpack_reverse_comm(int, int *, double *);
        // int pack_forward_comm(int, int *, double *, int, int *);
        // void unpack_forward_comm(int, int, double *);

        // UCG
        int pseudo_flag; // 0 for pseudo-likelihood update, 1 for likelihood update (Full SCE).
        int prior_flag; // 0 to use prior probabilities from chemical potentials, 1 to add some noise to the results from chemical potentials
        int method_flag;
        int seed; // random seed for the noise to generate prior probabilities
        double noise_level;
        double T, kT; // thermostatting temperature
        int nmax; // Maximum number of atoms a processor has.
        int n_formal_types; // total formal types, should be equal to atom->ntypes
        int n_actual_types;
        int max_states_per_type;
        int *n_states_per_type; // size (n_actual_types,)
        int *actual_types_from_formal;  // Mapper from formal type to actual type. size (n_formal_types,)
        int **formal_types_from_actual;  // Mapper from actual type to formal types. size (n_actual_types, max_states_per_type)
        double *chemical_potentials;
        double **prior_prob_from_type;
        class RanMars *random; // Random number generator for the noise
        
        void read_state_settings(const char *);
        
        double **prior_prob;
        // double **softmax_scores;
        double **post_prob;

   }; // class PairTable_UCG_Bethe
   
} // namespace LAMMPS_NS


#endif
#endif