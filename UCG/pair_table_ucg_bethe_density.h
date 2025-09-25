
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

   This pair style requests a FULL neighbor list.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(table_ucg_bethe_density, PairTable_UCG_Bethe_Density)

#else

#ifndef LMP_PAIR_TABLE_UCG_BETHE_DENSITY_H
#define LMP_PAIR_TABLE_UCG_BETHE_DENSITY_H

#include "pair.h"

namespace LAMMPS_NS {

class PairTable_UCG_Bethe_Density : public Pair {
      // In principle, we can also inherit from PairTable 
      // But all member functions may need to be overridden
      // which is useless.
      // So we directly copy and modify.
    public:  

        PairTable_UCG_Bethe_Density(class LAMMPS *);
        virtual ~PairTable_UCG_Bethe_Density();

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
        // Initialization for one pair of 
        void write_restart(FILE *);
        void read_restart(FILE *);
        void write_restart_settings(FILE *);
        void read_restart_settings(FILE *);
        double single(int, int, int, int, double, double, double, double &);
        void *extract(const char *, int &);

        enum{LOOKUP, LINEAR, SPLINE, BITMAP};
        // This line is under "protected:" in pair_table_rleucg_interface.h (version 17Nov16)
        // but is not protected in the current LAMMPS version (2Apr2025).

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
      //   int pack_reverse_comm(int, int, double *);
      //   void unpack_reverse_comm(int, int *, double *);
      //   int pack_forward_comm(int, int *, double *, int, int *);
      //   void unpack_forward_comm(int, int, double *);

        // UCG

        double T, kT; // thermostatting temperature
        int nmax; // Maximum number of atoms a processor has, including ghosts.
        int *real_jnum; // the number of neighbors for each atom within cutoff distance.
        int n_formal_types; // total formal types, should be equal to atom->ntypes
        int n_actual_types;
        int max_states_per_type;
        int *n_states_per_type; // size (n_actual_types,)
        int *actual_types_from_formal;  // Mapper from formal type to actual type. size (n_formal_types,)
        int **formal_types_from_actual;  // Mapper from actual type to formal types. size (n_actual_types, max_states_per_type)
        double *chemical_potentials;

        void read_state_settings(const char *);
        
        // For density single point probability calculations.
        int *use_state_entropy;
        int *use_density;
        double *cv_thresholds;
        double *threshold_radii;
        double compute_proximity_function(int, double);
        double compute_proximity_function_der(int, double);
        void threshold_prob_and_partial_from_cv(int, double, double&, double&);

        double **prior_prob, **prior_prob_partial, **prior_prob_force;
        double **prior_cv_backforce;
        // corresponding to the.
        // If no_density is set, 
        double **softmax_scores;
        double **post_prob;
        // This will be communicated to atom->ucgl in this pair style.

        /*
        Read a states settings file to establish the mapping of 
        all atom types -> actual types and UCG types.

        Expected input format: 

        4 6 2    --> 4 formal types, 6 actual types, maximum 2 states per type.

        1 1      --> Normal CG types. They do not require further specifications.
        2 1
        3 1
        4 2     --> A UCG type with 2 states (formal type 4 and 6). 3 lines are expected to follow.
        4 6 density no_entropy                            
        1.52 3.89 0.46 0.66 4           #--> This line has parameters for 'density' style single point probability calculations.
        0.0 0.0   --> Chemical potentials.
        5 2 
        5 7 single  entropy          #--> A UCG type with 2 states (formal type 5 and 7).
                                     #--> This line defines the single point probability of state 1 for the 'single' style. 
                                        #--> The two-point probabilities will be calculated based on Bethe approximation. 
        0.0 0.8   --> Chemical potentials.
        We do expect all UCG types to be defined AFTER all normal CG types.

        */

        /**/

   };
   

}

#endif
#endif