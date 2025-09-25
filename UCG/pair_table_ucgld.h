#ifdef PAIR_CLASS

PairStyle(table_ucgld, PairTable_UCGLD)

#else

#ifndef LMP_PAIR_TABLE_UCGLD_H
#define LMP_PAIR_TABLE_UCGLD_H

#include "pair.h"

namespace LAMMPS_NS {

class PairTable_UCGLD : public Pair {
      // In principle, we can also inherit from PairTable 
      // But all member functions may need to be overridden
      // which is useless.
      // So we directly copy and modify.

    public:  

        PairTable_UCGLD(class LAMMPS *); // yes
        virtual ~PairTable_UCGLD(); // yes

        virtual void compute(int, int); // yes

        void settings(int, char **); // yes
        // reads settings from the input pair_coeff command.

        
        void coeff(int, char **); // yes
        // void init_style();
        // I think this pair style does not require a separate init_style() function.
        // In RLE-UCG, this init_style() requests a FULL neighbor list.
        // But in Lambda Dynamics, only HALF neighbor list is required.
        // Additionally, in Bethe-UCG, this init_style() is used to obtain the thermostat temperature.
        // But in Lambda Dynamics, thermostat temperature is not required.
        
        void init_style();
        // Copy these
        double init_one(int, int);
        
        void write_restart(FILE *); // just copy
        void read_restart(FILE *); // just copy
        void write_restart_settings(FILE *); // just copy
        void read_restart_settings(FILE *); // just copy
        double single(int, int, int, int, double, double, double, double &); // can copy, can write on our own
        void *extract(const char *, int &); // just copy

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
        
        // Copy these functions from pair_table.cpp to pair_table_ucgld.cpp
        // They share the same functions as PairTable.
        // You can also explore ways to inherit these functions from PairTable.
        
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
        // This pairstyle do not require additional communication. Commenting these out.
        // int pack_reverse_comm(int, int, double *);
        // void unpack_reverse_comm(int, int *, double *);
        // int pack_forward_comm(int, int *, double *, int, int *);
        // void unpack_forward_comm(int, int, double *);

        // UCG-specific variables and functions

        double T, kT; // thermostatting temperature
        int nmax; // Maximum number of atoms a processor has.
        int n_formal_types; // total formal types, should be equal to atom->ntypes
        int n_actual_types;
        int max_states_per_type;
        int *n_states_per_type; // size (n_actual_types,)
        int *actual_types_from_formal;  // Mapper from formal type to actual type. size (n_formal_types,)
        int **formal_types_from_actual;  // Mapper from actual type to formal types. size (n_actual_types, max_states_per_type)
        double *chemical_potentials;
        
        void read_state_settings(const char *);

   }; // class PairTable_UCGLD
   
} // namespace LAMMPS_NS


#endif
#endif