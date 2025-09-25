

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

   Implements the Lambda Dynamics UCG pair style.
   Requires atom_style ucg. 
   This pair style is tailored for 2-state systems. 
   
   When doing UCG LD, should be used in tandem with fix ucgld.

   For multistate systems, use UCG-RLE for most cases,
   or UCG-TAP for a TAP approximation version. 
   The author believes that strongly correlated 3-state and more systems 
   are not necessary in terms of CG theory. 
   More internal states can be avoided by using 
   better resolution of the CG force field.
   
------------------------------------------------------------------------- */


#include "pair_table_ucgld.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "info.h"
#include "memory.h"
#include "neigh_list.h"
#include "table_file_reader.h"
#include "tokenizer.h"

#include "neighbor.h"     
#include "modify.h"       
#include "update.h"       
#include "fix.h" 

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

enum { NONE, RLINEAR, RSQ, BMP };
#define MAXLINE 1024 // Maximum line length for reading files

static constexpr double EPSILONR = 1.0e-6;

PairTable_UCGLD::PairTable_UCGLD(LAMMPS *lmp) : Pair(lmp) 
{
    ntables = 0;
    nmax = 0;
    tables = nullptr;
    unit_convert_flag = utils::get_supported_conversions(utils::ENERGY);

    // Double pointers may need to be initialized to nullptr here
    formal_types_from_actual = nullptr;
}

PairTable_UCGLD::~PairTable_UCGLD()
{
    if (copymode) return;

    for (int m = 0; m < ntables; m++) free_table(&tables[m]);
    memory->sfree(tables);

    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
        memory->destroy(tabindex);

        memory->destroy(n_states_per_type);
        memory->destroy(actual_types_from_formal);
        memory->destroy(formal_types_from_actual);

        memory->destroy(chemical_potentials);
    }
}

void PairTable_UCGLD::allocate()
{
  allocated = 1;
  const int nt = n_formal_types + 1;

  // Note that in our UCG_Bethe implementation,
  // atom->ntypes is about ACTUAL types.
  // We allocate FORMAL types.

  memory->create(setflag, nt, nt, "pair:setflag");
  memory->create(cutsq, nt, nt, "pair:cutsq");
  memory->create(tabindex, nt, nt,"pair:tabindex");

  memset(&setflag[0][0], 0, nt*nt*sizeof(int));
  memset(&cutsq[0][0], 0, nt*nt*sizeof(double));
  memset(&tabindex[0][0], 0, nt*nt*sizeof(int));
}

void PairTable_UCGLD::compute(int eflag, int vflag) {
    // from pair_table.cpp
    int i, j, ii, jj, inum, jnum, itype, jtype, itable, istate, jstate;
    // different from UCG-RLE implementation, itype and jtype are ACTUAL types.
    double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
    double rsq, factor_lj, fraction, value, a, b;
    int *ilist, *jlist, *numneigh, **firstneigh;
    Table *tb;

    int si, sj, sk; // variable to loop through substates
    int itype_si, jtype_sj, ktype_sk; // Formal types of i in substate s_i, j in substate s_j, and k in substate s_k
    // from Bethe approximation
    double jnum_f; 
   //  double pi0, pi1, pj0, pj1, pij00, pij11, pij10, pij01, Jij, aij, bij, Qij, Dij;
   double mui, muj; // chemical potentials for i and j

    // double f00x, f00y, f00z;
    // double f01x, f01y, f01z;
    // double f10x, f10y, f10z;
    // double f11x, f11y, f11z;

    double u00, u01, u10, u11;
    double fpair00, fpair01, fpair10, fpair11;
    double ldi, ldj; // lambda_i, lambda_j
    double temp;

    union_int_float_t rsq_lookup;
    int tlm1 = tablength - 1;

    evdwl = 0.0;
    ev_init(eflag, vflag);

    double **x = atom->x;
    double **f = atom->f;
    int *type = atom->type;

    int *ucgstate = atom->ucgstate;
    int *num_ucgstates = atom->num_ucgstates;
    double *ucgf = atom->ucgforce; // Lambda forces. Because lambda is a scaler, ucgf is a 1D array.
    double *ucgl = atom->ucgl;
    double *ucgp = atom->ucgp; // UCG state posterior probabilities, only logged in reverse communication.
    double **softmax_scores = atom->ucgsoftmaxscores;
    
    int nlocal = atom->nlocal;
    double *special_lj = force->special_lj;
    int newton_pair = force->newton_pair;
    int nall = atom->nlocal + atom->nghost;
    int comm_flag = 0;

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    // Initialize ucgf[i] to
    // This assumes EVERY atom is in interaction with one another,
    // so directly editing ucgf[i] here on every proc will cover ALL atoms.
    // Potential non-cover issues can be avoided by increasing the neighbor list cutoff.

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      itype = type[i];
      num_ucgstates[i] = n_states_per_type[itype];
      // pre-add atom->ucgforce with chemical potential differences
      if (n_states_per_type[itype] > 1) {
         mui = chemical_potentials[formal_types_from_actual[itype][1]] - chemical_potentials[formal_types_from_actual[itype][0]];
         ucgf[i] -= mui;
         softmax_scores[i][1] -= mui / kT;
      }
   }

   // compute two-body interactions

    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        itype = type[i];
        jlist = firstneigh[i];
        jnum = numneigh[i];
        jnum_f = 1. - jnum;
        itype_si = formal_types_from_actual[itype][0];
        istate = ucgstate[i];
        ldi = ucgl[i];

        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        // Loop over neighbors

        for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            factor_lj = special_lj[sbmask(j)];
            j &= NEIGHMASK;
            jtype = type[j];
            jtype_sj = formal_types_from_actual[jtype][0];
            jstate = ucgstate[j];
            ldj = ucgl[j];

            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            rsq = delx * delx + dely * dely + delz * delz;

            if (rsq < cutsq[itype][jtype]) {
                // Calculate two-point energies, energy forces, and probabilities.
                // Four scenarios: 

                // Scenario 1. i, j are non-UCG atoms.

                if (n_states_per_type[itype] == 1 && n_states_per_type[jtype] == 1) {
                    tb = &tables[tabindex[itype][jtype]];

                    // ------- For one table -------
                    if (rsq < tb->innersq)
                        error->one(FLERR, "Pair distance < table inner cutoff: ijtype {} {} dist {}", itype,
                                    jtype, sqrt(rsq));
                    if (tabstyle == LOOKUP) {
                        itable = static_cast<int>((rsq - tb->innersq) * tb->invdelta);
                        if (itable >= tlm1)
                            error->one(FLERR, "Pair distance > table outer cutoff: ijtype {} {} dist {}", itype,
                                    jtype, sqrt(rsq));
                        fpair = factor_lj * tb->f[itable];
                    } else if (tabstyle == LINEAR) {
                        itable = static_cast<int>((rsq - tb->innersq) * tb->invdelta);
                        if (itable >= tlm1)
                            error->one(FLERR, "Pair distance > table outer cutoff: ijtype {} {} dist {}", itype,
                                    jtype, sqrt(rsq));
                        fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
                        value = tb->f[itable] + fraction * tb->df[itable];
                        fpair = factor_lj * value;
                    } else if (tabstyle == SPLINE) {
                        itable = static_cast<int>((rsq - tb->innersq) * tb->invdelta);
                        if (itable >= tlm1)
                            error->one(FLERR, "Pair distance > table outer cutoff: ijtype {} {} dist {}", itype,
                                    jtype, sqrt(rsq));
                        b = (rsq - tb->rsq[itable]) * tb->invdelta;
                        a = 1.0 - b;
                        value = a * tb->f[itable] + b * tb->f[itable + 1] +
                            ((a * a * a - a) * tb->f2[itable] + (b * b * b - b) * tb->f2[itable + 1]) *
                                tb->deltasq6;
                        fpair = factor_lj * value;
                    } else {
                        rsq_lookup.f = rsq;
                        itable = rsq_lookup.i & tb->nmask;
                        itable >>= tb->nshiftbits;
                        fraction = (rsq_lookup.f - tb->rsq[itable]) * tb->drsq[itable];
                        value = tb->f[itable] + fraction * tb->df[itable];
                        fpair = factor_lj * value;
                    }

                     if (tabstyle == LOOKUP)
                           evdwl = tb->e[itable];
                     else if (tabstyle == LINEAR || tabstyle == BITMAP)
                           evdwl = tb->e[itable] + fraction * tb->de[itable];
                     else
                           evdwl = a * tb->e[itable] + b * tb->e[itable + 1] +
                              ((a * a * a - a) * tb->e2[itable] + (b * b * b - b) * tb->e2[itable + 1]) *
                              tb->deltasq6;
                     evdwl *= factor_lj;
                    // ------- End one table -------
                }

                // Scenario 2. i is a non-UCG atom, j is a UCG atom.

                else if (n_states_per_type[itype] == 1 && n_states_per_type[jtype] > 1) {
                    for (sj = 0; sj < n_states_per_type[jtype]; sj++) {
                        jtype_sj = formal_types_from_actual[jtype][sj];
                        tb = &tables[tabindex[itype][jtype_sj]];
                        
                        if (rsq < tb->innersq)
                            error->one(FLERR, "Pair distance < table inner cutoff: ijtype {} {} dist {}", itype,
                                        jtype_sj, sqrt(rsq));
                        if (tabstyle == LOOKUP) {
                            itable = static_cast<int>((rsq - tb->innersq) * tb->invdelta);
                            if (itable >= tlm1)
                                error->one(FLERR, "Pair distance > table outer cutoff: ijtype {} {} dist {}", itype,
                                        jtype_sj, sqrt(rsq));
                            fpair = factor_lj * tb->f[itable];
                        } else if (tabstyle == LINEAR) {
                            itable = static_cast<int>((rsq - tb->innersq) * tb->invdelta);
                            if (itable >= tlm1)
                                error->one(FLERR, "Pair distance > table outer cutoff: ijtype {} {} dist {}", itype,
                                        jtype_sj, sqrt(rsq));
                            fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
                            value = tb->f[itable] + fraction * tb->df[itable];
                            fpair = factor_lj * value;
                        } else if (tabstyle == SPLINE) {
                            itable = static_cast<int>((rsq - tb->innersq) * tb->invdelta);
                            if (itable >= tlm1)
                                error->one(FLERR, "Pair distance > table outer cutoff: ijtype {} {} dist {}", itype,
                                        jtype_sj, sqrt(rsq));
                            b = (rsq - tb->rsq[itable]) * tb->invdelta;
                            a = 1.0 - b;
                            value = a * tb->f[itable] + b * tb->f[itable + 1] +
                                ((a * a * a - a) * tb->f2[itable] + (b * b * b - b) * tb->f2[itable + 1]) *
                                    tb->deltasq6;
                            fpair = factor_lj * value;
                        } else {
                            rsq_lookup.f = rsq;
                            itable = rsq_lookup.i & tb->nmask;
                            itable >>= tb->nshiftbits;
                            fraction = (rsq_lookup.f - tb->rsq[itable]) * tb->drsq[itable];
                            value = tb->f[itable] + fraction * tb->df[itable];
                            fpair = factor_lj * value;
                        }

                        if (tabstyle == LOOKUP)
                           evdwl = tb->e[itable];
                        else if (tabstyle == LINEAR || tabstyle == BITMAP)
                              evdwl = tb->e[itable] + fraction * tb->de[itable];
                        else
                              evdwl = a * tb->e[itable] + b * tb->e[itable + 1] +
                                 ((a * a * a - a) * tb->e2[itable] + (b * b * b - b) * tb->e2[itable + 1]) *
                                 tb->deltasq6;
                        evdwl *= factor_lj;
                        
                        // Instead of directly tallying within this loop,
                        // we will calculate them explicitly later on our own.
                        // save fpair and evdwl from one table to our own temporary variables

                        if (si == 0) { fpair00 = fpair; u00 = evdwl; }
                        else if (si == 1) { fpair01 = fpair; u01 = evdwl; }
                        if (j < nlocal || newton_pair) softmax_scores[j][sj] -= evdwl / kT;
                        // Because j is UCG and i is not, this pair only contributes to the softmax score of j.
                        
                    }

                    // Reuse fpair as the total force on pair.
                    // Calculate fpair from fpair00 and fpair01.
                    fpair = (1. - ldj) * fpair00 + ldj * fpair01;
                    evdwl = (1. - ldj) * u00 + ldj * u01;

                    // dU/dlj = u01 - u00
                    // tally - dU/dLambda for i
                    if (j < nlocal || newton_pair) ucgf[j] -= u01 - u00;
                }

                // Scenario 3/4. i is a UCG atom, j is a non-UCG atom.

                else if (n_states_per_type[itype] > 1 && n_states_per_type[jtype] == 1) {
                    for (si = 0; si < n_states_per_type[itype]; si++) {
                        itype_si = formal_types_from_actual[itype][si];
                        tb = &tables[tabindex[itype_si][jtype]];

                        if (rsq < tb->innersq)
                            error->one(FLERR, "Pair distance < table inner cutoff: ijtype {} {} dist {}", itype_si,
                                        jtype, sqrt(rsq));
                        if (tabstyle == LOOKUP) {
                            itable = static_cast<int>((rsq - tb->innersq) * tb->invdelta);
                            if (itable >= tlm1)
                                error->one(FLERR, "Pair distance > table outer cutoff: ijtype {} {} dist {}", itype_si,
                                        jtype, sqrt(rsq));
                            fpair = factor_lj * tb->f[itable];
                        } else if (tabstyle == LINEAR) {
                            itable = static_cast<int>((rsq - tb->innersq) * tb->invdelta);
                            if (itable >= tlm1)
                                error->one(FLERR, "Pair distance > table outer cutoff: ijtype {} {} dist {}", itype_si,
                                        jtype, sqrt(rsq));
                            fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
                            value = tb->f[itable] + fraction * tb->df[itable];
                            fpair = factor_lj * value;
                        } else if (tabstyle == SPLINE) {
                            itable = static_cast<int>((rsq - tb->innersq) * tb->invdelta);
                            if (itable >= tlm1)
                                error->one(FLERR, "Pair distance > table outer cutoff: ijtype {} {} dist {}", itype_si,
                                        jtype, sqrt(rsq));
                            b = (rsq - tb->rsq[itable]) * tb->invdelta;
                            a = 1.0 - b;
                            value = a * tb->f[itable] + b * tb->f[itable + 1] +
                                ((a * a * a - a) * tb->f2[itable] + (b * b * b - b) * tb->f2[itable + 1]) *
                                    tb->deltasq6;
                            fpair = factor_lj * value;
                        } else {
                            rsq_lookup.f = rsq;
                            itable = rsq_lookup.i & tb->nmask;
                            itable >>= tb->nshiftbits;
                            fraction = (rsq_lookup.f - tb->rsq[itable]) * tb->drsq[itable];
                            value = tb->f[itable] + fraction * tb->df[itable];
                            fpair = factor_lj * value;
                        }

                        if (tabstyle == LOOKUP)
                           evdwl = tb->e[itable];
                        else if (tabstyle == LINEAR || tabstyle == BITMAP)
                              evdwl = tb->e[itable] + fraction * tb->de[itable];
                        else
                              evdwl = a * tb->e[itable] + b * tb->e[itable + 1] +
                                 ((a * a * a - a) * tb->e2[itable] + (b * b * b - b) * tb->e2[itable + 1]) *
                                 tb->deltasq6;
                        evdwl *= factor_lj;
                        
                        // Instead of directly tallying within this loop,
                        // we will calculate them explicitly later on our own.
                        // save fpair and evdwl from one table to our own temporary variables

                        if (si == 0) { fpair00 = fpair; u00 = evdwl; }
                        else if (si == 1) { fpair10 = fpair; u10 = evdwl; }
                        softmax_scores[i][si] -= evdwl / kT;
                        // Tally the contribution of neighbor j to the softmax score of i.
                        // we are using half neighbor list, so atom i is always owned
                        
                    } // end for all tables for i

                    // Reuse fpair as the total force on pair.
                    // Calculate fpair from fpair00 and fpair10.
                    fpair = (1. - ldi) * fpair00 + ldi * fpair10;
                    evdwl = (1. - ldi) * u00 + ldi * u10;

                    // dU/dli = u10 - u00
                    ucgf[i] -= u10 - u00;

                }

                // Scenario 4/4. i and j are both UCG atoms.
                else if (n_states_per_type[itype] > 1 && n_states_per_type[jtype] > 1) {
                    for (si = 0; si < n_states_per_type[itype]; si++) {
                        itype_si = formal_types_from_actual[itype][si];

                        for (sj = 0; sj < n_states_per_type[jtype]; sj++) {
                            jtype_sj = formal_types_from_actual[jtype][sj];
                            tb = &tables[tabindex[itype_si][jtype_sj]];

                            // Calculate the Bethe approximation here.
                            // We need to calculate the pairwise interaction
                            // between i and j, and then update the pmat matrix.

                            // ------- For one table -------
                            if (rsq < tb->innersq)
                                error->one(FLERR, "Pair distance < table inner cutoff: ijtype {} {} dist {}", itype_si,
                                            jtype_sj, sqrt(rsq));
                            if (tabstyle == LOOKUP) {
                                itable = static_cast<int>((rsq - tb->innersq) * tb->invdelta);
                                if (itable >= tlm1)
                                error->one(FLERR, "Pair distance > table outer cutoff: ijtype {} {} dist {}", itype_si,
                                            jtype_sj, sqrt(rsq));
                                fpair = factor_lj * tb->f[itable];
                            } else if (tabstyle == LINEAR) {
                                itable = static_cast<int>((rsq - tb->innersq) * tb->invdelta);
                                if (itable >= tlm1)
                                error->one(FLERR, "Pair distance > table outer cutoff: ijtype {} {} dist {}", itype_si,
                                            jtype_sj, sqrt(rsq));
                                fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
                                value = tb->f[itable] + fraction * tb->df[itable];
                                fpair = factor_lj * value;
                            } else if (tabstyle == SPLINE) {
                                itable = static_cast<int>((rsq - tb->innersq) * tb->invdelta);
                                if (itable >= tlm1)
                                error->one(FLERR, "Pair distance > table outer cutoff: ijtype {} {} dist {}", itype_si,
                                            jtype_sj, sqrt(rsq));
                                b = (rsq - tb->rsq[itable]) * tb->invdelta;
                                a = 1.0 - b;
                                value = a * tb->f[itable] + b * tb->f[itable + 1] +
                                ((a * a * a - a) * tb->f2[itable] + (b * b * b - b) * tb->f2[itable + 1]) *
                                        tb->deltasq6;
                                fpair = factor_lj * value;
                            } else {
                                rsq_lookup.f = rsq;
                                itable = rsq_lookup.i & tb->nmask;
                                itable >>= tb->nshiftbits;
                                fraction = (rsq_lookup.f - tb->rsq[itable]) * tb->drsq[itable];
                                value = tb->f[itable] + fraction * tb->df[itable];
                                fpair = factor_lj * value;
                            }

                           if (tabstyle == LOOKUP)
                              evdwl = tb->e[itable];
                           else if (tabstyle == LINEAR || tabstyle == BITMAP)
                                 evdwl = tb->e[itable] + fraction * tb->de[itable];
                           else
                                 evdwl = a * tb->e[itable] + b * tb->e[itable + 1] +
                                    ((a * a * a - a) * tb->e2[itable] + (b * b * b - b) * tb->e2[itable + 1]) *
                                    tb->deltasq6;
                           evdwl *= factor_lj;
                            
                            // Instead of directly tallying within this loop,
                            // we will calculate them explicitly later on our own.
                            // save fpair and evdwl from one table to our own temporary variables

                            if (si == 0 && sj == 0) { fpair00 = fpair; u00 = evdwl; }
                            else if (si == 1 && sj == 0) { fpair10 = fpair; u10 = evdwl; }
                            else if (si == 0 && sj == 1) { fpair01 = fpair; u01 = evdwl; }
                            else if (si == 1 && sj == 1) { fpair11 = fpair; u11 = evdwl; }
                            if (sj == jstate) {
                                // Tally the contribution of neighbor j to the softmax score of i.
                                // Note that we only tally the softmax score for the substate of j.
                                // Because we are using the pseudolikelihood method
                                // to assign the UCG states for the next sampling step.
                                softmax_scores[i][si] -= evdwl / kT;
                            }
                            // Also tally for j, because we are using Half neighbor list.
                            if (si == istate && (j < nlocal || newton_pair)) {
                                softmax_scores[j][sj] -= evdwl / kT;
                            }

                        } // end for all j substates
                    } // end for all i substates
                    
                    evdwl = (1.-ldi) * (1.-ldj) * u00 + (1.-ldi) * ldj * u01 + (1.-ldj) * ldi * u10 + ldi * ldj * u11;
                    // Exchange force, from nabla_U_{ij,sisj}
                    fpair = (1.-ldi) * (1.-ldj) * fpair00 + (1.-ldi) * ldj * fpair01 + (1.-ldj) * ldi * fpair10 + ldi * ldj * fpair11;
                    // The messy dU/dpij dpij/dJ nabla_J part vanishes because of pij11 satisfies the variational principle
                    // dUmix/dp11 = 0

                    // Tally ucg forces onto i and j
                    ucgf[i] -= ldj * (u11 - u01) + (1. - ldj) * (u10 - u00);
                    if (j < nlocal || newton_pair) {
                        ucgf[j] -= ldi * (u11 - u10) + (1. - ldi) * (u01 - u00);
                    }
                
                }
                // ------- End four scenarios -------
                // Now, fpair and evdwl are all mixed forces and energies.

                f[i][0] += delx * fpair;
                f[i][1] += dely * fpair;
                f[i][2] += delz * fpair;
                if (newton_pair || j < nlocal) {
                    f[j][0] -= delx * fpair;
                    f[j][1] -= dely * fpair;
                    f[j][2] -= delz * fpair;
                }
                if (evflag) {
                    ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
                }

            } // end if rsq < cutsq

         } // end for all j

    } // end for all i

}

/* ----------------------------------------------------------------------

    Read a states settings file to establish the mapping of 
    all atom types -> actual types and UCG types.

    Expected input format: 

    4 6 2     --> 4 formal types, 6 actual types, maximum 2 states per type.

    1 1       --> Normal CG types. They do not require further specifications.
    2 1
    3 1
    4 2      --> A UCG type with actual type 4, and 2 states. 2 lines are expected to follow this line.
    4 6          --> the formal types 4 and 6 are the two states of this UCG type.
    0.0 0.0      --> Chemical potentials.
    5 2      --> A UCG type with actual type 5, and 2 states. 2 lines are expected to follow this line.
    5 7          --> the formal types 5 and 7 are the two states of this UCG type.
    0.0 0.8      --> Chemical potentials.
    We do expect all UCG types to be defined AFTER all normal CG types.

---------------------------------------------------------------------- */

void PairTable_UCGLD::read_state_settings(const char *file) {
    char *eof;
    char line[MAXLINE];
    char state_type[MAXLINE];
    char entropy_spec[MAXLINE];

    // Open the state settings file.
    FILE* fp = fopen(file, "r");
    if (fp == NULL) {
        char str[128];
        sprintf(str, "Cannot open file %s", file);
        error->one(FLERR, str);
    }

    // Read the total number of actual types and total number of states.
    eof = fgets(line, MAXLINE, fp);
    if (eof == NULL) error->one(FLERR,"Unexpected end of RLEUCG state settings file");
    sscanf(line,"%d %d %d", &n_actual_types, &n_formal_types, &max_states_per_type);

    // Allocate memory for 1D pointers.

    memory->create(n_states_per_type, n_actual_types + 1, "pair:n_states_per_type");
    memory->create(actual_types_from_formal, n_formal_types + 1, "pair:n_states_per_type");
    memory->create(chemical_potentials, n_formal_types + 1, "pair:chemical_potentials");

    // Allocate memory for 2D pointers.
    memory->grow(formal_types_from_actual, n_actual_types + 1, max_states_per_type, "pair:formal_types_from_actual");
    // Note that for mode single, this is designated during initialization.

    for (int i = 0; i <= n_formal_types; i++) {
        chemical_potentials[i] = 0.0;
        actual_types_from_formal[i] = 0;
    }

    for (int i = 0; i <= n_actual_types; i++) {
        n_states_per_type[i] = 0;
        for (int j = 0; j < max_states_per_type; j++) {
            formal_types_from_actual[i][j] = 0;
        }
    }

    for (int i = 1; i <= n_actual_types; i++){
        eof = fgets(line, MAXLINE, fp);
        if (eof == NULL) error->one(FLERR,"Unexpected end of UCG state settings file");
        
        int this_type = 0;
        sscanf(line, "%d %d", &this_type, &n_states_per_type[i]);

        if (n_states_per_type[i] < 1 || n_states_per_type[i] > 2) {
            error->one(FLERR, "Invalid number of states for atom type %d: %d. Only 1 or 2 states are allowed.", i, n_states_per_type[i]);
        }
        else if (this_type != i) {
            error->one(FLERR, "Please write orderly. Invalid atom type %d in UCG state settings file. Expected %d.", this_type, i);
        }

        if (n_states_per_type[i] == 2) {
            // Read first additional line: <formal_type_i> <formal_type_j>
            eof = fgets(line, MAXLINE, fp);
            if (eof == NULL) error->one(FLERR, "Unexpected end of UCG state settings file");

            char *p = strtok(line, " ");
            for (int j = 0; j < n_states_per_type[i]; j++) {
                if (p == NULL) error->one(FLERR, "Not enough formal types specified for atom type %d.", i);
                sscanf(p, "%d", &formal_types_from_actual[i][j]);
                actual_types_from_formal[formal_types_from_actual[i][j]] = i;
                p = strtok(NULL, " ");
            }
            
            // Read second additional line: chemical potentials
            eof = fgets(line, MAXLINE, fp);
            if (eof == NULL) error->one(FLERR, "Unexpected end of UCG state settings file");
            p = strtok(line, " ");
            for (int j = 0; j < n_states_per_type[i]; j++) {
                if (p == NULL) error->one(FLERR, "Not enough formal types specified for atom type %d.", i);
                sscanf(p, "%lg", &chemical_potentials[formal_types_from_actual[i][j]]);
                p = strtok(NULL, " ");
            }

        }
        else if (n_states_per_type[i] == 1) {
            // Do nothing, just a placeholder in case something else is needed.
        }
        
    }

    fclose(fp);

}

void PairTable_UCGLD::settings(int narg, char **arg) {
    if (!atom->ucg_flag) {
        error->all(FLERR, "This pair style requires atom style ucg.");
    }

    if (narg < 2) utils::missing_cmd_args(FLERR, "pair_style table_ucgld", error);

    // new settings

    if (strcmp(arg[0], "lookup") == 0)
        tabstyle = LOOKUP;
    else if (strcmp(arg[0], "linear") == 0)
        tabstyle = LINEAR;
    else if (strcmp(arg[0], "spline") == 0)
        tabstyle = SPLINE;
    else if (strcmp(arg[0], "bitmap") == 0)
        tabstyle = BITMAP;
    else
        error->all(FLERR, "Unknown table style in pair_style command: {}", arg[0]);

    tablength = utils::inumeric(FLERR, arg[1], false, lmp);
    if (tablength < 2) error->all(FLERR, "Illegal number of pair table entries: {}", tablength);

    // optional keywords
    // assert the tabulation is compatible with a specific long-range solver

    // Read UCG Settings File
    read_state_settings(arg[2]);

    int iarg = 3;

    while (iarg < narg) {
        if (strcmp(arg[iarg], "ewald") == 0)
            ewaldflag = 1;
        else if (strcmp(arg[iarg], "pppm") == 0)
            pppmflag = 1;
        else if (strcmp(arg[iarg], "msm") == 0)
            msmflag = 1;
        else if (strcmp(arg[iarg], "dispersion") == 0)
            dispersionflag = 1;
        else if (strcmp(arg[iarg], "tip4p") == 0)
            tip4pflag = 1;
        else
            error->all(FLERR, "Unknown pair_style table keyword: {}", arg[iarg]);
        iarg++;
    }

    // delete old tables, since cannot just change settings

    for (int m = 0; m < ntables; m++) free_table(&tables[m]);
    memory->sfree(tables);

    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
        memory->destroy(tabindex);
    }
    allocated = 0;

    ntables = 0;
    tables = nullptr;

}


void PairTable_UCGLD::coeff(int narg, char **arg) {
   // Here, we require that all UCG interaction tables are provided in a single line of script.

   /* Pair_coeff command for pair_style table_ucgld:

                 [0]    [1]                         [2]           [3]               [4]                [5]                  [6]         ...
   pair_coeff <itype> <jtype> table_ucgld <num_states_i> <num_states_j> <table_file_si_sj> <table_title_si_sj> <table_cutoff_si_sj> ...

   total of num_states_i * num_states_j tables should be provided,
   in the order of: 

   num_states_i < num_states_j; (fewer states first)
   for (ii == 0; ii < num_states_i; ii++) {
      for (jj == 0; jj < num_states_j; jj++) {
         table_file[ii][jj], table_title[ii][jj], table_cutoff[ii][jj]
      }
   }
   for example, 0-0, 0-1, 1-0, 1-1 for 3 states each.
   
   Example: for the Actual Sites 4 and 5 with 2 states each ((4,6) for 4, (5,7) for 5), the command should be: 

   #          i j table_ucgld Ns_i Ns_j   table_ij_00          table_ij_01          table_ij_10          table_ij_11
   pair_coeff 4 5 table_ucgld 2    2      4-5.table 4-5 20.0   4-7.table 4-7 20.0   5-6.table 5-6 20.0   6-7.table 6-7 20.0
   pair_coeff 4 4 table_ucgld 2    2      4-4.table 4-4 20.0   4-6.table 4-6 20.0   4-6.table 4-6 20.0   6-6.table 6-6 20.0

   total number of arguments is 4 + 3 * 2 * 2 = 16.
   
   */
   if (narg < 7){
      if (narg == 6) {
         error->all(FLERR, "This pair style requires explicit definition of cutoff for each table.");
      }
      else error->all(FLERR, "Too few arguments.");
   }
   if (!allocated) allocate();

   int ilo,ihi,jlo,jhi;
   utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error); 
   utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error); 

   int me;
   MPI_Comm_rank(world, &me);

   int Ns_i = std::stoi(arg[2]);
   int Ns_j = std::stoi(arg[3]);
   
   // Just serves as a check.
   for (int temp = ilo; temp < ihi; temp++) {
      if (Ns_i != n_states_per_type[temp]) {
         error->one(FLERR, "Number of states for atom type %d does not match the number of states in the settings file.", temp);
      }
   }
   for (int temp = jlo; temp < jhi; temp++) {
      if (Ns_j != n_states_per_type[temp]) {
         error->one(FLERR, "Number of states for atom type %d does not match the number of states in the settings file.", temp);
      }
   }

   int ntables_this = Ns_i * Ns_j;

   if (narg != 4 + 3 * ntables_this) {
      error->all(FLERR, "Incorrect number of arguments for pair_coeff command. Expected 4 + 3 * n_states_i * n_states_j arguments.");
   }

   // Read all tables in this double loop.
   int this_i = 4;
   for (int s_i = 0; s_i < Ns_i; s_i++) {
      for (int s_j = 0; s_j < Ns_j; s_j++) {
         // Copy paste from pair_table.cpp in here.
         // Expected a table here.
         tables = (Table *) memory->srealloc(tables, (ntables + 1) * sizeof(Table), "pair:tables");
         Table *tb = &tables[ntables];
         null_table(tb);
         if (me == 0) read_table(tb, arg[this_i], arg[this_i + 1]);
         bcast_table(tb);

         tb->cut = utils::numeric(FLERR, arg[this_i + 2], false, lmp);

         // error check on table parameters
         // ensure cutoff is within table
         // for BITMAP tables, file values can be in non-ascending order

         if (tb->ninput <= 1) error->one(FLERR, "Invalid pair table length");
         double rlo, rhi;
         if (tb->rflag == 0) {
            rlo = tb->rfile[0];
            rhi = tb->rfile[tb->ninput - 1];
         } else {
            rlo = tb->rlo;
            rhi = tb->rhi;
         }
         if (tb->cut <= rlo || tb->cut > rhi) error->all(FLERR, "Pair table cutoff outside of table");
         if (rlo <= 0.0) error->all(FLERR, "Invalid pair table lower boundary");

         // match = 1 if don't need to spline read-in tables
         // this is only the case if r values needed by final tables
         //   exactly match r values read from file
         // for tabstyle SPLINE, always need to build spline tables

         tb->match = 0;
         if (tabstyle == LINEAR && tb->ninput == tablength && tb->rflag == RSQ && tb->rhi == tb->cut)
            tb->match = 1;
         if (tabstyle == BITMAP && tb->ninput == 1 << tablength && tb->rflag == BMP && tb->rhi == tb->cut)
            tb->match = 1;
         if (tb->rflag == BMP && tb->match == 0)
            error->all(FLERR, "Bitmapped table in file does not match requested table");

         // spline read-in values and compute r,e,f vectors within table

         if (tb->match == 0) spline_table(tb);
         compute_table(tb);

         // store ptr to table in tabindex

         int count = 0;
         int this_formal_type_i = 0; int this_formal_type_j = 0;
         for (int i = ilo; i <= ihi; i++) {
            for (int j = MAX(jlo, i); j <= jhi; j++) {
               this_formal_type_i = formal_types_from_actual[i][s_i];
               this_formal_type_j = formal_types_from_actual[j][s_j];
               if (this_formal_type_i == 0) 
                  error->all(FLERR, "Formal type not defined in pair_style command for actual type %d, state %d", i, s_i);
               else if (this_formal_type_j == 0)
                  error->all(FLERR, "Formal type not defined in pair_style command for actual type %d, state %d", j, s_j);

               tabindex[this_formal_type_i][this_formal_type_j] = ntables;
               setflag[this_formal_type_i][this_formal_type_j] = 1;
               // Then pair_coeff 4 4 command will define 4-4, 4-5, 5-5 all in one place.
               // pair_coeff 4 5 will define 4-5, 4-7, 5-6, 5-7 all in one place.
               count++;

            }
         }

         if (count == 0) error->all(FLERR, "Illegal pair_coeff command");

         // End of reading table. Moving to the next table.

         this_i += 3 ; 
         ntables ++ ;
      }
   }
   
   // after reading arg[3], check the number of remaining arguments
   // should be 3 * n_states_i * nstates_j remaining, so total narg should be 4 + 3 * arg[2] * arg[3]

}

void PairTable_UCGLD::init_style() {
    neighbor->add_request(this);
    int newton_pair = force->newton_pair;
    // For non-density calculations, default to HALF neighbor list.

    // obtain thermostat temperature
    double *pT = nullptr;
    int pdim;

    for(int ifix = 0; ifix < modify->nfix; ifix++)
    {
        pT = (double*) modify->fix[ifix]->extract("t_target", pdim);
        if(pT) { T = (*pT); break; }
    }
    kT = force->boltz * T;

    if (newton_pair != 1) error->all(FLERR, "Newton pair is turned off. It has to be turned ON in non-CV UCG Bethe simulation.");
}

double PairTable_UCGLD::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, Error::NOLASTLINE,
               "All pair coeffs are not set. Status:\n" + Info::get_pair_coeff_status(lmp));

  tabindex[j][i] = tabindex[i][j];

  return tables[tabindex[i][j]].cut;
}

void PairTable_UCGLD::read_table(Table *tb, char *file, char *keyword)
{
  TableFileReader reader(lmp, file, "pair", unit_convert_flag);

  // transparently convert units for supported conversions

  int unit_convert = reader.get_unit_convert();
  double conversion_factor = utils::get_conversion_factor(utils::ENERGY, unit_convert);
  char *line = reader.find_section_start(keyword);

  if (!line) error->one(FLERR, "Did not find keyword {} in table file", keyword);

  // read args on 2nd line of section
  // allocate table arrays for file values

  line = reader.next_line();
  param_extract(tb, line);
  memory->create(tb->rfile, tb->ninput, "pair:rfile");
  memory->create(tb->efile, tb->ninput, "pair:efile");
  memory->create(tb->ffile, tb->ninput, "pair:ffile");

  // setup bitmap parameters for table to read in

  tb->ntablebits = 0;
  int masklo, maskhi, nmask, nshiftbits;
  if (tb->rflag == BMP) {
    while (1 << tb->ntablebits < tb->ninput) tb->ntablebits++;
    if (1 << tb->ntablebits != tb->ninput)
      error->one(FLERR, "Bitmapped table is incorrect length in table file");
    init_bitmap(tb->rlo, tb->rhi, tb->ntablebits, masklo, maskhi, nmask, nshiftbits);
  }

  // read r,e,f table values from file
  // if rflag set, compute r
  // if rflag not set, use r from file

  double rfile, rnew;
  union_int_float_t rsq_lookup;

  int rerror = 0;
  reader.skip_line();
  for (int i = 0; i < tb->ninput; i++) {
    line = reader.next_line();
    if (!line)
      error->one(FLERR, "Data missing when parsing pair table '{}' line {} of {}.", keyword, i + 1,
                 tb->ninput);
    try {
      ValueTokenizer values(line);
      values.next_int();
      rfile = values.next_double();
      tb->efile[i] = conversion_factor * values.next_double();
      tb->ffile[i] = conversion_factor * values.next_double();
    } catch (TokenizerException &e) {
      error->one(FLERR, "Error parsing pair table '{}' line {} of {}. {}\nLine was: {}", keyword,
                 i + 1, tb->ninput, e.what(), line);
    }

    rnew = rfile;
    if (tb->rflag == RLINEAR)
      rnew = tb->rlo + (tb->rhi - tb->rlo) * i / (tb->ninput - 1);
    else if (tb->rflag == RSQ) {
      rnew = tb->rlo * tb->rlo + (tb->rhi * tb->rhi - tb->rlo * tb->rlo) * i / (tb->ninput - 1);
      rnew = sqrt(rnew);
    } else if (tb->rflag == BMP) {
      rsq_lookup.i = i << nshiftbits;
      rsq_lookup.i |= masklo;
      if (rsq_lookup.f < tb->rlo * tb->rlo) {
        rsq_lookup.i = i << nshiftbits;
        rsq_lookup.i |= maskhi;
      }
      rnew = sqrtf(rsq_lookup.f);
    }

    if (tb->rflag && fabs(rnew - rfile) / rfile > EPSILONR) rerror++;

    tb->rfile[i] = rnew;
  }

  // warn if force != dE/dr at any point that is not an inflection point
  // check via secant approximation to dE/dr
  // skip two end points since do not have surrounding secants
  // inflection point is where curvature changes sign

  double r, e, f, rprev, rnext, eprev, enext, fleft, fright;

  int ferror = 0;

  // bitmapped tables do not follow regular ordering, so we cannot check them here

  if (tb->rflag != BMP) {
    for (int i = 1; i < tb->ninput - 1; i++) {
      r = tb->rfile[i];
      rprev = tb->rfile[i - 1];
      rnext = tb->rfile[i + 1];
      e = tb->efile[i];
      eprev = tb->efile[i - 1];
      enext = tb->efile[i + 1];
      f = tb->ffile[i];
      fleft = -(e - eprev) / (r - rprev);
      fright = -(enext - e) / (rnext - r);
      if (f < fleft && f < fright) ferror++;
      if (f > fleft && f > fright) ferror++;
      //printf("Values %d: %g %g %g\n",i,r,e,f);
      //printf("  secant %d %d %g: %g %g %g\n",i,ferror,r,fleft,fright,f);
    }
  }

  if (ferror)
    error->warning(FLERR,
                   "{} of {} force values in table {} are inconsistent with -dE/dr.\n"
                   "WARNING:  Should only be flagged at inflection points",
                   ferror, tb->ninput, keyword);

  // warn if re-computed distance values differ from file values

  if (rerror)
    error->warning(FLERR,
                   "{} of {} distance values in table {} with relative error\n"
                   "WARNING:  over {} to re-computed values",
                   rerror, tb->ninput, EPSILONR, keyword);
}

void PairTable_UCGLD::bcast_table(Table *tb)
{
  MPI_Bcast(&tb->ninput, 1, MPI_INT, 0, world);

  int me;
  MPI_Comm_rank(world, &me);
  if (me > 0) {
    memory->create(tb->rfile, tb->ninput, "pair:rfile");
    memory->create(tb->efile, tb->ninput, "pair:efile");
    memory->create(tb->ffile, tb->ninput, "pair:ffile");
  }

  MPI_Bcast(tb->rfile, tb->ninput, MPI_DOUBLE, 0, world);
  MPI_Bcast(tb->efile, tb->ninput, MPI_DOUBLE, 0, world);
  MPI_Bcast(tb->ffile, tb->ninput, MPI_DOUBLE, 0, world);

  MPI_Bcast(&tb->rflag, 1, MPI_INT, 0, world);
  if (tb->rflag) {
    MPI_Bcast(&tb->rlo, 1, MPI_DOUBLE, 0, world);
    MPI_Bcast(&tb->rhi, 1, MPI_DOUBLE, 0, world);
  }
  MPI_Bcast(&tb->fpflag, 1, MPI_INT, 0, world);
  if (tb->fpflag) {
    MPI_Bcast(&tb->fplo, 1, MPI_DOUBLE, 0, world);
    MPI_Bcast(&tb->fphi, 1, MPI_DOUBLE, 0, world);
  }
}

void PairTable_UCGLD::spline_table(Table *tb)
{
  memory->create(tb->e2file, tb->ninput, "pair:e2file");
  memory->create(tb->f2file, tb->ninput, "pair:f2file");

  double ep0 = -tb->ffile[0];
  double epn = -tb->ffile[tb->ninput - 1];
  spline(tb->rfile, tb->efile, tb->ninput, ep0, epn, tb->e2file);

  if (tb->fpflag == 0) {
    tb->fplo = (tb->ffile[1] - tb->ffile[0]) / (tb->rfile[1] - tb->rfile[0]);
    tb->fphi = (tb->ffile[tb->ninput - 1] - tb->ffile[tb->ninput - 2]) /
        (tb->rfile[tb->ninput - 1] - tb->rfile[tb->ninput - 2]);
  }

  double fp0 = tb->fplo;
  double fpn = tb->fphi;
  spline(tb->rfile, tb->ffile, tb->ninput, fp0, fpn, tb->f2file);
}

void PairTable_UCGLD::param_extract(Table *tb, char *line)
{
  tb->ninput = 0;
  tb->rflag = NONE;
  tb->fpflag = 0;

  try {
    ValueTokenizer values(line);

    while (values.has_next()) {
      std::string word = values.next_string();
      if (word == "N") {
        tb->ninput = values.next_int();
      } else if ((word == "R") || (word == "RSQ") || (word == "BITMAP")) {
        if (word == "R")
          tb->rflag = RLINEAR;
        else if (word == "RSQ")
          tb->rflag = RSQ;
        else if (word == "BITMAP")
          tb->rflag = BMP;
        tb->rlo = values.next_double();
        tb->rhi = values.next_double();
      } else if (word == "FPRIME") {
        tb->fpflag = 1;
        tb->fplo = values.next_double();
        tb->fphi = values.next_double();
      } else {
        error->one(FLERR, "Invalid keyword {} in pair table parameters", word);
      }
    }
  } catch (TokenizerException &e) {
    error->one(FLERR, e.what());
  }

  if (tb->ninput == 0) error->one(FLERR, "Pair table parameters did not set N");
}


void PairTable_UCGLD::compute_table(Table *tb)
{
  int tlm1 = tablength - 1;

  // inner = inner table bound
  // cut = outer table bound
  // delta = table spacing in rsq for N-1 bins

  double inner;
  if (tb->rflag)
    inner = tb->rlo;
  else
    inner = tb->rfile[0];
  tb->innersq = inner * inner;
  tb->delta = (tb->cut * tb->cut - tb->innersq) / tlm1;
  tb->invdelta = 1.0 / tb->delta;

  // direct lookup tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // e,f = value at midpt of bin
  // e,f are N-1 in length since store 1 value at bin midpt
  // f is converted to f/r when stored in f[i]
  // e,f are never a match to read-in values, always computed via spline interp

  if (tabstyle == LOOKUP) {
    memory->create(tb->e, tlm1, "pair:e");
    memory->create(tb->f, tlm1, "pair:f");

    double r, rsq;
    for (int i = 0; i < tlm1; i++) {
      rsq = tb->innersq + (i + 0.5) * tb->delta;
      r = sqrt(rsq);
      tb->e[i] = splint(tb->rfile, tb->efile, tb->e2file, tb->ninput, r);
      tb->f[i] = splint(tb->rfile, tb->ffile, tb->f2file, tb->ninput, r) / r;
    }
  }

  // linear tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // rsq,e,f = value at lower edge of bin
  // de,df values = delta from lower edge to upper edge of bin
  // rsq,e,f are N in length so de,df arrays can compute difference
  // f is converted to f/r when stored in f[i]
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == LINEAR) {
    memory->create(tb->rsq, tablength, "pair:rsq");
    memory->create(tb->e, tablength, "pair:e");
    memory->create(tb->f, tablength, "pair:f");
    memory->create(tb->de, tlm1, "pair:de");
    memory->create(tb->df, tlm1, "pair:df");

    double r, rsq;
    for (int i = 0; i < tablength; i++) {
      rsq = tb->innersq + i * tb->delta;
      r = sqrt(rsq);
      tb->rsq[i] = rsq;
      if (tb->match) {
        tb->e[i] = tb->efile[i];
        tb->f[i] = tb->ffile[i] / r;
      } else {
        tb->e[i] = splint(tb->rfile, tb->efile, tb->e2file, tb->ninput, r);
        tb->f[i] = splint(tb->rfile, tb->ffile, tb->f2file, tb->ninput, r) / r;
      }
    }

    for (int i = 0; i < tlm1; i++) {
      tb->de[i] = tb->e[i + 1] - tb->e[i];
      tb->df[i] = tb->f[i + 1] - tb->f[i];
    }
  }

  // cubic spline tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // rsq,e,f = value at lower edge of bin
  // e2,f2 = spline coefficient for each bin
  // rsq,e,f,e2,f2 are N in length so have N-1 spline bins
  // f is converted to f/r after e is splined
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == SPLINE) {
    memory->create(tb->rsq, tablength, "pair:rsq");
    memory->create(tb->e, tablength, "pair:e");
    memory->create(tb->f, tablength, "pair:f");
    memory->create(tb->e2, tablength, "pair:e2");
    memory->create(tb->f2, tablength, "pair:f2");

    tb->deltasq6 = tb->delta * tb->delta / 6.0;

    double r, rsq;
    for (int i = 0; i < tablength; i++) {
      rsq = tb->innersq + i * tb->delta;
      r = sqrt(rsq);
      tb->rsq[i] = rsq;
      if (tb->match) {
        tb->e[i] = tb->efile[i];
        tb->f[i] = tb->ffile[i] / r;
      } else {
        tb->e[i] = splint(tb->rfile, tb->efile, tb->e2file, tb->ninput, r);
        tb->f[i] = splint(tb->rfile, tb->ffile, tb->f2file, tb->ninput, r);
      }
    }

    // ep0,epn = dh/dg at inner and at cut
    // h(r) = e(r) and g(r) = r^2
    // dh/dg = (de/dr) / 2r = -f/2r

    double ep0 = -tb->f[0] / (2.0 * sqrt(tb->innersq));
    double epn = -tb->f[tlm1] / (2.0 * tb->cut);
    spline(tb->rsq, tb->e, tablength, ep0, epn, tb->e2);

    // fp0,fpn = dh/dg at inner and at cut
    // h(r) = f(r)/r and g(r) = r^2
    // dh/dg = (1/r df/dr - f/r^2) / 2r
    // dh/dg in secant approx = (f(r2)/r2 - f(r1)/r1) / (g(r2) - g(r1))

    double fp0, fpn;
    double secant_factor = 0.1;
    if (tb->fpflag)
      fp0 = (tb->fplo / sqrt(tb->innersq) - tb->f[0] / tb->innersq) / (2.0 * sqrt(tb->innersq));
    else {
      double rsq1 = tb->innersq;
      double rsq2 = rsq1 + secant_factor * tb->delta;
      fp0 = (splint(tb->rfile, tb->ffile, tb->f2file, tb->ninput, sqrt(rsq2)) / sqrt(rsq2) -
             tb->f[0] / sqrt(rsq1)) /
          (secant_factor * tb->delta);
    }

    if (tb->fpflag && tb->cut == tb->rfile[tb->ninput - 1])
      fpn = (tb->fphi / tb->cut - tb->f[tlm1] / (tb->cut * tb->cut)) / (2.0 * tb->cut);
    else {
      double rsq2 = tb->cut * tb->cut;
      double rsq1 = rsq2 - secant_factor * tb->delta;
      fpn = (tb->f[tlm1] / sqrt(rsq2) -
             splint(tb->rfile, tb->ffile, tb->f2file, tb->ninput, sqrt(rsq1)) / sqrt(rsq1)) /
          (secant_factor * tb->delta);
    }

    for (int i = 0; i < tablength; i++) tb->f[i] /= sqrt(tb->rsq[i]);
    spline(tb->rsq, tb->f, tablength, fp0, fpn, tb->f2);
  }

  // bitmapped linear tables
  // 2^N bins from inner to cut, spaced in bitmapped manner
  // f is converted to f/r when stored in f[i]
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == BITMAP) {
    double r;
    union_int_float_t rsq_lookup;
    int masklo, maskhi;

    // linear lookup tables of length ntable = 2^n
    // stored value = value at lower edge of bin

    init_bitmap(inner, tb->cut, tablength, masklo, maskhi, tb->nmask, tb->nshiftbits);
    int ntable = 1 << tablength;
    int ntablem1 = ntable - 1;

    memory->create(tb->rsq, ntable, "pair:rsq");
    memory->create(tb->e, ntable, "pair:e");
    memory->create(tb->f, ntable, "pair:f");
    memory->create(tb->de, ntable, "pair:de");
    memory->create(tb->df, ntable, "pair:df");
    memory->create(tb->drsq, ntable, "pair:drsq");

    union_int_float_t minrsq_lookup;
    minrsq_lookup.i = 0 << tb->nshiftbits;
    minrsq_lookup.i |= maskhi;

    for (int i = 0; i < ntable; i++) {
      rsq_lookup.i = i << tb->nshiftbits;
      rsq_lookup.i |= masklo;
      if (rsq_lookup.f < tb->innersq) {
        rsq_lookup.i = i << tb->nshiftbits;
        rsq_lookup.i |= maskhi;
      }
      r = sqrtf(rsq_lookup.f);
      tb->rsq[i] = rsq_lookup.f;
      if (tb->match) {
        tb->e[i] = tb->efile[i];
        tb->f[i] = tb->ffile[i] / r;
      } else {
        tb->e[i] = splint(tb->rfile, tb->efile, tb->e2file, tb->ninput, r);
        tb->f[i] = splint(tb->rfile, tb->ffile, tb->f2file, tb->ninput, r) / r;
      }
      minrsq_lookup.f = MIN(minrsq_lookup.f, rsq_lookup.f);
    }

    tb->innersq = minrsq_lookup.f;

    for (int i = 0; i < ntablem1; i++) {
      tb->de[i] = tb->e[i + 1] - tb->e[i];
      tb->df[i] = tb->f[i + 1] - tb->f[i];
      tb->drsq[i] = 1.0 / (tb->rsq[i + 1] - tb->rsq[i]);
    }

    // get the delta values for the last table entries
    // tables are connected periodically between 0 and ntablem1

    tb->de[ntablem1] = tb->e[0] - tb->e[ntablem1];
    tb->df[ntablem1] = tb->f[0] - tb->f[ntablem1];
    tb->drsq[ntablem1] = 1.0 / (tb->rsq[0] - tb->rsq[ntablem1]);

    // get the correct delta values at itablemax
    // smallest r is in bin itablemin
    // largest r is in bin itablemax, which is itablemin-1,
    //   or ntablem1 if itablemin=0

    // deltas at itablemax only needed if corresponding rsq < cut*cut
    // if so, compute deltas between rsq and cut*cut
    //   if tb->match, data at cut*cut is unavailable, so we'll take
    //   deltas at itablemax-1 as a good approximation

    double e_tmp, f_tmp;
    int itablemin = minrsq_lookup.i & tb->nmask;
    itablemin >>= tb->nshiftbits;
    int itablemax = itablemin - 1;
    if (itablemin == 0) itablemax = ntablem1;
    int itablemaxm1 = itablemax - 1;
    if (itablemax == 0) itablemaxm1 = ntablem1;
    rsq_lookup.i = itablemax << tb->nshiftbits;
    rsq_lookup.i |= maskhi;
    if (rsq_lookup.f < tb->cut * tb->cut) {
      if (tb->match) {
        tb->de[itablemax] = tb->de[itablemaxm1];
        tb->df[itablemax] = tb->df[itablemaxm1];
        tb->drsq[itablemax] = tb->drsq[itablemaxm1];
      } else {
        rsq_lookup.f = tb->cut * tb->cut;
        r = sqrtf(rsq_lookup.f);
        e_tmp = splint(tb->rfile, tb->efile, tb->e2file, tb->ninput, r);
        f_tmp = splint(tb->rfile, tb->ffile, tb->f2file, tb->ninput, r) / r;
        tb->de[itablemax] = e_tmp - tb->e[itablemax];
        tb->df[itablemax] = f_tmp - tb->f[itablemax];
        tb->drsq[itablemax] = 1.0 / (rsq_lookup.f - tb->rsq[itablemax]);
      }
    }
  }
}



void PairTable_UCGLD::null_table(Table *tb)
{
  tb->rfile = tb->efile = tb->ffile = nullptr;
  tb->e2file = tb->f2file = nullptr;
  tb->rsq = tb->drsq = tb->e = tb->de = nullptr;
  tb->f = tb->df = tb->e2 = tb->f2 = nullptr;
}

void PairTable_UCGLD::free_table(Table *tb)
{
  memory->destroy(tb->rfile);
  memory->destroy(tb->efile);
  memory->destroy(tb->ffile);
  memory->destroy(tb->e2file);
  memory->destroy(tb->f2file);

  memory->destroy(tb->rsq);
  memory->destroy(tb->drsq);
  memory->destroy(tb->e);
  memory->destroy(tb->de);
  memory->destroy(tb->f);
  memory->destroy(tb->df);
  memory->destroy(tb->e2);
  memory->destroy(tb->f2);
}


void PairTable_UCGLD::spline(double *x, double *y, int n, double yp1, double ypn, double *y2)
{
  int i, k;
  double p, qn, sig, un;
  auto u = new double[n];

  if (yp1 > 0.99e30)
    y2[0] = u[0] = 0.0;
  else {
    y2[0] = -0.5;
    u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1);
  }
  for (i = 1; i < n - 1; i++) {
    sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
    p = sig * y2[i - 1] + 2.0;
    y2[i] = (sig - 1.0) / p;
    u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
    u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
  }
  if (ypn > 0.99e30)
    qn = un = 0.0;
  else {
    qn = 0.5;
    un = (3.0 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]));
  }
  y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0);
  for (k = n - 2; k >= 0; k--) y2[k] = y2[k] * y2[k + 1] + u[k];

  delete[] u;
}

/* ---------------------------------------------------------------------- */

double PairTable_UCGLD::splint(double *xa, double *ya, double *y2a, int n, double x)
{
  int klo, khi, k;
  double h, b, a, y;

  klo = 0;
  khi = n - 1;
  while (khi - klo > 1) {
    k = (khi + klo) >> 1;
    if (xa[k] > x)
      khi = k;
    else
      klo = k;
  }
  h = xa[khi] - xa[klo];
  a = (xa[khi] - x) / h;
  b = (x - xa[klo]) / h;
  y = a * ya[klo] + b * ya[khi] +
      ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * (h * h) / 6.0;
  return y;
}


void PairTable_UCGLD::write_restart(FILE *fp)
{
  write_restart_settings(fp);
}

void PairTable_UCGLD::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();
}

void PairTable_UCGLD::write_restart_settings(FILE *fp)
{
  fwrite(&tabstyle, sizeof(int), 1, fp);
  fwrite(&tablength, sizeof(int), 1, fp);
  fwrite(&ewaldflag, sizeof(int), 1, fp);
  fwrite(&pppmflag, sizeof(int), 1, fp);
  fwrite(&msmflag, sizeof(int), 1, fp);
  fwrite(&dispersionflag, sizeof(int), 1, fp);
  fwrite(&tip4pflag, sizeof(int), 1, fp);
}

void PairTable_UCGLD::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR, &tabstyle, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &tablength, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &ewaldflag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &pppmflag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &msmflag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &dispersionflag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &tip4pflag, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&tabstyle, 1, MPI_INT, 0, world);
  MPI_Bcast(&tablength, 1, MPI_INT, 0, world);
  MPI_Bcast(&ewaldflag, 1, MPI_INT, 0, world);
  MPI_Bcast(&pppmflag, 1, MPI_INT, 0, world);
  MPI_Bcast(&msmflag, 1, MPI_INT, 0, world);
  MPI_Bcast(&dispersionflag, 1, MPI_INT, 0, world);
  MPI_Bcast(&tip4pflag, 1, MPI_INT, 0, world);
}


double PairTable_UCGLD::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                         double /*factor_coul*/, double factor_lj, double &fforce)
{
  int itable;
  double fraction, value, a, b, phi;
  int tlm1 = tablength - 1;

  Table *tb = &tables[tabindex[itype][jtype]];
  if (rsq < tb->innersq) error->one(FLERR, "Pair distance < table inner cutoff");

  if (tabstyle == LOOKUP) {
    itable = static_cast<int>((rsq - tb->innersq) * tb->invdelta);
    if (itable >= tlm1) error->one(FLERR, "Pair distance > table outer cutoff");
    fforce = factor_lj * tb->f[itable];
  } else if (tabstyle == LINEAR) {
    itable = static_cast<int>((rsq - tb->innersq) * tb->invdelta);
    if (itable >= tlm1) error->one(FLERR, "Pair distance > table outer cutoff");
    fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
    value = tb->f[itable] + fraction * tb->df[itable];
    fforce = factor_lj * value;
  } else if (tabstyle == SPLINE) {
    itable = static_cast<int>((rsq - tb->innersq) * tb->invdelta);
    if (itable >= tlm1) error->one(FLERR, "Pair distance > table outer cutoff");
    b = (rsq - tb->rsq[itable]) * tb->invdelta;
    a = 1.0 - b;
    value = a * tb->f[itable] + b * tb->f[itable + 1] +
        ((a * a * a - a) * tb->f2[itable] + (b * b * b - b) * tb->f2[itable + 1]) * tb->deltasq6;
    fforce = factor_lj * value;
  } else {
    union_int_float_t rsq_lookup;
    rsq_lookup.f = rsq;
    itable = rsq_lookup.i & tb->nmask;
    itable >>= tb->nshiftbits;
    fraction = (rsq_lookup.f - tb->rsq[itable]) * tb->drsq[itable];
    value = tb->f[itable] + fraction * tb->df[itable];
    fforce = factor_lj * value;
  }

  if (tabstyle == LOOKUP)
    phi = tb->e[itable];
  else if (tabstyle == LINEAR || tabstyle == BITMAP)
    phi = tb->e[itable] + fraction * tb->de[itable];
  else
    phi = a * tb->e[itable] + b * tb->e[itable + 1] +
        ((a * a * a - a) * tb->e2[itable] + (b * b * b - b) * tb->e2[itable + 1]) * tb->deltasq6;
  return factor_lj * phi;
}

void* PairTable_UCGLD::extract(const char *str, int &dim)
{
  if (strcmp(str, "cut_coul") != 0) return nullptr;
  if (ntables == 0)
    error->all(FLERR, Error::NOLASTLINE,
               "All pair coeffs are not set. Status:\n" + Info::get_pair_coeff_status(lmp));

  // only check for cutoff consistency if claiming to be KSpace compatible

  if (ewaldflag || pppmflag || msmflag || dispersionflag || tip4pflag) {
    double cut_coul = tables[0].cut;
    for (int m = 1; m < ntables; m++)
      if (tables[m].cut != cut_coul)
        error->all(FLERR, Error::NOLASTLINE,
                   "Pair table cutoffs must all be equal to use with KSpace");
    dim = 0;
    return &tables[0].cut;
  } else
    return nullptr;
}
