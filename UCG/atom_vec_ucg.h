/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------
   Contributing authors: Weizhi Xue (University of Chicago)

   2025.6.23
------------------------------------------------------------------------- */

#ifdef ATOM_CLASS
// clang-format off
AtomStyle(ucg,AtomVecUCG);
// clang-format on
#else

#ifndef LMP_ATOM_VEC_UCG_H
#define LMP_ATOM_VEC_UCG_H

#include "atom_vec.h"

namespace LAMMPS_NS {

class AtomVecUCG : virtual public AtomVec {
 public:
  AtomVecUCG(class LAMMPS *);

  void grow_pointers() override;
  void force_clear(int, size_t) override;
  void data_atom_post(int) override;
  int property_atom(const std::string &name) override;
  void pack_property_atom(int, double*, int, int) override;

 protected:
 
  int *num_bond, *num_angle, *num_dihedral, *num_improper;
  int **bond_type, **angle_type, **dihedral_type, **improper_type;
  int **nspecial;

  int bond_per_atom, angle_per_atom, dihedral_per_atom, improper_per_atom;

  int *ucgstate, *num_ucgstates;
  double *ucgl, *ucgvl, *ucgml, *ucgp; // Now we only allow 2-state systems, so ucgl is a 1D pointer.
  // Change ucgl to double pointer **ucgl with (n-1) columns for n-state systems.
  double *ucgforce;
  double **ucgsoftmaxscores;   // per-atom UCG values

} ;

}

#endif
#endif
