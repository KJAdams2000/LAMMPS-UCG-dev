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

#include "atom_vec_ucg.h"

#include "atom.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecUCG::AtomVecUCG(LAMMPS *lmp): AtomVec(lmp) {
   molecular = Atom::MOLECULAR;
   bonds_allow = angles_allow = dihedrals_allow = impropers_allow = 1;
   mass_type = PER_TYPE;

   forceclearflag = 1;
   atom->molecule_flag = atom->q_flag = 1;
   atom->ucg_flag = 1;
   atom->max_ucgstates = 2; // Default to 2-state UCG, can be changed later.
   
   // strings with peratom variables to include in each AtomVec method
   // strings cannot contain fields in corresponding AtomVec default strings
   // order of fields in a string does not matter
   // except: fields_data_atom & fields_data_vel must match data file
   
   fields_grow = {
      "q", "molecule", 
      "num_bond", "bond_type", "bond_atom", 
      "num_angle", "angle_type", "angle_atom1", "angle_atom2", "angle_atom3", 
      "num_dihedral", "dihedral_type", "dihedral_atom1", "dihedral_atom2", "dihedral_atom3", "dihedral_atom4", 
      "num_improper", "improper_type", "improper_atom1", "improper_atom2", "improper_atom3", "improper_atom4", 
      "nspecial", "special", // above are from AtomVecFull
      "ucgstate", "ucgl", "ucgvl", "ucgml", "ucgp", "ucgforce", "ucgsoftmaxscores", "num_ucgstates" // This line is UCG customization
   };
   fields_copy = {
      "q", "molecule", 
      "num_bond", "bond_type", "bond_atom", 
      "num_angle", "angle_type", "angle_atom1", "angle_atom2", "angle_atom3", 
      "num_dihedral", "dihedral_type", "dihedral_atom1", "dihedral_atom2", "dihedral_atom3", "dihedral_atom4", 
      "num_improper", "improper_type", "improper_atom1", "improper_atom2", "improper_atom3", "improper_atom4", 
      "nspecial", "special",
      "ucgstate", "ucgl", "ucgvl", "ucgml", "ucgp", "ucgforce", "ucgsoftmaxscores", "num_ucgstates" // UCG customization
   };
   fields_border = {"q", "molecule", 
      "ucgstate", "num_ucgstates", "ucgl", "ucgp"};
   fields_border_vel = {"q", "molecule", 
      "ucgstate", "num_ucgstates", "ucgl", "ucgp", "ucgvl"};

   fields_comm = {"ucgstate", "ucgl", "ucgp"};  // When only in forward communication, exchange the UCG state and Lambda value.
   fields_comm_vel = {"ucgstate", "ucgl", "ucgvl", "ucgp"};
   fields_reverse = {"ucgforce", "ucgsoftmaxscores"}; // Only reverse forces, not probabilities 
   // When in reverse communication, exchange the forces.
   
   fields_exchange = {"q", "molecule", 
      "num_bond", "bond_type", "bond_atom", 
      "num_angle", "angle_type", "angle_atom1", "angle_atom2", "angle_atom3", 
      "num_dihedral", "dihedral_type", "dihedral_atom1", "dihedral_atom2", "dihedral_atom3", "dihedral_atom4", 
      "num_improper", "improper_type", "improper_atom1", "improper_atom2", "improper_atom3", "improper_atom4", 
      "nspecial", "special",
      "ucgstate", "ucgl", "ucgvl", "ucgml", "ucgp", "ucgforce", "ucgsoftmaxscores", "num_ucgstates"};
   // When exchanging, looks like you do not change the forces
   // ref: atom_vec_electron.cpp in src/EFF/
   fields_restart = {"ucgstate", "ucgl", "ucgml", "ucgvl", "ucgp"};

   fields_data_atom = {"id", "molecule", "type", "q", "x", "ucgstate", "ucgl", "ucgml"};
   // Note: we wish to both include the assigned UCG state and the UCG Lambda Value.
   // For Lambda Dynamics UCG, this ucgl serves as the Lambda degree of freedom.
   fields_data_vel = {"id", "v", "ucgvl"};

   setup_fields();
   // This single command initializes all the fields
   bond_per_atom = angle_per_atom = dihedral_per_atom = improper_per_atom = 0;

}

/* ----------------------------------------------------------------------
   set local copies of all grow ptrs used by this class, except defaults
   needed in replicate when 2 atom classes exist and it calls pack_restart()
------------------------------------------------------------------------- */

void AtomVecUCG::grow_pointers() {

   num_bond = atom->num_bond;
   bond_type = atom->bond_type;
   num_angle = atom->num_angle;
   angle_type = atom->angle_type;
   num_dihedral = atom->num_dihedral;
   dihedral_type = atom->dihedral_type;
   num_improper = atom->num_improper;
   improper_type = atom->improper_type;
   nspecial = atom->nspecial;

   ucgstate = atom->ucgstate;
   ucgl = atom->ucgl;
   ucgforce = atom->ucgforce;
   ucgsoftmaxscores = atom->ucgsoftmaxscores;
   ucgvl = atom->ucgvl;
   ucgp = atom->ucgp;
   ucgml = atom->ucgml;
   num_ucgstates = atom->num_ucgstates;
}

/* ----------------------------------------------------------------------
   clear UCG force values
   starting at atom N
   size is the number of atoms to clear
------------------------------------------------------------------------- */

void AtomVecUCG::force_clear(int n, size_t nbytes) {
   // Only clears out Forces
   memset(&ucgforce[n], 0, nbytes);
   memset(&ucgsoftmaxscores[n][0], 0, atom->max_ucgstates * nbytes);
}

/* ----------------------------------------------------------------------
   modify what AtomVec::data_atom() just unpacked
   or initialize other atom quantities.

   For example, we restrict the UCG Lambda value to be between 0 and 1,
   and ensure the UCG state is binary (0 or 1).
------------------------------------------------------------------------- */

void AtomVecUCG::data_atom_post(int ilocal)
{
     num_bond[ilocal] = 0;
      num_angle[ilocal] = 0;
      num_dihedral[ilocal] = 0;
      num_improper[ilocal] = 0;
      nspecial[ilocal][0] = 0;
      nspecial[ilocal][1] = 0;
      nspecial[ilocal][2] = 0;

   // Normalize the UCG Lambda value to be between 0 and 1
   if (ucgl[ilocal] < 0)  ucgl[ilocal] = 0.;
   else if (ucgl[ilocal] > 1)  ucgl[ilocal] = 1.;
   
   // For now, we only allow 2-state systems, so ucgl itself is a 1D pointer.
   // If you want to support n-state systems, please change ucgl to double pointer **ucgl with (n-1) columns.
   // Therefore, 

   if (ucgstate[ilocal] < 0) ucgstate[ilocal] = 0; 
   else if (ucgstate[ilocal] > 1) ucgstate[ilocal] = 1; 
   // Ensure UCG state is binary (0 or 1)

   ucgp[ilocal] = -1.0; // We set the posterior probability to -1.0, meaning that they are unassigned yet.
   // after the first force calculation, they will be updated with valid posterior probabilities.
   // In UCGLD, this is only a diagnostic value, not used in the simulation.
}

int AtomVecUCG::property_atom(const std::string &name)
{
   if (name == "ucgstate") return 0;
   if (name == "ucgl") return 1;
   if (name == "ucgforce") return 2;
   if (name == "ucgvl") return 3;
   if (name == "ucgp") return 4;
   if (name == "ucgml") return 5;
   return -1; // Not found
}

void AtomVecUCG::pack_property_atom(int index, double *buf, int nvalues, int groupbit)
{
   int *mask = atom->mask;
   int nlocal = atom->nlocal;
   int n = 0;

   if (index == 0) { // ucgstate
      for (int j = 0; j < nlocal; j++) {
         if (mask[j] & groupbit) {
            buf[n] = ucgstate[j];
         } else buf[n] = 0.0;
         n += nvalues;
      }
   } else if (index == 1) { // ucgl
      for (int j = 0; j < nlocal; j++) {
         if (mask[j] & groupbit) {
            buf[n] = ucgl[j];
         } else buf[n] = 0.0;
         n += nvalues;
      }
   } else if (index == 2) { // ucgforce
      for (int j = 0; j < nlocal; j++) {
         if (mask[j] & groupbit) {
            buf[n] = ucgforce[j];
         } else buf[n] = 0.0;
         n += nvalues;
      }
   } else if (index == 3) { // ucgvl
      for (int j = 0; j < nlocal; j++) {
         if (mask[j] & groupbit) {
            buf[n] = ucgvl[j];
         } else buf[n] = 0.0;
         n += nvalues;
      }
   } else if (index == 4) { // ucgp
      for (int j = 0; j < nlocal; j++) {
         if (mask[j] & groupbit) {
            buf[n] = ucgp[j];
         } else buf[n] = 0.0;
         n += nvalues;
      }
   } else if (index == 5) { // ucgml
      for (int j = 0; j < nlocal; j++) {
         if (mask[j] & groupbit) {
            buf[n] = ucgml[j];
         } else buf[n] = 0.0;
         n += nvalues;
      }
   } else {
      error->all(FLERR, "Unknown property_atom index in AtomVecUCG::pack_property_atom");
   }
}