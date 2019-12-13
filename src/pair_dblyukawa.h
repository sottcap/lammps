/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(dblyukawa,PairDblYukawa)

#else

#ifndef LMP_PAIR_DBLYUKAWA_H
#define LMP_PAIR_DBLYUKAWA_H

#include "pair_yukawa.h"

namespace LAMMPS_NS {

class PairDblYukawa : public PairYukawa {
 public:
  PairDblYukawa(class LAMMPS *);
  virtual ~PairDblYukawa() {}
  virtual void compute(int, int);
  void init_style();
  double init_one(int, int);
  double single(int, int, int, int, double, double, double, double &);

  void settings(int, char **);
  void coeff(int, char **);

  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);

 protected:
  double kappas[2];
  double **a1, **a2;

  virtual void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Pair yukawa/colloid requires atom style sphere

Self-explanatory.

E: Pair yukawa/colloid requires atoms with same type have same radius

Self-explanatory.

*/
