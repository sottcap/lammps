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

#include <cmath>
#include <cstdlib>
#include "pair_dlvo.h"
#include "atom.h"
#include "domain.h"
#include "force.h"
#include "comm.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairDLVO::PairDLVO(LAMMPS *lmp) : PairYukawa(lmp)
{
}

/* ---------------------------------------------------------------------- */

PairDLVO::~PairDLVO()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(rad);
    memory->destroy(cut);
    memory->destroy(a);
    memory->destroy(offset);
  }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairDLVO::settings(int narg, char **arg)
{
  if (narg < 3 || narg > 3) error->all(FLERR,"Illegal pair_style command");

  double bjerrum_length,vol, electrolyte_conc = 0;

  if (domain->dimension == 3)
    vol = domain->xprd * domain->yprd * domain->zprd;
  else
    vol = domain->xprd * domain->yprd;

  bjerrum_length = force->numeric(FLERR,arg[0]);
  cut_global = force->numeric(FLERR,arg[1]);

  if (narg == 3) { electrolyte_conc = force->numeric(FLERR,arg[2]); }

  double q_factor = 0;

  for(int i=1; i <= atom->ntypes; i++) {
    int atom_cnt = 0;

    for(int ii=0; ii <= atom->natoms; ii++)
      if (atom->type[ii] == i) atom_cnt++;
  
    double rho,Z;
    rho = atom_cnt/vol;
    Z = atom->q[i];

    q_factor += Z*rho;
  }

  q_factor += electrolyte_conc;

  kappa = sqrt(bjerrum_length * q_factor);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairDLVO::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    a[i][j] = mix_energy(a[i][i],a[j][j],1.0,1.0);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  if (offset_flag && (cut[i][j] > 0.0)) {
    double screening = exp(-kappa * cut[i][j]);
    offset[i][j] = a[i][j] * screening / cut[i][j];
  } else offset[i][j] = 0.0;

  a[j][i] = a[i][j];
  offset[j][i] = offset[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairDLVO::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR,"Pair dlvo requires atom style charge");
}

