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

#include "fix_bd_omp.h"
#include "atom.h"
#include "force.h"
#include "random_mars.h"
#include <math.h>

using namespace LAMMPS_NS;
using namespace FixConst;

typedef struct { double x,y,z; } dbl3_t;

/* ---------------------------------------------------------------------- */

FixBDOMP::FixBDOMP(LAMMPS *lmp, int narg, char **arg) :
  FixBD(lmp, narg, arg) { }

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixBDOMP::initial_integrate(int /* vflag */)
{
  // update v and x of atoms in group

  dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  dbl3_t * _noalias const v = (dbl3_t *) atom->v[0];
  const dbl3_t * _noalias const f = (dbl3_t *) atom->f[0];
  const int * const mask = atom->mask;
  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  const double damp_ = damp;
  const double * const ratio_ = this->ratio;
  int i;

  if (atom->rmass) {
    const double * const rmass = atom->rmass;
    const int * const type = atom->type;
#if defined (_OPENMP)
#pragma omp parallel for private(i) default(none) schedule(static)
#endif
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = dtf / rmass[i];
        #pragma omp critical
        v[i].x = (f[i].x / damp_ * ratio_[type[i]]);
        v[i].y = (f[i].y / damp_ * ratio_[type[i]]);
        v[i].z = (f[i].z / damp_ * ratio_[type[i]]);
        x[i].x += dtv * 0.5*v[i].x;
        x[i].y += dtv * 0.5*v[i].y;
        x[i].z += dtv * 0.5*v[i].z;
      }

  } else {
    const double * const mass = atom->mass;
    const int * const type = atom->type;
#if defined (_OPENMP)
#pragma omp parallel for private(i) default(none) schedule(static)
#endif
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = dtf / mass[type[i]];
        #pragma omp critical
        v[i].x = (f[i].x / damp_ * ratio_[type[i]]);
        v[i].y = (f[i].y / damp_ * ratio_[type[i]]);
        v[i].z = (f[i].z / damp_ * ratio_[type[i]]);
        x[i].x += dtv * 0.5*v[i].x;
        x[i].y += dtv * 0.5*v[i].y;
        x[i].z += dtv * 0.5*v[i].z;
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixBDOMP::final_integrate()
{
  // update v of atoms in group

  // update v and x of atoms in group

  dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  dbl3_t * _noalias const v = (dbl3_t *) atom->v[0];
  const dbl3_t * _noalias const f = (dbl3_t *) atom->f[0];
  const int * const mask = atom->mask;
  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  const double damp_ = damp;
  const double * const ratio_ = this->ratio;
  int i;

  if (atom->rmass) {
    const double * const rmass = atom->rmass;
    const int * const type = atom->type;
#if defined (_OPENMP)
#pragma omp parallel for private(i) default(none) schedule(static)
#endif
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = dtf / rmass[i];
        double rforce_x, rforce_y, rforce_z;
        #pragma omp critical
        {
        rforce_x = sqrt(gfac/ratio_[type[i]])*random->gaussian();
        rforce_y = sqrt(gfac/ratio_[type[i]])*random->gaussian();
        rforce_z = sqrt(gfac/ratio_[type[i]])*random->gaussian();
        }
        v[i].x = (f[i].x / damp_ / ratio_[type[i]] + rforce_x);
        v[i].y = (f[i].y / damp_ / ratio_[type[i]] + rforce_y);
        v[i].z = (f[i].z / damp_ / ratio_[type[i]] + rforce_z);
        x[i].x += dtv * (0.5*v[i].x + 0.5*rforce_x);
        x[i].y += dtv * (0.5*v[i].y + 0.5*rforce_y);
        x[i].z += dtv * (0.5*v[i].z + 0.5*rforce_z);
      }

  } else {
    const double * const mass = atom->mass;
    const int * const type = atom->type;
#if defined (_OPENMP)
#pragma omp parallel for private(i) default(none) schedule(static)
#endif
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = dtf / mass[type[i]];
        double rforce_x, rforce_y, rforce_z;
        #pragma omp critical
        {
        rforce_x = sqrt(gfac/ratio_[type[i]])*random->gaussian();
        rforce_y = sqrt(gfac/ratio_[type[i]])*random->gaussian();
        rforce_z = sqrt(gfac/ratio_[type[i]])*random->gaussian();
        }
        v[i].x = (f[i].x / damp_ / ratio_[type[i]] + rforce_x);
        v[i].y = (f[i].y / damp_ / ratio_[type[i]] + rforce_y);
        v[i].z = (f[i].z / damp_ / ratio_[type[i]] + rforce_z);
        x[i].x += dtv * (0.5*v[i].x + 0.5*rforce_x);
        x[i].y += dtv * (0.5*v[i].y + 0.5*rforce_y);
        x[i].z += dtv * (0.5*v[i].z + 0.5*rforce_z);
      }
  }
}

