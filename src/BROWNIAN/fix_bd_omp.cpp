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
  FixBD(lmp, narg, arg) 
{ 
  suffix_flag |= Suffix::OMP;
  random_thr = NULL;
  nthreads = 0;
}

FixBDOMP::~FixBDOMP()
{ 
  if (random_thr) {
    for (int i=1; i < nthreads; ++i)
      delete random_thr[i];

    delete[] random_thr;
    random_thr = NULL;
  }
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixBDOMP::initial_integrate(int /* vflag */)
{
  if (nthreads != comm->nthreads) {
    if (random_thr) {
      for (int i=1; i < nthreads; ++i)
        delete random_thr[i];

      delete[] random_thr;
    }

    nthreads = comm->nthreads;
    random_thr = new RanMars*[nthreads];
    for (int i=1; i < nthreads; ++i)
      random_thr[i] = NULL;

    // to ensure full compatibility with the serial Brownian style
    // we use is random number generator instance for thread 0
    random_thr[0] = random;
  }

#if defined(_OPENMP)
  tid = omp_get_thread_num();
#endif

#if defined(_OPENMP)
#pragma omp parallel default(none) shared(eflag,vflag)
#endif
  if ((tid > 0) && (random_thr[tid] == NULL))
    random_thr[tid] = new RanMars(Pair::lmp, seed + comm->me
                                  + comm->nprocs*tid);
  // update v and x of atoms in group

  dbl3_t * _noalias const x = (dbl3_t *) atom->x[0];
  dbl3_t * _noalias const v = (dbl3_t *) atom->v[0];
  const dbl3_t * _noalias const f = (dbl3_t *) atom->f[0];
  const int * const mask = atom->mask;
  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
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
        rforce_x = sqrt(gfac*ratio[type[i]])*random_thr[tid]->gaussian();
        rforce_y = sqrt(gfac*ratio[type[i]])*random_thr[tid]->gaussian();
        rforce_z = sqrt(gfac*ratio[type[i]])*random_thr[tid]->gaussian();
        v[i].x = (f[i].x * damp * ratio[type[i]] + rforce_x);
        v[i].y = (f[i].y * damp * ratio[type[i]] + rforce_y);
        v[i].z = (f[i].z * damp * ratio[type[i]] + rforce_z);
        x[i].x += dtv * v[i].x;
        x[i].y += dtv * v[i].y;
        x[i].z += dtv * v[i].z;
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
        rforce_x = sqrt(gfac*ratio[type[i]])*random_thr[tid]->gaussian();
        rforce_y = sqrt(gfac*ratio[type[i]])*random_thr[tid]->gaussian();
        rforce_z = sqrt(gfac*ratio[type[i]])*random_thr[tid]->gaussian();
        v[i].x = (f[i].x * damp * ratio[type[i]] + rforce_x);
        v[i].y = (f[i].y * damp * ratio[type[i]] + rforce_y);
        v[i].z = (f[i].z * damp * ratio[type[i]] + rforce_z);
        x[i].x += dtv * v[i].x;
        x[i].y += dtv * v[i].y;
        x[i].z += dtv * v[i].z;
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixBDOMP::final_integrate()
{
  // update v of atoms in group

#if 0
  dbl3_t * _noalias const v = (dbl3_t *) atom->v[0];
  const dbl3_t * _noalias const f = (dbl3_t *) atom->f[0];
  const int * const mask = atom->mask;
  const int nlocal = (igroup == atom->firstgroup) ? atom->nfirst : atom->nlocal;
  int i;

  if (atom->rmass) {
    const double * const rmass = atom->rmass;
#if defined (_OPENMP)
#pragma omp parallel for private(i) default(none) schedule(static)
#endif
    for (i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        const double dtfm = dtf / rmass[i];
        /*
        v[i].x += dtfm * f[i].x;
        v[i].y += dtfm * f[i].y;
        v[i].z += dtfm * f[i].z;
        */
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
        /*
        v[i].x += dtfm * f[i].x;
        v[i].y += dtfm * f[i].y;
        v[i].z += dtfm * f[i].z;
        */
      }
  }
#endif // 0
}

