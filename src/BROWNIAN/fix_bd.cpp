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

#include <cstdio>
#include <cstring>
#include <math.h>
#include "fix_bd.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "respa.h"
#include "error.h"
#include "comm.h"
#include "input.h"
#include "variable.h"
#include "random_mars.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixBD::FixBD(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  random(NULL), seed(0), damp(0), tstr(NULL), ratio(NULL)
{
  if (narg < 5)
    error->all(FLERR,"Illegal fix bd command");

  dynamic_group_allow = 1;
  time_integrate = 1;

  // initialize Marsaglia RNG with processor-unique seed

  ratio = new double[atom->ntypes+1];

  for(int i=0;i<atom->ntypes;i++) {
    ratio[i] = 1.0;
  }

  if (strstr(arg[3],"v_") == arg[3]) {
    int n = strlen(&arg[3][2]) + 1;
    tstr = new char[n];
    strcpy(tstr,&arg[3][2]);
  } else {
    temp = force->numeric(FLERR,arg[3]);
  }

  seed = force->inumeric(FLERR,arg[4]);
  damp = force->numeric(FLERR,arg[5]);

  random = new RanMars(lmp,seed + comm->me);

  int iarg = 5;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"scale") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix langevin command");
      int itype = force->inumeric(FLERR,arg[iarg+1]);
      double scale = force->numeric(FLERR,arg[iarg+2]);
      if (itype <= 0 || itype > atom->ntypes)
        error->all(FLERR,"Illegal fix langevin command");
      ratio[itype] = scale;
      iarg += 3;
    }
    iarg++;
  }
}

/* ---------------------------------------------------------------------- */

FixBD::~FixBD()
{
  delete random;
  delete [] ratio;
  
  if(tstr) {
    delete [] tstr;
  }
}

/* ---------------------------------------------------------------------- */

int FixBD::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBD::init()
{
  double boltz = force->boltz;

  dtv = update->dt;
  dtf = update->dt * force->ftm2v;
  gfac = 2*boltz*temp*damp/update->dt/force->mvv2e;

  if (strstr(update->integrate_style,"respa"))
    step_respa = ((Respa *) update->integrate)->step;

  if (tstr) {
    temp = input->variable->find(tstr);
    if (temp < 0)
      error->all(FLERR,"Variable name for fix langevin does not exist");
  }
}

/* ----------------------------------------------------------------------
   allow for both per-type and per-atom mass
------------------------------------------------------------------------- */

void FixBD::initial_integrate(int /*vflag*/)
{
  double dtfm;
  double rforce;
  double boltz = force->boltz;

  // update v and x of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    const int * const type = atom->type;
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        for(int d=0;d<3;d++) {
          rforce = sqrt(gfac*rmass[i])*random->gaussian();
          v[i][d] = (f[i][d] * damp * ratio[type[i]] + rforce);
          x[i][d] += dtv * v[i][d];
        }
      }

  } else {
    const int * const type = atom->type;
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        for(int d=0;d<3;d++) {
          rforce = sqrt(gfac*mass[type[i]])*random->gaussian();
          v[i][d] = (f[i][d] * damp  * ratio[type[i]] + rforce);
          x[i][d] += dtv * v[i][d];
        }
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixBD::final_integrate()
{
  double dtfm;

  // update v of atoms in group

  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (rmass) {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / rmass[i];
        /*
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
        */
      }

  } else {
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit) {
        dtfm = dtf / mass[type[i]];
        /*
        v[i][0] += dtfm * f[i][0];
        v[i][1] += dtfm * f[i][1];
        v[i][2] += dtfm * f[i][2];
        */
      }
  }
}

/* ---------------------------------------------------------------------- */

void FixBD::initial_integrate_respa(int vflag, int ilevel, int /*iloop*/)
{
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;

  // innermost level - BD update of v and x
  // all other levels - BD update of v

  if (ilevel == 0) initial_integrate(vflag);
  else final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixBD::final_integrate_respa(int ilevel, int /*iloop*/)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixBD::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}
