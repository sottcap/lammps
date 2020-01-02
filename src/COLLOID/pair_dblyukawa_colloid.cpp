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
   Contributing authors: Randy Schunk (Sandia)
------------------------------------------------------------------------- */

#include <cmath>
#include <cstdlib>
#include "pair_dblyukawa_colloid.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairDblYukawaColloid::PairDblYukawaColloid(LAMMPS *lmp) : PairYukawa(lmp)
{
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

void PairDblYukawaColloid::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair,radi,radj;
  double rsq,r,rinv,rinv2,screening1, screening2,forceyukawa,factor;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  double *radius = atom->radius;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  tagint *tag = atom->tag;
  tagint *molecule = atom->molecule;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    radi = radius[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      if (molecule[ii] == molecule[jj]) continue;
      j = jlist[jj];
      factor = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];
      radj = radius[j];

      if (rsq < cutsq[itype][jtype]) {
//        rsq = (rsq > (radi+radj)*(radi+radj)*0.9*0.9) ? rsq : (radi+radj)*(radi+radj)*0.9*0.9;
        r = sqrt(rsq);
        rinv = 1.0/r;
        rinv2 = 1.0/rsq;
        screening1 = exp(-kappas[0]*(r-(radi+radj)));
        screening2 = exp(-kappas[1]*(r-(radi+radj)));
//        forceyukawa = a1[itype][jtype] * screening1 + a2[itype][jtype] * screening2;
        forceyukawa = a1[itype][jtype] * screening1 * (kappas[0] + rinv) + a2[itype][jtype] * screening2  * (kappas[1] + rinv);

        fpair = factor*forceyukawa * rinv2;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = a1[itype][jtype]/kappas[0] * screening1 + a2[itype][jtype]/kappas[1] * screening2 - offset[itype][jtype];
          evdwl *= factor;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairDblYukawaColloid::init_style()
{
  if (!atom->sphere_flag)
    error->all(FLERR,"Pair dblyukawa/colloid requires atom style sphere");

  neighbor->request(this,instance_me);

  // require that atom radii are identical within each type

  for (int i = 1; i <= atom->ntypes; i++)
    if (!atom->radius_consistency(i,rad[i]))
      error->all(FLERR,"Pair dblyukawa/colloid requires atoms with same type "
                 "have same radius");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairDblYukawaColloid::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    a1[i][j] = mix_energy(a1[i][i],a1[j][j],1.0,1.0);
    a2[i][j] = mix_energy(a2[i][i],a2[j][j],1.0,1.0);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  if (offset_flag && (kappas[0] != 0.0 && kappas[1] != 0.0)) {
    double screening1 = exp(-kappas[0] * (cut[i][j] - (rad[i]+rad[j])));
    double screening2 = exp(-kappas[1] * (cut[i][j] - (rad[i]+rad[j])));
    offset[i][j] = a1[i][j]/kappas[0] * screening1 + a2[i][j]/kappas[1] * screening2;
  } else offset[i][j] = 0.0;

  a1[j][i] = a1[i][j];
  a2[j][i] = a2[i][j];
  offset[j][i] = offset[i][j];

  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairDblYukawaColloid::single(int /*i*/, int /*j*/, int itype, int jtype,
                                 double rsq,
                                 double /*factor_coul*/, double factor_lj,
                                 double &fforce)
{
  double r,rinv,screening1,screening2,forceyukawa,phi;

  r = sqrt(rsq);
  rinv = 1.0/r;
  screening1 = exp(-kappas[0]*(r-(rad[itype]+rad[jtype])));
  screening2 = exp(-kappas[1]*(r-(rad[itype]+rad[jtype])));
  forceyukawa = a1[itype][jtype] * screening1 + a2[itype][jtype] * screening2;
  fforce = factor_lj*forceyukawa * rinv;

  phi = a1[itype][jtype]/kappas[0] * screening1 + a2[itype][jtype]/kappas[1] * screening2  - offset[itype][jtype];
  return factor_lj*phi;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairDblYukawaColloid::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(rad,n+1,"pair:rad");
  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(a1,n+1,n+1,"pair:a1");
  memory->create(a2,n+1,n+1,"pair:a2");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairDblYukawaColloid::settings(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR,"Illegal pair_style command");

  kappas[0] = force->numeric(FLERR,arg[0]);
  kappas[1] = force->numeric(FLERR,arg[1]);
  cut_global = force->numeric(FLERR,arg[2]);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairDblYukawaColloid::coeff(int narg, char **arg)
{
  if (narg < 4 || narg > 5)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double a1_one = force->numeric(FLERR,arg[2]);
  double a2_one = force->numeric(FLERR,arg[3]);

  double cut_one = cut_global;
  if (narg == 5) cut_one = force->numeric(FLERR,arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      a1[i][j] = a1_one;
      a2[i][j] = a2_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairDblYukawaColloid::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&a1[i][j],sizeof(double),1,fp);
        fwrite(&a2[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairDblYukawaColloid::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&a1[i][j],sizeof(double),1,fp);
          fread(&a2[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&a1[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&a2[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairDblYukawaColloid::write_restart_settings(FILE *fp)
{
  fwrite(&kappas[0],sizeof(double),1,fp);
  fwrite(&kappas[1],sizeof(double),1,fp);
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairDblYukawaColloid ::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&kappas[0],sizeof(double),1,fp);
    fread(&kappas[1],sizeof(double),1,fp);
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&kappas[0],1,MPI_DOUBLE,0,world);
  MPI_Bcast(&kappas[1],1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

