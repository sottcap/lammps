.. index:: fix bond/react

fix bond/react command
======================

Syntax
""""""


.. parsed-literal::

   fix ID group-ID bond/react common_keyword values ...
     react react-ID react-group-ID Nevery Rmin Rmax template-ID(pre-reacted) template-ID(post-reacted) map_file individual_keyword values ...
     react react-ID react-group-ID Nevery Rmin Rmax template-ID(pre-reacted) template-ID(post-reacted) map_file individual_keyword values ...
     react react-ID react-group-ID Nevery Rmin Rmax template-ID(pre-reacted) template-ID(post-reacted) map_file individual_keyword values ...
     ...

* ID, group-ID are documented in :doc:`fix <fix>` command. Group-ID is ignored.
* bond/react = style name of this fix command
* the common keyword/values may be appended directly after 'bond/react'
* this applies to all reaction specifications (below)
* common\_keyword = *stabilization*
  
  .. parsed-literal::
  
       *stabilization* values = *no* or *yes* *group-ID* *xmax*
         *no* = no reaction site stabilization
         *yes* = perform reaction site stabilization
           *group-ID* = user-assigned prefix for the dynamic group of atoms not currently involved in a reaction
           *xmax* = xmax value that is used by an internally-created :doc:`nve/limit <fix_nve_limit>` integrator

* react = mandatory argument indicating new reaction specification
* react-ID = user-assigned name for the reaction
* react-group-ID = only atoms in this group are considered for the reaction
* Nevery = attempt reaction every this many steps
* Rmin = bonding pair atoms must be separated by more than Rmin to initiate reaction (distance units)
* Rmax = bonding pair atoms must be separated by less than Rmax to initiate reaction (distance units)
* template-ID(pre-reacted) = ID of a molecule template containing pre-reaction topology
* template-ID(post-reacted) = ID of a molecule template containing post-reaction topology
* map\_file = name of file specifying corresponding atom-IDs in the pre- and post-reacted templates
* zero or more individual keyword/value pairs may be appended to each react argument
* individual\_keyword = *prob* or *max\_rxn* or *stabilize\_steps* or *update\_edges*
  
  .. parsed-literal::
  
         *prob* values = fraction seed
           fraction = initiate reaction with this probability if otherwise eligible
           seed = random number seed (positive integer)
         *max_rxn* value = N
           N = maximum number of reactions allowed to occur
         *stabilize_steps* value = timesteps
           timesteps = number of timesteps to apply the internally-created :doc:`nve/limit <fix_nve_limit>` fix to reacting atoms
         *update_edges* value = *none* or *charges* or *custom*
           none = do not update topology near the edges of reaction templates
           charges = update atomic charges of all atoms in reaction templates
           custom = force the update of user-specified atomic charges



Examples
""""""""

For unabridged example scripts and files, see examples/USER/misc/bond\_react.


.. parsed-literal::

   molecule mol1 pre_reacted_topology.txt
   molecule mol2 post_reacted_topology.txt
   fix 5 all bond/react react myrxn1 all 1 0 3.25 mol1 mol2 map_file.txt

   molecule mol1 pre_reacted_rxn1.txt
   molecule mol2 post_reacted_rxn1.txt
   molecule mol3 pre_reacted_rxn2.txt
   molecule mol4 post_reacted_rxn2.txt
   fix 5 all bond/react stabilization yes nvt_grp .03 &
     react myrxn1 all 1 0 3.25 mol1 mol2 map_file_rxn1.txt prob 0.50 12345 &
     react myrxn2 all 1 0 2.75 mol3 mol4 map_file_rxn2.txt prob 0.25 12345
   fix 6 nvt_grp_REACT nvt temp 300 300 100 # set thermostat after bond/react

Description
"""""""""""

Initiate complex covalent bonding (topology) changes. These topology
changes will be referred to as 'reactions' throughout this
documentation. Topology changes are defined in pre- and post-reaction
molecule templates and can include creation and deletion of bonds,
angles, dihedrals, impropers, bond types, angle types, dihedral types,
atom types, or atomic charges. In addition, reaction by-products or
other molecules can be identified and deleted.

Fix bond/react does not use quantum mechanical (eg. fix qmmm) or
pairwise bond-order potential (eg. Tersoff or AIREBO) methods to
determine bonding changes a priori. Rather, it uses a distance-based
probabilistic criteria to effect predetermined topology changes in
simulations using standard force fields.

This fix was created to facilitate the dynamic creation of polymeric,
amorphous or highly cross-linked systems. A suggested workflow for
using this fix is: 1) identify a reaction to be simulated 2) build a
molecule template of the reaction site before the reaction has
occurred 3) build a molecule template of the reaction site after the
reaction has occurred 4) create a map that relates the
template-atom-IDs of each atom between pre- and post-reaction molecule
templates 5) fill a simulation box with molecules and run a simulation
with fix bond/react.

Only one 'fix bond/react' command can be used at a time. Multiple
reactions can be simultaneously applied by specifying multiple *react*
arguments to a single 'fix bond/react' command. This syntax is
necessary because the 'common keywords' are applied to all reactions.

The *stabilization* keyword enables reaction site stabilization.
Reaction site stabilization is performed by including reacting atoms
in an internally-created fix :doc:`nve/limit <fix_nve_limit>` time
integrator for a set number of timesteps given by the
*stabilize\_steps* keyword. While reacting atoms are being time
integrated by the internal nve/limit, they are prevented from being
involved in any new reactions. The *xmax* value keyword should
typically be set to the maximum distance that non-reacting atoms move
during the simulation.

Fix bond/react creates and maintains two important dynamic groups of
atoms when using the *stabilization* keyword. The first group contains
all atoms currently involved in a reaction; this group is
automatically thermostatted by an internally-created
:doc:`nve/limit <fix_nve_limit>` integrator. The second group contains
all atoms currently not involved in a reaction. This group should be
used by a thermostat in order to time integrate the system. The name
of this group of non-reacting atoms is created by appending '\_REACT'
to the group-ID argument of the *stabilization* keyword, as shown in
the second example above.

.. note::

   When using reaction stabilization, you should generally not have
   a separate thermostat which acts on the 'all' group.

The group-ID set using the *stabilization* keyword can be an existing
static group or a previously-unused group-ID. It cannot be specified
as 'all'. If the group-ID is previously unused, the fix bond/react
command creates a :doc:`dynamic group <group>` that is initialized to
include all atoms. If the group-ID is that of an existing static
group, the group is used as the parent group of new,
internally-created dynamic group. In both cases, this new dynamic
group is named by appending '\_REACT' to the group-ID, e.g.
nvt\_grp\_REACT. By specifying an existing group, you may thermostat
constant-topology parts of your system separately. The dynamic group
contains only atoms not involved in a reaction at a given timestep,
and therefore should be used by a subsequent system-wide time
integrator such as nvt, npt, or nve, as shown in the second example
above (full examples can be found at examples/USER/misc/bond\_react).
The time integration command should be placed after the fix bond/react
command due to the internal dynamic grouping performed by fix
bond/react.

.. note::

   If the group-ID is an existing static group, react-group-IDs
   should also be specified as this static group, or a subset.

The following comments pertain to each *react* argument (in other
words, can be customized for each reaction, or reaction step):

A check for possible new reaction sites is performed every *Nevery*
timesteps.

Three physical conditions must be met for a reaction to occur. First,
a bonding atom pair must be identified within the reaction distance
cutoffs. Second, the topology surrounding the bonding atom pair must
match the topology of the pre-reaction template. Finally, any reaction
constraints listed in the map file (see below) must be satisfied. If
all of these conditions are met, the reaction site is eligible to be
modified to match the post-reaction template.

A bonding atom pair will be identified if several conditions are met.
First, a pair of atoms I,J within the specified react-group-ID of type
itype and jtype must be separated by a distance between *Rmin* and
*Rmax*\ . It is possible that multiple bonding atom pairs are
identified: if the bonding atoms in the pre-reacted template are  1-2
neighbors, i.e. directly bonded, the farthest bonding atom partner is
set as its bonding partner; otherwise, the closest potential partner
is chosen. Then, if both an atom I and atom J have each other as their
bonding partners, these two atoms are identified as the bonding atom
pair of the reaction site. Once this unique bonding atom pair is
identified for each reaction, there could two or more reactions that
involve a given atom on the same timestep. If this is the case, only
one such reaction is permitted to occur. This reaction is chosen
randomly from all potential reactions. This capability allows e.g. for
different reaction pathways to proceed from identical reaction sites
with user-specified probabilities.

The pre-reacted molecule template is specified by a molecule command.
This molecule template file contains a sample reaction site and its
surrounding topology. As described below, the bonding atom pairs of
the pre-reacted template are specified by atom ID in the map file. The
pre-reacted molecule template should contain as few atoms as possible
while still completely describing the topology of all atoms affected
by the reaction. For example, if the force field contains dihedrals,
the pre-reacted template should contain any atom within three bonds of
reacting atoms.

Some atoms in the pre-reacted template that are not reacting may have
missing topology with respect to the simulation. For example, the
pre-reacted template may contain an atom that, in the simulation, is
currently connected to the rest of a long polymer chain. These are
referred to as edge atoms, and are also specified in the map file. All
pre-reaction template atoms should be linked to a bonding atom, via at
least one path that does not involve edge atoms. When the pre-reaction
template contains edge atoms, not all atoms, bonds, charges, etc.
specified in the reaction templates will be updated. Specifically,
topology that involves only atoms that are 'too near' to template
edges will not be updated. The definition of 'too near the edge'
depends on which interactions are defined in the simulation. If the
simulation has defined dihedrals, atoms within two bonds of edge atoms
are considered 'too near the edge.' If the simulation defines angles,
but not dihedrals, atoms within one bond of edge atoms are considered
'too near the edge.' If just bonds are defined, only edge atoms are
considered 'too near the edge.'

.. note::

   Small molecules, i.e. ones that have all their atoms contained
   within the reaction templates, never have edge atoms.

Note that some care must be taken when a building a molecule template
for a given simulation. All atom types in the pre-reacted template
must be the same as those of a potential reaction site in the
simulation. A detailed discussion of matching molecule template atom
types with the simulation is provided on the :doc:`molecule <molecule>`
command page.

The post-reacted molecule template contains a sample of the reaction
site and its surrounding topology after the reaction has occurred. It
must contain the same number of atoms as the pre-reacted template. A
one-to-one correspondence between the atom IDs in the pre- and
post-reacted templates is specified in the map file as described
below. Note that during a reaction, an atom, bond, etc. type may
change to one that was previously not present in the simulation. These
new types must also be defined during the setup of a given simulation.
A discussion of correctly handling this is also provided on the
:doc:`molecule <molecule>` command page.

.. note::

   When a reaction occurs, it is possible that the resulting
   topology/atom (e.g. special bonds, dihedrals, etc.) exceeds that of
   the existing system and reaction templates. As when inserting
   molecules, enough space for this increased topology/atom must be
   reserved by using the relevant "extra" keywords to the
   :doc:`read\_data <read_data>` or :doc:`create\_box <create_box>` commands.

The map file is a text document with the following format:

A map file has a header and a body. The header of map file the
contains one mandatory keyword and four optional keywords. The
mandatory keyword is 'equivalences':


.. parsed-literal::

   N *equivalences* = # of atoms N in the reaction molecule templates

The optional keywords are 'edgeIDs', 'deleteIDs', 'customIDs' and
'constraints':


.. parsed-literal::

   N *edgeIDs* = # of edge atoms N in the pre-reacted molecule template
   N *deleteIDs* = # of atoms N that are specified for deletion
   N *customIDs* = # of atoms N that are specified for a custom update
   N *constraints* = # of specified reaction constraints N

The body of the map file contains two mandatory sections and four
optional sections. The first mandatory section begins with the keyword
'BondingIDs' and lists the atom IDs of the bonding atom pair in the
pre-reacted molecule template. The second mandatory section begins
with the keyword 'Equivalences' and lists a one-to-one correspondence
between atom IDs of the pre- and post-reacted templates. The first
column is an atom ID of the pre-reacted molecule template, and the
second column is the corresponding atom ID of the post-reacted
molecule template. The first optional section begins with the keyword
'EdgeIDs' and lists the atom IDs of edge atoms in the pre-reacted
molecule template. The second optional section begins with the keyword
'DeleteIDs' and lists the atom IDs of pre-reaction template atoms to
delete. The third optional section begins with the keyword 'Custom
Edges' and allows for forcing the update of a specific atom's atomic
charge. The first column is the ID of an atom near the edge of the
pre-reacted molecule template, and the value of the second column is
either 'none' or 'charges.' Further details are provided in the
discussion of the 'update\_edges' keyword. The fourth optional section
begins with the keyword 'Constraints' and lists additional criteria
that must be satisfied in order for the reaction to occur. Currently,
there are three types of constraints available, as discussed below.

A sample map file is given below:


----------



.. parsed-literal::

   # this is a map file

   7 equivalences
   2 edgeIDs

   BondingIDs

   3
   5

   EdgeIDs

   1
   7

   Equivalences

   1   1
   2   2
   3   3
   4   4
   5   5
   6   6
   7   7


----------


Any number of additional constraints may be specified in the
Constraints section of the map file. The constraint of type 'distance'
has syntax as follows:


.. parsed-literal::

   distance *ID1* *ID2* *rmin* *rmax*

where 'distance' is the required keyword, *ID1* and *ID2* are
pre-reaction atom IDs, and these two atoms must be separated by a
distance between *rmin* and *rmax* for the reaction to occur.

The constraint of type 'angle' has the following syntax:


.. parsed-literal::

   angle *ID1* *ID2* *ID3* *amin* *amax*

where 'angle' is the required keyword, *ID1*\ , *ID2* and *ID3* are
pre-reaction atom IDs, and these three atoms must form an angle
between *amin* and *amax* for the reaction to occur (where *ID2* is
the central atom). Angles must be specified in degrees. This
constraint can be used to enforce a certain orientation between
reacting molecules.

The constraint of type 'arrhenius' imposes an additional reaction
probability according to the temperature-dependent Arrhenius equation:

.. image:: Eqs/fix_bond_react.jpg
   :align: center

The Arrhenius constraint has the following syntax:


.. parsed-literal::

   arrhenius *A* *n* *E_a* *seed*

where 'arrhenius' is the required keyword, *A* is the pre-exponential
factor, *n* is the exponent of the temperature dependence, *E\_a* is
the activation energy (:doc:`units <units>` of energy), and *seed* is a
random number seed. The temperature is defined as the instantaneous
temperature averaged over all atoms in the reaction site, and is
calculated in the same manner as for example
:doc:`compute\_temp\_chunk <compute_temp_chunk>`. Currently, there are no
options for additional temperature averaging or velocity-biased
temperature calculations. A uniform random number between 0 and 1 is
generated using *seed*\ ; if this number is less than the result of the
Arrhenius equation above, the reaction is permitted to occur.

Once a reaction site has been successfully identified, data structures
within LAMMPS that store bond topology are updated to reflect the
post-reacted molecule template. All force fields with fixed bonds,
angles, dihedrals or impropers are supported.

A few capabilities to note: 1) You may specify as many *react*
arguments as desired. For example, you could break down a complicated
reaction mechanism into several reaction steps, each defined by its
own *react* argument. 2) While typically a bond is formed or removed
between the bonding atom pairs specified in the pre-reacted molecule
template, this is not required. 3) By reversing the order of the pre-
and post- reacted molecule templates in another *react* argument, you
can allow for the possibility of one or more reverse reactions.

The optional keywords deal with the probability of a given reaction
occurring as well as the stable equilibration of each reaction site as
it occurs:

The *prob* keyword can affect whether or not an eligible reaction
actually occurs. The fraction setting must be a value between 0.0 and
1.0. A uniform random number between 0.0 and 1.0 is generated and the
eligible reaction only occurs if the random number is less than the
fraction. Up to N reactions are permitted to occur, as optionally
specified by the *max\_rxn* keyword.

The *stabilize\_steps* keyword allows for the specification of how many
timesteps a reaction site is stabilized before being returned to the
overall system thermostat. In order to produce the most physical
behavior, this 'reaction site equilibration time' should be tuned to
be as small as possible while retaining stability for a given system
or reaction step. After a limited number of case studies, this number
has been set to a default of 60 timesteps. Ideally, it should be
individually tuned for each fix reaction step. Note that in some
situations, decreasing rather than increasing this parameter will
result in an increase in stability.

The *update\_edges* keyword can increase the number of atoms whose
atomic charges are updated, when the pre-reaction template contains
edge atoms. When the value is set to 'charges,' all atoms' atomic
charges are updated to those specified by the post-reaction template,
including atoms near the edge of reaction templates. When the value is
set to 'custom,' an additional section must be included in the map
file that specifies whether or not to update charges, on a per-atom
basis. The format of this section is detailed above. Listing a
pre-reaction atom ID with a value of 'charges' will force the update
of the atom's charge, even if it is near a template edge. Atoms not
near a template edge are unaffected by this setting.

A few other considerations:

Many reactions result in one or more atoms that are considered
unwanted by-products. Therefore, bond/react provides the option to
delete a user-specified set of atoms. These pre-reaction atoms are
identified in the map file. A deleted atom must still be included in
the post-reaction molecule template, in which it cannot be bonded to
an atom that is not deleted. In addition to deleting unwanted reaction
by-products, this feature can be used to remove specific topologies,
such as small rings, that may be otherwise indistinguishable.

Optionally, you can enforce additional behaviors on reacting atoms.
For example, it may be beneficial to force reacting atoms to remain at
a certain temperature. For this, you can use the internally-created
dynamic group named "bond\_react\_MASTER\_group", which consists of all
atoms currently involved in a reaction. For example, adding the
following command would add an additional thermostat to the group of
all currently-reacting atoms:


.. parsed-literal::

   fix 1 bond_react_MASTER_group temp/rescale 1 300 300 10 1

.. note::

   This command must be added after the fix bond/react command, and
   will apply to all reactions.

Computationally, each timestep this fix operates, it loops over
neighbor lists (for bond-forming reactions) and computes distances
between pairs of atoms in the list. It also communicates between
neighboring processors to coordinate which bonds are created and/or
removed. All of these operations increase the cost of a timestep. Thus
you should be cautious about invoking this fix too frequently.

You can dump out snapshots of the current bond topology via the dump
local command.


----------


**Restart, fix\_modify, output, run start/stop, minimize info:**

Cumulative reaction counts for each reaction are written to :doc:`binary restart files <restart>`. These values are associated with the
reaction name (react-ID). Additionally, internally-created per-atom
properties are stored to allow for smooth restarts. None of the
:doc:`fix\_modify <fix_modify>` options are relevant to this fix.

This fix computes one statistic for each *react* argument that it
stores in a global vector, of length 'number of react arguments', that
can be accessed by various :doc:`output commands <Howto_output>`. The
vector values calculated by this fix are "intensive".

These is 1 quantity for each react argument:

* (1) cumulative # of reactions occurred

No parameter of this fix can be used with the *start/stop* keywords
of the :doc:`run <run>` command.  This fix is not invoked during :doc:`energy minimization <minimize>`.

When fix bond/react is 'unfixed,' all internally-created groups are
deleted. Therefore, fix bond/react can only be unfixed after unfixing
all other fixes that use any group created by fix bond/react.

Restrictions
""""""""""""


This fix is part of the USER-MISC package.  It is only enabled if
LAMMPS was built with that package.  See the
:doc:`Build package <Build_package>` doc page for more info.

Related commands
""""""""""""""""

:doc:`fix bond/create <fix_bond_create>`,
:doc:`fix bond/break <fix_bond_break>`,
:doc:`fix bond/swap <fix_bond_swap>`,
:doc:`dump local <dump>`, :doc:`special\_bonds <special_bonds>`

Default
"""""""

The option defaults are stabilization = no, prob = 1.0, stabilize\_steps = 60,
update\_edges = none


----------


.. _Gissinger:



**(Gissinger)** Gissinger, Jensen and Wise, Polymer, 128, 211 (2017).


.. _lws: http://lammps.sandia.gov
.. _ld: Manual.html
.. _lc: Commands_all.html
