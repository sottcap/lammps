.. index:: kim\_init

kim\_init command
=================

kim\_interactions command
=========================

kim\_query command
==================

Syntax
""""""


.. parsed-literal::

   kim_init model user_units unitarg
   kim_interactions typeargs
   kim_query variable formatarg query_function queryargs

* model = name of the KIM interatomic model (the KIM ID for models archived in OpenKIM)
* user\_units = the LAMMPS :doc:`units <units>` style assumed in the LAMMPS input script
* unitarg = *unit\_conversion\_mode* (optional)
* typeargs = atom type to species mapping (one entry per atom type) or *fixed\_types* for models with a preset fixed mapping
* variable = name of a (string style) variable where the result of the query is stored
* formatarg = *split* (optional)
* query\_function = name of the OpenKIM web API query function to be used
* queryargs = a series of *keyword=value* pairs that represent the web query; supported keywords depend on the query function

Examples
""""""""


.. parsed-literal::

   kim_init SW_StillingerWeber_1985_Si__MO_405512056662_005 metal
   kim_interactions Si
   kim_init Sim_LAMMPS_ReaxFF_StrachanVanDuinChakraborty_2003_CHNO__SM_107643900657_000 real
   kim_init Sim_LAMMPS_ReaxFF_StrachanVanDuinChakraborty_2003_CHNO__SM_107643900657_000 metal unit_conversion_mode
   kim_interactions C H O
   Sim_LAMMPS_IFF_PCFF_HeinzMishraLinEmami_2015Ver1v5_FccmetalsMineralsSolvents Polymers__SM_039297821658_000 real
   kim_interactions fixed_types
   kim_query a0 get_lattice_constant_cubic crystal=["fcc"] species=["Al"] units=["angstrom"]

Description
"""""""""""

The set of *kim\_commands* provide a high-level wrapper around the
`Open Knowledgebase of Interatomic Models (OpenKIM) <https://openkim.org>`_
repository of interatomic models (IMs) (potentials and force fields),
so that they can be used by LAMMPS scripts.  These commands do not implement
any computations directly, but rather generate LAMMPS input commands based
on the information retrieved from the OpenKIM repository to initialize and
activate OpenKIM IMs and query their predictions for use in the LAMMPS script.
All LAMMPS input commands generated and executed by *kim\_commands* are
echoed to the LAMMPS log file.

Benefits of Using OpenKIM IMs
-----------------------------

Employing OpenKIM IMs provides LAMMPS users with multiple benefits:

Reliability
^^^^^^^^^^^

* All content archived in OpenKIM is reviewed by the `KIM Editor <https://openkim.org/governance/>`_ for quality.
* IMs in OpenKIM are archived with full provenance control. Each is associated with a maintainer responsible for the integrity of the content. All changes are tracked and recorded.
* IMs in OpenKIM are exhaustively tested using `KIM Tests <https://openkim.org/doc/evaluation/kim-tests/>`_ that compute a host of material properties, and `KIM Verification Checks <https://openkim.org/doc/evaluation/kim-verification-checks/>`_ that provide the user with information on various aspects of the IM behavior and coding correctness. This information is displayed on the IM's page accessible through the  `OpenKIM browse interface <https://openkim.org/browse>`_.

Reproducibility
^^^^^^^^^^^^^^^

* Each IM in OpenKIM is issued a unique identifier (`KIM ID <https://openkim.org/doc/schema/kim-ids/>`_), which includes a version number (last three digits).  Any changes that can result in different numerical values lead to a version increment in the KIM ID. This makes it possible to reproduce simulations since the specific version of a specific IM used can be retrieved using its KIM ID.
* OpenKIM is a member organization of `DataCite <https://datacite.org/>`_ and issues digital object identifiers (DOIs) to all IMs archived in OpenKIM. This makes it possible to cite the IM code used in a simulation in a publications to give credit to the developers and further facilitate reproducibility.

Convenience
^^^^^^^^^^^

* IMs in OpenKIM are distributed in binary form along with LAMMPS and can be used in a LAMMPS input script simply by providing their KIM ID in the *kim\_init* command documented on this page.
* The *kim\_query* web query tool provides the ability to use the predictions of IMs for supported material properties (computed via `KIM Tests <https://openkim.org/doc/evaluation/kim-tests/>`_) as part of a LAMMPS input script setup and analysis.
* Support is provided for unit conversion between the :doc:`unit style <units>` used in the LAMMPS input script and the units required by the OpenKIM IM. This makes it possible to use a single input script with IMs using different units without change and minimizes the likelihood of errors due to incompatible units.

.. _IM\_types:



Types of IMs in OpenKIM
-----------------------

There are two types of IMs archived in OpenKIM:

1. The first type is called a *KIM Portable Model* (PM). A KIM PM is an independent computer implementation of an IM written in one of the languages supported by KIM (C, C++, Fortran) that conforms to the KIM Application Programming Interface (`KIM API <https://openkim.org/kim-api/>`_) Portable Model Interface (PMI) standard. A KIM PM will work seamlessly with any simulation code that supports the KIM API/PMI standard (including LAMMPS; see `complete list of supported codes <https://openkim.org/projects-using-kim/>`_).
2. The second type is called a *KIM Simulator Model* (SM). A KIM SM is an IM that is implemented natively within a simulation code (\ *simulator*\ ) that supports the KIM API Simulator Model Interface (SMI); in this case LAMMPS. A separate SM package is archived in OpenKIM for each parameterization of the IM, which includes all of the necessary parameter files, LAMMPS commands, and metadata (supported species, units, etc.) needed to run the IM in LAMMPS.

With these two IM types, OpenKIM can archive and test almost all IMs that
can be used by LAMMPS. (It is easy to contribute new IMs to OpenKIM, see
the `upload instructions <https://openkim.org/doc/repository/adding-content/>`_.)

OpenKIM IMs are uniquely identified by a
`KIM ID <https://openkim.org/doc/schema/kim-ids/>`_.
The extended KIM ID consists of
a human-readable prefix identifying the type of IM, authors, publication year,
and supported species, separated by two underscores from the KIM ID itself,
which begins with an IM code
(\ *MO* for a KIM Portable Model, and *SM* for a KIM Simulator Model)
followed by a unique 12-digit code and a 3-digit version identifier.
By convention SM prefixes begin with *Sim\_* to readily identify them.


.. parsed-literal::

   SW_StillingerWeber_1985_Si__MO_405512056662_005
   Sim_LAMMPS_ReaxFF_StrachanVanDuinChakraborty_2003_CHNO__SM_107643900657_000

Each OpenKIM IM has a dedicated "Model Page" on `OpenKIM <https://openkim.org>`_
providing all the information on the IM including a title, description,
authorship and citation information, test and verification check results,
visualizations of results, a wiki with documentation and user comments, and
access to raw files, and other information.
The URL for the Model Page is constructed from the
`extended KIM ID <https://openkim.org/doc/schema/kim-ids/>`_ of the IM:


.. parsed-literal::

   https://openkim.org/id/extended_KIM_ID

For example for the Stillinger-Weber potential
listed above the Model Page is located at:


.. parsed-literal::

   `https://openkim.org/id/SW_StillingerWeber_1985_Si__MO_405512056662_005 <https://openkim.org/id/SW_StillingerWeber_1985_Si__MO_405512056662_005>`_

See the `current list of KIM PMs and SMs archived in OpenKIM <https://openkim.org/browse/models/by-species>`_.
This list is sorted by species and can be filtered to display only
IMs for certain species combinations.

See `Obtaining KIM Models <http://openkim.org/doc/usage/obtaining-models>`_ to
learn how to install a pre-build binary of the OpenKIM Repository of Models.

.. note::

   It is also possible to locally install IMs not archived in OpenKIM,
   in which case their names do not have to conform to the KIM ID format.

Using OpenKIM IMs with LAMMPS
-----------------------------

Two commands are employed when using OpenKIM IMs, one to select the
IM and perform necessary initialization (*kim\_init*), and the second
to set up the IM for use by executing any necessary LAMMPS commands
(*kim\_interactions*). Both are required.

See the *examples/kim* directory for example input scripts that use KIM PMs
and KIM SMs.

OpenKIM IM Initialization (*kim\_init*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *kim\_init* mode command must be issued **before**
the simulation box is created (normally at the top of the file).
This command sets the OpenKIM IM that will be used and may issue
additional commands changing LAMMPS default settings that are required
for using the selected IM (such as :doc:`units <units>` or
:doc:`atom\_style <atom_style>`). If needed, those settings can be overridden,
however, typically a script containing a *kim\_init* command
would not include *units* and *atom\_style* commands.

The required arguments of *kim\_init* are the *model* name of the
IM to be used in the simulation (for an IM archived in OpenKIM this is
its `extended KIM ID <https://openkim.org/doc/schema/kim-ids/>`_, and
the *user\_units*, which are the LAMMPS :doc:`units style <units>` used
in the input script.  (Any dimensioned numerical values in the input
script and values read in from files are expected to be in the
*user\_units* system.)

The selected IM can be either a :ref:`KIM PM or a KIM SM <IM_types>`.
For a KIM SM, the *kim\_init* command verifies that the SM is designed
to work with LAMMPS (and not another simulation code).
In addition, the LAMMPS version used for defining
the SM and the LAMMPS version being currently run are
printed to help diagnose any incompatible changes to input script or
command syntax between the two LAMMPS versions.

Based on the selected model *kim\_init* may modify the
:doc:`atom\_style <atom_style>`.
Some SMs have requirements for this setting. If this is the case, then
*atom\_style* will be set to the required style. Otherwise, the value is left
unchanged (which in the absence of an *atom\_style* command in the input script
is the :doc:`default atom\_style value <atom_style>`).

Regarding units, the *kim\_init* command behaves in different ways depending
on whether or not *unit conversion mode* is activated as indicated by the
optional *unitarg* argument.
If unit conversion mode is **not** active, then *user\_units* must
either match the required units of the IM or the IM must be able
to adjust its units to match. (The latter is only possible with some KIM PMs;
SMs can never adjust their units.) If a match is possible, the LAMMPS
:doc:`units <units>` command is called to set the units to
*user\_units*. If the match fails, the simulation is terminated with
an error.

Here is an example of a LAMMPS script to compute the cohesive energy
of a face-centered cubic (fcc) lattice for the Ercolessi and Adams (1994)
potential for Al:


.. parsed-literal::

   kim_init         EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005 metal
   boundary         p p p
   lattice          fcc 4.032
   region           simbox block 0 1 0 1 0 1 units lattice
   create_box       1 simbox
   create_atoms     1 box
   mass             1 26.981539
   kim_interactions Al
   run              0
   variable         Ec equal (pe/count(all))/${_u_energy}
   print            "Cohesive Energy = ${EcJ} eV"

The above script will end with an error in the *kim\_init* line if the
IM is changed to another potential for Al that does not work with *metal*
units. To address this *kim\_init* offers the *unit\_conversion\_mode*.
If unit conversion mode *is* active, then *kim\_init* calls the LAMMPS
:doc:`units <units>` command to set the units to the IM's required or
preferred units. Conversion factors between the IM's units and the *user\_units*
are defined for all :doc:`physical quantities <units>` (mass, distance, etc.).
(Note that converting to or from the "lj" unit style is not supported.)
These factors are stored as :doc:`internal style variables <variable>` with
the following standard names:


.. parsed-literal::

   _u_mass
   _u_distance
   _u_time
   _u_energy
   _u_velocity
   _u_force
   _u_torque
   _u_temperature
   _u_pressure
   _u_viscosity
   _u_charge
   _u_dipole
   _u_efield
   _u_density

If desired, the input script can be designed to work with these conversion
factors so that the script will work without change with any OpenKIM IM.
(This approach is used in the
`OpenKIM Testing Framework <https://openkim.org/doc/evaluation/kim-tests/>`_.)
For example, the script given above for the cohesive energy of fcc Al
can be rewritten to work with any IM regardless of units. The following
script constructs an fcc lattice with a lattice parameter defined in
meters, computes the total energy, and prints the cohesive energy in
Joules regardless of the units of the IM.


.. parsed-literal::

   kim_init         EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005 si unit_conversion_mode
   boundary         p p p
   lattice          fcc 4.032e-10\*${_u_distance}
   region           simbox block 0 1 0 1 0 1 units lattice
   create_box       1 simbox
   create_atoms     1 box
   mass             1 4.480134e-26\*${_u_mass}
   kim_interactions Al
   run              0
   variable         Ec_in_J equal (pe/count(all))/${_u_energy}
   print            "Cohesive Energy = ${Ec_in_J} J"

Note the multiplication by ${\_u_distance} and ${\_u_mass} to convert
from SI units (specified in the *kim\_init* command) to whatever units the
IM uses (metal in this case), and the division by ${\_u_energy}
to convert from the IM's energy units to SI units (Joule). This script
will work correctly for any IM for Al (KIM PM or SM) selected by the
*kim\_init* command.

Care must be taken to apply unit conversion to dimensional variables read in
from a file. For example if a configuration of atoms is read in from a
dump file using the :doc:`read\_dump <read_dump>` command, the following can
be done to convert the box and all atomic positions to the correct units:


.. parsed-literal::

   variable xyfinal equal xy\*${_u_distance}
   variable xzfinal equal xz\*${_u_distance}
   variable yzfinal equal yz\*${_u_distance}
   change_box all x scale ${_u_distance} &
                          y scale ${_u_distance} &
                          z scale ${_u_distance} &
                          xy final ${xyfinal} &
                          xz final ${xzfinal} &
                          yz final ${yzfinal} &
                          remap

.. note::

   Unit conversion will only work if the conversion factors are placed in
   all appropriate places in the input script. It is up to the user to do this
   correctly.

OpenKIM IM Execution (*kim\_interactions*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The second and final step in using an OpenKIM IM is to execute the
*kim\_interactions* command. This command must be preceded by a *kim\_init*
command and a command that defines the number of atom types *N* (such as
:doc:`create\_box <create_box>`).
The *kim\_interactions* command has one argument *typeargs*\ . This argument
contains either a list of *N* chemical species, which defines a mapping between
atom types in LAMMPS to the available species in the OpenKIM IM, or the
keyword *fixed\_types* for models that have a preset fixed mapping (i.e.
the mapping between LAMMPS atom types and chemical species is defined by
the model and cannot be changed). In the latter case, the user must consult
the model documentation to see how many atom types there are and how they
map to the chemical species.

For example, consider an OpenKIM IM that supports Si and C species.
If the LAMMPS simulation has four atom types, where the first three are Si,
and the fourth is C, the following *kim\_interactions* command would be used:


.. parsed-literal::

   kim_interactions Si Si Si C

Alternatively, for a model with a fixed mapping the command would be:


.. parsed-literal::

   kim_interactions fixed_types

The *kim\_interactions* command performs all the necessary steps to set up
the OpenKIM IM selected in the *kim\_init* command. The specific actions depend
on whether the IM is a KIM PM or a KIM SM.  For a KIM PM,
a :doc:`pair\_style kim <pair_kim>` command is executed followed by
the appropriate *pair\_coeff* command. For example, for the
Ercolessi and Adams (1994) KIM PM for Al set by the following commands:


.. parsed-literal::

   kim_init EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005 metal
   ...
   ...  box specification lines skipped
   ...
   kim_interactions Al

the *kim\_interactions* command executes the following LAMMPS input commands:


.. parsed-literal::

   pair_style kim EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005
   pair_coeff \* \* Al

For a KIM SM, the generated input commands may be more complex
and require that LAMMPS is built with the required packages included
for the type of potential being used. The set of commands to be executed
is defined in the SM specification file, which is part of the SM package.
For example, for the Strachan et al. (2003) ReaxFF SM
set by the following commands:


.. parsed-literal::

   kim_init Sim_LAMMPS_ReaxFF_StrachanVanDuinChakraborty_2003_CHNO__SM_107643900657_000 real
   ...
   ...  box specification lines skipped
   ...
   kim_interactions C H N O

the *kim\_interactions* command executes the following LAMMPS input commands:


.. parsed-literal::

   pair_style reax/c lmp_control safezone 2.0 mincap 100
   pair_coeff \* \* ffield.reax.rdx C H N O
   fix reaxqeq all qeq/reax 1 0.0 10.0 1.0e-6 param.qeq

Note that the files *lmp\_control*, *ffield.reax.rdx* and *param.qeq*
are specific to the Strachan et al. (2003) ReaxFF parameterization
and are archived as part of the SM package in OpenKIM.
Note also that parameters like cutoff radii and charge tolerances,
which have an effect on IM predictions, are also included in the
SM definition ensuring reproducibility.

.. note::

   When using *kim\_init* and *kim\_interactions* to select
   and set up an OpenKIM IM, other LAMMPS commands
   for the same functions (such as pair\_style, pair\_coeff, bond\_style,
   bond\_coeff, fixes related to charge equilibration, etc.) should normally
   not appear in the input script.

Using OpenKIM Web Queries in LAMMPS (*kim\_query*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *kim\_query* command performs a web query to retrieve the predictions
of the IM set by *kim\_init* for material properties archived in
`OpenKIM <https://openkim.org>`_.  The *kim\_query* command must be preceded
by a *kim\_init* command. The result of the query is stored in a
:doc:`string style variable <variable>`, the name of which is given as the first
argument of the *kim\_query command*.  (For the case of multiple
return values, the optional *split* keyword can be used after the
variable name to separate the results into multiple variables; see
the :ref:`example <split_example>` below.)
The second required argument *query\_function* is the name of the
query function to be called (e.g. *get\_lattice\_constant\_cubic*).
All following :doc:`arguments <Commands_parse>` are parameters handed over to
the web query in the format *keyword=value*\ , where *value* is always
an array of one or more comma-separated items in brackets.
The list of supported keywords and the type and format of their values
depend on the query function used. The current list of query functions
is available on the OpenKIM webpage at
`https://openkim.org/doc/usage/kim-query <https://openkim.org/doc/usage/kim-query>`_.

.. note::

   All query functions require the *model* keyword, which identifies
   the IM whose predictions are being queried. This keyword is automatically
   generated by *kim\_query* based on the IM set in *kim\_init* and must not
   be specified as an argument to *kim\_query*.

.. note::

   Each *query\_function* is associated with a default method (implemented
   as a `KIM Test <https://openkim.org/doc/evaluation/kim-tests/>`_)
   used to compute this property. In cases where there are multiple
   methods in OpenKIM for computing a property, a *method* keyword can
   be provided to select the method of choice.  See the
   `query documentation <https://openkim.org/doc/repository/kim-query>`_
   to see which methods are available for a given *query function*\ .

*kim\_query* Usage Examples and Further Clarifications:
"""""""""""""""""""""""""""""""""""""""""""""""""""""""

The data obtained by *kim\_query* commands can be used as part of the setup
or analysis phases of LAMMPS simulations. Some examples are given below.

**Define an equilibrium fcc crystal**


.. parsed-literal::

   kim_init         EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005 metal
   boundary         p p p
   kim_query        a0 get_lattice_constant_cubic crystal=["fcc"] species=["Al"] units=["angstrom"]
   lattice          fcc ${a0}
   ...

The *kim\_query* command retrieves from `OpenKIM <https://openkim.org>`_
the equilibrium lattice constant predicted by the Ercolessi and Adams (1994)
potential for the fcc structure and places it in
variable *a0*\ . This variable is then used on the next line to set up the
crystal. By using *kim\_query*, the user is saved the trouble and possible
error of tracking this value down, or of having to perform an energy
minimization to find the equilibrium lattice constant.

Note that in *unit\_conversion\_mode* the results obtained from a
*kim\_query* would need to be converted to the appropriate units system.
For example, in the above script, the lattice command would need to be
changed to: "lattice fcc ${a0}\*${\_u_distance}".

.. _split\_example:



**Define an equilibrium hcp crystal**


.. parsed-literal::

   kim_init         EAM_Dynamo_Mendelev_2007_Zr__MO_848899341753_000 metal
   boundary         p p p
   kim_query        latconst split get_lattice_constant_hexagonal crystal=["hcp"] species=["Zr"] units=["angstrom"]
   variable         a0 equal latconst_1
   variable         c0 equal latconst_2
   variable         c_to_a equal ${c0}/${a0}
   lattice          custom ${a0} a1 0.5 -0.866025 0 a2 0.5 0.866025 0 a3 0 0 ${c_to_a} &
                    basis 0.333333 0.666666 0.25 basis 0.666666 0.333333 0.75
   ...

In this case the *kim\_query* returns two arguments (since the hexagonal
close packed (hcp) structure has two independent lattice constants).
The default behavior of *kim\_query* returns the result as a string
with the values separated by commas. The optional keyword *split*
separates the result values into individual variables of the form
*prefix\_I*, where *prefix* is set to the the *kim\_query* *variable* argument
and *I* ranges from 1 to the number of returned values. The number and order of
the returned values is determined by the type of query performed.

**Define a crystal at finite temperature accounting for thermal expansion**


.. parsed-literal::

   kim_init         EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005 metal
   boundary         p p p
   kim_query        a0 get_lattice_constant_cubic crystal=["fcc"] species=["Al"] units=["angstrom"]
   kim_query        alpha get_linear_thermal_expansion_coefficient_cubic  crystal=["fcc"] species=["Al"] units=["1/K"] temperature=[293.15] temperature_units=["K"]
   variable         DeltaT equal 300
   lattice          fcc ${a0}\*${alpha}\*${DeltaT}
   ...

As in the previous example, the equilibrium lattice constant is obtained
for the Ercolessi and Adams (1994) potential. However, in this case the
crystal is scaled to the appropriate lattice constant at room temperature
(293.15 K) by using the linear thermal expansion constant predicted by the
potential.

.. note::

   When passing numerical values as arguments (as in the case
   of the temperature in the above example) it is also possible to pass a
   tolerance indicating how close to the value is considered a match.
   If no tolerance is passed a default value is used. If multiple results
   are returned (indicating that the tolerance is too large), *kim\_query*
   will return an error. See the
   `query documentation <https://openkim.org/doc/repository/kim-query>`_
   to see which numerical arguments and tolerances are available for a
   given *query function*\ .

**Compute defect formation energy**


.. parsed-literal::

   kim_init         EAM_Dynamo_ErcolessiAdams_1994_Al__MO_123629422045_005 metal
   ...
   ... Build fcc crystal containing some defect and compute the total energy
   ... which is stored in the variable *Etot*
   ...
   kim_query        Ec get_cohesive_energy_cubic crystal=["fcc"] species=["Al"] units=["eV"]
   variable         Eform equal ${Etot} - count(all)\*${Ec}
   ...

The defect formation energy *Eform* is computed by subtracting from *Etot* the
ideal fcc cohesive energy of the atoms in the system obtained from
`OpenKIM <https://openkim.org>`_ for the Ercolessi and Adams (1994) potential.

.. note::

   *kim\_query* commands return results archived in
   `OpenKIM <https://openkim.org>`_. These results are obtained
   using programs for computing material properties
   (KIM Tests and KIM Test Drivers) that were contributed to OpenKIM.
   In order to give credit to Test developers, the number of times results
   from these programs are queried is tracked. No other information about
   the nature of the query or its source is recorded.

Citation of OpenKIM IMs
-----------------------

When publishing results obtained using OpenKIM IMs researchers are requested
to cite the OpenKIM project :ref:`(Tadmor) <kim-mainpaper>`, KIM API
:ref:`(Elliott) <kim-api>`, and the specific IM codes used in the simulations,
in addition to the relevant scientific references for the IM.
The citation format for an IM is displayed on its page on
`OpenKIM <https://openkim.org>`_ along with the corresponding BibTex file,
and is automatically added to the LAMMPS *log.cite* file.

Citing the IM software (KIM infrastructure and specific PM or SM codes)
used in the simulation gives credit to the researchers who developed them
and enables open source efforts like OpenKIM to function.

Restrictions
""""""""""""


The set of *kim\_commands* is part of the KIM package.  It is only enabled if
LAMMPS is built with that package. A requirement for the KIM package,
is the KIM API library that must be downloaded from the
`OpenKIM website <https://openkim.org/kim-api/>`_ and installed before
LAMMPS is compiled. When installing LAMMPS from binary, the kim-api package
is a dependency that is automatically downloaded and installed. See the KIM
section of the :doc:`Packages details <Packages_details>` for details.

Furthermore, when using *kim\_commands* to run KIM SMs, any packages required
by the native potential being used or other commands or fixes that it invokes
must be installed.

Related commands
""""""""""""""""

:doc:`pair\_style kim <pair_kim>`


----------


.. _kim-mainpaper:



**(Tadmor)** Tadmor, Elliott, Sethna, Miller and Becker, JOM, 63, 17 (2011).
doi: `https://doi.org/10.1007/s11837-011-0102-6 <https://doi.org/10.1007/s11837-011-0102-6>`_

.. _kim-api:



**(Elliott)** Elliott, Tadmor and Bernstein, `https://openkim.org/kim-api <https://openkim.org/kim-api>`_ (2011)
doi: `https://doi.org/10.25950/FF8F563A <https://doi.org/10.25950/FF8F563A>`_


.. _lws: http://lammps.sandia.gov
.. _ld: Manual.html
.. _lc: Commands_all.html
