.. index:: pair\_style spin/exchange

pair\_style spin/exchange command
=================================

Syntax
""""""


.. parsed-literal::

   pair_style spin/exchange cutoff

* cutoff = global cutoff pair (distance in metal units)


Examples
""""""""


.. parsed-literal::

   pair_style spin/exchange 4.0
   pair_coeff \* \* exchange 4.0 0.0446928 0.003496 1.4885
   pair_coeff 1 2 exchange 6.0 -0.01575 0.0 1.965

Description
"""""""""""

Style *spin/exchange* computes the exchange interaction between
pairs of magnetic spins:

.. image:: Eqs/pair_spin_exchange_interaction.jpg
   :align: center

where si and sj are two neighboring magnetic spins of two particles,
rij = ri - rj is the inter-atomic distance between the two particles,
and J(rij) is a function defining the intensity and the sign of the exchange
interaction for different neighboring shells. This function is defined as:

.. image:: Eqs/pair_spin_exchange_function.jpg
   :align: center

where a, b and d are the three constant coefficients defined in the associated
"pair\_coeff" command (see below for more explanations).

The coefficients a, b, and d need to be fitted so that the function above matches with
the value of the exchange interaction for the N neighbor shells taken into account.
Examples and more explanations about this function and its parameterization are reported
in :ref:`(Tranchida) <Tranchida3>`.

From this exchange interaction, each spin i will be submitted
to a magnetic torque omega, and its associated atom can be submitted to a
force F for spin-lattice calculations (see :doc:`fix\_nve\_spin <fix_nve_spin>`),
such as:

.. image:: Eqs/pair_spin_exchange_forces.jpg
   :align: center

with h the Planck constant (in metal units), and eij = (ri - rj)/\|ri-rj\| the unit
vector between sites i and j.

More details about the derivation of these torques/forces are reported in
:ref:`(Tranchida) <Tranchida3>`.

For the *spin/exchange* pair style, the following coefficients must be defined
for each pair of atoms types via the :doc:`pair\_coeff <pair_coeff>` command as in
the examples above, or in the data file or restart files read by the
:doc:`read\_data <read_data>` or :doc:`read\_restart <read_restart>` commands, and
set in the following order:

* rc (distance units)
* a  (energy units)
* b  (adim parameter)
* d  (distance units)

Note that rc is the radius cutoff of the considered exchange interaction,
and a, b and d are the three coefficients performing the parameterization
of the function J(rij) defined above.

None of those coefficients is optional. If not specified, the
*spin/exchange* pair style cannot be used.


----------


Restrictions
""""""""""""


All the *pair/spin* styles are part of the SPIN package.  These styles
are only enabled if LAMMPS was built with this package, and if the
atom\_style "spin" was declared.  See the :doc:`Build package <Build_package>` doc page for more info.

Related commands
""""""""""""""""

:doc:`atom\_style spin <atom_style>`, :doc:`pair\_coeff <pair_coeff>`,
:doc:`pair\_eam <pair_eam>`,

**Default:** none


----------


.. _Tranchida3:



**(Tranchida)** Tranchida, Plimpton, Thibaudeau and Thompson,
Journal of Computational Physics, 372, 406-425, (2018).


.. _lws: http://lammps.sandia.gov
.. _ld: Manual.html
.. _lc: Commands_all.html
