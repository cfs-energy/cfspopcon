.. _authoringaplugin:

Authoring a Plugin
*******************

cfspopcon can run algorithms it knows nothing about. A plugin is an ordinary importable Python
package which registers its algorithms when imported; an input file then names its plugins in a
``plugins`` section and uses their algorithms exactly like builtins. The list is explicit and
per-case: the input file states its own dependencies, and a case is reproducible from the file
alone.

This page walks through a complete, deliberately non-physics example: a plugin which plans widget
production.

Package layout
====================

.. code:: text

  my_popcon_plugin/
  ├── __init__.py
  ├── algorithms.py
  └── plugin_variables.yaml

The package must be importable wherever cfspopcon runs — installed into the same environment
(e.g. ``pip install``, or a ``git`` dependency of your project) or reachable via ``PYTHONPATH``.
Note that it is the *import* name that a case names, which need not match the *distribution*
name: a distribution ``my-popcon-plugin`` providing the package ``my_popcon_plugin`` is fine, and
the case says ``my_popcon_plugin``.

An algorithm
====================

Any module in the package may register algorithms; registration walks the whole package. One
function with :meth:`~cfspopcon.algorithm_class.Algorithm.register_algorithm` is all it takes — ``algorithms.py``:

.. code::

  from cfspopcon.algorithm_class import Algorithm
  from cfspopcon.unit_handling import Unitfull


  @Algorithm.register_algorithm(return_keys=["widgets_per_shift"])
  def calc_widgets_per_shift(widget_rate: Unitfull, shift_length: Unitfull) -> Unitfull:
      """Compute the widgets produced in one shift."""
      return widget_rate * shift_length

A composite spanning your algorithms (or cfspopcon's) can be declared next to them with
:meth:`~cfspopcon.algorithm_class.CompositeAlgorithm.register_from_list`.

The variables file
====================

A file called ``plugin_variables.yaml`` in the package root declares the default units of the
variables the plugin introduces, in the shape of cfspopcon's own ``variables.yaml`` (see
:data:`~cfspopcon.plugins.PLUGIN_VARIABLES_FILE`). With this file, the plugin needs no units code
at all:

.. code::

  widget_rate:
    default_units: 1 / hour
    description:
    - Widgets produced per hour.
  shift_length:
    default_units: hour
    description:
    - Duration of one production shift.
  widgets_per_shift:
    default_units: dimensionless
    description:
    - Widgets produced in one shift.

A plugin may only *add*: redefining a variable, unit or algorithm that cfspopcon (or an earlier
plugin) already defines raises a :class:`~cfspopcon.plugins.PluginClashError` naming the offender, and
the whole registration is rolled back.

A case using the plugin
========================

The ``plugins`` section is read before the algorithm names are resolved and before input units
are looked up — ``widget_case/input.yaml``:

.. code::

  plugins:
    - my_popcon_plugin

  algorithms:
    - calc_widgets_per_shift

  widget_rate: 6.0
  shift_length: 8.0

Run it like any other case; the result lands in ``widget_case/output/dataset.nc``:

.. code::

  >>> popcon widget_case
  Done

To see what a plugin offers alongside the builtins, pass it (repeatably) to the algorithm
listing:

.. code::

  >>> popcon_algorithms --plugin my_popcon_plugin

Registering from Python
========================

:func:`~cfspopcon.plugins.register_plugins` registers packages directly — for several plugins, pass them
in one call, so a composite in one may name algorithms from another. It returns a
:class:`~cfspopcon.plugins.PluginReport` per plugin listing what was registered, and
:func:`~cfspopcon.algorithm_class.algorithms_setting` / :func:`~cfspopcon.algorithm_class.algorithms_using` answer "which
algorithms set / use this variable?" with any registered plugins included:

.. code::

  >>> import cfspopcon
  >>> cfspopcon.register_plugins("my_popcon_plugin")
  >>> cfspopcon.algorithms_setting("widgets_per_shift")
  ['calc_widgets_per_shift']
