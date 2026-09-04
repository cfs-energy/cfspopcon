.. _authoringaplugin:

Authoring a Plugin
*******************

A plugin is an importable Python package which provides algorithms, composites, and variables
of its own. A cfspopcon *case* (the ``input.yaml`` describing one POPCON run) uses a plugin by
listing it in its ``plugins`` section; the plugin's algorithms are then available exactly like
the built-in ones. Because the case names the plugins it needs, it is reproducible from its
input file alone.

This page walks through a complete example: a plugin which plans widget production.

Package layout
====================

.. code:: text

  my_popcon_plugin/
  ├── __init__.py
  ├── algorithms.py
  └── variables.yaml

The package must be importable wherever cfspopcon runs — installed into the same environment
(e.g. ``pip install``, or a ``git`` dependency of your project) or reachable via ``PYTHONPATH``.
The name in the input file is the plugin's *import* name, which can differ from the name you
install: after ``pip install my-popcon-plugin``, the package is imported (and registered) as
``my_popcon_plugin``.

The ``__init__.py`` can be empty: a plugin contains no registration code. Registration is
triggered by name, from a case's ``plugins`` section or from Python with
``cfspopcon.register_plugin("my_popcon_plugin")``, and covers the plugin's default units,
algorithms, and composites in one step. Importing the package, or any function from it,
registers nothing.

An algorithm
====================

Algorithms are defined with the :meth:`~cfspopcon.algorithm_class.Algorithm.register_algorithm`
decorator, in any module of the package: registration walks the whole package, so
``__init__.py`` needs no imports of its own. ``algorithms.py``:

.. code::

  from cfspopcon.algorithm_class import Algorithm
  from cfspopcon.unit_handling import Unitfull


  @Algorithm.register_algorithm(return_keys=["widgets_per_shift"])
  def calc_widgets_per_shift(widget_rate: Unitfull, shift_length: Unitfull) -> Unitfull:
      """Compute the widgets produced in one shift."""
      return widget_rate * shift_length

A composite combines several algorithms into one. It is declared next to your algorithms and
assigned at module level, e.g. ``full_shift_analysis = CompositeAlgorithm.declare([...], name="full_shift_analysis")``
(see :meth:`~cfspopcon.algorithm_class.CompositeAlgorithm.declare`). The composite is built when
your plugin's registration finishes, so it may include your own algorithms, the built-in ones
(which are always registered before your plugin), and those of any plugin registered earlier.

The variables file
====================

A file called ``variables.yaml`` in the package root declares the default units of the
variables the plugin introduces, in the shape of cfspopcon's own ``variables.yaml``. It is read
during registration, before the plugin's algorithms are registered:

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

Name collisions
====================

By default a plugin may only *add*. Registering an algorithm under a name which is already
taken is an error, and the error message states the name; pass ``override=True`` to
:meth:`~cfspopcon.algorithm_class.Algorithm.register_algorithm` or
:meth:`~cfspopcon.algorithm_class.CompositeAlgorithm.declare` to replace the registered
algorithm deliberately, or rename yours. Changing the default units of a variable which is
already defined is also an error, and units cannot be overridden; re-declaring a variable with
identical units is allowed.

A failed registration is rolled back completely, so you can fix the plugin and register it
again in the same session.

Depending on another plugin
============================

If your composites include another plugin's algorithms, declare that plugin with
``__popcon_requires__ = ("other_plugin",)`` in your ``__init__.py``; it is then registered
before yours. This declaration is only needed for composites: to call another package's
functions directly, an ordinary ``from other_plugin import ...`` is enough.

A case using the plugin
========================

The listed plugins are registered before the ``algorithms`` names are resolved and before
input units are looked up. ``widget_case/input.yaml``:

.. code::

  plugins:
    - my_popcon_plugin

  algorithms:
    - calc_widgets_per_shift

  widget_rate: 6.0
  shift_length: 8.0

Run it like any other case; the result is written to ``widget_case/output/dataset.nc``:

.. code:: console

  $ popcon widget_case
  Done

To list a plugin's algorithms alongside the built-in ones, pass its name to
``popcon_algorithms``; the option may be repeated for several plugins:

.. code:: console

  $ popcon_algorithms --plugin my_popcon_plugin

Registering from Python
========================

From Python, :func:`~cfspopcon.algorithm_class.register_plugin` registers the plugin by name.
The queries :func:`~cfspopcon.algorithm_class.algorithms_setting` and
:func:`~cfspopcon.algorithm_class.algorithms_using` then report which registered algorithms set
or use a given variable, plugins included:

.. code::

  >>> import cfspopcon
  >>> cfspopcon.register_plugin("my_popcon_plugin")
  >>> cfspopcon.algorithms_setting("widgets_per_shift")
  ['calc_widgets_per_shift']

The ``plugins`` section of a case makes exactly this call for each name it lists.
