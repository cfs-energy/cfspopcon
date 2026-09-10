.. _plotstyles:

Plot Styles
************

Each entry in an input file's ``plots`` section names a yaml "plot style" file, which
``popcon`` renders with ``cfspopcon.plotting.make_plot``. The committed examples are
``example_cases/SPARC_PRD/plot_popcon.yaml`` (a filled POPCON on the input grid) and
``plot_remapped.yaml`` (the same, remapped onto computed variables). This page lists the keys
those files may use.

Top-level keys
====================

``type``
  Only ``popcon`` is implemented.
``figsize``
  ``[width, height]`` in inches.
``show_dpi``
  Figure resolution.
``legend_loc``
  A matplotlib legend location string; defaults to ``best``.
``coords`` or ``new_coords``
  Exactly one of the two — the plot axes, below.
``fill``, ``contour``, ``points``
  The plot's layers, below; each is optional.

Axes: ``coords`` / ``new_coords``
==================================

Both take ``x`` and ``y`` entries, each with a ``dimension`` (the dataset variable to put on
that axis), a ``label`` and optionally ``units``. The ``label`` is used verbatim as the axis
label, so write the units into it yourself (e.g. ``"$<T_e>$ [$keV$]"``).

``coords`` plots against existing dimensions of the dataset. ``new_coords`` instead remaps the
field onto *computed* variables (e.g. ``P_auxiliary_launched``) by interpolation; each entry
additionally accepts ``min``, ``max`` and ``resolution`` for the new axis, and ``new_coords``
itself accepts ``max_distance``, the (grid-normalised) distance beyond which points too far
from any sample are masked.

``fill``
====================

A single filled (colormapped) variable:

``variable``
  The dataset variable to fill with.
``units``
  Units to convert to before plotting; defaults to the variable's own.
``cbar_label``
  Colorbar label; defaults to the variable name. The variable's units are appended
  automatically, so a label which already ends in units renders them twice — write
  ``cbar_label: "$P_{fusion}$"``, not ``"$P_{fusion}$ [MW]"``.
``labelpad``
  Padding of the colorbar label; defaults to ``15.0``.
``where``
  A mask, below.

``contour``
====================

A mapping of dataset variable to contour settings:

``label``
  Legend entry, used verbatim (units are *not* appended, unlike ``cbar_label``).
``levels``
  List of contour levels, in ``units``.
``color``
  A matplotlib color.
``line``
  Line style; defaults to ``solid``.
``format``
  Format spec for the in-line contour labels; ``fontsize`` sets their size (default ``10.0``).
``units``, ``where``
  As for ``fill``.

``points``
====================

A mapping of point name — matching a point in the input file's ``points`` section — to marker
style: ``label`` (defaults to the point name), ``marker``, ``color`` and ``size``.

``where``
====================

A mapping of dataset variable to ``min`` / ``max`` bounds (with optional ``units``), hiding the
regions outside every bound — used to mask the inaccessible parts of the operational space:

.. code::

  where:
    greenwald_fraction:
      max: 0.9
    P_auxiliary_launched:
      min: 0.0
      max: 25.0
      units: MW
