.. _gettingstarted:

Getting Started
===================

Installation
^^^^^^^^^^^^^

The cfspopcon package is available on the `Python Package Index <https://pypi.org/>`, thus installation is as simple as:

.. code::

  >>> pip install cfspopcon --with dev
  >>> radas -d ./radas_dir

The second step is to generate a folder of OpenADAS atomic data files. We can't ship these files with
cfsPOPCON due to licensing issues, but they're easy to make with `radas`. You only need to do this once.

Running cfspopcon from the command line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you've installed :code:`cfspopcon`, you can run it from the command line using

.. code::

  >>> popcon example_cases/SPARC_PRD --show

This will run the :code:`run_popcon_cli` function from :code:`cfspopcon/cli.py`. The first argument to :code:`popcon` should be a path to a folder containing an :code:`input.yaml` file, which
sets the parameters for the POPCON analysis. Have a look at :code:`example_cases/SPARC_PRD/input.yaml` to see how this file is structured.

The results of the POPCON analysis are stored in a :code:`output` folder in the directory where :code:`input.yaml` was read from. For the example case above, you can find the outputs in
:code:`example_cases/SPARC_PRD/outputs`. These include a NetCDF dataset containing the results of the run, a JSON file representing points in plain-text, as well as any plots requested in the
:code:`input.yaml` file.

Getting started with Jupyter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can also run :code:`cfspopcon` via a Jupyter notebook. If you've installed :code:`cfspopcon` using :code:`pip`, launch :code:`jupyter` from the environment you installed :code:`cfspopcon`
into. You can also run :code:`cfspopcon` without installing anything at all, by using the `binder interface <https://mybinder.org/v2/gh/cfs-energy/cfspopcon/HEAD>`_. If you've installed using
:code:`poetry`, you can either :code:`poetry run jupyter lab` or open Jupyter (either directly or via an IDE like VSCode) and select :code:`.venv/bin/python` as your kernel.

An example notebook in the `docs folder <https://github.com/cfs-energy/cfspopcon/blob/main/docs/doc_sources/getting_started.ipynb>`_. The contents and results of this can be compared to the
static representation below:

.. toctree::

  getting_started

Other example notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Jupyter notebooks provide a convenient way of demonstrating and documenting features of the code. You can find a collection of demonstration notebooks listed below. If you add features to
:code:`cfspopcon`, or have a nice notebook which documents some pre-existing functionality, please add it below (this is a great way to start developing :code:`cfspopcon`).

.. toctree::

  separatrix_operational_space
  time_independent_inductances_and_fluxes
