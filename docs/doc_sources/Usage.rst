.. _gettingstarted:

Getting Started
===================

Installation
^^^^^^^^^^^^^

The cfspopcon package is available on the `Python Package Index <https://pypi.org/>`, thus installation is as simple as:

.. code::

  >>> pip install cfspopcon
.. warning::
   The :code:`cfspopcon.atomic_data` module requires data files produced by the `radas project <https://github.com/cfs-energy/radas>`_. Radas produces these files by processing `OpenADAS <https://open.adas.ac.uk/adf11>`_ data. These files are not shipped as part of :code:`cfspopcon`, thus the below steps need to be run once after installing :code:`cfspopcon`. Please follow the below instructions from within the python environement :code:`cfspopcon` is installed into.

   .. code:: bash

    >>> export RADAS=$(python -c "from cfspopcon import atomic_data;from pathlib import Path; print(Path(atomic_data.__file__).parent)")
    >>> git clone https://github.com/cfs-energy/radas.git
    >>> pushd radas
    >>> PYTHONPATH=$PWD:$PYTHONPATH python adas_data/fetch_adas_data.py
    >>> PYTHONPATH=$PWD:$PYTHONPATH python run_radas.py
    >>> cp ./cases/*/output/*.nc $RADAS
    >>> popd


Example Notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example notebook linked below can be found within the `docs folder <https://github.com/cfs-energy/cfspopcon/blob/main/docs/doc_sources/getting_started.ipynb>`_ of our github repository.

.. toctree::

  getting_started
  time_independent_inductance_and_fluxes
