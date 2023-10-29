.. _gettingstarted:

Getting Started
===================

Installation
^^^^^^^^^^^^^

The cfspopcon package is available on the `Python Package Index <https://pypi.org/>`, thus installation is as simple as:

.. code::

  >>> pip install cfspopcon

.. warning::
   The :code:`cfspopcon.atomic_data` module requires data files produced by the `radas project <https://github.com/cfs-energy/radas>`_. Radas produces these files by processing `OpenADAS <https://open.adas.ac.uk/adf11>`_ data. These files are not shipped as part of :code:`cfspopcon`. Follow the below steps after the :code:`pip install cfspopcon` command (we will try to make this smoother in the future. N.b. this only has to be done once).

   .. code:: bash
     
    >>> export RADAS=$(python -c "from cfspopcon import atomic_data;from pathlib import Path; print(Path(atomic_data.__file__).parent)")
    >>> pushd /tmp
    >>> git clone https://github.com/cfs-energy/radas.git
    >>> pushd radas
    >>> poetry install --only main
    >>> poetry run fetch_adas
    >>> poetry run run_radas
    >>> cp ./cases/*/output/*.nc $RADAS
    >>> popd && popd


Example Notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example notebook linked below can be found within the `docs folder <https://github.com/cfs-energy/cfspopcon/blob/main/docs/doc_sources/getting_started.ipynb>`_ of our github repository.

.. toctree::

  getting_started
