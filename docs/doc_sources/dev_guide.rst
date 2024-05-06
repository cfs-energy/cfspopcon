.. _devguide:

Developer's Guide
*******************

The cfspopcon team uses `Poetry <https://python-poetry.org/>`_ to develop.
If you are familiar with the usual poetry based development workflow, feel free to skip right ahead to the `Contribution Guidelines`_.

Development Setup
====================

For more information and help installing Poetry, please refer to `their documentation <https://python-poetry.org/docs/>`_.
Once you have Poetry installed we are ready to start developing. First we clone the repository and enter into the folder.

.. code::

  >>> git clone https://github.com/cfs-energy/cfspopcon.git
  >>> cd cfspopcon

Setting up a virtual environment and installing all dependencies required to develop, is done in just one command:

.. code::

  >>> poetry install
  >>> poetry run radas -d ./radas_dir

If you are new to Poetry, we suggest that you at least read their brief introduction on `how to use this virtual environment <https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment>`_.
You can verify that everything worked as expected by following the :ref:`Getting Started <gettingstarted>` guide.

At this point you are ready to read our `Contribution Guidelines`_ and start making changes to the code. We are looking forward to your contribution!


Contribution Guidelines
========================

If you have a question or found a bug, please feel free to raise an issue on GitHub.

If you would like to make changes to the code, we ask that you follow the below guidelines:

1. Please follow our `Style Guide`_
2. The `Pre-Commit Checks`_ should all pass
3. Make sure tests in the test suite are still passing, see `Running the Test Suite`_
4. If adding new functionality, please try to add a unit test for it, if applicable.
5. Please ensure that any changes are correctly reflected in the documentation, see `Building The Documentation`_



Style Guide
=============

The set of tools configured to run as pre-commit hooks should cover the simple style decisions.
For everything else, we follow the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_, but make some exceptions for science / math variables.
The Google style guide and PEP8 encourage long, descriptive, lower case variables names. However, these can make science / math equations hard to read.
There is a case to be made for using well-established mathematical symbols as **local** variable names, e.g. :code:`T` for temperature or :code:`r` for position.
Subscripts can be added, e.g. :code:`r_plasma`.

To make reading & validating formulas easier, we additionally follow the below guidelines:

- Add a descriptive comment when using short variable names.
- When local variables are declared, specify their units in a comment next to the declaration, like :code:`x = 1.0  # [m]`
- Use basic SI units, unless you have a good reason not to. (e.g. prefer :code:`[A]` over :code:`[kA]`).
- Explicitly call out dimensionless physical quantities, e.g. :code:`reynolds_number = 1e3  # [~]`.
- Functions that handle dimensional quantities should use :class:`pint.Quantity`.

Please note, that while we have some checks for docstrings, those checks do not cover all aspects.
So let's look at a basic example, the :func:`~cfspopcon.formulas.geometry.calc_plasma_volume` function:

.. literalinclude:: ../../cfspopcon/formulas/geometry/analytical.py
  :language: python
  :linenos:
  :pyobject: calc_plasma_volume

To summarize the important points of the above example:

1. Include short descriptive one-liner.
2. If applicable, add a more detailed description.
3. List the arguments with a short description, and include their units.
4. Each return value should come with a brief explanation and unit.
5. Do **not** include any type annotations within the docstring. These will be added automatically by sphinx.

Aside from the units annotations in the docstring, you'll notice the parameters are annotated with the type :class:`~cfspopcon.unit_handling.Unitfull`.
This is because all calculations in cfspopcon use explicit unit handling to better ensure that calculations are correct and no units handling errors sneak into a formula.
The units handling cfspopcon is powered by the `pint <https://pint.readthedocs.io/en/stable/>`_ and `pint-xarray <https://github.com/xarray-contrib/pint-xarray>`_ python packages.
The type :class:`~cfspopcon.unit_handling.Unitfull`, used in the above function as type annotation, is an alias of :code:`pint.Quantity | xarray.DataArray`.

In addition to the above example, we also recommend having a look at the :code:`cfspopcon.formulas` module, which holds many good examples.


Pre-Commit Checks
===================

As the name suggests, these are a list of checks that should be run before making a commit.
We use the `pre-commit <https://pre-commit.com/>`_ framework to ensure these checks are run for every commit.
You already installed the :code:`pre-commit` tool as a development dependency during the `Development Setup`_.

Run all configured checks by executing:

.. code::

  >>> poetry run pre-commit run --all-files

But instead of trying to remember to run this command before every commit, we suggest you follow the `pre-commit documentation <https://pre-commit.com/#3-install-the-git-hook-scripts>`_ and install the git hooks.

.. code::

  >>> poetry run pre-commit install

The installed git hooks will now automatically run the required checks when you try to :code:`git commit` some changes.
An added benefit is that this will usually be faster than running over all files, as :code:`pre-commit` is pretty smart at figuring out which files it needs to check for a given commit.

If you are curious, you can see all the automatic checks that we have configured to run in the file :code:`.pre-commit-config.yaml`:

.. literalinclude:: ../../.pre-commit-config.yaml




Running the Test Suite
=======================

We use `pytest <https://docs.pytest.org/>`_ and the `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ plugin for our test suite.
All tests can be found in the :code:`tests` subfolder.
The configuration can be found in the :code:`pyproject.toml` file.

Running the entire test suit can be done via:

.. code::

  >>> poetry run pytest

Adding a new Test
-------------------

When adding new functionality it is best to also add a test for it.
If the category of the added functionality fits within one of the existing files, please add your test to that file.
Otherwise feel free to create a new test file. The name should follow the convention :code:`test_{description}.py`.


Building The Documentation
===============================

Our documentation is build and hosted on `Read The Docs <https://readthedocs.org/>`_ and previews are available on each PR.
But when extending the documentation it is most convenient to first build it locally yourself to check that everything is included & rendered correctly.

.. warning::
   Building the documentation unfortunately requires a non-python dependency: `pandoc <https://pandoc.org/>`_.
   Please ensure that the :code:`pandoc` executable is available before proceeding.
   This package can easily be installed via :code:`sudo apt-get install pandoc` (Linux) or :code:`brew install pandoc` (MacOS). 
   For more details please see `pandoc's installation guide <https://pandoc.org/installing.html>`_.

Starting from inside the project folder you can trigger the build by running:

.. code::

  >>> poetry run make -C docs html

Once that build is finished, open the file :code:`./docs/_build/html/index.html` to view the documentation.

As part of our CI we also run the `sphinx-doctest <https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html>`_ and `sphinx-linkcheck <https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-the-linkcheck-builder>`_ extensions.
The :code:`sphinx-doctest` extension checks that python snippets used in docstrings are actually valid python code and produce the expected output. And :code:`sphinx-linkcheck` is used to ensure that any links used within our documentation are correct and accessible.

To avoid having failures in the CI it's a good idea to run these locally first as well:

.. code::

  poetry run make -C docs doctest
  poetry run make -C docs linkcheck

