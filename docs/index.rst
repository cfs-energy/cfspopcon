.. cfspopcon documentation master file, created by
   sphinx-quickstart on Mon Nov 14 16:09:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#####################################
Welcome to cfspopcon's documentation!
#####################################

POPCONs (Plasma OPerating CONtours) is a tool developed to explore the performance and constraints of tokamak designs based on 0D scaling laws, model plasma kinetic profiles, and physics assumptions on the properties and behavior of the core plasma.

POPCONs was initially described in :cite:`prd` where it was applied to the design of the SPARC tokamak.
Further, :cite:`prd` also introduced the SPARC Primary Reference Discharge (PRD) defining the highest fusion gain achievable on SPARC.
Since that paper was released, the design point has been refined and the latest design point is now available as :code:`example_cases/SPARC_PRD`, which is covered in the :ref:`Getting Started <gettingstarted>` guide.
A document explaining the changes between the SPARC Physics Basis and the current version are detailed in :download:`Changes-to-SPARC-PRD.pdf<doc_sources/Changes-to-SPARC-PRD.pdf>`.

To start generating your fist plasma operating contours with cfspopcon, head over to the :ref:`Getting Started <gettingstarted>` guide.

A useful resource is our :ref:`Physics Glossary <physics_glossary>` which lists all input and output variables, including definitions and citations for formulas where relevant.

If you are interested in how to setup a development environment to make changes to cfspopcon, we suggest you checkout the :ref:`Developer's Guide <devguide>`.

.. toctree::
  :maxdepth: 1

  doc_sources/Usage
  doc_sources/physics_glossary
  doc_sources/dev_guide
  doc_sources/api
  doc_sources/bib
