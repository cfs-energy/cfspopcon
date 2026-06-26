"""Formulas used for the POPCON analysis.

Submodules are no longer imported by hand here: their algorithms are found automatically by
:func:`cfspopcon._discovery.discover_builtin_algorithms` (a ``pkgutil`` walk of this package),
which runs lazily the first time the registry is queried. Adding a new ``formulas/...`` module is
therefore sufficient to register its algorithms — no edit to this file is required.
"""
