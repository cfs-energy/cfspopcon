from cfspopcon.algorithm_class import Algorithm
import cfspopcon
from importlib import import_module


def import_all_submodules(importable, module, prefix):

    for module in module.__all__:
        prefix = f"cfspopcon.formulas.{module}"
        importable.append(module)
        try:
            submodule = import_module(prefix)
            if hasattr(submodule, "__all__"):
                import_all_submodules(importable, submodule, prefix=f"cfspopcon.formulas.{module}")
        except ModuleNotFoundError:
            pass


def test_for_anonymous_algorithms():

    importable = []
    import_all_submodules(importable, cfspopcon.formulas, "cfspopcon.formulas")

    not_found = 0
    for name, _ in Algorithm.instances.items():

        if name not in importable:
            print(f"Cannot import {name} from cfspopcon.formulas. Algorithms must be importable.")
            not_found += 1

    assert not_found == 0
