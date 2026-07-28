"""Build importable throwaway packages, for tests which exercise algorithm discovery."""

import sys
from pathlib import Path


def write_package(root: Path, name: str, modules: dict[str, str]) -> Path:
    """Create an importable package under `root`, as {submodule name: source}.

    The "__init__" key, if present, becomes the package's ``__init__.py``; every other key becomes a
    submodule of that name.
    """
    pkg = root / name
    pkg.mkdir(parents=True, exist_ok=True)
    modules = dict(modules)
    (pkg / "__init__.py").write_text(modules.pop("__init__", ""))
    for module, source in modules.items():
        (pkg / f"{module}.py").write_text(source)
    return pkg


def forget_packages(*package_names: str) -> None:
    """Drop the named packages, and everything under them, from the import cache."""
    for name in [m for m in sys.modules if m.split(".")[0] in package_names]:
        sys.modules.pop(name, None)
