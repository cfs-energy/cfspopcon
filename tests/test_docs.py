import pytest
import subprocess

pytest.importorskip("sphinx")
from importlib.resources import files


@pytest.mark.docs
def test_docs():
    "Test the Sphinx documentation."
    popcon_directory = files("cfspopcon")

    subprocess.run(args=["make", "-C", str(popcon_directory.joinpath("../docs")), "doctest"])
    subprocess.run(args=["make", "-C", str(popcon_directory.joinpath("../docs")), "linkcheck"])
