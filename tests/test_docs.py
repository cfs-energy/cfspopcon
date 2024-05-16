import pytest
import subprocess

pytest.importorskip("sphinx")
from importlib.resources import files


@pytest.mark.docs
def test_docs():
    "Test the Sphinx documentation."
    popcon_directory = files("cfspopcon")

    doctest_output = subprocess.run(
        args=["make", "-C", str(popcon_directory.joinpath("../docs")), "doctest"], capture_output=True, check=True
    )

    linkcheck_output = subprocess.run(
        args=["make", "-C", str(popcon_directory.joinpath("../docs")), "linkcheck"], capture_output=True, check=True
    )

    if len(doctest_output.stderr) > 0:
        raise RuntimeError(doctest_output)

    if "term not in glossary" in str(linkcheck_output.stderr):
        raise RuntimeError(linkcheck_output)
