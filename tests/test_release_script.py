"""Tests for the release helper's text rewriting."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("release", Path(__file__).parents[1] / "scripts" / "release.py")
release = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release)


def test_bump_version_rewrites_the_single_field():
    """The sole version field is rewritten; zero or several fields are refused."""
    text, old = release._bump_version('name = "x"\nversion = "8.0.0"\n', "9.0.0")
    assert old == "8.0.0"
    assert 'version = "9.0.0"' in text
    with pytest.raises(ValueError, match="exactly one"):
        release._bump_version("name = 'x'\n", "9.0.0")
    with pytest.raises(ValueError, match="exactly one"):
        release._bump_version('version = "1"\nversion = "2"\n', "9.0.0")


def test_open_release_cuts_unreleased_and_keeps_its_body():
    """The Unreleased body becomes the new version's section, dated."""
    opened = release._open_release("# Changelog\n\n## Unreleased\n\n- an entry\n", "9.0.0")
    assert "## Unreleased\n\n## 9.0.0 - " in opened
    assert release._changelog_section(opened, "9.0.0") == "- an entry"
    with pytest.raises(ValueError, match="Unreleased"):
        release._open_release("# Changelog\n", "9.0.0")


def test_changelog_section_extraction_and_guards():
    """A version's section is extracted exactly; missing, empty, and duplicate sections are refused."""
    text = "## 9.0.0 - 2026-09-11\n\n- entry\n\n## 8.0.0\n\n- old\n"
    assert release._changelog_section(text, "9.0.0") == "- entry"
    assert release._changelog_section(text, "8.0.0") == "- old"
    with pytest.raises(ValueError, match="no '## 7.0.0'"):
        release._changelog_section(text, "7.0.0")
    with pytest.raises(ValueError, match="empty"):
        release._changelog_section("## 9.0.0\n\n## 8.0.0\n\n- old\n", "9.0.0")
    with pytest.raises(ValueError, match="2 '## 9.0.0'"):
        release._changelog_section("## 9.0.0\n- a\n## 9.0.0\n- b\n", "9.0.0")
    # 9.0.2 must not match 9.0.20's heading.
    assert release._changelog_section("## 9.0.20\n\n- twenty\n\n## 9.0.2\n\n- two\n", "9.0.2") == "- two"
