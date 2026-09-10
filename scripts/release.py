"""cfspopcon release helper: prepare, publish, and verify a release."""

# ruff: noqa: T201, INP001
import argparse
import contextlib
import datetime
import json
import re
import subprocess
import urllib.request
from collections.abc import Iterator
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_DIR / "pyproject.toml"
CHANGELOG = REPO_DIR / "CHANGELOG.md"
UNRELEASED = "## Unreleased"
VERSION_RE = re.compile(r'^(version\s*=\s*")([^"]+)(")', re.MULTILINE)
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
RTD_BUILDS_URL = "https://readthedocs.org/api/v3/projects/cfspopcon/versions/stable/builds/?limit=1"
PYPI_URL = "https://pypi.org/pypi/cfspopcon/json"


# The text helpers take file contents and know nothing of where they came from. They
# reject what they cannot rewrite with a ValueError; `_parsing` turns that into an
# abort naming the file, so the operator learns which one needs a hand.
@contextlib.contextmanager
def _parsing(source: object) -> Iterator[None]:
    """Abort, naming `source`, if the enclosed helper rejects its contents.

    Raises:
        SystemExit: If the enclosed block raises `ValueError`.
    """
    try:
        yield
    except ValueError as exc:
        raise SystemExit(f"{source}: {exc}") from exc


def _check_version(version: str) -> None:
    """Reject a version the release process cannot carry end to end.

    The version is written verbatim into the pyproject; the git tag adds the `v` prefix
    itself, so passing one here would double it.

    Raises:
        SystemExit: If the version is not a bare `X.Y.Z`.
    """
    if not SEMVER_RE.match(version):
        raise SystemExit(f"Version must be a bare X.Y.Z with no 'v' prefix; got {version!r}.")


def _bump_version(text: str, version: str) -> tuple[str, str]:
    """Rewrite the sole `version = "..."` field of a pyproject.

    Returns the rewritten text and the version it replaced.

    Raises:
        ValueError: If the text holds anything other than exactly one version field.
    """
    matches = VERSION_RE.findall(text)
    # Bumping zero fields ships the old version; bumping several rewrites whichever
    # tables happen to carry one. Both need a human.
    if len(matches) != 1:
        raise ValueError(f"expected exactly one 'version = \"...\"' field, found {len(matches)}")
    return VERSION_RE.sub(rf"\g<1>{version}\g<3>", text), matches[0][1]


def _open_release(text: str, version: str) -> str:
    """Insert a dated `## <version>` heading directly below `## Unreleased`.

    The entries accumulated under `## Unreleased` become the body of the new version's
    section, leaving `## Unreleased` empty for the next cycle.

    Raises:
        ValueError: If the changelog has no `## Unreleased` heading.
    """
    if UNRELEASED not in text:
        raise ValueError(f"no '{UNRELEASED}' section found")
    today = datetime.date.today().isoformat()
    return text.replace(UNRELEASED, f"{UNRELEASED}\n\n## {version} - {today}", 1)


def _changelog_section(text: str, version: str) -> str:
    """Extract the changelog section for a given version, for use as release notes.

    A heading may carry a trailing ` - <date>`, so the Keep a Changelog date form is
    recognised alongside the bare heading.

    Raises:
        ValueError: If the changelog holds anything other than exactly one section for
            the version, or if that section is empty.
    """
    # The optional ` - ...` tail must not swallow the rest of the line, or `9.0.2`
    # would match the heading of `9.0.20`.
    heading = re.compile(rf"^## {re.escape(version)}(?: +-.*)?[ \t]*$", re.MULTILINE)
    matches = list(heading.finditer(text))
    if not matches:
        raise ValueError(f"no '## {version}' section found")
    # Picking one of several would publish notes for whichever came first.
    if len(matches) > 1:
        raise ValueError(
            f"found {len(matches)} '## {version}' sections; expected exactly one. "
            f"Only `prepare` may add a version heading; remove the hand-written one."
        )

    rest = text[matches[0].end() :]
    next_header = re.search(r"^## ", rest, re.MULTILINE)
    notes = (rest[: next_header.start()] if next_header else rest).strip()
    if not notes:
        raise ValueError(f"the '## {version}' section is empty; nothing to release")
    return notes


def _run(*cmd: str, capture: bool = False) -> str:
    """Run a shell command from the repo root, returning its output when captured.

    Raises:
        SystemExit: If the command exits non-zero.
    """
    print(f"  $ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, cwd=REPO_DIR, capture_output=capture, text=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}") from exc
    return result.stdout.strip() if capture else ""


def _confirm(prompt: str) -> None:
    """Ask for confirmation, exit if declined.

    Raises:
        SystemExit: If the user declines.
    """
    if input(f"\n{prompt} [y/N] ").strip().lower() != "y":
        raise SystemExit("Aborted.")


def _get_json(url: str) -> dict:
    """Fetch a JSON document.

    Raises:
        SystemExit: If the request fails.
    """
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            return json.load(response)
    except OSError as exc:
        raise SystemExit(f"Could not fetch {url}: {exc}") from exc


def prepare(version: str) -> None:
    """Bump the pyproject version, cut the changelog, commit, and push.

    Every rewrite is computed before any of them is written, so declining the prompt or
    tripping a guard leaves the working tree untouched.
    """
    _check_version(version)

    with _parsing(PYPROJECT.name):
        bumped, current = _bump_version(PYPROJECT.read_text(encoding="utf-8"), version)
    print(f"Bumping version: {current} -> {version}")

    with _parsing(CHANGELOG.name):
        opened = _open_release(CHANGELOG.read_text(encoding="utf-8"), version)
        # An empty section has to fail here, while the release is still nothing but a
        # working-tree diff, rather than at `publish` once the bump is on main.
        _changelog_section(opened, version)
    print(f"Updating changelog: {UNRELEASED} -> ## {version}")

    _confirm("Write, commit and push?")

    PYPROJECT.write_text(bumped, encoding="utf-8")
    CHANGELOG.write_text(opened, encoding="utf-8")
    _run("git", "add", PYPROJECT.name, CHANGELOG.name)
    _run("git", "commit", "-m", f"chore: prepare release {version}")
    _run("git", "push")

    print(f"\nDone! Once CI is green, run:\n  python scripts/release.py publish {version}")


def publish(version: str) -> None:
    """Create a draft GitHub release, tagged `v<version>`, with the changelog section as notes.

    Publishing the draft on GitHub is what triggers the build-and-publish pipeline.
    """
    _check_version(version)

    with _parsing(CHANGELOG.name):
        notes = _changelog_section(CHANGELOG.read_text(encoding="utf-8"), version)
    print(f"Release notes for v{version}:\n")
    print(notes)

    _confirm("Create draft release?")

    _run("gh", "release", "create", f"v{version}", "--title", f"v{version}", "--notes", notes, "--target", "main", "--draft")
    print(f"\nDraft release v{version} created! Review and publish it on GitHub, then run:\n  python scripts/release.py verify {version}")


def verify(version: str) -> None:
    """Check that a published release actually reached PyPI and readthedocs.

    readthedocs serves the last *successful* `stable` build, so a failed tag build
    leaves the previous version's docs up with no other signal.
    """
    _check_version(version)
    ok = True

    on_pypi = version in _get_json(PYPI_URL).get("releases", {})
    print(f"PyPI has {version}: {'yes' if on_pypi else 'NO (publishing can lag a few minutes; re-run, then check the release workflow)'}")
    ok &= on_pypi

    tag_commit = _run("git", "rev-list", "-n1", f"v{version}", capture=True)
    build = _get_json(RTD_BUILDS_URL)["results"][0]
    docs_ok = bool(build.get("success")) and build.get("commit") == tag_commit
    print(f"readthedocs stable built from v{version}: {'yes' if docs_ok else 'NO'}")
    if not docs_ok:
        print(f"  last stable build: commit {str(build.get('commit'))[:9]}, success={build.get('success')}")
        print("  Rebuild 'stable' at https://app.readthedocs.org/projects/cfspopcon/ and re-run this check.")
    ok &= docs_ok

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    commands = {"prepare": prepare, "publish": publish, "verify": verify}
    parser = argparse.ArgumentParser(prog="release", description="cfspopcon release helper.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in commands.items():
        p = sub.add_parser(name, help=fn.__doc__)
        p.add_argument("version", help="Version to release (e.g. 9.0.0).")
    args = parser.parse_args()
    commands[args.command](args.version)
