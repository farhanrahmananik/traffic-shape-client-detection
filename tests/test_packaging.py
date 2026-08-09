"""
test_packaging.py
-----------------
Tests for pyproject.toml.

No network and no install: `pyproject.toml` is parsed with `tomllib` and
compared against what `src/tsd/` actually imports.

The dependency list is checked in BOTH directions, because each
direction fails somewhere else and neither fails here:

    a missing dependency   -> ImportError on someone else's machine,
                              after `pip install`, with the package
                              apparently installed correctly
    a stale dependency     -> a heavier install than the library needs,
                              and a claim about the code that stopped
                              being true

Both are invisible in this repository, where every package is already in
the virtualenv. So the assertion is not "does it import" -- it always
does here -- but "does the declared list still describe the code".
"""

from __future__ import annotations

import ast
import importlib
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_ROOT = REPO_ROOT / "src" / "tsd"

# Where the import name and the distribution name differ. Listed
# explicitly rather than guessed: the mapping is not derivable, and a
# guess that happened to work for `requests` would quietly mis-handle
# `sklearn`.
IMPORT_TO_DISTRIBUTION = {
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
}


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


def third_party_imports() -> dict[str, set[str]]:
    """Every non-stdlib, non-local module imported under src/tsd/."""
    found: dict[str, set[str]] = {}

    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative: part of this package
                    continue
                names = [(node.module or "").split(".")[0]]
            else:
                continue

            for name in names:
                if not name or name == "tsd" or name in sys.stdlib_module_names:
                    continue
                found.setdefault(name, set()).add(path.name)

    return found


def declared_distributions(pyproject: dict) -> set[str]:
    """Distribution names from [project] dependencies, without the bounds."""
    names = set()

    for requirement in pyproject["project"]["dependencies"]:
        for separator in (">=", "==", "<=", "~=", ">", "<", "!=", "["):
            requirement = requirement.split(separator)[0]
        names.add(requirement.strip())

    return names


# --------------------------------------------------------------
# Metadata
# --------------------------------------------------------------

def test_project_metadata(pyproject):
    project = pyproject["project"]

    assert project["name"] == "traffic-shape-client-detection"
    assert project["requires-python"] == ">=3.12"
    assert project["description"]
    assert project["dynamic"] == ["version"]


def test_version_is_single_sourced(pyproject):
    """
    Two hardcoded versions drift, and the one nobody looks at is the one
    that ships.
    """
    import tsd

    assert pyproject["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "tsd.__version__"
    }
    assert isinstance(tsd.__version__, str)
    assert tsd.__version__.count(".") >= 2

    assert "version" not in pyproject["project"], (
        "version must come from tsd.__version__, not be written here too"
    )


def test_src_layout_is_declared(pyproject):
    setuptools = pyproject["tool"]["setuptools"]

    assert setuptools["package-dir"] == {"": "src"}
    assert setuptools["packages"]["find"]["where"] == ["src"]


def test_build_backend_is_setuptools(pyproject):
    build = pyproject["build-system"]

    assert build["build-backend"] == "setuptools.build_meta"
    assert any(item.startswith("setuptools>=") for item in build["requires"])


# --------------------------------------------------------------
# The console script
# --------------------------------------------------------------

def test_entry_point_target_is_exact(pyproject):
    assert pyproject["project"]["scripts"] == {"tsd-classify": "tsd.cli:main"}


def test_entry_point_resolves_and_is_callable(pyproject):
    """
    A typo here produces a script that installs cleanly and fails on
    first use, which is the worst moment to find out.
    """
    target = pyproject["project"]["scripts"]["tsd-classify"]
    module_name, _, attribute = target.partition(":")

    module = importlib.import_module(module_name)
    entry = getattr(module, attribute)

    assert callable(entry)


def test_entry_point_returns_an_int_exit_code(tmp_path):
    """
    setuptools wraps the target as `sys.exit(main())`, so returning an
    int IS the exit-code contract. A main() that returned None would
    exit 0 on every failure.
    """
    from tsd.cli import main

    code = main([str(tmp_path / "absent.pcap"), "--model",
                 str(tmp_path / "absent.joblib")])

    assert isinstance(code, int)
    assert code != 0


# --------------------------------------------------------------
# Dependencies, both directions
# --------------------------------------------------------------

def test_every_third_party_import_is_declared(pyproject):
    """
    Missing here means an ImportError after `pip install`, on a machine
    where the package looks correctly installed.
    """
    declared = declared_distributions(pyproject)
    undeclared = {}

    for name, modules in third_party_imports().items():
        distribution = IMPORT_TO_DISTRIBUTION.get(name, name)
        if distribution not in declared:
            undeclared[distribution] = sorted(modules)

    assert not undeclared, f"imported under src/tsd/ but not declared: {undeclared}"


def test_every_declared_dependency_is_imported(pyproject):
    """
    Stale here means a heavier install than the library needs, and a
    claim about the code that stopped being true.
    """
    imported = {
        IMPORT_TO_DISTRIBUTION.get(name, name) for name in third_party_imports()
    }
    unused = declared_distributions(pyproject) - imported

    assert not unused, f"declared but not imported under src/tsd/: {sorted(unused)}"


def test_dependencies_use_lower_bounds_not_pins(pyproject):
    """
    pyproject declares what the library needs to RUN;
    requirements.lock.txt records the environment the results were
    MEASURED in. Pinning in both would give two places to update, and
    one of them would rot.
    """
    for requirement in pyproject["project"]["dependencies"]:
        assert "==" not in requirement, f"{requirement} pins; use >= here"
        assert ">=" in requirement, f"{requirement} has no lower bound"


def test_plotting_is_not_a_library_dependency(pyproject):
    """
    matplotlib belongs to scripts/explain_model.py, which is operational
    tooling. Someone installing the classifier should not be installing
    a plotting stack.
    """
    assert "matplotlib" not in declared_distributions(pyproject)
    assert "matplotlib" not in third_party_imports()


def test_the_import_name_mapping_is_actually_needed():
    """
    The mapping exists for names that genuinely differ. An entry that
    maps a name to itself would be noise pretending to be care.
    """
    for import_name, distribution in IMPORT_TO_DISTRIBUTION.items():
        assert import_name != distribution


# --------------------------------------------------------------
# --version
# --------------------------------------------------------------

def test_version_flag_prints_the_package_version(capsys):
    import tsd
    from tsd.cli import main

    with pytest.raises(SystemExit) as raised:
        main(["--version"])

    assert raised.value.code == 0
    assert tsd.__version__ in capsys.readouterr().out


def test_version_is_not_in_the_verdict_schema():
    """
    The verdict document's schema is published. Adding a field is a
    schema change, and packaging is not the reason to make one -- if
    provenance is wanted there it belongs next to the model sha256, as
    its own decision.
    """
    from test_verdict import page_load, write_artefact
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        from tsd.verdict import classify_pcap, load_artefact

        artefact = load_artefact(write_artefact(root / "m.joblib"))
        document = classify_pcap(page_load(root / "t.pcap"), artefact)

    assert "version" not in document
    assert list(document) == ["schema_version", "pcap", "verdict", "model"]
