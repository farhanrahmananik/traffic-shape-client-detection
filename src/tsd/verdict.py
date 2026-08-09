"""
verdict.py
----------
Classify one PCAP with the trained model, and say so in JSON.

Pure logic: no argparse, no `sys.argv`, no printing. The step-8 CLI is a
thin shell over this, the same way `scrape_corpus.py` sits over
`mirror.py` and `explain_model.py` over `shap_explain.py`.

It does no feature arithmetic of its own. `tsd.features.read_trace()`
and `tsd.features.extract_features()` are called directly, so the tool
that ships computes the same numbers the model was trained on. If this
module reimplemented even one statistic, the drift would surface as a
tool that disagrees with the published metrics -- and it would disagree
quietly, because both halves would still return plausible floats.

Why the artefact is validated rather than trusted
-------------------------------------------------
A model is a file, and files outlive the code that wrote them. The
failure this module exists to prevent is a model trained on one feature
set being fed a vector built from another: scikit-learn does not check
column *names*, only the count, so a renamed or reordered feature
produces a confident prediction from mismatched inputs. Nothing raises,
and the verdict looks exactly like a correct one.

So `load_artefact()` compares the stored feature list against
`features.feature_names()` and refuses on any difference, naming what
moved. And `build_vector()` reads columns strictly in the artefact's
order, never in the order a dict happens to iterate.

The artefact's sha256 travels in every verdict, so a JSON file can be
traced back to the exact model that produced it -- `models/` is
gitignored, and a verdict that cannot name its model is not evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import joblib

from .features import (
    SERVER_PORT,
    TraceError,
    extract_features,
    feature_names,
    read_trace,
)

SCHEMA_VERSION = 1

# Below this, a trace is not a page load. A handshake alone is three
# packets; anything shorter than a handshake plus one exchange is a
# connection attempt, a stray retransmission, or a capture that never
# ran -- and the model has never seen such a thing, so a prediction
# from it would be an extrapolation dressed as a verdict.
MIN_PACKETS = 4

# Everything float in the output is rounded to this many decimals, so
# two verdicts on the same input diff cleanly.
ROUND_DECIMALS = 6

REQUIRED_KEYS = ("pipeline", "features", "classes", "rounds")


class VerdictError(RuntimeError):
    """Unusable input, or a model artefact that does not match this code."""


@dataclass(frozen=True)
class Artefact:
    """A loaded model, with everything needed to use and identify it."""

    pipeline: object
    features: tuple[str, ...]
    classes: tuple[str, ...]
    rounds: tuple[int, ...]
    path: Path
    sha256: str


# --------------------------------------------------------------
# Loading
# --------------------------------------------------------------

def load_artefact(path: str | Path) -> Artefact:
    """
    Load the joblib bundle written by `scripts/train_model.py`.

    Raises `VerdictError` with an actionable message rather than letting
    a mismatch through: the whole point of the check is that the failure
    it guards against is silent.
    """
    path = Path(path)

    if not path.is_file():
        raise VerdictError(
            f"model artefact {path} not found.\n"
            f"models/ is gitignored, so a fresh clone has none. Train one:\n"
            f"    pip install -e . && python scripts/train_model.py"
        )

    try:
        bundle = joblib.load(path)
    except Exception as error:  # joblib raises a wide variety
        raise VerdictError(
            f"{path} could not be loaded: {type(error).__name__}: {error}"
        ) from error

    if not isinstance(bundle, dict):
        raise VerdictError(
            f"{path} contains a {type(bundle).__name__}, not the dict written "
            f"by scripts/train_model.py. Retrain to regenerate it."
        )

    missing = [key for key in REQUIRED_KEYS if key not in bundle]
    if missing:
        raise VerdictError(
            f"{path} is missing {', '.join(missing)}. It was written by a "
            f"different version of scripts/train_model.py; retrain to "
            f"regenerate it."
        )

    stored = tuple(bundle["features"])
    _check_feature_set(stored, path)

    return Artefact(
        pipeline=bundle["pipeline"],
        # The artefact's list is the AUTHORITY for column order: the
        # pipeline was fitted with the columns in exactly this sequence,
        # and scikit-learn checks only how many there are. feature_names()
        # is cross-checked against it above and never substituted for it
        # -- if the two ever disagree the load has already failed, which
        # is the point.
        features=stored,
        classes=tuple(bundle["classes"]),
        rounds=tuple(bundle["rounds"]),
        path=path,
        sha256=file_sha256(path),
    )


def _check_feature_set(stored: tuple[str, ...], path: Path) -> None:
    """
    Refuse a model whose feature set is not the one this code produces.

    scikit-learn validates the number of columns, never their names, so
    a renamed or reordered feature yields a confident prediction built
    from the wrong inputs. Nothing raises and the verdict looks correct.
    """
    current = set(feature_names())
    stored_set = set(stored)

    if stored_set == current:
        return

    removed = sorted(stored_set - current)
    added = sorted(current - stored_set)

    details = []
    if removed:
        details.append(
            f"the model expects {len(removed)} feature(s) this code no longer "
            f"produces: {', '.join(removed)}"
        )
    if added:
        details.append(
            f"this code produces {len(added)} feature(s) the model never saw: "
            f"{', '.join(added)}"
        )

    raise VerdictError(
        f"{path} was trained on a different feature set.\n"
        + "\n".join(f"  - {detail}" for detail in details)
        + f"\nA prediction from mismatched columns would still look like a "
        f"verdict, so this is refused. Retrain:\n"
        f"    pip install -e . && python scripts/train_model.py --force"
    )


def file_sha256(path: str | Path) -> str:
    """Streamed, so a large PCAP is not read into memory whole."""
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


# --------------------------------------------------------------
# Vector building
# --------------------------------------------------------------

def build_vector(feats: dict[str, float], artefact: Artefact) -> list[float]:
    """
    Order the feature values the way the pipeline was fitted.

    Iterates `artefact.features`, never the dict. `extract_features()`
    does return a deterministic order today, and relying on that would
    couple the shipped tool to an implementation detail of a module it
    only calls -- a coupling that breaks silently, since a permuted
    vector is still a vector of the right length.
    """
    vector: list[float] = []

    for name in artefact.features:
        if name not in feats:
            raise VerdictError(
                f"feature {name!r} is missing from the extracted features. "
                f"The model and tsd.features have diverged; retrain with "
                f"scripts/train_model.py."
            )
        vector.append(float(feats[name]))

    return vector


# --------------------------------------------------------------
# Classification
# --------------------------------------------------------------

def classify_pcap(
    pcap_path: str | Path,
    artefact: Artefact,
    server_port: int = SERVER_PORT,
    include_features: bool = False,
) -> dict:
    """
    Classify one PCAP and return the published verdict document.

    The key order below is the schema; it is written literally rather
    than assembled, because the output is a published artefact and a
    reordering would show up in every diff.
    """
    pcap_path = Path(pcap_path)

    try:
        trace = read_trace(pcap_path, server_port=server_port)
    except TraceError as error:
        # One exception type for callers: the CLI should not have to
        # know which module refused.
        raise VerdictError(str(error)) from error
    except OSError as error:
        raise VerdictError(f"{pcap_path}: {error}") from error

    if len(trace) < MIN_PACKETS:
        raise VerdictError(
            f"{pcap_path}: only {len(trace)} packet(s) on port {server_port}, "
            f"fewer than the {MIN_PACKETS} needed for a page load.\n"
            f"Check the capture filter and the port -- a trace this short is "
            f"a connection attempt or an empty capture, and the model has "
            f"never seen one."
        )

    feats = extract_features(trace)
    vector = build_vector(feats, artefact)

    predicted = str(artefact.pipeline.predict([vector])[0])

    verdict: dict = {"client": predicted}

    probabilities = _probabilities(artefact, vector)
    if probabilities is not None:
        verdict["probabilities"] = probabilities

    document = {
        "schema_version": SCHEMA_VERSION,
        "pcap": {
            "path": str(pcap_path),
            "sha256": file_sha256(pcap_path),
            "packets": len(trace),
            "duration_s": _rounded(feats["duration"]),
            "server_port": server_port,
        },
        "verdict": verdict,
        "model": {
            "path": str(artefact.path),
            "sha256": artefact.sha256,
            "trained_on_rounds": list(artefact.rounds),
            "n_features": len(artefact.features),
        },
    }

    if include_features:
        document["features"] = {
            name: _rounded(feats[name]) for name in artefact.features
        }

    return document


def _probabilities(artefact: Artefact, vector: list[float]) -> dict | None:
    """
    Class probabilities, or None when the estimator cannot give them.

    Omitted rather than invented: a fabricated 1.0 for the predicted
    class would read as certainty the model never expressed.
    """
    predict_proba = getattr(artefact.pipeline, "predict_proba", None)
    if predict_proba is None:
        return None

    scores = predict_proba([vector])[0]
    labels = [str(label) for label in getattr(
        artefact.pipeline, "classes_", artefact.classes
    )]

    return {
        label: _rounded(score) for label, score in zip(labels, scores)
    }


def _rounded(value) -> float:
    return round(float(value), ROUND_DECIMALS)
