#!/usr/bin/env python3
"""
classify_pcap.py
----------------
Classify one PCAP as a browser or an automated client, and print the
verdict as JSON.

A shim. Everything lives in src/tsd/: `tsd.cli` parses the arguments,
`tsd.verdict` loads the model and produces the document, and
`tsd.features` computes the numbers -- the same code the model was
trained on. Run `python -m tsd.cli --help` for the arguments; they are
defined once, in `tsd.cli.build_parser()`.

After `pip install -e .` the console script is the shorter route, and
`tsd.cli` names itself after whichever one you used:

    tsd-classify capture.pcap

This shim does the same thing and keeps working either way:

    python scripts/classify_pcap.py capture.pcap | jq .verdict

stdout is the JSON document and nothing else; diagnostics go to stderr.
Exit codes: 0 a verdict, 2 usage error, 3 the PCAP or the model could
not be used. The exit code never encodes the predicted class.
"""

from __future__ import annotations

import sys

from tsd.cli import main

if __name__ == "__main__":
    sys.exit(main())
