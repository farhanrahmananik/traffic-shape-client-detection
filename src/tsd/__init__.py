"""
tsd -- traffic-shape client detection.

Classify an encrypted HTTPS page-load capture as a browser or an
automated client, using traffic shape alone: packet sizes, directions,
inter-arrival times and burst structure. Never payload.

The version lives here and nowhere else. pyproject.toml reads it via
[tool.setuptools.dynamic], because two hardcoded versions drift and the
one nobody looks at is the one that ships.
"""

__version__ = "0.1.0"
