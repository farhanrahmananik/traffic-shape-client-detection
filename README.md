# traffic-shape-client-detection

Given an encrypted HTTPS page-load capture, decide whether the client was a
real browser (Firefox) or an automated tool (wget), from **traffic shape
alone** — packet sizes, directions, inter-arrival times, burst structure. Both
classes load the same pages from the same local server, so the model cannot
separate them on content. Deliberately **not** in scope: payload inspection
(tcpdump keeps only the IP and TCP headers, so page content never reaches a
capture file), per-page fingerprinting, and real-time capture — the tool
classifies a PCAP that already exists.

**Method, results and the full limitations are on the case-study page:**
<https://farhanrahmananik.github.io/traffic-shape-client-detection/>
Everything below is operational.

<!-- TODO: screenshots — the page is not deployed yet. Add one of the page
     and one of the CLI output once it is. -->

## Measured environment

| Component | Version |
|---|---|
| OS | Ubuntu 24.04.4 LTS (WSL2) |
| Python | 3.12.3 |
| tcpdump | 4.99.4 (libpcap 1.10.4) |
| OpenSSL | 3.0.13 |
| **wget** | **GNU Wget 1.21.4 — not wget2** |
| **Firefox** | **153.0.3 — Mozilla APT deb, not snap** |

Per-round versions are in `results/capture_rounds/*.json` under `versions`.

- **wget2 is a different client**: multi-threaded, HTTP/2, so a completely
  different traffic shape. It will not match these numbers.
- **Snap Firefox is a different client too**: the sandbox adds startup and
  network-path overhead, landing on one class only.

Captures run without sudo, which needs:

```sh
sudo setcap cap_net_raw,cap_net_admin=eip "$(which tcpdump)"
```

## Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt   # the environment the numbers were measured in
pip install -e .                       # installs tsd-classify
pytest                                 # 432 tests, no network, no captures needed
```

`tsd-classify` installs into the venv's `bin/`, so keep the venv activated.

## Pipeline

```sh
# 1. corpus     → data/mirror/, results/corpus_manifest.json
python scripts/scrape_corpus.py --seed 42 --target-pages 100

# 2. cert + server  → certs/, results/provenance/tls_cert.txt
scripts/make_cert.sh
python scripts/serve.py --web-root data/mirror --port 8443

# 3. capture    → data/pcaps/round_NN_YYYYMMDD/, results/capture_rounds/
sudo -v && python scripts/capture_round.py --round 1

# 4. features   → data/features/features.csv
python scripts/extract_features.py --force

# 5. train      → results/metrics.json, models/client_classifier.joblib
python scripts/train_model.py --force --ablate-groups

# 6. explain    → results/shap_summary.json + three PNGs in results/
python scripts/explain_model.py --seed 42 --compare-seed 7

# 7. classify   → JSON on stdout
tsd-classify capture.pcap | jq .verdict
```

Worth knowing:

- **(1)** robots.txt obeyed fail-closed, 1.5 s between requests, seeded. Exit 1
  means a *local* failure, the class that breaks reproducibility — and piping
  hides it, so use `set -o pipefail`.
- **(2)** refuses to overwrite an existing certificate: identical certificate
  bytes across rounds is what makes rounds comparable. `serve.py` is for
  inspecting the mirror by hand; the capture harness starts its own server
  inside the namespace.
- **(3)** **probes external hosts before recording** and aborts the round if
  one resolves. **~16 min per round** for 200 traces (15.5–15.7 min across the
  four in `results/capture_rounds/`). Take rounds on **different days** — the
  split is by round, so a round means a different condition.
- **(4)** writes **800 rows × 53 features** (`results/metrics.json`,
  `dataset`), and errors, not warns, if the traces on disk disagree with the
  round metadata.
- **(5)** `LeaveOneGroupOut` on `round`, never random.
- **(7)** JSON on stdout only, diagnostics on stderr, empty stdout on failure,
  exit **0 / 2 / 3** — never the predicted class. `--model` resolves against
  the working directory, so run from the repository root.
  `python scripts/verify_cli_parity.py` checks the shipped and training feature
  paths agree (`results/cli_parity.json`); it reports no accuracy, deliberately.

## Not published

`data/` (mirror, PCAPs, features), `certs/` and `models/*.joblib` are
gitignored — `data/**` is default-deny, because the mirror path is a CLI
argument and an allowlist cannot cover a path chosen at runtime. The pages
belong to the university they were mirrored from; the captures derive from
them.

Published in their place: **`results/corpus_manifest.json`** — per page and per
asset, the URL, local filename, byte size, **SHA-256**, HTTP status and fetch
timestamp, and no BTU content. The hashes are the point: anyone who re-runs
`scrape_corpus.py` can compare their corpus against the manifest file by file
without either side distributing a byte of that content, which makes
*"regenerate it by running the scripts"* checkable rather than a promise.
`results/provenance/` holds a timestamped robots.txt and the certificate
fingerprints.

Caveat: b-tu.de's link graph moves, so the same seed on a changed site yields a
*different* corpus, invalidating captures already taken.

## The case-study page

```sh
python scripts/build_site_data.py            # results/ → docs/data/case_study.json
python scripts/build_site_data.py --check    # fails if the two are out of sync
```

`docs/index.html` contains no measured number as a literal — each is a
`<span data-bind="dotted.path">` filled by `docs/app.js` from that one file,
and `tests/test_site_bindings.py` fails if any path does not resolve.

## Limitations

Treated properly on the page; in short:

- **Loopback is not the wire.** ~0.03 ms RTT, so inter-arrival features
  measure client-side processing rather than network latency — and those are
  the features the model leans on hardest. The kernel also batches loopback
  writes, so recorded packets exceed any real frame. Equal across classes, but
  these results do not transfer to WAN captures.
- **Four rounds span three distinct days** — rounds 1 and 2 are the same local
  day, ten hours apart (`started_at` in `results/capture_rounds/*.json`).
- **One site, two clients.** Nothing shows the features generalise to another
  browser or scraper.
- **A teardown asymmetry remains**: FIN and RST were removed as features, but
  those packets still contribute to the size, count and burst statistics.
- **Two deliberate gaps in the mirror**: b-tu.de's sibling document host is not
  crawled, and responses over 8 MB are refused. Missing for both clients alike.
- **Attribution is stable at family level, unstable at feature level.** No
  single feature may be called the strongest discriminator.
- **Packaging**: `dependencies` is one flat set, so installing the CLI also
  pulls the scraping and explanation libraries.
