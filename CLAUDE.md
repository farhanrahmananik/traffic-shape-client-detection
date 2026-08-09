# traffic-shape-client-detection

## What this project is

Given an encrypted HTTPS page-load capture, decide whether the client was a
real browser or an automated tool — using **only traffic shape** (packet
sizes, directions, inter-arrival times, burst structure), **never payload**.
This is the same signal behind bot detection, scraper detection and C2
beaconing analysis.

Two classes: **Firefox** vs **wget**, loading the same pages, so the model
cannot cheat on content differences.

## Who I am

Farhan — M.Sc. Cyber Security student at BTU Cottbus-Senftenberg, RHCSA and
RHCE certified, 16 years in IT. Targeting SOC Analyst and System/Cloud
Administration roles in Germany. This repo is portfolio work: it will be read
by interviewers, so code quality and honest reporting matter more than
impressive numbers.

---

## How to work with me

- **Micro-steps. ONE step at a time.** Never lay out the whole plan — I lose
  momentum when I can see the entire road. Finish a step, show it, wait for
  confirmation before the next.
- **Write complete, working code.** Not sketches, not "you could do X".
- **Explain key decisions briefly** — complex logic, architecture, critical
  technical choices — so I can speak to them in interviews. Explain the
  *reasoning*, not the syntax.
- **Best practices and security.** Industry standards, clean code.
- **Scope control.** Stay on the core requirements. Push back on scope creep,
  including my own.
- **Language:** I write in Banglish or English. **Reply in Bengali script
  using standard English technical terms.**
- Run the tests yourself after changing code. Don't ask me to paste output
  you can read directly.

---

## Scope — this and nothing more

1. **Corpus** — a Python scraper mirrors ~100 unique public pages from
   b-tu.de. Honest User-Agent, respects robots.txt, polite delay.
2. **Serve** the mirror locally over HTTPS with a self-signed certificate,
   so captured traffic is genuinely encrypted.
3. **Capture** — one PCAP per page load per client. tcpdump filtered to the
   local server host and port only, small snaplen so payload is never
   stored, no artificial padding at either end. Several capture rounds on
   different days.
4. **Feature extraction** — per-trace features from sizes, directions,
   timings and bursts. No payload inspection anywhere.
5. **Classifier** — scikit-learn. Split train/test **BY CAPTURE ROUND, never
   randomly**. A random split leaks, because traces from one round share
   conditions. Use `GroupKFold` / `LeaveOneGroupOut` so the API enforces it
   rather than my own care.
6. **Explainability** — SHAP, showing which features drive the decision.
7. **CLI tool** — takes a PCAP, returns a JSON verdict.
8. **README + GitHub Pages case study** — semantic HTML, vanilla CSS,
   vanilla JS, no framework — publishing the measured results honestly,
   including what the model gets wrong.

### NOT in scope

Real-time capture, dashboards, deep learning, web UI, and per-page website
fingerprinting (100 classes with a handful of samples each is not trainable;
deliberately not attempting it).

---

## Publishing constraints

**Publish:** scraper, capture scripts, feature code, model code, results.

**Do NOT publish:** the mirrored site content, or the PCAPs derived from it —
that content belongs to BTU. The README explains how to regenerate everything
by running the scripts.

This is enforced in `.gitignore`, not just promised in prose:
`data/mirror/`, `data/pcaps/`, `data/features/`, `certs/`, `models/*.joblib`.

Because the mirror is not published, **the scripts must be able to regenerate
it deterministically**. That makes seeded, reproducible discovery a
correctness requirement, not a nicety.

## Definition of finished

Working code, a public GitHub repo, a README with screenshots and honestly
stated limitations, and a case-study page. **Measured numbers are published as
measured — never tuned to look better.**

---

## Background

I built a version of this corpus and these captures once before, in a
university simulation, but that environment was deleted. Everything is being
rebuilt from scratch on a fresh WSL Ubuntu. The rebuild fixes three flaws in
the original captures:

1. tcpdump ran unfiltered
2. full payload was stored
3. every trace had 3 seconds of padding at each end

These three are the reason the project is being redone. Do not reintroduce
them.

---

## Environment (measured, goes in the README)

| Component | Version |
|---|---|
| OS | Ubuntu 24.04.4 LTS (WSL2, VHDX on E:\) |
| Kernel | 6.18.33.2-microsoft-standard-WSL2 |
| Python | 3.12.3 |
| tcpdump | 4.99.4 (libpcap 1.10.4) |
| OpenSSL | 3.0.13 |
| **wget** | **GNU Wget 1.21.4 — NOT wget2** |
| **Firefox** | **153.0.3 — Mozilla APT deb, NOT snap** |
| gcc | 13.3.0 |

Two of these are load-bearing:

- **wget2 is a different client.** It is multi-threaded and speaks HTTP/2, so
  it produces a completely different traffic shape. Anyone reproducing with
  wget2 will not match these numbers.
- **Firefox must not be the snap build.** Snap runs Firefox in a confined
  sandbox with extra startup and network-path overhead. That overhead would
  land systematically on the Firefox class only, so the model could learn
  "snap is slow" instead of "browsers load pages this way" — a confounding
  variable, and exactly the kind of artefact this rebuild exists to remove.

tcpdump has `cap_net_raw,cap_net_admin=eip` set, so captures run **without
sudo**. This is deliberate: root-owned PCAPs would push the whole pipeline to
run as root.

Also worth remembering: WSL2 loopback RTT is ~0.03 ms. Inter-arrival features
therefore measure **client-side processing time**, not network latency. Good
for signal clarity, but it must be stated in the README limitations, and it
means results do not directly transfer to WAN conditions.

It has a size-side twin, measured later: kernel GSO batches loopback writes,
so single "packets" of 32768 and 47616 bytes appear in the captures. See
"Loopback is not the wire" under the `capture.py` decisions. Both limitations
say the same thing from different directions, and both go in the README.

---

## Repo layout

```
traffic-shape-client-detection/
├── scripts/       # operational: run once to produce data
├── src/tsd/       # importable library: features, model, CLI
├── tests/
├── data/
│   ├── mirror/    # BTU content        → gitignored
│   ├── pcaps/     # raw captures       → gitignored
│   └── features/  # extracted CSV      → gitignored (revisit at step 4)
├── certs/         # self-signed key + cert → gitignored
├── models/
├── results/       # metrics, SHAP plots, provenance → published
└── docs/          # GitHub Pages case study
```

`scripts/` vs `src/` is deliberate: `src/tsd/` is the actual library — it is
importable, unit-testable, and the CLI depends on it. `scripts/` is
run-once operational tooling. The practical payoff is that someone who clones
the repo **without the PCAPs** can still run and verify everything in `src/`.

Run tests with `pytest` (config in `pytest.ini`, `pythonpath = src`).

---

## Decisions already made — do not silently undo these

### `src/tsd/robots.py`

- **Python's `urllib.robotparser` matches paths by plain prefix.** b-tu.de's
  robots.txt contains `Disallow: /*/wiki/`, which under prefix matching can
  never fire — every department wiki would be silently treated as allowed.
  So this module layers a compiled-regex wildcard check on top of
  `RobotFileParser`. A path is fetched only if **both** layers allow it.
  `tests/test_robots.py::test_wildcard_rule_blocks_department_wikis` is the
  regression guard. Do not "simplify" this back to plain robotparser.
- **Fail closed.** If robots.txt cannot be fetched or parsed, nothing is
  allowed. A crawler that keeps going when it cannot read the rules is not
  compliant.
- **Plain prefix rules are NOT duplicated** in the wildcard layer. Every rule
  has exactly one owner; two copies of the same logic drift apart.
- **User-Agent** is honest and contactable:
  `traffic-shape-client-detection/1.0 (research mirror; +https://github.com/farhanrahmananik/traffic-shape-client-detection)`
  Never a browser string, never a library default — b-tu.de explicitly blocks
  generic library agents (`Python-urllib`, `httplib`, `lwp-trivial`). The
  token was checked against the site's ~130-agent blocklist (matching is
  substring-based on the part before `/`). If a WAF ever 403s us for this,
  the answer is **not** browser impersonation.
- b-tu.de declares **no `Crawl-delay`**. We impose 1.5 s anyway. A declared
  longer delay wins.
- A timestamped copy of robots.txt lives in `results/provenance/`. Since the
  mirror is not published, small provenance artefacts like this carry the
  verifiability.

### `src/tsd/fetcher.py`

- **`PoliteFetcher.get()` is the single chokepoint** for every outbound
  request. Politeness spread across call sites breaks silently — each new
  call site is a chance to forget one. Enforces, in order: same-host →
  robots policy → crawl delay → streamed size ceiling.
- **The delay is measured from when the previous response FINISHED**, not
  when it started. If BTU takes 2 s to answer, a start-anchored delay would
  collapse to zero — meaning a struggling server gets hit hardest. Backwards.
- **Host is re-checked after redirects.** An allowed URL that redirects
  off-host must not land in the mirror.
- **Body is streamed with a size ceiling** (8 MB). `response.content` would
  buffer a huge file entirely before the size could be checked.
- **`get()` returns a `FetchResult`, not a `requests.Response`.** An earlier
  version wrote to `Response._content`, coupling the module to requests
  internals; a test caught it. Nothing from requests escapes this module.
- **Every attempt is logged** — allowed, blocked and failed alike. That log
  becomes `results/corpus_manifest.json`, which is what gets published in
  place of the mirror.

### `src/tsd/urls.py`

- **URL normalisation is part of the experimental design, not tidiness.**
  Uniqueness of the ~100 pages is decided entirely by URL comparison. Count
  `/fakultaet1` and `/fakultaet1/?utm_source=x` separately and the same page
  gets mirrored twice, produces several traces, and can land on **both sides
  of the train/test split** — the exact leakage the round-based split exists
  to prevent.
- Canonicalisation: scheme and **host** lowercased, fragment dropped,
  tracking params dropped, remaining params sorted, trailing slash
  normalised, default ports stripped.
- **Path case is PRESERVED.** b-tu.de is TYPO3 on Linux, where paths are
  case-sensitive, so `/Fakultaet1/` and `/fakultaet1/` may be genuinely
  different resources. Over-eager normalisation would shrink the corpus
  silently. `test_path_case_is_preserved` guards this in the opposite
  direction from the equivalence tests.
- **robots.txt is deliberately NOT re-checked here.** `is_corpus_page()` only
  answers "does this look like a German HTML page on b-tu.de". Whether it may
  be fetched is `PoliteFetcher`'s sole responsibility. Two places deciding
  the same thing will eventually disagree.
- sha256, not md5, for filename digests. No security claim is being made, but
  a security portfolio should not model bad habits and the cost is nil.

### `src/tsd/discover.py`

- **Discovery is separate from mirroring, and runs first.** Two reasons:
  (a) a walk that finds 105 pages when the target is 100 would otherwise have
  already pulled assets for 5 discarded pages — load on someone else's server
  for nothing; (b) more importantly, rewriting a page's internal links
  requires knowing the **final** page set. A link to a page outside the
  corpus must stay absolute, or the mirrored site has a dead link → a 404
  during capture → and a 404's traffic shape is nothing like a page load.
  That would be an artefact injected into the dataset by the scraper itself.
- **Candidates are `sorted()` before `random.choice()`.** Set iteration order
  is not stable across runs, so without the sort the same seed would produce
  different corpora — and the README's "regenerate by running the scripts"
  claim would be false.
- **Content-Type is checked**, not just the file extension. TYPO3 serves PDFs
  and downloads from extensionless paths.
- **Refused URLs are remembered** so the walk never retries a dead link on
  every subsequent walk.
- **Documented limitation:** the walk never re-enters a page it has already
  collected, so a page reachable *only* through an already-collected page is
  never found. Asserted in
  `test_walk_finds_pages_reachable_without_revisiting` rather than engineered
  away. The corpus is "100 pages the walk happened to reach", not "the 100
  most important pages". This goes in the README.

### `src/tsd/mirror.py`

- **Pages come from `DiscoveryResult.html_cache`, never re-fetched.** The
  walk already paid for that HTML. Only assets are fetched here.
- **A failed asset is recorded AND neutralised.** `MirrorResult.failures`
  gets `(url, outcome, reason)`, and the reference is deleted from the HTML
  (`url("about:blank")` inside CSS). Recording alone would not be enough:
  the original `save_asset()` bug did damage by leaving the *live* URL in
  the page, so that during capture the browser fetched from the real
  b-tu.de — outside traffic in a supposedly isolated capture, invisible to
  the loopback filter. An empty `url("")` is not usable as the dead value
  either; it makes the browser re-request the page itself.
- **A non-empty `failures` list means the mirror is suspect.** The caller
  decides; the module never decides for it by staying quiet.
- **Every failure carries an `outcome` value, not just a reason string.**
  `outcome` comes from `FetchRecord.outcome`, plus three the module raises
  itself (`write_error`, `missing_html`, `depth_exceeded`). Callers branch
  on the outcome; the reason is prose for humans and will get reworded.
  `scripts/scrape_corpus.py` classifies upstream vs local on it — see
  below. Never classify a failure by matching on its reason text.
- **`<base href>` is removed.** Left in, that one tag would re-point every
  carefully rewritten relative reference back at the live site at load
  time. It is used to resolve references, then dropped.
- **Two prefixes, one asset map.** Pages sit at the mirror root, assets in
  `assets/`. So a page refers to `assets/x.png` while a stylesheet — which
  already lives in `assets/` — refers to the sibling `x.png`. Getting this
  backwards yields a mirror that looks right in a file listing and 404s in
  the browser.
- **Assets are deduplicated by normalised URL and fetched exactly once.**
  A site-wide stylesheet referenced from 100 pages is 1 request. Failures
  are cached too, so a dead URL is reported once, not per page.
- **Circular `@import` terminates** because the asset map entry is written
  *before* a stylesheet's own contents are walked. `MAX_CSS_DEPTH = 3` is a
  separate guard; exceeding it is a recorded failure, not a silent
  absolute URL left in place.
- **`srcset` is parsed per the HTML algorithm, not `split(",")`** — a
  `data:` URI contains commas. Descriptors (`1x`, `800w`) are preserved:
  they decide which image the browser actually requests, so dropping them
  would change Firefox's request pattern and leave wget's untouched — an
  artefact landing on one class only.
- **`<style>` blocks are read with `get_text()`, not `tag.string`.**
  `.string` is `None` whenever the block is not exactly one child, which a
  CSS comment or CDATA section can cause — and the block would then be
  skipped in silence with every `url()` in it still absolute. Same bug
  class as above, reintroduced by an idiom. Rewritten content is put back
  as a `Stylesheet` string so no formatter escapes a `>` child selector.
- **Anything a browser auto-fetches is handled**: `<link rel=stylesheet|
  icon>`, `<script src>`, `<img src|srcset|data-src|data-original|
  data-lazy-src>`, `<source>`, `<input type=image>`, `<iframe>`,
  `<embed>`, `<object data>`, `<video src|poster>`, `<audio>`, `<track>`,
  inline `style=`, `<style>` blocks, and `url()` inside CSS files.
  `<link>` relations that a browser fetches but we do not mirror
  (`preload`, `prefetch`, `preconnect`, `apple-touch-icon`, `manifest`, …)
  have their tags **stripped**; metadata relations (`canonical`,
  `alternate`) are harmless and stay.
- **Third-party embeds neutralise themselves.** A YouTube or OSM `<iframe>`
  is off-host, so `PoliteFetcher` refuses it with `blocked_host`, the
  failure is recorded, and the attribute is deleted.
- **Link rewriting knows the frozen page set**: in-corpus → local
  filename; anything else → absolute URL to the live site. Relative would
  resolve against the local server and 404 during capture, and a 404's
  traffic shape is nothing like a page load.
- **Deterministic**: pages iterated `sorted()` (the caller may pass a set,
  and set order is not stable), assets in document order, filenames from
  `urls.py` digests, no timestamp written anywhere. Same cache in, same
  bytes out — which is what makes "regenerate it by running the scripts"
  true.

#### Known issue carried into step 4 — measured, not hypothetical

The mirror stores b-tu.de's own JavaScript, and **Firefox will run it
during capture**. Link rewriting cannot reach inside that JavaScript, so
live URLs survive in it. This was **measured**, not assumed: five BTU
pages were mirrored and scanned, and live `b-tu.de` URLs remain in

- `data-cookieman-settings` — the cookie-consent JSON config, present on
  **every page**, with b-tu.de URLs inside it
- `data-condition-uri="https://www.b-tu.de/barrierefreiheit?type=3132"` on
  a `<script>` tag — almost certainly fetched by AJAX at runtime

**mirror.py deliberately does not rewrite these.** `data-*` attributes are
JavaScript's private namespace; the browser never auto-fetches them, and
guessing which ones hold URLs is an unbounded surface where a wrong guess
corrupts a working page. The three lazy-loading `data-*` names it does
handle are there because they are a documented convention with a known
meaning — not because the module tries to parse arbitrary data.

This is **not fixable in the scraper**, and it is no longer a prediction.

#### MEASURED: Firefox does reach the live site — 2026-08-06

Loading the mirror's root page in Firefox with the Network panel
recording and the cache disabled, two requests left the machine:

| Request | Result |
|---|---|
| `GET https://www.b-tu.de/…/matomo.js` | 200, 68.45 kB, **409 ms**, initiator `line 518 > injectedScript` |
| `POST https://www.b-tu.de/…/matomo.php?action_name=…` | beacon, `NS_BINDING_ABORTED` after **190 ms** |

Both are BTU's own Matomo analytics, surviving inside JavaScript that
link rewriting cannot reach. **Neither appears in the PCAP**, because
tcpdump is filtered to the local host and port — but their DNS, connect
and TLS time lands inside Firefox's inter-arrival times, and wget
executes no JavaScript, so the cost falls on **one class only**. 409 ms
is not a rounding error next to a loopback RTT of ~0.03 ms.

**Network isolation during capture is therefore a measured requirement,
not a precaution.** Step 4 blocks all traffic except loopback for the
duration of each capture — network namespace, or firewall rules on the
capture host.

#### Also measured on that load — what step 4 and step 5 must decide

- **Firefox issued 62 requests; `wget --page-requisites` issued 116 for
  the same page.** wget fetches every `srcset` variant, Firefox picks one
  per set and defers lazy-loaded images. This is **genuine client
  behaviour — signal, not artefact. Do not try to equalise it.**
- **Firefox requests `/favicon.ico` (404); wget requests `/robots.txt`
  (404).** One trivially learnable marker per class, produced by how each
  client is invoked rather than by how it loads a page. **Step 4 must
  decide how each client is invoked, and the README must state it** — a
  classifier that scores well by finding a 404 has learned the harness.
- **Firefox kept requesting carousel images after the load completed.**
  **Step 5 must decide when a trace ends.** Cutting too early removes
  exactly the Firefox signature; cutting too late measures idle time.

Related, and worth stating separately in the README: the **cookie banner
runs under Firefox and not under wget**. That difference is genuine client
behaviour and belongs in the data — a browser executing page JavaScript is
exactly what separates the two classes. What must not be in the data is
that banner's JavaScript reaching the outside world, because then its
latency enters the Firefox class as contamination rather than as client
behaviour. Isolation keeps the first and removes the second.

### `scripts/scrape_corpus.py`

- **Mirror failures are split into three classes, and only one is an
  alarm.** The question being asked is not "did everything work" but
  "will the next run produce the same corpus".
  - *upstream* (`http_error`, `blocked_robots`) — properties of b-tu.de:
    its own CSS references jQuery-UI images that are not deployed, and
    robots.txt withholds a script. **29** on the first full run.
  - *excluded* (`blocked_host`, `too_large`) — deterministic too, but
    withheld by **our** policy: off-host assets and responses over the
    8 MB ceiling. **7** on the first full run.
  - *local* (`error`, `write_error`, plus anything unrecognised) — vary
    between runs, so the corpus stops being reproducible. **1**.
  Upstream and excluded → **exit 0**, one summary line each. Local →
  **exit 1**, loud.
- **Why three and not two:** the first split called all 8 non-upstream
  failures local, and 7 of them were in fact deterministic. A gate that
  fires on every run is a gate people learn to scroll past — and then the
  one that matters scrolls past too. An alarm that is never silent is not
  an alarm.
- **Classification is on `outcome`, never on the reason string**, and an
  **unrecognised outcome counts as local**. A new failure mode should be
  noticed, not absorbed into a list of things already decided not to care
  about.
- **Known limit of the `blocked_host` bucket:** `PoliteFetcher` uses that
  one outcome both for a URL that was off-host to begin with
  (deterministic) and for a request that *redirected* off-host mid-flight
  (potentially a real anomaly). Both are counted as excluded for now.
  Separating them means a new outcome in `fetcher.py` — not parsing the
  reason text.
- **Piping hides the exit code.** `… | tee scrape.log` reports tee's
  status, so the gate fired on the first full run while `$?` read 0. Use
  `set -o pipefail` or `${PIPESTATUS[0]}`.
- **The manifest is the published substitute for the mirror**, so it
  carries no BTU content — only URLs, local filenames, sizes, sha256
  hashes and timestamps. The hashes are the point: they let anyone
  re-running the script check they got the same corpus without either
  side publishing a byte of b-tu.de's content.
- **`status_code`, `content_type` and `fetched_at` come from
  `fetcher.log`**, and the page/asset inventory from `mirror.py`'s
  `on_event` callback. Neither is re-derived here: a second source of
  truth for the same facts is a pair of sources that will disagree later.
- **`bytes` and `sha256` are read back off disk**, not taken from memory,
  so the manifest describes the file that will actually be served.

#### Two deliberate gaps in the mirror — both go in the README limitations

**`www-docs.b-tu.de` is not mirrored.** BTU serves documents from that
sibling host, and `PoliteFetcher` is single-host by design, so those
assets are refused as `blocked_host` (5 on the first run). This is a
**deficiency of the mirror against the live site, not a class-confounding
artefact**: both clients load the same mirror, so anything missing from it
is missing for Firefox and wget alike, and affects them equally. Widening
the fetcher to a second host would mean a second robots.txt, a second
crawl-delay budget and a second trust boundary — cost paid on the crawl
side for no gain on the measurement side.

**The 8 MB response ceiling stays as it is** (2 assets refused,
`too_large`). It exists so a stray video or large PDF cannot walk into a
corpus that is loaded ~100 times per capture round. Same reasoning as
above: a missing large asset is missing for both clients. Raising the
ceiling would inflate every trace on both sides without separating them.

Neither is a bug to fix. Both are honest limitations to state: the mirror
is *the site minus these*, and the measured numbers describe traffic
shapes on that corpus.

#### Future improvement — deliberately NOT now (scope)

A targeted repair path, so one failed asset does not imply a full
re-scrape. Two things block it today:

1. `DiscoveryResult.html_cache` lives only in memory, so redoing one
   page's rewrite means re-fetching that page.
2. Failures record the asset URL but not the page that referenced it,
   and the neutralised reference leaves no trace on disk to grep for.

Persisting the cache (gitignored) and recording the referring page would
reduce a repair to a handful of requests. It would also need `SiteMirror`
to separate "pages to write" from "the known page set used for link
rewriting" — today one argument does both, so re-mirroring a single page
reports the other 99 as `missing_html`.

Worth doing only if a repair is ever actually needed. Re-scraping is the
worse option in the meantime, not because of the load but because BTU's
link graph moves: the same seed on a changed site yields a *different*
corpus, which would silently invalidate captures already taken.

### `src/tsd/server.py`

- **The per-connection request cap is 1000, and it is deliberately
  generous. Do not lower it.** At 100 it fired on the real corpus:
  `wget --page-requisites` sent 116 requests down one sequential
  connection and got closed at 100, then reconnected — a full TLS
  handshake injected into the middle of the trace. Firefox spreads the
  same 116 across ~6 parallel connections and never comes near the cap.
  **A limit only one client can reach is a server property the model
  would learn as a client property.** The cap stays as a DoS guard, but
  it now warns loudly — `--quiet` does not silence it — and only when
  the client did not itself ask to close.

#### TLS, as measured

All measured 2026-08-06, `data/mirror` served on `127.0.0.1:8443`.

- **No version pin in the `SSLContext`.** Both clients negotiate TLS 1.3
  unprompted — Firefox 153.0.3 and GNU Wget 1.21.4. Pinning would
  constrain the clients without changing what they do, and a pin that
  silently stops matching a future client is worse than no pin.
- **The cipher difference does not matter, and that was checked rather
  than assumed.** Firefox picks `TLS_AES_128_GCM_SHA256`, wget picks
  `TLS_AES_256_GCM_SHA384`. TLS 1.3 AEAD record overhead is identical
  either way — 5-byte header plus 16-byte tag — so the key length
  changes neither ciphertext length nor packet boundaries. The
  difference is in the ClientHello, which is client behaviour and
  therefore signal, not an artefact of the setup.
- **Certificate: a local root CA plus an IP-only leaf**, fingerprints
  pinned in `results/provenance/tls_cert.txt`. Firefox accepts the
  825-day leaf through a user-imported CA; the 398-day cap applies only
  to roots in the Mozilla program. **Verified before any capture was
  taken** — deliberately, because a certificate problem discovered
  mid-round invalidates the round.
- **Firefox resumes TLS sessions; wget never does.** Observed:
  "Reused, TLSv1.3" with 8 session-cache hits for Firefox, while wget
  starts a fresh process per fetch. A resumed handshake omits the server
  certificate, so every Firefox trace after the first would be ~1 KB
  lighter than the first — a **process-lifetime artefact, not client
  behaviour**. Handled at capture time with a fresh Firefox profile per
  page load, which the HTTP cache requires anyway.
  **Do not "fix" this by disabling session tickets on the server.** That
  would make the server behave unlike any real server, and it would hide
  a resumption bug in the capture harness rather than prevent one.

### `src/tsd/capture.py`

Everything here was measured on the way to a working round, not designed
in advance. Where a decision cost a debugging session, that is recorded —
the wrong turn is the useful part.

- **`sudo unshare -n`, never `unshare -rn`.** The `-r` flag adds a user
  namespace that maps the caller to uid 0. tcpdump then drops privileges
  and tries to `chown` the savefile to a uid that is not mapped in that
  namespace, fails, and exits — leaving a **24-byte pcap with a valid
  header and zero packets**. The filename is right, the header is right,
  and the trace is empty. Nothing else in the pipeline would notice.
  This is why `count_pcap_packets()` exists at all, and why zero packets
  is a **recorded failure rather than a silent success**. Two assertions
  hold the invocation (`command[:3] == ["sudo", "unshare", "-n"]` and
  `"-rn" not in joined`) — a comment would not have survived the next
  edit.
- **The server runs inside the namespace.** Each namespace has its own
  loopback, so a server started outside is simply unreachable from
  inside — which presents as a broken server rather than a broken
  invocation. **One server per round, not per page**: per-page startup
  would land in every trace.
- **Isolation is verified per round**, by probing external hosts, and a
  host that resolves **aborts the round** rather than warning. `--no-netns`
  waives only the marker, never the check: **a flag is a claim, a failed
  connection is a measurement.**
- **Firefox's profile directory must be created before launch.**
  `--profile` does not create it, and a missing one kills Firefox with
  **SIGKILL and no useful message**. This cost a debugging round: the
  failure was first attributed to the network namespace, and the control
  experiment — the same command with no namespace — is what showed the
  namespace was innocent. **The wrong fix would have been
  `MOZ_DISABLE_CONTENT_SANDBOX=1`**, which changes Firefox's process
  structure and would have put a timing artefact into one class only.
  Worth remembering as a method: when something fails inside a new
  environment, test the same thing outside it before blaming it.
- **Firefox runs under Wayland** (`MOZ_ENABLE_WAYLAND=1`). X11 uses
  abstract unix sockets, which network namespaces isolate; Wayland's
  socket is a filesystem path and survives. Verified working inside the
  namespace, which is why Wayland is forced rather than autodetected.
- **wget runs with `-e robots=off`.** In recursive mode it requests
  `/robots.txt` first and gets a 404 that Firefox never produces — a
  property of **how the harness invokes wget**, not of wget being
  automation. A classifier that scores well by finding a 404 has learned
  the harness. **Firefox's own `/favicon.ico` request is deliberately NOT
  suppressed**: the browser does that by itself, unasked, so it is
  genuine client behaviour and belongs in the data.
- **End of load is the capture going quiet, not a fixed timeout**,
  because Firefox keeps requesting carousel images after the page reports
  itself loaded. Measured on the smoke run: the quiet wait is
  **wall-clock only and does not extend the trace** — traces ran
  0.38–0.50 s while `quiet_seconds` was 3.0, so no padding enters the
  packet timestamps. The old captures' 3 s of padding at each end is not
  reintroduced.
- **`pages_attempted` is counted from the traces that exist, never from
  `--limit`.** Limit is what was asked for; traces are what happened.
  The first smoke run wrote `"pages": 100` for a 4-trace round — intent
  recorded as outcome, and unreadable as such six months later.

#### MEASURED: smoke run, 2026-08-07 (2 pages, both clients)

| client | SYNs | packets | duration |
|---|---|---|---|
| firefox | 6 | 159 | 0.39–0.50 s |
| wget | 1 | 223–227 | 0.38–0.42 s |

Firefox opens ~6 parallel connections; wget uses one sequential
connection. **This is visible in the PCAP, and it is the reason the
server had to be threaded** — a sequential accept loop would have
serialised the six and recorded the server's queuing as the client's
behaviour.

Firefox produced **exactly 159 packets on both pages**. Not a bug: both
pages reference exactly 13 assets, because BTU's interior pages share
nearly all their chrome. That shared structure is **favourable here** —
it leaves little for a model to learn about the page rather than the
client, which is the whole point of loading the same pages with both
clients.

#### MEASURED: four rounds across three distinct days

Read from `results/capture_rounds/*.json` and cross-checked against the
`date` column of `data/features/features.csv`; the two sources agree.

| round | `date` (local) | `started_at` (UTC) |
|---|---|---|
| 1 | 20260807 | 2026-08-06T22:46:56+00:00 |
| 2 | 20260807 | 2026-08-07T08:51:09+00:00 |
| 3 | 20260808 | 2026-08-08T03:04:24+00:00 |
| 4 | 20260809 | 2026-08-08T22:09:39+00:00 |

**`date` is local time (UTC+2) while `started_at` is UTC**, which is why
round 1's filename says 07 against a UTC date of 06, and round 4's says
09 against a UTC date of 08. Confusing, but not a bug — and **do not
"fix" the field or rename the directories**. Both are already referenced
by `extract_features.py`'s metadata cross-check and by committed
metadata; renaming them would make the published record disagree with
the data it describes, which is the exact failure that cross-check
exists to catch.

**The README must print the timestamp, not just the phrase "a different
day".** A reader looking at the UTC column will otherwise ask the
question themselves and have no answer in front of them.

**The finding, and it still stands: rounds 1 and 2 are the same local
day, about 10 hours apart.** Four rounds span **three** distinct days,
not four. The scope statement in this file requires "several capture
rounds on different days" — the round is supposed to be a *different
condition*, not a second name for one. Round 4 does not erase that
limitation for rounds 1 and 2; it adds a genuinely independent day
alongside them.

**The evaluation was never invalid.** `LeaveOneGroupOut` ran over
genuine groups with no leak, and the metrics stand as measured. What was
weakened is what the group *meant*: two of the folds hold out conditions
nearly identical to each other. Round 4 is what makes that statement
testable rather than merely admitted — see "Round 4, and what it
answered" below.

It also **retroactively explained the near-zero per-fold SHAP spread**
recorded under `shap_explain.py`. That section called the rounds "near
replicates" — an inference at the time, then a measurement: two of them
are literally the same day, the same uptime, the same machine state.

#### Round 4 — captured 2026-08-09 (local)

`started_at` 2026-08-08T22:09:39+00:00 = local 2026-08-09 00:09.
**200 traces, 0 failures, 47,906 packets.** Metadata:
`results/capture_rounds/round_04_20260809.json`.

Taken ~19 hours after round 3, **after a machine restart and a fresh WSL
session** — a different calendar day *and* a different machine state,
which is what the requirement was actually about. A round taken an hour
later on the same uptime would satisfy the calendar and none of the
intent.

**Why it was taken, recorded because the reason shapes the write-up:**
not for accuracy. The classifier was already at 1.0000 and a fourth
round was never going to move it; adding data to improve a number
already at ceiling would have been the wrong reason, and the wrong
reason produces the wrong write-up. It was taken for the open question
the seed experiment created — whether attribution instability is a
property of **the model fitting** or of **the capture conditions**.

**Comparability was verified, not assumed.** Before any of round 4's
numbers were trusted:

- `server_cert_sha256` **identical to round 3** — the certificate is
  transmitted in every handshake, so a different one would change the
  first bytes of every trace
- the full `versions` dict **identical**: Firefox 153.0.3, Wget 1.21.4,
  tcpdump 4.99.4, OpenSSL 3.0.13, Python 3.12.3
- all capture parameters unchanged: snaplen 96, same filter,
  `quiet_seconds` 3.0, `max_load_seconds` 90.0, port 8443, same
  `web_root`

Packet count is **135 above round 3 (~0.3%)** — the same order of
variation already seen between rounds 1 and 3, so nothing about the
round asks for an explanation.

This is the check the overwrite guard in `make_cert.sh` exists to make
possible: identical certificate bytes across rounds by construction, and
verified here rather than trusted.

#### After round 4 — checklist, all done

1. [x] **`scripts/extract_features.py --force`** — all four rounds
   re-extracted, metadata cross-check passed.
2. [x] **`scripts/train_model.py --force --ablate-groups`** —
   `LeaveOneGroupOut` and the ablation sweep re-run over four folds.
   Numbers under "MEASURED: step 6" below.
3. [x] **Explanation re-run, three quantities compared, not two.**

**Step 3 was the point of round 4.** The obvious version of it would not
have answered the question: with four rounds every fold trains on three
rounds and therefore mixes days, so **no fold represents a single day's
conditions**. Re-running under two seeds alone measures the seed and
nothing else. The comparison had to be between round *pairs*.

#### Round 4, and what it answered — MEASURED

All from `fold_importance_table()` / `importance_spread()`, seed 42
unless stated.

| | quantity | pair | max spread |
|---|---|---|---|
| **(a)** | same-day | r1 vs r2 | **0.00335** (`iat_down_std`) |
| **(b)** | cross-day | r3 vs r4 | **0.00308** (`count_ratio_up_down`) |
| **(c)** | seed | 42 vs 7 | **0.02128** (`iat_down_max`) |

Largest rank movement from seed 42 to seed 7: **24 places** across the
53 features, with **accuracy 1.0000 under both seeds**.

**The reading, following the checklist's own third branch.** (b) is if
anything *slightly smaller* than (a): the capture day has **no
measurable effect** on attribution. Both are six to seven times smaller
than (c). **The instability belongs to Shapley credit allocation among
redundant features, not to the capture rig.**

That is the more favourable of the possible outcomes for the write-up —
the limitation attaches to **the explanation method** rather than to
**the measurement apparatus** — and it is recorded because it was
measured, not because it reads better. The experiment was set up to be
able to say the opposite.

**Refinement of the earlier seed finding, and it sharpens the claim.**
The top **ten** features are the **same set** under both seeds, only
reordered, and `iat_max` is rank 0 under both:

| seed 42 | seed 7 |
|---|---|
| `iat_max` | `iat_max` |
| `iat_down_max` | `size_up_std` |
| `size_up_max` | `iat_up_max` |
| `iat_up_max` | `size_up_p90` |
| `size_up_p90` | `size_up_max` |
| `size_up_std` | `iat_down_max` |
| `burst_len_mean` | `syn_ack_count` |
| `iat_median` | `ack_down_count` |
| `ack_down_count` | `burst_len_mean` |
| `syn_count` | `syn_count` |

So attribution is **stable at family level and unstable at feature
level**. That is a sharper and more useful statement than the
three-round version, which could only say that the ranking moved.

**What the README and case study MAY claim:**

- the timing and upstream-size families together carry most of the
  attribution, across four rounds and both seeds
- which individual feature receives credit within a family is
  seed-dependent, moving up to **24 places** while accuracy stays at
  **1.0000**
- the capture day affects none of this — measured by comparing a
  same-day round pair against a cross-day pair, not asserted

**What they MAY NOT claim:** that any single feature is the strongest
discriminator.

#### Loopback is not the wire — for the README limitations

Kernel GSO batches writes on loopback, so the captures contain single
"packets" of **32768 and 47616 bytes**. This affects both classes
equally and is **not class-confounding**, but it means the size and
burst features describe **application-layer write behaviour rather than
MTU-bounded wire patterns**.

It is the size-side twin of the ~0.03 ms loopback RTT already recorded
under Environment, which does the same thing to the timing features.
**Both belong in the README**: together they say that these results
characterise client behaviour under ideal local conditions, and do not
transfer directly to WAN captures.

### `src/tsd/features.py`

- **Two layers, kept apart.** `read_trace()` owns dpkt and the file I/O;
  `extract_features()` is a pure function over the records. The step-8
  CLI calls the second one directly, so **the training path and the
  inference path cannot drift apart** — if each did its own parsing and
  its own arithmetic, the drift would surface as a model that scores
  well in evaluation and badly in the tool that ships.
- **Payload length comes from the IP and TCP headers, never from the
  captured frame.** Snaplen is 96, so the frame on disk is clipped:
  using its length would make **every packet larger than 96 bytes
  identical**, destroying the entire size family while the feature table
  still looked perfectly plausible. `ip.len` minus both header lengths
  is present in those 96 bytes and is the real number. A test builds
  exactly that situation — headers claiming 1448 and 32768 bytes, frames
  clipped to 96 — because this is a failure that would never raise.
- **A burst is a run of packets in the same direction, with no time
  threshold.** This is a **modelling choice, not a fact about the
  data**, so it is stated rather than buried. The obvious alternative —
  same direction *and* less than T seconds apart — was rejected because
  T has no principled value here: with a loopback RTT of ~0.03 ms, any T
  separates "the client thinking" from "the network working" at a point
  chosen by us, and a threshold picked by looking at a dataset this size
  is a threshold that can be tuned toward the answer, knowingly or not.
  Direction changes need no parameter and are decided by the protocol.
  The cost — a long pause inside one direction does not split a burst —
  is carried by the `iat_*` features, where that pause appears as a
  large inter-arrival.

#### `fin_count` and `rst_count`: implemented, measured, removed

Measured across all 200 round-1 traces: **Firefox 0/100 FINs and 0/100
RSTs; wget 100/100 of each.** No exceptions.

That is not client behaviour, it is the harness. wget exits by itself,
so its teardown lands inside the capture window; Firefox is killed only
after tcpdump has already stopped, so its teardown is never recorded.
The feature measures **how we stop each client**. Kept, it would have
been a perfect separator sitting at the top of the SHAP plots explaining
the wrong thing.

This one was **harder to catch than the `/robots.txt` 404**, because
"connection teardown feature" sounds legitimate — nothing about the name
suggests it is about the harness. What caught it was running the
features over the real captures and looking at a separation that was too
clean.

**Record that as method: a feature that separates perfectly is a reason
to check the harness before celebrating.** The same reflex applies to
step 6 — an accuracy that looks too good is a hypothesis about the rig,
not a result.

`syn_count` **stays**: 6 against 1 happens during the load, not at
teardown, and it is the parallelism that made the threaded server
necessary.

*Residual, measured and not hidden:* the FIN and RST packets are still
in the wget traces — **200 packets, 0.73% of wget's total** — and they
still contribute to the count, size and burst features. Removing them
would mean discarding packets under an arbitrary rule, a heavier
intervention than the problem warrants. **This goes in the README
limitations**: a small teardown asymmetry remains in the data even
though no feature names it.

#### Excluded on purpose

Ports, addresses, absolute timestamps, TCP window size, MSS, option
ordering, initial sequence numbers.

They are **real client fingerprints, and that is exactly why they cannot
be used**. The claim under test is that traffic *shape* alone separates a
browser from a scraper. A stack fingerprint would win without ever
testing that claim, and the SHAP plots would then faithfully explain a
different experiment. Absolute timestamps are excluded for a second
reason: they encode the capture round, which is the split group.

The guard test is **token-based, not substring** — `'rst'` matches
`'burst'`, so a substring check would either ban the whole burst family
or, once someone "fixed" it by dropping `rst` from the list, stop
guarding the thing it was written for.

### `scripts/extract_features.py`

- **`round` is written into the CSV, not reconstructed from a path
  later.** The split is the one thing in this project that **cannot be
  validated by looking at the result**: a leaked group produces *good*
  numbers, not bad ones, and nothing downstream complains.
- **The trace count on disk is cross-checked against `traces_ok` in the
  round metadata, and a mismatch is an error, not a warning.** The
  metadata is what is published in place of the PCAPs; if they disagree,
  either the published record is wrong or the data changed outside the
  harness, and **re-reading the disk cannot tell you which**. The CSV is
  still written, for investigation, with stderr saying not to train on
  it. Both the mismatch and the missing-metadata paths were exercised by
  removing a PCAP and by pointing at an empty metadata directory — an
  unexercised check is not a check.
- **Constant features are reported, never dropped automatically.** Round
  1 found three: `size_up_min`, `size_up_p25`, `size_down_min`, all zero
  because every trace carries pure ACKs in both directions. **They are
  kept.** Dropping a feature because it is constant is a decision made by
  looking at the whole dataset *including the test rounds*, which is a
  mild leak; and a feature that is constant in round 1 may not be in
  round 3. The cost of keeping them is nil — tree models ignore
  zero-variance columns and SHAP assigns them zero. What the report is
  for is the *question*: a constant feature is usually a bug in
  extraction and occasionally a fact about the traffic, and dropping it
  silently answers neither.
- **`data/features/` is gitignored because it derives from unpublished
  PCAPs, not because it leaks anything.** Payload was never captured, so
  the CSV holds only counts, sizes and timings. Publishing a derivative
  of unpublished data would be inconsistent; that is the whole reason.

### `src/tsd/model.py`

- **`iter_round_folds()` is the only splitter in the project.** The
  `LeaveOneGroupOut` loop was extracted into one shared function because
  step 7 computes SHAP over held-out rounds and must use the **same**
  folds. Two splitters in two modules would not fail loudly if they
  drifted: the metrics would stay honest while the SHAP plots quietly
  described training data — and those plots would look **better**, not
  worse, because a model explaining data it was fitted on gives tidier,
  more confident attributions.
- **The per-fold assertions are redundant on purpose.**
  `LeaveOneGroupOut` already guarantees one held-out group and no
  overlap. They are asserted anyway because this is the one invariant in
  the project whose violation makes the numbers look *better* — the
  class of bug that no amount of staring at results will reveal.
- **No shuffle, no `random_state` in the split path.** Fold order
  follows the group values, so folds are reproducible without a seed at
  all. That matters beyond tidiness: SHAP is computed per fold, and a
  published plot that cannot be regenerated from the same inputs is not
  evidence.
- **The refactor was proven behaviour-preserving, not assumed.** The
  pre-refactor module was loaded side by side via
  `git show HEAD:src/tsd/model.py`, both were run on the same synthetic
  dataset, and `to_dict()` plus `build_metrics()` output was compared as
  JSON with `sort_keys=True`. All three comparisons byte-identical.
  **Record the method, because it generalises:** when changing something
  whose failure mode is invisible in the output, compare against the old
  implementation directly instead of inspecting the new one's results.

#### MEASURED: step 6, `LeaveOneGroupOut` on `round` — four rounds

**800 traces, 4 rounds, 53 features.** Constant features unchanged
(`size_up_min`, `size_up_p25`, `size_down_min`) — round 4 introduced no
new ones, which is itself a small check that the fourth round is the
same kind of data as the first three.

| model | pooled accuracy | notes |
|---|---|---|
| random_forest | **1.0000** | per fold r1/r2/r3/r4 all 1.000; **no page misclassified at all** |
| logistic_regression | 0.9950 | 4 errors, **all in the round-2 fold** |

The linear model's four errors are `bibliothek`,
`lausitz-science-park`, `studieninteressierte` (senftenberg) and
`universitaet`.

Ablation, random_forest, same protocol:

| configuration | accuracy |
|---|---|
| all features | 1.0000 |
| without `syn_count` | 1.0000 |
| without connections | 1.0000 |
| without sizes | 1.0000 |
| without timing | 1.0000 |
| without bursts | 1.0000 |
| **only `syn_count`** | **0.9900** |

**No single family carries the result.** The signal is redundant:
several families would do the job alone.

`only syn_count` now scores **0.990 in every one of the four folds**,
and the failures are the same two pages every time: **`ikmz_xwiki` and
`webmail`**. Those two reference no assets, so Firefox opens no parallel
connections — SYN = 1, exactly as wget. CLAUDE.md predicted these would
be the hardest pages; the prediction has now **held four times,
including on an independent day**. Record it as **established**, not as
a pattern that might be coincidence.

On those two pages the difference is a constant **3 packets** (firefox
20/12/8, wget 17/10/7) — 2 extra upstream requests, most likely
Firefox's `/favicon.ico`, which is deliberately not suppressed because
the browser issues it unasked.

The teardown residual recorded under `features.py` left **no visible
trace**: parity 149/300 against 151/300 on the three-round run.

##### The one result the forest could not show

For **logistic_regression**, `without timing` scores **1.0000** while
`all features` scores **0.9950**. **Removing the timing family improves
the linear model.** The forest cannot show this, because it is already
at ceiling with everything included — a model at 1.0000 has no room to
get better, so an ablation on it can only ever detect harm.

Offered as a reading rather than a claim: a few outlying `iat_*` values
pull a linear decision boundary while leaving tree splits unaffected,
which is what one would expect from a family whose distribution has a
long tail.

**This is the same story as the SHAP result, from the opposite end.**
The family the forest leans on hardest — `iat_*` holds the top SHAP
ranks — is the family that costs the linear model its only errors. And
this file already records that on loopback, `iat_*` measures
**client-side processing rather than network latency**. Three separate
observations pointing at one family, from three different directions.
**This belongs in the case study**, and it is a better story than the
headline 1.0000.

### `src/tsd/shap_explain.py`

Library layer only: no plotting, no file writing, no CLI.

1. **SHAP is computed per fold, on the held-out round only**, reusing
   `iter_round_folds()`. Explaining a model on data it was fitted on
   reintroduces, *in the explanation*, exactly the leak the round-based
   split prevents in the evaluation — and it does so invisibly, because
   the plots come out cleaner rather than obviously wrong.
2. **`feature_perturbation="tree_path_dependent"`**, not the
   "interventional" default. With 53 correlated features, the
   interventional scheme evaluates each against a marginal background
   and so breaks correlated groups apart: a redundant-but-rarely-split
   feature receives near-zero attribution, producing a confident "one
   feature does everything" picture that would **directly contradict the
   measured ablation**. It also needs no background dataset, so there is
   **no knob to tune toward a nicer plot**.
3. **The class index is read from the fitted classifier's `classes_`,
   never assumed.** Measured: `positive_class` is `wget` at index 1,
   because `sorted(["firefox", "wget"])` puts it there — an
   implementation detail of label sorting. Relabelling one class to
   `ff` would invert every plot and **nothing would raise**. Same
   failure class as the snaplen bug in `features.py`: plausible output,
   no exception.
4. **The SHAP array shape is asserted** — `ndim == 3` and
   `(n_test, n_features, n_classes)`. shap has changed this return
   shape between versions, and a 2-D array indexed as though it were
   3-D would silently take a column along the wrong axis. Measured
   environment: **shap 0.52.0, scikit-learn 1.9.0, numpy 2.4.6**, with
   `expected_value` of shape `(2,)` summing to 1.0 — probability space,
   not log-odds.
5. **Trees only; explaining `logistic_regression` raises.**
   `TreeExplainer` does not apply, and that pipeline carries a
   `StandardScaler`, so the values would be in scaled units while the
   feature names still claim bytes, seconds and packets. A second
   assertion checks the forest pipeline contains **only** a classifier
   step — the name `random_forest` would not catch a scaler added later
   for tidiness.

#### MEASURED: what SHAP measured

Positive SHAP means "toward wget". Base value **0.4984** in all three
folds, consistent with perfectly balanced folds.

Pooled top features by mean |SHAP|:

| feature | mean |SHAP| |
|---|---|
| `iat_down_max` | 0.05977 |
| `iat_max` | 0.05787 |
| `size_up_max` | 0.05268 |
| `iat_up_max` | 0.05125 |
| `size_up_p90` | 0.05109 |
| `size_up_std` | 0.04711 |
| `burst_len_mean` | 0.02919 |
| `ack_down_count` | 0.02626 |
| `syn_count` | 0.02566 |

Total pooled mean |SHAP| across all features: **0.50547, spread over 49
of 53 features**. The single largest is roughly a tenth of the total
displacement — **no feature dominates**. The ablation reached the same
conclusion by a completely different method, and **the two agreeing is
the point**.

**`syn_count` ranks ninth by SHAP, yet reaches 0.9900 alone in the
ablation.** That is the cleanest available illustration that the two
methods answer different questions: SHAP says *what the model used*,
ablation says *what was necessary*. When alternatives exist the trees
split on `iat_*` and `size_up_*` instead, so `syn_count` earns little
credit; remove the alternatives and it does almost the whole job by
itself. **These results are not in conflict, and the case study must
present them side by side rather than picking one.**

**The top two features are `iat_*`** — and loopback RTT is ~0.03 ms, so
inter-arrival measures client-side processing rather than network
latency. The model therefore leans hardest on the family that is most
environment-dependent. This makes the existing README limitation
**specific rather than generic**: "these results do not transfer to WAN
conditions" is now a SHAP-supported statement, not a disclaimer.

**`size_up_*` is strong while `size_down_*` is nearly absent** from the
top ranks. Upstream is client-to-server, i.e. requests: Firefox's
headers are larger and more varied than wget's. Genuine client
behaviour, not harness.

**Four features have zero pooled importance, and they are two different
findings.** Three are the constants already predicted under
`extract_features.py` — `size_up_min`, `size_up_p25`, `size_down_min`,
all identically 0. The fourth, **`size_down_median`, is not constant**:
it takes 255, 839, 839.5, 840, 841.5 and still no tree ever split on it,
because correlated neighbours (`size_down_p90`, `size_down_mean`)
already did the work. That is direct single-feature evidence of
redundancy, and it **must not be listed alongside the constants** —
*uninformative* and *unnecessary* are different findings.

#### STABILITY — the important negative result

Per-fold spread of mean |SHAP| is tiny — on the original three-round
run, max **0.00327** and **0.00042** on the top feature. **That does not
mean the attributions are robust.** `random_state` is fixed at 42 in
every fold, so the spread measures data variation only, and the rounds
are near replicates of each other: same 100 pages, same server, same
machine.

**"Near replicates" was measured rather than inferred.** Rounds 1 and 2
fall on the same local day, about 10 hours apart — see "MEASURED: four
rounds across three distinct days" under `capture.py`. The per-fold
spread was small partly *because* two folds hold out nearly the same
conditions, which is exactly why round 4 was taken.

Re-running the whole explanation with `random_state=7` moves the ranking
substantially. **Accuracy stays 1.0000 under both seeds.** The model is
stable; the explanation's ordering is not.

**Round 4 settled what the spread was actually varying over.** The
three-round version of this section could only say the ranking moved.
With a same-day pair and a cross-day pair to compare, the numbers say
where the movement comes from — seed, not capture day — and the top ten
features turn out to be the same set under both seeds, only reordered.
The measured comparison, and the sharper claim it licenses, is under
**"Round 4, and what it answered"** in the `capture.py` section. Read
that before quoting any ranking from here.

**Therefore the README and case study MAY NOT claim that any single
feature is "the strongest discriminator".** What they may claim: the
timing and upstream-size families together carry most of the
attribution, but **which individual feature receives credit within a
family is seed-dependent** — exactly what is expected when the signal is
redundant. State the seed sensitivity as a **measured result**, not as a
caveat in small print.

#### Method note, alongside the `fin_count` story

The near-zero per-fold spread **looked like evidence of stability and
was not**. It was checked, and the check is what found the seed
sensitivity.

The existing rule was *a feature that separates perfectly is a reason to
check the harness*. The companion rule: **a stability number that looks
reassuring is a reason to ask what it actually varies over.** Both come
from the same habit — treating a comfortable number as a hypothesis
rather than a result.

### `scripts/explain_model.py`

- **Three plots, not one, and they are one argument in order.**
  Measured: re-fitting at `random_state=7` moves individual feature
  ranks by up to **24 places** while accuracy stays **1.0000** — but the
  top ten features are the same **set** under both seeds. Attribution is
  **stable at family level and unstable at feature level**, so the
  output is built to say exactly that and nothing stronger:
  1. **family bar chart**, both seeds paired — the claim that survives
  2. **feature beeswarm** — informative, but seed-dependent
  3. **stability chart** — the measurement that bounds plot 2

  A reader who stops after plot 1 has the honest headline. A reader who
  reaches plot 2 sees plot 3 next to it.
- **The family mapping has one owner.** The script uses
  `model.FEATURE_GROUPS` and `model.group_of()` rather than defining
  families again, and a test asserts the family totals sum to the
  per-feature pooled total — nothing dropped, nothing double-counted.

#### Captions are computed, not written — and that was a bug twice

`stability_reading()` and `family_reading()` derive their text from the
measured values. This is not a stylistic preference; hardcoded captions
were wrong **twice**, in both directions:

- **Plot 3's caption was hardcoded to "seed dominates".** On synthetic
  data where `cross_day` (0.02816) exceeded `seed` (0.02448) it printed
  the seed conclusion anyway. The number disagreed with the sentence and
  **nothing failed**.
- **Plot 1's title asserted "Attribution is stable at family level /
  timing and upstream size carry most of it, under both seeds"** while
  the bars underneath showed the two families **swapping order** between
  seeds — seed 42: timing 0.21299 > sizes 0.20230; seed 7: sizes 0.21208
  > timing 0.19919. **The figure contradicted its own headline.** That
  title was written when only three rounds existed and was never
  revisited when the fourth arrived.

**The rule, next to the `fin_count` and stability-number entries: every
other number in this project is measured rather than declared, and a
caption is not exempt.** A stale caption is worse than a stale comment,
because it ships **inside the published artefact** — the comment stays
in the repository, the caption goes into someone's slide deck.

#### What plot 1 may and may not say

**Stable under both seeds:** timing + sizes together carry **81–82%** of
total attribution (0.8246 at seed 42, 0.8126 at seed 7), and the other
three families are far behind.

**Not stable:** which of the two leads. The subtitle **states the
seed-dependence** rather than leaving the reader to notice it from the
bars.

The range is written **81–82%, not ~82%**: the two shares are different
numbers, and rounding them into one would be declaring a figure that was
never measured.

#### Captions are wrapped, and the figure grows to fit

Plot 1's subtitle clipped at the right edge, losing **"dependent"** from
"seed-dependent" — the caveat the title had just been rewritten to
carry.

Fixed by **wrapping before render, not by shortening the wording**:
shorter text fits today and clips silently on the next dataset, and what
gets cut is the **end** of the sentence, which is exactly where the
qualifying clause lives. Figure height grows per wrapped line, so the
fix does not take space from the bars — otherwise the caption would stop
clipping at the data's expense.

All **five** caption branches (two from `family_reading()`, three from
`stability_reading()`) are rendered and measured against the figure
bounds in a parametrized test, **including branches the current data
does not trigger**. The longest plot 3 branch — *"Cross-day spread
exceeds both the same-day and the seed spread — the rig, not just the
method"*, 93 characters — **would have clipped**, and that branch is
precisely the one that fires when there **is** a finding about the
capture rig to report.

#### Deterministic output

shap's beeswarm jitters overlapping points using the **global numpy
RNG**, so the PNG differed byte-for-byte between runs on identical
input. `np.random.seed()` is set before the plot, and a test compares
the sha256 of all three PNGs across two runs.

Without it, "regenerate everything by running the scripts" would have
been false for the figures — quietly, since the plots looked the same.

#### The published artefacts

`results/shap_family_importance.png`, `shap_feature_beeswarm.png`,
`shap_stability.png`, `shap_summary.json`. All publishable: feature
names and numbers only, and no payload was ever captured.

Measured family importance, pooled mean |SHAP| summed per family:

| family | seed 42 | seed 7 |
|---|---|---|
| timing | **0.21299** | 0.19919 |
| sizes | 0.20230 | **0.21208** |
| bursts | 0.04493 | 0.04038 |
| connections | 0.04079 | 0.04904 |
| counts | 0.00458 | 0.00545 |

**Zero-importance features are now six, not three**, and the JSON keeps
the two findings apart:

- **constant** (*uninformative* — never varies): `size_up_min`,
  `size_up_p25`, `size_down_min`
- **varying but unused** (*unnecessary* — a correlated neighbour was
  always split on instead): `size_down_median`, `burst_gap_max`,
  `iat_down_min`

**The second list grew from one to three when the fourth round was
added.** More data made the trees rely on **fewer** features — further
evidence of redundancy, arriving from a direction neither SHAP nor the
ablation supplies. The list is **detected from the dataset, never
hardcoded**, so a future round can change it again.

The summary JSON records the direction explicitly — `positive_class`
`wget`, `classes_` `['firefox', 'wget']`, `base_value` **0.4983** — so a
reader knows which way a positive SHAP value points **without opening an
image**.

### `src/tsd/verdict.py` + `src/tsd/cli.py` + `scripts/classify_pcap.py`

**Three layers, and the seam is deliberate.** `verdict.py` is pure logic
and knows nothing of argv; `cli.py` owns argparse and the exit codes;
`scripts/classify_pcap.py` is a shim that imports `main` and calls it.
The CLI lives in `src/tsd/` rather than in `scripts/` for two reasons:
`scripts/` is run-once operational tooling while **the CLI is the
shipped deliverable**, and `main(argv)` inside the library is
unit-testable without a subprocess — where a failure keeps its
traceback instead of collapsing into an exit code and a silent shell.

**The artefact's feature list is the authority; `feature_names()` is
only cross-checked against it.** The pipeline was fitted with the
columns in the artefact's order, and this is the failure being defended
against: scikit-learn takes a **positional array** and validates the
column *count*, never the names. A permuted or mismatched vector is
still a valid vector, so the model returns a **confident wrong answer
with nothing raised**. `load_artefact()` therefore refuses on any
difference in the feature set and names what moved, and `build_vector()`
iterates the artefact rather than the feature dict.

**stdout carries JSON and nothing else; every diagnostic goes to
stderr.** On any failure stdout is left **completely empty** — verified
against the real binary at **0 bytes**, not only through `capsys`. A
half-written JSON object is worse than none, because a consuming
pipeline would not notice it; an empty stream at least fails honestly.
The tool has to pipe into `jq` without the caller filtering anything
first.

**The exit code never encodes the predicted class.** 0 a verdict was
produced, 2 usage, 3 `VerdictError`. Exit status answers *"did the tool
work"*, not *"what did it find"*. Overloading it would make

    classify_pcap x.pcap || echo failed

report failure whenever the correct answer happened to be `wget` — a
right answer treated as an error. The class is in the JSON, where a
caller has to read it deliberately.

**`MIN_PACKETS = 4`.** The handshake alone is three packets, so anything
shorter is a connection attempt, a stray retransmission, or a capture
that never ran — and the model has never seen such a trace. A prediction
there would be **extrapolation wearing the costume of a verdict**, which
is worse than a refusal because it looks like an answer.

**Rounding is presentation only.** The model receives full precision;
the six-decimal rounding exists so two verdicts on the same input diff
cleanly. Two tests hold this from both sides — one asserts the JSON's
order *after* rounding, the other asserts the model **never** receives
pre-rounded values. A later "consistency" fix that moved the rounding
upstream would change the prediction path for a presentation concern,
and would look like tidying.

**`--json` was deliberately not added.** JSON is the only output format,
so a flag that switches nothing would be a promise of a second one.
`--output` and directory/batch input are out of scope for the same
reason: the shell already has redirection and `find`.

#### Packaging

**`pyproject.toml` declares what the library needs to run;
`requirements.lock.txt` records the environment the published numbers
were measured in.** Lower bounds in one, pins in the other. Pinning in
both would be two places to update and one of them would rot — silently,
because a stale lower bound still installs. The lockfile has to stay
exact, since it is what makes the measured results reproducible; the
dependency list has to stay loose, since it describes what the code
imports rather than what one machine happened to have.

**The version is single-sourced** from `src/tsd/__init__.py` via
`[tool.setuptools.dynamic]`, and `--version` renders `%(prog)s` rather
than a constant. Two hardcoded versions drift, and the one nobody looks
at is the one that ships.

**`prog` is not hardcoded — it was, briefly, and the result was
`tsd-classify --help` printing `usage: classify_pcap`**: a help text
naming a command that does not exist on the reader's system. The
hardcoding existed to stop argparse showing `pytest` when `main()` is
called from the suite. That is **one caller's problem**, so it now lives
in the tests as an explicit `prog=` argument instead of in the shipped
parser. The stderr prefix reads `parser.prog` rather than a literal:
**the prefix on an error and the name in `--help` are the same fact, and
two copies of one fact eventually disagree.**

**`pytest.ini` keeps `pythonpath = src` alongside the install, and the
two are not redundant.** The suite must run on a fresh clone *before*
anything is installed; `pip install -e .` is what makes the console
script exist. They serve the uninstalled and the installed case, and
removing either breaks a case the other never covered.

**A bidirectional test ties the dependency list to the actual imports
under `src/tsd/`.** An added import or a stale dependency fails the suite
here, rather than surfacing as an `ImportError` on a reader's machine
after a `pip install` that looked successful. Neither direction is
visible in this repository, where every package is already in the
virtualenv — which is exactly why it is asserted rather than noticed.

#### Known packaging limitations — for the README

**`dependencies` is one flat set.** Installing the CLI therefore also
pulls `shap`, `requests` and `beautifulsoup4` — and transitively
`numba`/`llvmlite` — because those really are imported under `src/tsd/`.
The correct fix is `[project.optional-dependencies]` splitting `scrape`
and `explain` extras, which means rearranging module boundaries so the
classifier path imports none of them. **Out of scope for step 8, and
stated rather than hidden.**

**The default `--model` path is relative to the working directory**, so
the installed tool finds it only when run from the repository root; from
anywhere else `--model` must be given. Deliberate: `models/` is
gitignored and per-user, so resolving it relative to the *package* would
point confidently at a file that does not exist.

**The console script installs into the venv's `bin/`**, so the README
quickstart must include activating the venv — otherwise a reader
following it lands on `command not found` and blames the install.

#### MEASURED: training path and inference path are identical — 2026-08-09

`scripts/verify_cli_parity.py` compares, for every real capture, the
features the **shipped inference path** produces against the row already
recorded in `data/features/features.csv` by the **training path**.

| | |
|---|---|
| PCAPs found | 800 |
| CSV rows | 800 |
| compared | 800 |
| agreeing on **every** one of 53 features | **800** |
| total comparisons | **42,400** |
| mismatches | **0** |

Exit 0. Record in `results/cli_parity.json`.

**This is a parity check, not an evaluation.** The model in `models/` is
fitted on all four rounds, so every PCAP here is a **training row**, and
any accuracy figure from this script would measure only how well a model
reproduces data it has already seen. The published accuracy remains the
`LeaveOneGroupOut` number from step 6. The script computes and records
**no accuracy, label or agreement field at all** — asserted by a test
over the record's keys.

**Method note: exact equality after applying the same rounding to both
sides, never a tolerance.** Choosing a tolerance decides in advance how
much drift is acceptable, which is precisely the question being asked.
Differences, had there been any, are recorded **with their magnitude
rather than approved by it**.

#### Two self-referential test traps, because both generalise

Writing that script produced two failing assertions that were wrong in
the same way:

- an assertion that the record **contains no accuracy** failed on the
  record's own sentence *declaring that it measures none*
- an assertion that a trace key **held no client name** failed because
  the key is derived from the capture directory — `round_01_20260807/
  wget/index` — where the client name is the path, not an answer

**A naive substring assertion tests the prose, not the structure.** Both
are now checked against parsed keys rather than raw text. This is the
same lesson as the token-based guard test in `features.py` — where
`'rst'` matches `'burst'` — arrived at from the opposite direction:
there the substring was too broad, here it was matching the right word
in the wrong place.

---

## Reusing my earlier scripts

`scraper.py` and `custom_server.py` (in the uploads) are my own earlier work.
They are being rewritten into this structure, not dropped.

### `custom_server.py` → `src/tsd/server.py` + `scripts/serve.py` — DONE

All six changes are implemented and covered by `tests/test_server.py`. The
reasoning is kept because it is why the code looks the way it does; see also
the `src/tsd/server.py` decisions entry above.

1. **It was HTTP; this project needs HTTPS.** Now `ssl.SSLContext` with the
   local CA's leaf certificate. The TLS record layer *is* the observable size
   structure — without it the premise collapses.
2. **The worst problem was the single-threaded sequential accept loop.**
   Firefox opens up to ~6 parallel connections per page load; wget goes one at
   a time. A server handling one socket at a time would have artificially
   serialised Firefox's parallelism, so the model would have learned *the
   server's queuing behaviour* rather than real client concurrency. Now one
   daemon thread per connection — and, the part that is easy to get wrong,
   **the listening socket is not wrapped in TLS**. Wrapping it would move
   every handshake inside `accept()` on one thread, which serialises Firefox
   exactly as before while the code still looks concurrent.
   `test_connections_are_served_in_parallel` fails if that regression is
   reintroduced.
3. **`recv(4096)` once was not enough.** Firefox's headers are large and TCP
   gives no message boundaries. Now loops until `\r\n\r\n`, with a 16 KB /
   100-line ceiling → 431, because an unbounded read is a one-line denial of
   service.
4. **Keep-alive was ambiguous.** It advertised HTTP/1.1 (persistent implied)
   but closed after each response with no `Connection: close`, so Firefox
   would try to reuse, fail and reconnect while wget barely noticed. The
   policy is now explicit in every response and **identical for both
   clients**.
5. **The path traversal check is hardened.** Percent-decoding happens *before*
   the check (otherwise `%2e%2e%2f` walks straight past it), then
   `realpath()` so a symlink inside the mirror cannot point out, then
   `commonpath()` rather than `startswith()` — a prefix test accepts
   `/data/mirror-evil` for a root of `/data/mirror`. All three cases are
   tested.
6. **Response headers are constant-length.** No `Date` (its digits change
   length), and no `Server`, `ETag` or `Last-Modified` either: without the
   validators there is no conditional-request path for the two clients to
   take differently. Headers and body go out in **one** `sendall()`, so Nagle
   cannot make identical responses land on different packet boundaries.

### `scraper.py` → rewritten across `src/tsd/` + `scripts/` — DONE

All five changes are in place, and the corpus was scraped with them
(2026-08-06). Each line says what was wrong and which module owns that
concern now, so the flaw and its fix can be read together.

1. **The UA was a fake Chrome string.** Now the honest, contactable UA
   above — it lives in `robots.py` next to the robots parsing it is
   matched against, and `fetcher.py` is what actually sets the header, so
   there is one place to change it and no call site that can forget.
   Never a browser string, never a library default: b-tu.de blocks
   generic library agents, and if a WAF ever 403s us the answer is not
   impersonation.
2. **robots.txt was never checked.** Now `robots.py` (prefix layer via
   `RobotFileParser`, plus a compiled-regex wildcard layer, because
   `Disallow: /*/wiki/` can never fire under prefix matching) enforced by
   `fetcher.py`. `PoliteFetcher.get()` is the single chokepoint for every
   outbound request — politeness spread across call sites breaks
   silently. It fails closed: unreadable robots.txt means nothing is
   crawled.
3. **Output path `btu_mirror` → `data/mirror`.** `urls.py` decides the
   local filenames (readable prefix plus a sha256 digest, so two long
   URLs sharing a prefix cannot collide), `mirror.py` writes them, and
   the directory is a CLI argument of `scrape_corpus.py`. That last part
   is why `.gitignore` denies all of `data/` by default rather than
   listing paths: a run into `data/mirror_test/` once left 289 files of
   BTU content that git was ready to commit.
4. **`save_asset()` swallowed failures** (`except Exception: return
   None`), silently leaving the original absolute URL in the HTML. During
   capture the browser would then fetch that asset **from the real
   b-tu.de** — outside network traffic contaminating a supposedly local,
   isolated capture, and invisible to the loopback filter. Now
   `mirror.py` **records the failure and neutralises the reference**:
   recording alone would leave the live request in place, so the
   attribute is deleted or the CSS reference becomes
   `url("about:blank")`. Each failure carries an `outcome`, and
   `scrape_corpus.py` classifies it upstream / excluded / local so the
   alarm only sounds for the class that matters. Rewriting a page's links
   correctly also required knowing the final page set first, which is why
   `discover.py` runs as a separate pass before `mirror.py`.
5. **It needed `results/corpus_manifest.json`.** Built by
   `scrape_corpus.py` and committed: URL, local filename, byte size,
   sha256, HTTP status and fetch timestamp per page and per asset, over
   **100 pages and 1701 assets**. Publishable, leaks no BTU content, and
   makes the corpus verifiable — the hashes are what turn "regenerate it
   by running the scripts" from a claim into something a reader can
   check.

---

## Progress

- [x] **Step 1** — environment, toolchain, repo skeleton, `.gitignore`, venv,
      pinned dependencies (`requirements.lock.txt`)
- [x] **Step 2a–2i** — `robots.py`, `fetcher.py`, `urls.py`, `discover.py`,
      all with tests (79 passing, no network required)
- [x] **Step 2j** — `src/tsd/mirror.py`: write pages + assets to
      `data/mirror/`, rewrite links against the frozen page set, deterministic
      output, loud failures (26 tests, no network required)
- [x] **Step 2k** — `scripts/scrape_corpus.py` entry point + manifest
- [x] **Scope 1 — corpus scraped**, 2026-08-06, seed 42, 19 of 20 walks:
      **100 pages, 1701 assets, 220,636,729 bytes (210 MiB)**.
      37 mirror failures: **29 upstream** (18 robots.txt refusals, 11 site
      404s), **7 excluded by policy** (5 off-host `www-docs.b-tu.de`,
      2 over the 8 MB ceiling), **1 local** (one `ConnectionError` on a
      TYPO3 thumbnail — inspected and accepted, see
      `results/provenance/scrape_notes.md`). Plus 7 pages refused during
      discovery. Manifest: `results/corpus_manifest.json`.
- [x] **Step 3 — HTTPS server, self-signed cert** (Scope 2), 2026-08-06.
      `scripts/make_cert.sh` (local root CA + IP-only 825-day leaf,
      fingerprints in `results/provenance/tls_cert.txt`),
      `src/tsd/server.py` + `scripts/serve.py` (49 tests). All six
      `custom_server.py` problems fixed. Measured serving `data/mirror` on
      `127.0.0.1:8443`: both clients negotiate **TLS 1.3** unprompted
      (Firefox 153.0.3 `TLS_AES_128_GCM_SHA256`, Wget 1.21.4
      `TLS_AES_256_GCM_SHA384` — identical AEAD record overhead);
      Firefox 62 requests vs wget 116 for the same page; Firefox resumes
      sessions, wget never does.
- [x] **Step 4 — capture harness**, `src/tsd/capture.py` +
      `scripts/capture_round.py`. **Four rounds captured**, 100 pages ×
      2 clients each = **800 traces total, 0 failures**:

      | round | `date` (local) | `started_at` (UTC) |
      |---|---|---|
      | 1 | 20260807 | 2026-08-06T22:46:56+00:00 |
      | 2 | 20260807 | 2026-08-07T08:51:09+00:00 |
      | 3 | 20260808 | 2026-08-08T03:04:24+00:00 |
      | 4 | 20260809 | 2026-08-08T22:09:39+00:00 |

      Metadata in `results/capture_rounds/`. Round 1 per trace: firefox
      6 SYNs / ~159–170 packets, wget 1 SYN / ~222–228 packets.
      **Four rounds span three distinct days**: rounds 1 and 2 are the
      same local day, and that limitation stands. See "MEASURED: four
      rounds across three distinct days" under the `capture.py`
      decisions.
- [x] **Round 4 captured 2026-08-09** (local; `started_at`
      2026-08-08T22:09:39+00:00): 200 traces, 0 failures, 47,906
      packets, taken ~19 h after round 3 after a machine restart and a
      fresh WSL session. Comparability **verified before use** —
      identical `server_cert_sha256` and identical `versions` dict.
      Taken **not for accuracy** but for the seed question; what it
      answered is under "Round 4, and what it answered".
- [x] **Step 5 — feature extraction** — `src/tsd/features.py` (53
      features, 30 tests) + `scripts/extract_features.py`. All four
      rounds extracted to `data/features/features.csv`: **800 rows, 57
      columns** (4 labels + 53 features), metadata cross-check passed,
      0 parse failures. Round 4 introduced **no new constant features**.
- [x] **Step 6 — classifier with round-based split** —
      `src/tsd/model.py` + `scripts/train_model.py`.
      `LeaveOneGroupOut` on `round`, **4 rounds, 800 traces**.
      **random_forest 1.0000 pooled** (r1/r2/r3/r4 all 1.000, no page
      misclassified), logistic_regression 0.9950 (4 errors, all in the
      round-2 fold). Ablation: removing **any** single feature family
      still gives 1.0000 for the forest; `syn_count` alone gives 0.9900
      in every fold. **New:** for the linear model, `without timing`
      scores 1.0000 — removing a family *improves* it. Full numbers
      under "MEASURED: step 6" below. `results/metrics.json` and
      `models/client_classifier.joblib` regenerated over four rounds.
- [x] **Step 7 — SHAP**, complete: `src/tsd/shap_explain.py` (library,
      computed per fold on held-out rounds only) +
      `scripts/explain_model.py` (three plots + `shap_summary.json`).
      Attribution is **stable at family level, unstable at feature
      level**: the top ten features are the same set under seed 42 and
      seed 7, reordered by up to 24 places, while the capture day has no
      measurable effect. Timing + sizes carry **81–82%** of attribution
      under both seeds, but **which of the two leads is seed-dependent**.
      Every plot caption is **computed from the measured values**, not
      written — hardcoded ones were wrong twice. See "Round 4, and what
      it answered", "MEASURED: what SHAP measured", and the
      `explain_model.py` decisions.
- [x] **Step 8 — CLI**. Three layers: `src/tsd/verdict.py` (pure logic),
      `src/tsd/cli.py` (argparse and exit codes) and
      `scripts/classify_pcap.py` (shim). **Installable as
      `tsd-classify`** via `pyproject.toml`, so `PYTHONPATH=src` is no
      longer required. JSON on stdout and nothing else, diagnostics on
      stderr, exit codes **0 / 2 / 3** — never the predicted class.
      **Parity verified: 800 PCAPs × 53 features = 42,400 comparisons,
      0 mismatches** (`results/cli_parity.json`), confirming the shipped
      inference path and the training path produce identical features.
      Parity was **re-run after the editable install, without
      `PYTHONPATH`, and still gave 800/800** — so the install changed
      how the tool is reached, not what it computes. Known packaging
      limitations are listed under the decisions entry and go in the
      README.
- [ ] **Step 9** — README + case-study page

The corpus **has been scraped** (2026-08-06). `data/mirror/` is gitignored;
what is published in its place is `results/corpus_manifest.json` (per-file
sha256, so a re-run can be checked against it) and
`results/provenance/`.
