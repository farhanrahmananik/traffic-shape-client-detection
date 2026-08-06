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
- [ ] **Step 4** — capture harness
- [ ] **Step 5** — feature extraction
- [ ] **Step 6** — classifier with round-based split
- [ ] **Step 7** — SHAP
- [ ] **Step 8** — CLI
- [ ] **Step 9** — README + case-study page

The corpus **has been scraped** (2026-08-06). `data/mirror/` is gitignored;
what is published in its place is `results/corpus_manifest.json` (per-file
sha256, so a re-run can be checked against it) and
`results/provenance/`.
