#!/usr/bin/env bash
#
# make_cert.sh
# ------------
# Generate the local root CA and the server certificate for the local
# HTTPS capture server. Run once; the certificates then stay fixed for
# the whole project.
#
# Why a local CA instead of a bare self-signed leaf certificate
# ------------------------------------------------------------
# A self-signed leaf can only be accepted through Firefox's "add a
# security exception" path. That is not just a nuisance: an exception is
# a per-profile override stored outside normal chain validation, and it
# puts the connection on a different code path from an ordinary trusted
# one. This project measures the traffic shape of a page load down to
# inter-arrival times, so a page load that carries extra validation
# machinery on one client and not the other is a difference introduced
# by the setup, landing entirely on the Firefox class -- exactly the
# kind of confounder this rebuild exists to remove.
#
# With a CA in Firefox's Authorities store, both clients validate the
# chain the ordinary way, and both are pointed at the SAME trust anchor
# by the same kind of interface: an imported CA for Firefox,
# --ca-certificate for wget. Symmetry between the two classes is the
# whole point. It also means the leaf can be re-issued later (expiry, a
# second SAN) without re-trusting anything in either client.
#
# Why the SAN is IP-only
# ----------------------
# The server is reached as https://127.0.0.1:PORT and nothing else. With
# no DNS name in the certificate there is no name to resolve, so no
# resolver step exists in the load path at all. A hostname would add one
# -- and Firefox and wget do not resolve names the same way, so its
# latency would be asymmetric between the classes, on top of being
# unrelated to what is being measured.
#
# It also means this certificate cannot accidentally be valid for any
# real site, and it keeps the tcpdump filter to one host and one port.
#
# EC P-256 rather than RSA: a compact, fixed-size handshake. The same
# certificate is served to both clients, so it cannot differentiate them
# -- but a smaller constant is a smaller constant to explain.
#
# Nothing here touches the network. This script generates certificates
# only; the server itself is step 3.

set -euo pipefail

# Resolve everything from the script's own location, so the script works
# from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CERT_DIR="${REPO_ROOT}/certs"
PROVENANCE_DIR="${REPO_ROOT}/results/provenance"

CA_KEY="${CERT_DIR}/ca.key"
CA_CRT="${CERT_DIR}/ca.crt"
CA_SRL="${CERT_DIR}/ca.srl"
SERVER_KEY="${CERT_DIR}/server.key"
SERVER_CRT="${CERT_DIR}/server.crt"
SERVER_CSR="${CERT_DIR}/server.csr"
PROVENANCE_FILE="${PROVENANCE_DIR}/tls_cert.txt"

CA_DAYS=3650
# 825 days: the longest leaf lifetime browsers still accept. Long enough
# that the certificate never changes mid-project, which matters because
# a new certificate means new handshake bytes, and capture rounds taken
# on different days must stay comparable.
SERVER_DAYS=825

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

# ------------------------------------------------------------------
# Overwrite protection
# ------------------------------------------------------------------
# The certificate is part of the measurement apparatus. Its size and
# contents are transmitted in the TLS handshake of every single page
# load, so regenerating it between capture rounds would change the byte
# pattern at the start of every trace -- and the split is BY ROUND, so
# that change would line up exactly with the train/test boundary. The
# model could learn the certificate instead of the client.
if [[ -e "${SERVER_CRT}" && "${FORCE}" -eq 0 ]]; then
    cat >&2 <<'REFUSED'
ERROR: certs/server.crt already exists. Refusing to overwrite it.

The certificate is transmitted in the handshake of every page load, so
regenerating it changes the first bytes of every trace. Capture rounds
recorded before and after would differ for a reason that has nothing to
do with the client -- and since the train/test split is BY ROUND, that
difference would line up with the split itself.

If you genuinely need new certificates, every capture round taken with
the old ones must be discarded and recaptured:

    scripts/make_cert.sh --force
REFUSED
    exit 1
fi

# ------------------------------------------------------------------
# Directories and cleanup
# ------------------------------------------------------------------
mkdir -p "${CERT_DIR}"
chmod 700 "${CERT_DIR}"
mkdir -p "${PROVENANCE_DIR}"

# The CSR is an intermediate product and the extension file is written
# to a temporary path; neither should survive the run.
EXT_FILE="$(mktemp)"
cleanup() {
    rm -f "${SERVER_CSR}" "${EXT_FILE}"
}
trap cleanup EXIT

echo "openssl: $(openssl version)"
echo "output:  ${CERT_DIR}"
echo

# ------------------------------------------------------------------
# Root CA
# ------------------------------------------------------------------
echo "==> CA key (EC P-256)"
openssl genpkey \
    -algorithm EC \
    -pkeyopt ec_paramgen_curve:P-256 \
    -out "${CA_KEY}" 2>/dev/null

echo "==> CA certificate (${CA_DAYS} days)"
openssl req -x509 -new \
    -key "${CA_KEY}" \
    -sha256 \
    -days "${CA_DAYS}" \
    -subj "/CN=traffic-shape-client-detection local CA" \
    -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
    -addext "keyUsage=critical,keyCertSign,cRLSign" \
    -out "${CA_CRT}"

# ------------------------------------------------------------------
# Server certificate
# ------------------------------------------------------------------
echo "==> server key (EC P-256)"
openssl genpkey \
    -algorithm EC \
    -pkeyopt ec_paramgen_curve:P-256 \
    -out "${SERVER_KEY}" 2>/dev/null

echo "==> server CSR"
openssl req -new \
    -key "${SERVER_KEY}" \
    -subj "/CN=127.0.0.1" \
    -out "${SERVER_CSR}"

# Deliberately absent from these extensions: authorityInfoAccess (OCSP)
# and crlDistributionPoints. Either one would name a URL, and Firefox
# would try to fetch it during the handshake. That request leaves the
# machine, so it is invisible to a tcpdump filtered to the local server
# -- but the DNS lookup and connect time it costs land inside Firefox's
# inter-arrival times. wget does no revocation checking, so the cost
# would fall on one class only. A revocation endpoint for a certificate
# that never leaves this machine buys nothing and risks exactly the
# artefact this project is built to avoid.
cat > "${EXT_FILE}" <<'EXTENSIONS'
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = IP:127.0.0.1
EXTENSIONS

echo "==> server certificate, signed by the CA (${SERVER_DAYS} days)"
openssl x509 -req \
    -in "${SERVER_CSR}" \
    -CA "${CA_CRT}" \
    -CAkey "${CA_KEY}" \
    -CAserial "${CA_SRL}" \
    -CAcreateserial \
    -days "${SERVER_DAYS}" \
    -sha256 \
    -extfile "${EXT_FILE}" \
    -out "${SERVER_CRT}" 2>/dev/null

# ------------------------------------------------------------------
# Permissions
# ------------------------------------------------------------------
chmod 600 "${CA_KEY}" "${SERVER_KEY}"
chmod 644 "${CA_CRT}" "${SERVER_CRT}"

# ------------------------------------------------------------------
# Verification -- a certificate that does not validate is worse than no
# certificate, because the failure would surface as a browser warning
# in the middle of a capture round.
# ------------------------------------------------------------------
echo
echo "==> verifying"

openssl verify -CAfile "${CA_CRT}" "${SERVER_CRT}"

SERVER_TEXT="$(openssl x509 -in "${SERVER_CRT}" -noout -text)"

if ! grep -q "IP Address:127.0.0.1" <<<"${SERVER_TEXT}"; then
    echo "ERROR: subjectAltName IP:127.0.0.1 missing from server.crt" >&2
    exit 1
fi
echo "subjectAltName: IP Address:127.0.0.1 OK"

# The two extensions that must NOT be there. Asserted rather than
# assumed: they are absent because nothing added them, and an assertion
# is what catches the day something does.
for forbidden in "Authority Information Access" "CRL Distribution Points"; do
    if grep -q "${forbidden}" <<<"${SERVER_TEXT}"; then
        echo "ERROR: ${forbidden} present in server.crt" >&2
        echo "       Firefox would fetch that URL during the handshake." >&2
        exit 1
    fi
done
echo "no OCSP / CRL endpoints OK"

# ------------------------------------------------------------------
# Provenance
# ------------------------------------------------------------------
# certs/ is gitignored, so this file is what gets published about the
# certificates. It contains public certificate metadata only -- NEVER
# any private key material, and never the certificates themselves.
describe_cert() {
    local label="$1" path="$2" text
    text="$(openssl x509 -in "${path}" -noout -text)"

    echo "## ${label}"
    echo
    openssl x509 -in "${path}" -noout \
        -subject -issuer -startdate -enddate -fingerprint -sha256
    echo "signature_algorithm= $(grep -m1 'Signature Algorithm' <<<"${text}" \
        | sed 's/.*Signature Algorithm: //')"
    echo "public_key_algorithm= $(grep -m1 'Public Key Algorithm' <<<"${text}" \
        | sed 's/.*Public Key Algorithm: //')"
    echo "public_key_size= $(grep -m1 'Public-Key:' <<<"${text}" \
        | sed 's/.*Public-Key: //')"
    echo "curve= $(grep -m1 'NIST CURVE' <<<"${text}" \
        | sed 's/.*NIST CURVE: //')"
    echo
}

{
    echo "# TLS certificates for the local capture server"
    echo
    echo "Generated by scripts/make_cert.sh. certs/ is gitignored, so this"
    echo "file is the published record of what the capture server presented."
    echo "Public certificate metadata only -- no private key material."
    echo
    echo "generated_at= $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "openssl_version= $(openssl version)"
    echo
    describe_cert "Root CA (certs/ca.crt)" "${CA_CRT}"
    describe_cert "Server certificate (certs/server.crt)" "${SERVER_CRT}"
} > "${PROVENANCE_FILE}"

echo "provenance: ${PROVENANCE_FILE#"${REPO_ROOT}/"}"

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
CA_FINGERPRINT="$(openssl x509 -in "${CA_CRT}" -noout -fingerprint -sha256 \
    | sed 's/.*=//')"
SERVER_FINGERPRINT="$(openssl x509 -in "${SERVER_CRT}" -noout -fingerprint -sha256 \
    | sed 's/.*=//')"

cat <<SUMMARY

done

  CA      SHA-256  ${CA_FINGERPRINT}
  server  SHA-256  ${SERVER_FINGERPRINT}

Trust the CA in each client -- both point at the same anchor, which is
what keeps the two classes comparable:

  Firefox   Settings -> Privacy & Security -> Certificates ->
            View Certificates -> Authorities -> Import -> certs/ca.crt ->
            tick "Trust this CA to identify websites"

  wget      --ca-certificate=certs/ca.crt

WARNING: do NOT install this CA system-wide (update-ca-certificates).
A research CA in the system trust store can vouch for any host to every
program on the machine -- unnecessary blast radius -- and per-client
explicit trust is also what makes this setup reproducible for someone
else, since it is visible in the commands rather than in system state.
SUMMARY
