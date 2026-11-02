#!/usr/bin/env bash
# Generate a self-signed TLS certificate so browsers allow getUserMedia
# (the attendance camera page) when the service is opened from another
# machine over the LAN. Browsers show a one-time warning for self-signed
# certificates; the user proceeds past it once per browser.
#
# Usage:
#   ./make_dev_cert.sh [LAN-IP]      # default 127.0.0.1
#
# Then set in .env:
#   SSL_CERTFILE=face_compare_api/certs/cert.pem
#   SSL_KEYFILE=face_compare_api/certs/key.pem
# and restart the service.

set -euo pipefail

CERT_DIR="$(cd "$(dirname "$0")" && pwd)/certs"
DAYS="${CERT_DAYS:-825}"
ADDRESS="${1:-127.0.0.1}"

mkdir -p "$CERT_DIR"
openssl req -x509 -newkey rsa:2048 -sha256 -days "$DAYS" -nodes \
  -keyout "$CERT_DIR/key.pem" \
  -out "$CERT_DIR/cert.pem" \
  -subj "/CN=face-compare" \
  -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:${ADDRESS}"

chmod 600 "$CERT_DIR/key.pem"
echo "Wrote ${CERT_DIR}/cert.pem and ${CERT_DIR}/key.pem (valid ${DAYS} days)."
