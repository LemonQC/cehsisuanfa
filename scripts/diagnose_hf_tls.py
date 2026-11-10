#!/usr/bin/env python3
"""Credential-safe diagnosis of Hugging Face TLS and proxy routing.

Never downloads model weights, sets verify=False, or prints proxy passwords,
tokens or Authorization headers. Run in the conda env that failed.
"""
from __future__ import annotations
import os
import socket
import ssl
from urllib.parse import urlsplit


def proxy_origin(value: str) -> str:
    parsed = urlsplit(value if "://" in value else "http://" + value)
    return f"{parsed.scheme}://{parsed.hostname or '?'}:{parsed.port or '(default)'}"


def main():
    print("HF_TLS_DIAGNOSTIC_ONLY_NO_DOWNLOAD")
    print("python_ssl", ssl.OPENSSL_VERSION)
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy",
                "ALL_PROXY", "all_proxy"):
        if os.environ.get(key):
            print(key, "configured:", proxy_origin(os.environ[key]))
    for key in ("NO_PROXY", "no_proxy", "REQUESTS_CA_BUNDLE",
                "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        print(key, "SET" if os.environ.get(key) else "unset")
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    p = urlsplit(endpoint)
    print("HF_ENDPOINT host", p.hostname or "invalid")
    host = "huggingface.co"
    try:
        addresses = sorted({entry[4][0] for entry in socket.getaddrinfo(
            host, 443, type=socket.SOCK_STREAM)})
        print("DNS huggingface.co", ", ".join(addresses[:8]))
    except Exception as err:
        print("DNS ERROR", type(err).__name__, str(err)[:220])
    try:
        with socket.create_connection((host, 443), timeout=7) as sock:
            with ssl.create_default_context().wrap_socket(
                    sock, server_hostname=host) as tls:
                cert = tls.getpeercert()
                print("DIRECT TLS OK", "subject", cert.get("subject"),
                      "issuer", cert.get("issuer"))
    except Exception as err:
        print("DIRECT TLS ERROR", type(err).__name__, str(err)[:300])
    try:
        import requests
        resp = requests.head(
            "https://huggingface.co/api/models/LogosRoboticsLab/roboreel_models",
            timeout=10, allow_redirects=False)
        print("REQUESTS (env-aware proxy route)", resp.status_code)
    except Exception as err:
        print("REQUESTS ERROR", type(err).__name__, str(err)[:300])
    print("No model artifacts downloaded; SSL verification always enabled.")


if __name__ == "__main__":
    main()
