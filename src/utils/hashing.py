"""Simple hashing utilities."""

import hashlib


def sha256_hash_file(path: str) -> str:
    """Compute SHA-256 of a file in a streaming way."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
