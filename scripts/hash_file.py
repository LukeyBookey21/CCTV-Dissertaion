"""Small helper script to compute SHA-256 from the command line."""

import sys
from pathlib import Path

try:
    from cctv_dissertation.utils.hashing import sha256_hash_file
except ImportError:
    # If the package isn't installed, add the project root to sys.path so the
    # local package can be imported for quick local runs.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from cctv_dissertation.utils.hashing import sha256_hash_file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/hash_file.py <file>")
        sys.exit(2)
    print(sha256_hash_file(sys.argv[1]))
