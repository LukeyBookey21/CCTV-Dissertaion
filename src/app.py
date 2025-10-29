"""Minimal CLI entrypoint for the project."""

import argparse
from cctv_dissertation.utils.hashing import sha256_hash_file


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Forensics video analysis toolbox "
            "(skeleton)"
        ),
    )
    parser.add_argument(
        "--file",
        "-f",
        help="Path to a file to hash",
        required=False,
    )
    args = parser.parse_args()

    if args.file:
        h = sha256_hash_file(args.file)
        print(f"SHA-256: {h}")
    else:
        print("No file provided. Use --file <path> to compute SHA-256.")


if __name__ == "__main__":
    main()
