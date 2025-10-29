from cctv_dissertation.utils.hashing import sha256_hash_file


def test_sha256_hash_file(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"hello world")
    h = sha256_hash_file(str(p))

    # SHA-256 for the bytes string b"hello world"
    expected = (
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )
    assert h == expected
