from nexis_ml import load_config


def test_load_plain_toml(tmp_path):
    p = tmp_path / "train.toml"
    p.write_bytes(b"[train]\nepochs = 5\n")
    assert load_config(str(p)) == {"train": {"epochs": 5}}


def test_load_toml_with_utf8_bom(tmp_path):
    # Notepad / PowerShell Set-Content write a BOM; tomllib alone chokes on it
    p = tmp_path / "train.toml"
    p.write_bytes(b"\xef\xbb\xbf[train]\nepochs = 5\n")
    assert load_config(str(p)) == {"train": {"epochs": 5}}
