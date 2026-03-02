import textwrap

from main import load_config


def test_load_config_reads_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        textwrap.dedent(
            """
            training:
              n_splits: 3
            """
        ),
        encoding="utf-8",
    )
    cfg = load_config(str(p))
    assert "training" in cfg
    assert cfg["training"]["n_splits"] == 3
