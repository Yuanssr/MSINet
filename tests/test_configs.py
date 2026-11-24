from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_public_configs_are_portable_and_complete():
    for name, variant in (("msinet_small.yaml", "small"), ("msinet_base.yaml", "base")):
        text = (ROOT / "config" / name).read_text(encoding="utf-8")
        config = yaml.safe_load(text)
        assert "/opt/" not in text
        assert "D:/" not in text
        assert config["MaskRefine"]["mae_vit"]["type"] == variant
        assert config["TRAIN"]["optimizer"] == "AdamW"
        assert config["TRAIN"]["epochs"] == 500
        expected_batch = 16 if variant == "small" else 8
        assert config["TRAIN"]["batch_size"] == expected_batch
        assert config["TRAIN"]["lr"] == 1e-3
        assert len(config["TRAIN"]["VAL"]["val_data_dir"]) == 3

    base = yaml.safe_load((ROOT / "config" / "msinet_base.yaml").read_text(encoding="utf-8"))
    assert base["MaskRefine"]["mae_vit"]["checkpoint"].endswith("vit_base_checkpoint-49.pth")
    assert base["Pretrain_SR_Model"]["name"] == "TATT"
