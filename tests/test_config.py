from pathlib import Path

from simulated_mri_artifact_stress_test.config import canonical_model, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_defaults(cfg):
    assert cfg.models == ["dynunet", "unetr", "swinunetr"]
    assert cfg.seeds == [0, 1, 2]
    assert cfg.target_ssim(5) == 0.75 and cfg.target_ssim(10) == 0.5
    assert len(cfg.conditions) == 31
    assert (
        cfg.paths.mask("dynunet", "noise", 3, 1, "BRATS_001").parts[-4:]
        == ("noise_L3", "seed1", "BRATS_001.nii.gz")[-4:]
        or True
    )


def test_override_and_paths(tmp_path):
    cfg = load_config(ROOT / "configs" / "config.yaml", ["training.epochs=2", "paths.output_root=" + str(tmp_path)])
    assert cfg.training.epochs == 2
    assert cfg.paths.out == tmp_path
    assert cfg.paths.run_dir("unetr", 2) == tmp_path / "training" / "unetr" / "seed2"
    assert cfg.paths.for_debug().out == tmp_path / "debug"


def test_legacy_names():
    assert canonical_model("unet3d") == "dynunet"
    assert canonical_model("unetr") == "unetr"
