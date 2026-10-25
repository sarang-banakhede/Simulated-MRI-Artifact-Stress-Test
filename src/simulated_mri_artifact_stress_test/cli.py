from __future__ import annotations

import argparse
import sys
from pathlib import Path

from simulated_mri_artifact_stress_test import __version__
from simulated_mri_artifact_stress_test.config import canonical_model, load_config
from simulated_mri_artifact_stress_test.utils import atomic_json, environment_info, log, setup_logging

STAGES = ["prepare", "train", "calibrate", "infer", "analyze"]
DEFAULT_CONFIG = Path("configs/config.yaml")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG, help="path to config.yaml")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set training.epochs=2 (repeatable)",
    )


def _selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--models", nargs="+", help="subset of models (default: all configured)")
    p.add_argument("--seeds", nargs="+", type=int, help="subset of seeds (default: all configured)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="simulated-mri-artifact-stress-test", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("prepare", help="build the patient-level split and the preprocessed training cache")
    _common(s)
    s = sub.add_parser("train", help="train models (resumable; one run per model and seed)")
    _common(s)
    _selection(s)
    s = sub.add_parser("calibrate", help="SSIM-match the artifact strengths on the validation cases")
    _common(s)
    s = sub.add_parser("infer", help="run every condition (clean + artifacts x levels) on the test set")
    _common(s)
    _selection(s)
    s.add_argument("--debug", action="store_true", help="smoke test on inference.debug_max_cases cases")
    s = sub.add_parser("analyze", help="metrics, statistics and mask-based post-inference analyses")
    _common(s)
    s.add_argument("--force", action="store_true", help="recompute the cached mask tables")
    s.add_argument("--limit-cases", type=int, default=None)
    s = sub.add_parser("reader-study", help="clean and degraded volumes (optionally blinded) for a radiologist")
    _common(s)
    s.add_argument("--blind-only", action="store_true")
    s = sub.add_parser("run-all", help="run several stages in order")
    _common(s)
    _selection(s)
    s.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    s = sub.add_parser("status", help="show which training runs and inference results exist")
    _common(s)
    return p


def _status(cfg) -> None:
    ps = cfg.paths
    print(f"output root : {ps.out}")
    print(f"splits      : {'yes' if ps.splits.exists() else 'no'}")
    print(f"calibration : {'yes' if ps.degradation_config.exists() else 'no'}")
    for m in cfg.models:
        for s in cfg.seeds:
            d = ps.run_dir(m, s)
            tr = "trained" if (d / "DONE").exists() else ("partial" if (d / "resume.pt").exists() else "-")
            inf = "inferred" if ps.result_csv(m, s).exists() else "-"
            print(f"{m:10s} seed{s}: {tr:8s} {inf}")


def _stage_prepare(cfg) -> None:
    from simulated_mri_artifact_stress_test.data.splits import get_splits
    from simulated_mri_artifact_stress_test.data.training_cache import build_training_cache

    cfg.paths.ensure()
    splits = get_splits(cfg)
    log.info("split: train %d, val %d, test %d", splits["num_train"], splits["num_val"], splits["num_test"])
    build_training_cache(cfg)


def _stage_train(cfg, models, seeds) -> bool:
    from simulated_mri_artifact_stress_test.training.trainer import train_models

    return train_models(cfg, models, seeds)


def _stage_calibrate(cfg) -> None:
    from simulated_mri_artifact_stress_test.artifacts.calibration import calibrate, frequency_footprint

    cfg.paths.ensure()
    deg = calibrate(cfg)
    frequency_footprint(cfg, deg)


def _stage_infer(cfg, models, seeds, debug=False) -> bool:
    from simulated_mri_artifact_stress_test.evaluation.inference import run_inference

    return run_inference(cfg, models, seeds, debug)


def _stage_analyze(cfg, force=False, limit_cases=None) -> None:
    from simulated_mri_artifact_stress_test.analysis import operating, post_masks, results, sensitivity, statistics

    cfg.paths.ensure()
    df = results.summarize(cfg)
    statistics.run(cfg, df)
    operating.run(cfg, df)
    post_masks.run(cfg, df, force=force, limit_cases=limit_cases)
    sensitivity.run(cfg, df)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    cfg.paths.logs.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.paths.logs / f"{args.command}.log")
    if args.command != "status":
        atomic_json(environment_info(), cfg.paths.logs / "environment.json")
    models, seeds = (
        cfg.select(getattr(args, "models", None), getattr(args, "seeds", None))
        if hasattr(args, "models")
        else (cfg.models, cfg.seeds)
    )
    models = [canonical_model(m) for m in models]
    cmd = args.command
    ok = True
    if cmd == "status":
        _status(cfg)
    elif cmd == "prepare":
        _stage_prepare(cfg)
    elif cmd == "train":
        ok = _stage_train(cfg, models, seeds)
    elif cmd == "calibrate":
        _stage_calibrate(cfg)
    elif cmd == "infer":
        ok = _stage_infer(cfg, models, seeds, args.debug)
    elif cmd == "analyze":
        _stage_analyze(cfg, args.force, args.limit_cases)
    elif cmd == "reader-study":
        from simulated_mri_artifact_stress_test import reader_study

        reader_study.run(cfg, args.blind_only)
    elif cmd == "run-all":
        for stage in args.stages:
            log.info("=== stage: %s ===", stage)
            if stage == "prepare":
                _stage_prepare(cfg)
            elif stage == "train":
                ok = _stage_train(cfg, models, seeds)
            elif stage == "calibrate":
                _stage_calibrate(cfg)
            elif stage == "infer":
                ok = _stage_infer(cfg, models, seeds)
            elif stage == "analyze":
                _stage_analyze(cfg)
            if not ok:
                log.info("stage %s stopped early (time limit); rerun the same command to resume", stage)
                break
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
