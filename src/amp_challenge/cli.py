from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

from .constants import (
    DEFAULT_CONFIG,
    DEFAULT_LIBRARY_SIZE,
    DEFAULT_REFERENCE,
    DEFAULT_SEED,
    DEFAULT_TOP_SIZE,
)
from .data_registry import fetch_source, source_status
from .freeze import freeze_run
from .kaggle import run_read_command
from .linear_oracle import train_linear_oracle
from .pipeline import run_pipeline
from .submission import submit_kaggle
from .training_data import prepare_training_data
from .training_preflight import training_preflight
from .utils import find_project_root, sha256_file
from .validation import validate_submission


def _project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-sequences", type=int, default=DEFAULT_LIBRARY_SIZE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--candidate-fasta",
        type=Path,
        help="Use candidates produced by AMP-Diffusion, HydrAMP, or another generator",
    )
    parser.add_argument(
        "--external-scores",
        action="append",
        default=[],
        type=Path,
        help="Normalized oracle CSV; repeat to form an ensemble",
    )
    parser.add_argument(
        "--external-activity-scores",
        action="append",
        default=[],
        type=Path,
        help="Normalized CSV used for activity only; repeat to form an ensemble",
    )


def _run_generate(args: argparse.Namespace, *, default_output: str) -> int:
    root = find_project_root()
    output_dir = args.output_dir or Path(default_output)
    result = run_pipeline(
        project_root=root,
        config_path=_project_path(root, args.config),
        output_dir=output_dir,
        reference_path=_project_path(root, args.reference),
        n_sequences=args.n_sequences,
        top_k=args.top_k,
        seed=args.seed,
        candidate_fasta=args.candidate_fasta,
        external_score_paths=args.external_scores,
        external_activity_score_paths=args.external_activity_scores,
    )
    print(
        json.dumps(
            {
                "status": "generated",
                "output_dir": str(result.output_dir),
                "library": str(result.library_path),
                "top": str(result.top_path),
                "selection_scanned": result.selection.scanned,
                "rejections": result.selection.rejection_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def generate_entrypoint() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an AMP Challenge 2027 library and ranked Top-100"
    )
    _add_generation_arguments(parser)
    args = parser.parse_args()
    command_name = Path(sys.argv[0]).stem
    raise SystemExit(_run_generate(args, default_output=command_name))


def _add_validation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--expected-library-size", type=int, default=DEFAULT_LIBRARY_SIZE)
    parser.add_argument("--expected-top-size", type=int, default=DEFAULT_TOP_SIZE)
    parser.add_argument("--report", type=Path)


def _run_validate(args: argparse.Namespace) -> int:
    root = find_project_root()
    report = validate_submission(
        library_path=args.run_dir / "library.fasta",
        top_path=args.run_dir / "top.fasta",
        reference_path=_project_path(root, args.reference),
        expected_library_size=args.expected_library_size,
        expected_top_size=args.expected_top_size,
    )
    rendered = json.dumps(report.as_dict(), indent=2, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report.valid else 1


def _run_doctor() -> int:
    root = find_project_root()
    reference = root / DEFAULT_REFERENCE
    checks = {
        "project_root": str(root),
        "python": sys.version.split()[0],
        "levenshtein": importlib.metadata.version("levenshtein"),
        "uv": shutil.which("uv"),
        "git": shutil.which("git"),
        "git_lfs": shutil.which("git-lfs"),
        "kaggle_optional": shutil.which("kaggle"),
        "reference_exists": reference.is_file(),
        "reference_sha256": sha256_file(reference) if reference.is_file() else None,
    }
    print(json.dumps(checks, indent=2, sort_keys=True))
    required_ok = bool(checks["uv"] and checks["git"] and checks["reference_exists"])
    return 0 if required_ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amp", description="AMP Challenge pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate, score, and select")
    _add_generation_arguments(generate_parser)

    validate_parser = subparsers.add_parser("validate", help="Validate a generated run")
    _add_validation_arguments(validate_parser)

    freeze_parser = subparsers.add_parser("freeze", help="Validate and freeze a run")
    _add_validation_arguments(freeze_parser)
    freeze_parser.add_argument("--submission-dir", type=Path, default=Path("submission"))

    submit_parser = subparsers.add_parser("submit", help="Guarded Kaggle CLI transport")
    submit_parser.add_argument("--run-dir", type=Path, required=True)
    submit_parser.add_argument("--artifact", type=Path, required=True)
    submit_parser.add_argument("--run-id", required=True)
    submit_parser.add_argument("--message", required=True)
    submit_parser.add_argument("--competition", default="amp-challenge")
    submit_parser.add_argument("--execute", action="store_true")

    data_parser = subparsers.add_parser("data", help="Inspect or fetch registered data")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    data_subparsers.add_parser("status", help="Verify registered source checksums")
    fetch_parser = data_subparsers.add_parser("fetch", help="Fetch one approved source")
    fetch_parser.add_argument("source")
    fetch_parser.add_argument("--force", action="store_true")
    prepare_parser = data_subparsers.add_parser(
        "prepare", help="Build normalized, clustered oracle training tables"
    )
    prepare_parser.add_argument("--config", default="configs/training_data.json")
    prepare_parser.add_argument("--raw", type=Path)
    prepare_parser.add_argument("--output-dir", type=Path)

    kaggle_parser = subparsers.add_parser(
        "kaggle", help="Use the official Kaggle CLI for competition reads/downloads"
    )
    kaggle_subparsers = kaggle_parser.add_subparsers(dest="kaggle_command", required=True)
    for action in ("files", "submissions"):
        action_parser = kaggle_subparsers.add_parser(action)
        action_parser.add_argument("--competition", default="amp-challenge")
    download_parser = kaggle_subparsers.add_parser("download")
    download_parser.add_argument("--competition", default="amp-challenge")
    download_parser.add_argument("--destination", type=Path, default=Path("data/competition"))

    train_parser = subparsers.add_parser("train", help="Validate the handoff to model training")
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)
    preflight_parser = train_subparsers.add_parser(
        "preflight", help="Verify data, splits, hashes, and pinned model config without training"
    )
    preflight_parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/processed/dramp-oracle-v1")
    )
    preflight_parser.add_argument("--config", default="configs/oracle_train.json")
    preflight_parser.add_argument("--report", type=Path)
    linear_parser = train_subparsers.add_parser(
        "linear", help="Train the deterministic physicochemical activity/safety baseline"
    )
    linear_parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/processed/dramp-oracle-v1")
    )
    linear_parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/linear-physchem-v1.json")
    )
    linear_parser.add_argument("--seed", type=int, default=42)
    linear_parser.add_argument("--ensemble-members", type=int, default=5)
    linear_parser.add_argument("--execute", action="store_true")
    esm_parser = train_subparsers.add_parser(
        "esm", help="Extract frozen ESM2 embeddings and benchmark an organism-aware MIC head"
    )
    esm_parser.add_argument(
        "--dataset-dir", type=Path, default=Path("data/processed/dramp-oracle-v1")
    )
    esm_parser.add_argument("--config", default="configs/oracle_train.json")
    esm_parser.add_argument("--model-source", default="facebook/esm2_t12_35M_UR50D")
    esm_parser.add_argument("--embeddings", type=Path, default=Path("runs/esm2-dramp-v1.npz"))
    esm_parser.add_argument("--output", type=Path, default=Path("checkpoints/esm2-mic16-v2.json"))
    esm_parser.add_argument("--report", type=Path, default=Path("runs/esm2-mic16-v2-report.json"))
    esm_parser.add_argument("--batch-size", type=int, default=64)
    esm_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    esm_parser.add_argument("--seed", type=int, default=42)
    esm_parser.add_argument("--execute", action="store_true")
    esm_score_parser = train_subparsers.add_parser(
        "esm-score", help="Coarse-to-fine ESM2 scoring for an existing candidate library"
    )
    esm_score_parser.add_argument("--candidate-fasta", type=Path, required=True)
    esm_score_parser.add_argument("--prefilter-scores", type=Path, required=True)
    esm_score_parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/esm2-mic16-v2.json")
    )
    esm_score_parser.add_argument("--model-source", default="facebook/esm2_t12_35M_UR50D")
    esm_score_parser.add_argument(
        "--panel", type=Path, default=Path("configs/challenge_panel_proxy.json")
    )
    esm_score_parser.add_argument(
        "--embeddings", type=Path, default=Path("runs/esm2-candidate-prefilter.npz")
    )
    esm_score_parser.add_argument(
        "--output", type=Path, default=Path("runs/esm2-candidate-scores.csv")
    )
    esm_score_parser.add_argument("--prefilter-k", type=int, default=5000)
    esm_score_parser.add_argument("--exploration-k", type=int, default=1000)
    esm_score_parser.add_argument("--batch-size", type=int, default=64)
    esm_score_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    esm_score_parser.add_argument("--seed", type=int, default=42)
    esm_score_parser.add_argument("--execute", action="store_true")

    subparsers.add_parser("doctor", help="Check the local runtime")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = find_project_root()
    registry = root / "configs/data_sources.json"

    try:
        if args.command == "generate":
            code = _run_generate(args, default_output="generate_broad_spectrum")
        elif args.command == "validate":
            code = _run_validate(args)
        elif args.command == "freeze":
            record, artifact, _ = freeze_run(
                run_dir=args.run_dir,
                reference_path=_project_path(root, args.reference),
                submission_dir=args.submission_dir,
                expected_library_size=args.expected_library_size,
                expected_top_size=args.expected_top_size,
            )
            print(
                json.dumps(
                    {
                        "status": "frozen",
                        "run_id": record["run_id"],
                        "artifact": str(artifact),
                        "artifact_sha256": record["artifact"]["sha256"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            code = 0
        elif args.command == "submit":
            print(
                submit_kaggle(
                    run_dir=args.run_dir,
                    artifact=args.artifact,
                    run_id=args.run_id,
                    message=args.message,
                    competition=args.competition,
                    execute=args.execute,
                )
            )
            code = 0
        elif args.command == "data" and args.data_command == "status":
            print(json.dumps(source_status(project_root=root, registry_path=registry), indent=2))
            code = 0
        elif args.command == "data" and args.data_command == "fetch":
            path = fetch_source(
                name=args.source,
                project_root=root,
                registry_path=registry,
                force=args.force,
            )
            print(json.dumps({"status": "ready", "path": str(path)}, indent=2))
            code = 0
        elif args.command == "data" and args.data_command == "prepare":
            output_dir = (
                _project_path(root, args.output_dir) if args.output_dir is not None else None
            )
            raw_path = _project_path(root, args.raw) if args.raw is not None else None
            manifest = prepare_training_data(
                project_root=root,
                config_path=_project_path(root, args.config),
                registry_path=registry,
                output_dir=output_dir,
                raw_path=raw_path,
            )
            print(
                json.dumps(
                    {
                        "status": "prepared",
                        "dataset_id": manifest["dataset_id"],
                        "counts": manifest["counts"],
                        "clustering": manifest["clustering"],
                        "fold_counts": manifest["splits"]["fold_counts"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            code = 0
        elif args.command == "kaggle":
            print(
                run_read_command(
                    action=args.kaggle_command,
                    competition=args.competition,
                    destination=getattr(args, "destination", None),
                )
            )
            code = 0
        elif args.command == "doctor":
            code = _run_doctor()
        elif args.command == "train" and args.train_command == "preflight":
            dataset_dir = _project_path(root, args.dataset_dir)
            report_path = (
                _project_path(root, args.report)
                if args.report is not None
                else dataset_dir / "preflight.json"
            )
            report = training_preflight(
                dataset_dir=dataset_dir,
                train_config_path=_project_path(root, args.config),
                report_path=report_path,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            code = 0 if report["ready"] else 1
        elif args.command == "train" and args.train_command == "linear":
            dataset_dir = _project_path(root, args.dataset_dir)
            checkpoint_path = _project_path(root, args.output)
            if not args.execute:
                print(
                    json.dumps(
                        {
                            "status": "dry_run",
                            "dataset_dir": str(dataset_dir),
                            "output": str(checkpoint_path),
                            "seed": args.seed,
                            "ensemble_members": args.ensemble_members,
                            "execute_required": True,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                code = 0
            else:
                preflight = training_preflight(
                    dataset_dir=dataset_dir,
                    train_config_path=root / "configs/oracle_train.json",
                    report_path=dataset_dir / "preflight.json",
                )
                if not preflight["ready"]:
                    raise RuntimeError("training preflight is not ready")
                payload = train_linear_oracle(
                    mic_csv=dataset_dir / "mic_measurements.csv",
                    hc50_csv=dataset_dir / "hc50_measurements.csv",
                    checkpoint_path=checkpoint_path,
                    seed=args.seed,
                    ensemble_members=args.ensemble_members,
                    dataset_manifest_path=dataset_dir / "manifest.json",
                )
                print(
                    json.dumps(
                        {
                            "status": "trained",
                            "checkpoint": str(checkpoint_path),
                            "checkpoint_id": payload["checkpoint_id"],
                            "metrics": {
                                task: payload["training"]["reports"][task][
                                    "diagnostic_calibrated_metrics"
                                ]
                                for task in ("activity", "safety")
                            },
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                code = 0
        elif args.command == "train" and args.train_command == "esm":
            dataset_dir = _project_path(root, args.dataset_dir)
            config_path = _project_path(root, args.config)
            embedding_path = _project_path(root, args.embeddings)
            checkpoint_path = _project_path(root, args.output)
            report_path = _project_path(root, args.report)
            model_source_path = _project_path(root, args.model_source)
            model_source = model_source_path if model_source_path.exists() else args.model_source
            if not args.execute:
                print(
                    json.dumps(
                        {
                            "status": "dry_run",
                            "dataset_dir": str(dataset_dir),
                            "model_source": str(model_source),
                            "embeddings": str(embedding_path),
                            "checkpoint": str(checkpoint_path),
                            "report": str(report_path),
                            "execute_required": True,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                code = 0
            else:
                preflight = training_preflight(
                    dataset_dir=dataset_dir,
                    train_config_path=config_path,
                    report_path=dataset_dir / "preflight.json",
                )
                if not preflight["ready"]:
                    raise RuntimeError("training preflight is not ready")
                from .esm_oracle import (
                    extract_esm_embeddings,
                    read_split_sequences,
                    train_esm_mic16_oracle,
                )

                config = json.loads(config_path.read_text(encoding="utf-8"))
                backbone = config["backbone"]
                embedding_manifest = extract_esm_embeddings(
                    sequences=read_split_sequences(dataset_dir / "sequence_splits.csv"),
                    model_source=model_source,
                    model_name=backbone["name"],
                    revision=backbone["revision"],
                    output_path=embedding_path,
                    batch_size=args.batch_size,
                    device=args.device,
                )
                checkpoint, report = train_esm_mic16_oracle(
                    mic_csv=dataset_dir / "mic_measurements.csv",
                    embeddings_path=embedding_path,
                    checkpoint_path=checkpoint_path,
                    report_path=report_path,
                    dataset_manifest_path=dataset_dir / "manifest.json",
                    oracle_config_path=config_path,
                    seed=args.seed,
                )
                print(
                    json.dumps(
                        {
                            "status": "trained",
                            "checkpoint": str(checkpoint_path),
                            "checkpoint_id": checkpoint["checkpoint_id"],
                            "embedding_manifest": embedding_manifest,
                            "candidate_minus_baseline": report["candidate_minus_baseline"],
                            "development_gate": report["development_gate"],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                code = 0
        elif args.command == "train" and args.train_command == "esm-score":
            paths = {
                key: _project_path(root, getattr(args, key))
                for key in (
                    "candidate_fasta",
                    "prefilter_scores",
                    "checkpoint",
                    "panel",
                    "embeddings",
                    "output",
                )
            }
            model_source_path = _project_path(root, args.model_source)
            model_source = model_source_path if model_source_path.exists() else args.model_source
            if not args.execute:
                print(
                    json.dumps(
                        {
                            "status": "dry_run",
                            **{key: str(value) for key, value in paths.items()},
                            "model_source": str(model_source),
                            "prefilter_k": args.prefilter_k,
                            "exploration_k": args.exploration_k,
                            "execute_required": True,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                code = 0
            else:
                from .esm_oracle import score_esm_mic16_candidates

                score_report = score_esm_mic16_candidates(
                    candidate_fasta=paths["candidate_fasta"],
                    prefilter_scores_path=paths["prefilter_scores"],
                    checkpoint_path=paths["checkpoint"],
                    panel_path=paths["panel"],
                    model_source=model_source,
                    embeddings_path=paths["embeddings"],
                    output_path=paths["output"],
                    prefilter_k=args.prefilter_k,
                    exploration_k=args.exploration_k,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed=args.seed,
                )
                print(json.dumps(score_report, indent=2, sort_keys=True))
                code = 0
        else:
            parser.error("unsupported command")
    except (FileNotFoundError, KeyError, PermissionError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
