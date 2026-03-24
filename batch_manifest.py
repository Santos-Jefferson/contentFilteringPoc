import argparse
import json

from src.batch_runner import run_manifest


DEFAULT_CONFIG_PATH = "config/bad_terms_categories.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process a batch manifest to build catalog artifacts asynchronously.",
    )
    parser.add_argument("manifest_path", help="Path to a JSON manifest containing jobs or a top-level jobs array.")
    parser.add_argument(
        "--config-path",
        default=None,
        help=(
            "Optional path to term config JSON. When omitted, uses the manifest's config_path "
            f"or falls back to {DEFAULT_CONFIG_PATH}."
        ),
    )
    parser.add_argument(
        "--artifacts-root",
        default="artifacts",
        help="Base directory where per-job artifacts and batch summaries are written.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent jobs. Defaults to the manifest value or 1.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_manifest(
        manifest_path=args.manifest_path,
        config_path=args.config_path,
        artifacts_root=args.artifacts_root,
        concurrency=args.concurrency,
    )
    print(json.dumps({
        "batch_id": summary.batch_id,
        "summary_path": summary.summary_path,
        "total_jobs": summary.total_jobs,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
    }, indent=2))


if __name__ == "__main__":
    main()

