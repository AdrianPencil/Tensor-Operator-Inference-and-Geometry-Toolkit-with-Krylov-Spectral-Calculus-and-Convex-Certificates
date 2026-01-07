"""
Command line interface for tig (minimal).

Provides:
- list-pipelines
- run-pipeline --name <pipeline> --config <json-string>

The CLI is intentionally small: notebooks and docs are the primary UX.
"""

import argparse
import json
from typing import Any, Dict

from tig.workflows.pipelines import list_pipelines, run_pipeline

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tig")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list-pipelines", help="List registered pipelines")

    p_run = sub.add_parser("run-pipeline", help="Run a named pipeline")
    p_run.add_argument("--name", required=True, type=str, help="Pipeline name")
    p_run.add_argument(
        "--config",
        required=False,
        type=str,
        default="{}",
        help='JSON string config, e.g. \'{"m":64,"n":32}\'',
    )

    args = parser.parse_args(argv)

    if args.cmd == "list-pipelines":
        for name in list_pipelines():
            print(name)
        return 0

    if args.cmd == "run-pipeline":
        cfg: Dict[str, Any] = json.loads(args.config)
        out = run_pipeline(args.name, cfg)
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    return 2
