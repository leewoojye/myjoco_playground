from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from transition_core import DEFAULT_XML, run_transition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--no-push", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, _ = run_transition(xml_path=args.xml, push_time=None if args.no_push else 6.55)
    print(json.dumps(metrics.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
