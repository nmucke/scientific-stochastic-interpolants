#!/usr/bin/env python
"""Classify an untouched or completed training stage; refuse partial artifacts."""

import argparse
import json
from pathlib import Path


def stage_status(directory, expected_updates):
    path = Path(directory)
    if expected_updates < 1:
        raise ValueError("Expected update count must be positive")
    if not path.exists() or (path.is_dir() and not any(path.iterdir())):
        return "missing"
    required = ("config.yaml", "metadata.json", "history.json", "model.pth")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Existing stage {path} is incomplete (missing {missing}); "
            "inspect it and resume explicitly before restarting orchestration"
        )
    history = json.loads((path / "history.json").read_text())
    if not history or history[-1].get("step") != expected_updates:
        raise RuntimeError(
            f"Existing stage {path} has incomplete/unexpected updates "
            f"(expected {expected_updates}); inspect and resume explicitly"
        )
    return "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("expected_updates", type=int)
    args = parser.parse_args()
    print(stage_status(args.directory, args.expected_updates))


if __name__ == "__main__":
    main()
