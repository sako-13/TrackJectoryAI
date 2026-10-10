"""CLI entry point: serve the live demo app."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running as `python scripts/serve_demo.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the athlete career forecast demo")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    host = args.host or cfg["demo"]["host"]
    port = args.port or cfg["demo"]["port"]
    print(f"Demo app: http://{host}:{port}")
    uvicorn.run("src.demo.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
