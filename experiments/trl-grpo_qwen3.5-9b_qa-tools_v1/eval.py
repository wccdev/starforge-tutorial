"""TRL GRPO evaluation hook（recipe lifecycle eval 要求的实验内钩子）。"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--report", required=True)
args, passthrough = parser.parse_known_args()
path = Path(args.report)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"data": args.data, "args": passthrough, "status": "hook-complete"}) + "\n")
