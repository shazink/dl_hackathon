"""Emit a JSON inspection report to stdout; exit 2 for blocked validation."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from tafr_ids.data.inspection import inspect_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing the eight downloaded files")
    parser.add_argument("--split-manifest", help="Portable TOML with verified logical roles and hashes")
    parser.add_argument("--output", type=Path, help="Write JSON report outside the dataset directory")
    args = parser.parse_args()
    try:
        if args.output:
            output = args.output.expanduser().resolve()
            root = Path(args.data_dir).expanduser().resolve()
            if output == root or root in output.parents:
                raise ValueError("Output must be outside the downloaded dataset directory")
            if args.split_manifest and output == Path(args.split_manifest).resolve():
                raise ValueError("Output must not overwrite the split manifest")
            if output.suffix != ".json":
                raise ValueError("Output must be a .json report")
        report = inspect_dataset(args.data_dir, args.split_manifest)
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            output.parent.mkdir(parents=True, exist_ok=True)
            # Replace the report atomically; never mutate an existing hard-linked
            # inode that might also name a downloaded source file.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(payload)
                os.replace(temporary, output)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    except (OSError, ValueError) as error:
        print(f"Inspection failed: {error}", file=sys.stderr)
        return 2
    if args.output:
        print(f"Inspection {report['status']}; report: {output}")
    else:
        print(payload, end="")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
