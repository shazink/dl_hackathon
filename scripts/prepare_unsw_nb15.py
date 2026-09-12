"""Prepare verified training-only arrays with the locked experience protocol."""

import argparse
import sys

from tafr_ids.data.artifacts import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--experience-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        metadata = prepare(args.data_dir, args.split_manifest, args.experience_config, args.output_dir)
    except (OSError, ValueError) as error:
        print(f"Preparation blocked: {error}", file=sys.stderr)
        return 2
    print(f"Prepared eight subsets; E1 fit rows: {metadata['fit_rows']}; features: {len(metadata['feature_names'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
