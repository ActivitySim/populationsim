#!/usr/bin/env python
# PopulationSim
# See full license in LICENSE.txt.

import sys
import argparse
from pathlib import Path

from populationsim import run, add_run_args


def main():
    """
    Command-line entry point for populationsim.

    This allows running the package directly with:
    python -m populationsim
    or
    populationsim (if installed with the entry point)
    """
    parser = argparse.ArgumentParser(
        description="PopulationSim: Population Synthesis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_run_args(parser)
    args = parser.parse_args()

    # If no working_dir is specified, use the current directory
    if not args.working_dir:
        args.working_dir = Path.cwd()

    try:
        sys.exit(run(args))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
