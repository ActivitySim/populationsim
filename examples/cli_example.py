#!/usr/bin/env python
# Example script demonstrating how to use the populationsim CLI

import subprocess
import sys
from pathlib import Path


def main():
    """
    Example of how to use the populationsim command-line interface.

    This script demonstrates how to call populationsim with command-line arguments
    using subprocess, which tests the actual command-line interface.

    It uses the example_test configuration as an example, which runs faster.
    """
    # Get the path to the example_test directory
    example_dir = Path(__file__).parent / "example_test"

    # Define the paths for configs, data, and output
    configs_dir = example_dir / "configs"
    data_dir = example_dir / "data"
    output_dir = example_dir / "output"

    # Ensure the output directory exists
    output_dir.mkdir(exist_ok=True)

    print("Running populationsim CLI example...")
    print(f"Using example directory: {example_dir}")
    print(f"Config directory: {configs_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Build the command to run populationsim
    # This uses python -m populationsim to run the module directly
    command = [
        sys.executable,
        "-m",
        "populationsim",
        "-c",
        str(configs_dir),
        "-d",
        str(data_dir),
        "-o",
        str(output_dir),
    ]

    print(f"\nRunning command: {' '.join(command)}")
    print("\nOutput will appear below (press Ctrl+C to interrupt):\n")
    print("-" * 80)

    # Run the command using subprocess
    # We don't capture the output so it appears in the terminal in real-time
    try:
        result = subprocess.run(command, check=True, text=True)
        print("-" * 80)
        print(f"\nCommand completed with exit code: {result.returncode}")

    except subprocess.CalledProcessError as e:
        print("-" * 80)
        print(f"\nCommand failed with exit code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCommand was interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
