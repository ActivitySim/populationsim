# PopulationSim Examples

This directory contains example projects for PopulationSim.

## Command-Line Interface

PopulationSim now supports a command-line interface (CLI) that allows you to run the package directly from the command line:

```bash
populationsim -c /path/to/configs -d /path/to/data -o /path/to/output
```

### CLI Options

- `-c, --config`: Path to config directory (can be specified multiple times)
- `-d, --data`: Path to data directory (can be specified multiple times)
- `-o, --output`: Path to output directory
- `-w, --working_dir`: Path to example/project directory (default: current directory)
- `-r, --resume`: Resume after step
- `-p, --pipeline`: Pipeline file name
- `-s, --settings_file`: Settings file name
- `--households_sample_size`: Households sample size
- `-m, --multiprocess`: Run multiprocess (optionally specify number of processes)
- `-e, --ext`: Package of extension modules to load
- `--fast`: Do not limit process to one thread

### Examples

1. Basic usage:

```bash
populationsim -c ./configs -d ./data -o ./output
```

2. Using a working directory:

```bash
populationsim -w ./my_project
```

3. Using multiple config and data directories:

```bash
populationsim -c ./configs -c ./configs_mp -d ./data -d ./additional_data -o ./output
```

4. Running with multiprocessing:

```bash
populationsim -c ./configs -d ./data -o ./output -m 4
```

5. Resuming from a checkpoint:

```bash
populationsim -c ./configs -d ./data -o ./output -r some_step
```

## Example Projects

- `example_calm`: Example using CALM data
- `example_calm_repop`: Example using CALM data with repopulation
- `example_oceanside_repop`: Example using Oceanside data with repopulation
- `example_survey_weighting`: Example of survey weighting
- `example_test`: Test example with various configurations

## CLI Example Script

The `cli_example.py` script demonstrates how to use the PopulationSim CLI from Python code. You can run it with:

```bash
python cli_example.py
```

This will run the Oceanside repopulation example using the CLI.
