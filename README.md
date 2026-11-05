PopulationSim
=============

[![Python package](https://github.com/ActivitySim/populationsim/actions/workflows/python-package.yml/badge.svg?branch=master)](https://github.com/ActivitySim/populationsim/actions/workflows/python-package.yml)<a href="https://medium.com/zephyrfoundation/populationsim-the-synthetic-commons-670e17383048"><img src="https://github.com/ZephyrTransport/zephyr-website/blob/gh-pages/img/badging/project_pages/populationsim/PopulationSim.png" width="72.6" height="19.8"></a>


PopulationSim is an open platform for population synthesis.  It emerged
from Oregon DOT's desire to build a shared, open, platform that could be
easily adapted for statewide, regional, and urban transportation planning
needs. PopulationSim includes its own modeling core, originally derived from
[ActivitySim](https://github.com/ActivitySim/activitysim).

## Command-Line Interface

Use Python 3.12 for the released application. Install
[uv 0.13.4 or newer](https://docs.astral.sh/uv/getting-started/installation/), then install the
released command-line application:

```bash
uv tool install --python 3.12 populationsim
```

PopulationSim can then be run directly from the command line. The config, data,
and output directories must already exist:

```bash
populationsim -c /path/to/configs -d /path/to/data -o /path/to/output
```

See [Getting Started](https://activitysim.github.io/populationsim/getting_started.html)
for the uv setup and example commands, or browse the [examples directory](examples/).
For existing environments, follow the
[certificate security update instructions](https://activitysim.github.io/populationsim/getting_started.html#certificate-security-update)
for the October 2026 eMudhra/emSign incident before installing packages.

## Documentation

https://activitysim.github.io/populationsim/
