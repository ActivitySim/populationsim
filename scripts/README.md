
## Scripts

  - validation.ipynb - Jupyter validation script to generate advanced summary statistics and validation plots. This validation script takes summaries and outputs from a PopulationSim run. The script is configured to run for the CALM region example and includes notes on inputs and configuration settings
  - calm_verification.yaml - YAML file to specify the controls for which the summaries should be generated
  - benchmark_incidence_table.py - times build_incidence_table and build_grouped_incidence_table against their previous implementations on synthetic seed data (`python scripts/benchmark_incidence_table.py [num_households ...]`) and checks that both produce identical tables
