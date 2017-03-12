
## convert CALM popsim inputs to ActivitySim format

  - all required inputs are in the data folder:
    - ss10hor.csv - PUMS HHs
    - ss10por.csv - PUMS persons
    - CALMtractData.csv - CALM tract data
    - CALMtazData.csv - CALM TAZ data
    - OSU.csv - CALM OSU TAZ data
    - geographicCwalk.csv - geographic crosswalk file
  - run with Python 2.7 (with pandas and numpy libraries): `python calm_data.py`
  - outputs `calm_psim.h5` HDF5 data store with the following pandas DataFrames:
    - seed_households
    - seed_persons
    - meta_control_data
    - mid_control_data
    - low_control_data
