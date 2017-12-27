
::~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:: Runs PopulationSim. User should specify the following
:: 		- Local Anaconda installation directory
::		- Assumes SetUpPopulationSim.bat has been run and Conda environment "popsim" exists
:: Binny Paul, binny.mathewpaul@rsginc.com, 121517
::~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@ECHO OFF

:: USER INPUTS
:: ---------------------------------------------------------------------
:: Local Anaconda installation directory
SET ANACONDA_DIR=E:\projects\clients\odot\PopulationSim\Anaconda2
:: ---------------------------------------------------------------------


:: setup paths to Python application, Conda script, etc.
SET CONDA_ACT=%ANACONDA_DIR%\envs\popsim\Scripts\activate.bat
ECHO CONDA_ACT: %CONDA_ACT%

SET CONDA_DEA=%ANACONDA_DIR%\envs\popsim\Scripts\deactivate.bat
ECHO CONDA_DEA: %CONDA_DEA%

SET PYTHON=%ANACONDA_DIR%\envs\popsim\python.exe
ECHO PYTHON: %PYTHON%


:: run populationsim
ECHO Running PopulationSim....
CALL %CONDA_ACT% popsim
%PYTHON% run_populationsim.py -m repop
CALL %CONDA_DEA%

ECHO PopulationSim run complete!!
PAUSE