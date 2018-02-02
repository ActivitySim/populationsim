
::~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:: Runs PopulationSim. User should specify the following
:: 		- Local Anaconda installation directory
::		- Assumes SetUpPopulationSim.bat has been run and Conda environment "popsim" exists
:: Binny Paul, binny.mathewpaul@rsginc.com, 121517
::~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@ECHO OFF
ECHO %startTime%%Time%
SET BATCH_DIR=%~dp0

:: USER INPUTS
:: ---------------------------------------------------------------------
:: Local Anaconda installation directory
SET ANACONDA_DIR=E:\projects\clients\odot\PopulationSim\Anaconda2
:: ---------------------------------------------------------------------

SET PATH=%ANACONDA_DIR%\Library\bin;%PATH%

:: setup paths to Python application, Conda script, etc.
SET CONDA_ACT=%ANACONDA_DIR%\Scripts\activate.bat
ECHO CONDA_ACT: %CONDA_ACT%

SET CONDA_DEA=%ANACONDA_DIR%\Scripts\deactivate.bat
ECHO CONDA_DEA: %CONDA_DEA%

SET PYTHON=%ANACONDA_DIR%\envs\popsim\python.exe
ECHO PYTHON: %PYTHON%


:: run populationsim
ECHO Running PopulationSim....
CD %ANACONDA_DIR%\envs\popsim\Scripts
CALL %CONDA_ACT% popsim
CD %BATCH_DIR%

%PYTHON% run_populationsim.py

CD %ANACONDA_DIR%\envs\popsim\Scripts
CALL %CONDA_DEA%
CD %BATCH_DIR%

ECHO PopulationSim run complete!!
ECHO %startTime%%Time%
PAUSE
