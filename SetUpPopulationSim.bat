
::~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:: Setup PopulationSim
:: 		- Creates Conda environment
::		- Installs required packages (tools, zbox, orca, openmatrix, activitysim, ortools)
::		- Installs PopulationSim package
:: User should specify the following
::      - Local Anaconda installation directory
::      - PopulationSim Working Directory [should contain populationsim source code folder and this batch file]
:: Binny Paul, binny.mathewpaul@rsginc.com, 121517
::~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@ECHO OFF


:: USER INPUTS
:: ---------------------------------------------------------------------
:: Local Anaconda installation directory
SET ANACONDA_DIR=E:\projects\clients\odot\PopulationSim\Anaconda2
:: ---------------------------------------------------------------------


:: setup paths to Python application, Conda script, etc.
SET CONDA_APP=%ANACONDA_DIR%\envs\popsim\Scripts\conda.exe
ECHO CONDA_APP: %CONDA_APP%

SET CONDA_ACT=%ANACONDA_DIR%\envs\popsim\Scripts\activate.bat
ECHO CONDA_ACT: %CONDA_ACT%

SET CONDA_DEA=%ANACONDA_DIR%\envs\popsim\Scripts\deactivate.bat
ECHO CONDA_DEA: %CONDA_DEA%

SET PIP_INSTALL=%ANACONDA_DIR%\envs\popsim\Scripts\pip.exe
ECHO PIP_INSTALL: %PIP_INSTALL%

SET PYTHON=%ANACONDA_DIR%\envs\popsim\python.exe
ECHO PYTHON: %PYTHON%


:: Remove existing environment [uncomment lines below to remove previously created evironment]
::%ANACONDA_DIR%\Scripts\conda.exe env remove -n popsim

:: Create Conda environment if doesnt exists
ECHO Creating Conda environment - popsim
%ANACONDA_DIR%\Scripts\conda.exe env list>env_list.txt
findstr /e %ANACONDA_DIR%\envs\popsim env_list.txt
if %errorlevel%==0 (
echo Conda env already exists
) else (
echo Conda env does not exists. Creating Conda env...
%ANACONDA_DIR%\Scripts\conda.exe create -n popsim python=2.7
)


:: install dependencies
ECHO Installing dependencies....
CALL %CONDA_ACT% popsim
%PIP_INSTALL% install --upgrade toolz
%PIP_INSTALL% install --upgrade tables
%PIP_INSTALL% install --upgrade zbox
%PIP_INSTALL% install --upgrade orca
%PIP_INSTALL% install --upgrade openmatrix
%PIP_INSTALL% install --upgrade ortools
%PIP_INSTALL% install --upgrade activitysim
CALL %CONDA_DEA%


:: install PopulationSim package
ECHO Installing PopulationSim package....
CALL %CONDA_ACT% popsim
%PIP_INSTALL% install --upgrade https://github.com/RSGInc/populationsim/zipball/master
CALL %CONDA_DEA%

DEL /Q env_list.txt

ECHO PopulationSim Set Up Complete
