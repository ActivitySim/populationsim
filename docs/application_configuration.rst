.. PopulationSim documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Application & Configuration
=============================

This section describes how to set up a new PopulationSim implementation. The first step is to understand the requirements of the project in terms of geographic resolution and details desired in the synthetic population. Once the requirements of the project have been established, the next step is to prepare the inputs to PopulationSim. Next, PopulationSim needs to be configured for available inputs and features desired in the final synthetic population. After this, the user needs to run PopulationSim and resolve any data related errors. Finally, the output synthetic population needs to be validated against the controls for of precision and variance.

Geographies
-----------

Traditionally, travel forecasting models have followed the sequential four-step model framework. This required the modeling region to be divided into zones. The 4-step forecasting process starts with *trip generation* in each zone using the available demographic data. Next, *trip distribution* between zones, and finally, *mode choice* and *route assignment*. The zones used in four-step process are typically known as Traffic Analysis Zones (TAZs). The spatial boundaries of TAZs varies across modeling region and ranges from a city block to a large area in the suburb within a modeling region. Smaller TAZs result in additional zones and thus adds to the computational burden. 

ABMs on the other hand are computationally efficient and operate in a micro-simulation framework at the level of persons and households. Most of the advanced ABMs (e.g., DaySim, CT-RAMP, etc.) operate at a finer spatial resolution wherein all location choices (e.g., usual work location, tour destination choice) are modeled at a finer geography. This finer geography typically is the Micro-Analysis Zones (MAZs) which are smaller zones nested within TAZs. This requires the synthetic population to be produced at the level of MAZs.

As discussed earlier, two main inputs to a population synthesizer are a seed sample and controls. The seed sample can come from a household travel survey or from ACS PUMS with latter being the most common source. In case of a household travel survey, the geographic resolution of the seed sample is determined by the geocoding geography. The PUMS data contains a sample of actual responses to ACS. While PUMS data contain records from disaggregate geographies but data is made available at an aggregate geography called Public Use Microdata Area (PUMA). PUMAs are special non-overlapping areas that partition each state into contiguous geographic units containing no fewer than 100,000 people each.

Ideally, it is desired that all the controls and seed sample should be available at the same  level of geographic resolution as the travel model. However, this is typically not the case. Some important demographic, socio-economic and land-use development distributions (e.g., employment or occupation data) that dictates population synthesis are only available at a more aggregate geography (e.g., County, District, Region, etc.). Moreover, some distributions which are available at a finer geographic level in the base year may not be available at the same geographic level for a future forecast year. In some cases, even if a control is available at a finer geography, the modeler might want to specify that control (e.g., population by age) at an aggregate geography.

The flexible number of geographies feature in PopulationSim enables user to make use of data available at different geographic resolutions. In summary, the **choice of geographies for PopulationSim** is decided based on following:

:Travel Model Spatial Resolution:
	For most ABMs, this is **MAZ** but can also be **TAZ** or **Block Group**
	
:Seed Sample Geography:
	Typically, this would be **PUMA** but can be some other geography depending on the source of seed sample
	
:Marginal Controls Geographies: 
	User would need to specify at least one control at the travel model spatial resolution. Geography of other controls would depend upon the availability of data and user preferences. 

The hierarchy of geographies is important when making a decision regarding controls. The hierarchy of geographies in PopulationSim framework is as follows:
	
  * Meta (e.g., County)
  * Seed (e.g., PUMA)
  * Sub-Seed (e.g., TAZ, Block Group, MAZ)
 
Seed geography is the geographic resolution of the seed data. PopulationSim starts at this geography and moves up to the Meta geography. Currently, PopulationSim can handle only one Meta geography. After incorporating controls at the Meta geography, PopulationSim moves down to Sub-Seed geographies. PopulationSim can handle any number of Sub-Seed geographies. While selecting control geographies user should ensure to not select more than one Meta geography. More information on PopulationSim algorithm can be found from the PopulationSim specifications in the *Documents & Resources* section.

  
Controls
---------

 

  * Data Sources
  * How to specify controls
  
Configuration
-------------

  * Tables
  
Checks on data inputs
---------------------


Inputs & Outputs
----------------


Error Handling & Debugging
--------------------------


