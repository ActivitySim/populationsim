.. PopulationSim documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Introduction
=============

PopulationSim is an open platform for population synthesis and survey weighting.  It emerged from
`Oregon DOT <https://www.oregon.gov/odot>`_'s desire to build a shared, open, platform that could 
be easily adapted for statewide, regional, and urban transportation planning needs.
      
What is population synthesis?
-----------------------------
Activity based travel demand models such as `ActivitySim <http://www.activitysim.org>`_ operate at an individual
level, wherein the travel choices of person and household decision-making agents are predicted by applying 
Monte Carlo methods to behavioral models. This requires a data set of households and persons representing 
the entire population in the modeling region. Population synthesis refers to the process used to create this data.

The required inputs to population synthesis are a population sample and marginal distributions (or control totals). 
The population sample is commonly referred to as the *seed or reference sample* and the marginal distributions are 
commonly referred to as *controls or targets*. **The process of expanding the seed sample to match the marginal 
distribution is termed population synthesis.** The software tool which implements this population synthesis process 
is termed as a **Population Synthesizer**.

What does a Population Synthesizer produce?
-------------------------------------------
The objective of a population synthesizer is to generate a synthetic population for 
a modeling region. The main outputs from a population synthesizer include tables of persons and households 
representing the entire population of the modeling region. These tables also include household and person-level 
attributes of interest. Examples of attributes at the household level include household income, household size, housing 
type, and number of vehicles. Examples of person attributes include  
age, gender, work\school status, and occupation. Depending on the use case, a population synthesizer may also 
produce multi-way distribution of demographic variables at different geographies to be used as an input 
to aggregate (four-step) travel models. In the case of PopulationSim specifically, an additional option is also included to 
modify an existing regional synthetic population for a smaller geographical area. In this case, the outputs are a modified 
set of persons and households.

How does a population synthesizer work?
---------------------------------------
The main inputs to a population synthesizer are disaggregate population samples and marginal control
distributions. In the United States, the disaggregate population sample is typically obtained from the `Census Public Use 
Microdata Sample (PUMS) <https://www.census.gov/programs-surveys/acs/microdata.html>`_, but other sources, such as a household 
travel survey, can also be used. The seed sample should include demographic variables corresponding to each marginal control 
termed as *controlled variables* (e.g., household size, household income, etc.). The seed sample could also include other 
variables of interest but not necessarily controlled via marginal controls. These are termed as *uncontrolled variables*. 
The seed sample should also include an initial weight on each household record. 

Current year marginal distributions of person and household-level attributes of interest are available from Census. For 
future years, marginal distributions are either held constant, or forecasted.  Marginal distributions can be for both 
household or person level variables and are specified at a specific geography (e.g., Block Groups, Traffic Analysis Zone 
or County). PopulationSim allows controls to be specified at multiple geographic levels. 

The objective of a population synthesizer is to generate household weights which satisfies the marginal control 
distributions. This is achieved by use of a data fitting technique. The most common fitting technique used by various 
population synthesizers is the Iterative Proportional Fitting (IPF) procedure. Generally, the IPF procedure is used 
to obtain joint distributions of demographic  variables. Then, random sampling from PUMS generates the baseline synthetic 
population. 

One of the limitations of the simple IPF method is that it does not incorporate both household and person 
level attributes simulatenously. Some population synthesizers use a heuristic algorithm called the 
Iterative Proportional Updating Algorithm (IPU) to incorporate both person and household-level variables in the fitting procedure. 

Besides IPF, entropy maximization algorithms have been used as a fitting technique. In most of the entropy based methods, 
the relative entropy is used as the objective function. The relative entropy based optimization ensures 
that the least amount of new information is introduced in finding a feasible solution. The base entropy 
is defined by the initial weights in the seed sample. The weights generated by the entropy maximization 
algorithm preserves the distribution of initial weights while matching the marginal controls. This is an 
advantage of the entropy maximization based procedures over the IPF based procedures. PopulationSim uses the entropy maximization 
based list balancing to match controls specified at various geographic levels.

Once the final weights have been assigned, the seed sample is expanded using these weights to generate a synthetic population. Most 
population synthesizers create distributions using final weights and employ random sampling to expand the
seed sample. PopulationSim uses Linear Programming to convert the final weights to integer values and expands 
the seed sample using these integer weights. For detailed description of PopulationSim algorithm, please refer to the TRB paper 
link in the :ref:`docs` section. For information on software implementation refer to :ref:`core_components` and :ref:`model_steps`. To 
learn more about PopulationSim application and configuration, please follow the content index below. 

How does population synthesis work for survey weighting?
--------------------------------------------------------
PopulationSim can also be used to solve the household travel survey weighting problem of developing final weights.  Travel surveys typically
include a set of initial household weights based on the sampling plan.  Often the initial weights are revised to match a set of control totals 
that describe the overall survey population.  This exercise is like population synthesis, except that the geographic allocation of households is 
not needed because household locations are surveyed and not synthesized. 


Contents
--------

.. toctree::
   :maxdepth: 2
   
   getting_started
   application_configuration
   validation
   software
   docs


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
