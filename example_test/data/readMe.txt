# The sample data package contains the following files:
1. geogXWalk.csv: 
	geographic crosswalk between the four geographies - Low, Mid, Seed and Meta
	Fields - LowZoneID	MidZoneID	SeedID	MetaID
2. low_Controls.csv:
	controls specified at lower geographies
	LowZoneID: low zone ID
	POP: population for each low zone 
	HH: number of households in each low zone 
	HHSIZE1: number of HHs in each low zone with HH size=1
	HHSIZE2: number of HHs in each low zone with HH size=2
	HHSIZE3: number of HHs in each low zone with HH size=3	
	HHSIZE4p: number of HHs in each low zone with HH size>=4
3. mid_Controls.csv:
	controls specified at middle geography
	MidZoneID: mid zone ID 	
	POP: population for each mid zone 
	HH: number of households in each mid zone 
	HHWORKER0: number of households with 0 workers
	HHWORKER1: number of households with 1 workers	
	HHWORKER2: number of households with 2 workers	
	HHWORKER3p: number of households with 3+ workers
4. meta_controls.csv
	controls specified at meta geography level
	region: meta geography is whole region, so region ID=1
	POP: total regional population
5. hh_sample.csv
	seed sample of HH records
	SERIALNO: unique HH ID 	
	SeedID: ID of the seed geography this record belongs to 
	ADJINC: Income adjustment factor
	WGTP: initial seed weight	
	NP: number of persons in the hosehold [0,1,2,...20]	
	TYPE: type of HH record (all residential so TYPE==1)	
	HINCP: household income
6. per_sample.csv
	seed sample of person records
	SERIALNO: unique HH ID	
	SPORDER: person order within each HH
	SeedID: ID of the seed geography this record belongs to	
	WGTP: initial hh seed weight	
	AGEP: person age [0-99+]	
	ESR: employment status recode []
	    0 .N/A (less than 16 years old)
        1 .Civilian employed, at work
        2 .Civilian employed, with a job but not at work
        3 .Unemployed
        4 .Armed forces, at work
        5 .Armed forces, with a job but not at work
        6 .Not in labor force
	SEX: gender
	    1 .Male
        2 .Female 
	SCHG: schooling level
        0 .N/A (not attending school)
        1 .Nursery school/preschool
        2 .Kindergarten
        3 .Grade 1 to grade 4
        4 .Grade 5 to grade 8
        5 .Grade 9 to grade 12
        6 .College undergraduate
        7 .Graduate or professional school



