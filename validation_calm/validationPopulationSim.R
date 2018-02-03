##############################################################################################################
# Validation script for PopulationSim
#
# binny.paul@rsginc.com, Jan 2018
# 
# This script uses summary outputs from a PopulationSim run to generate validation summaries and plots
#
# User needs to specify the following inputs and settings:-
#
#         PopulationSim working directory [the directory containing data, configs and output folders]
#         Validation directory [the directory where you would want all your validation summaries]
#         scenario name
#         region name
#         List of geographies - from highest to lower geography [e.g., REGION > PUMA > TAZ > MAZ]
#                               First Meta, second Seed and then all Sub-Seed from the highest to the lowest
#         Plot geographies - any geography other than Seed geography. Plots will be generated only for the 
#                            geographies listed here for the controls for that geography
#         Geographic crosswalk file name [assumed to be inside the data folder in PopulationSim working dir]
#         Column Map CSV - CSV file to specify the controls for which the summaries should be generated. FOllowing 
#                          columns need to be specified:
#             NAME       : Name of the control to be used for labels
#             GEOGRAPHY  : Geography at which the control was specified
#             CONTROL    : Control value column name in the summary_GEOGRAPHY file [summary_PUMA file for Meta controls] 
#             SUMMARY    : Estimate/Result value column name in the summary_GEOGRAPHY file [summary_PUMA file for Meta controls]
#
#         Input seed household sample file
#         List of column names in the input seed household sample (seed_col) for following:
#             Seed geography name
#             Unique household ID
#             Initial household weight
#         
#         Column name of the unique HH ID (expanded_hhid_col) in the expanded household id file (expanded_household_ids.csv). 
#                            This is the column name assigned to the unique household ID in the initial seed sample in the 
#                            PopulationSim YAML settings file.
#         
#         The user should also configure the PopulationSim to produce following summary files in the output folder:
#                 expanded_household_ids.csv
#                 summary_GEOGRAPHY.csv (for all sub-seed geographies, e.g., summary_TRACT.csv)
#                 summary_LOWEST_PUMA.csv (PUMA level summaries for the lowest gepgraphy, e.g., summary_TAZ_PUMA.csv)
#
#
# List of outputs:-
# 
# CSV summary Statistics file - It has the following statistics:
#         controlName - Name of the control
#         geography - Geography at which the control is specified
#         Observed - Regional total specified
#         Predicted - Regional total synthesized
#         Difference - Predicted - Observed
#         pcDifference - Perecntage difference at a regional level
#         N - Number of geographies (MAZ/TAZ/META) with non-zero control
#         RMSE - Percentage root mean square error for the control at the specified geography
#         SDEV - Standard deviation of precentage difference
#
# Plots (JPEGs) - convergence plots:
#
#         Plot showing mean percentage difference and STDEV for each control
#         Plot showing frequency distribution of differences b/w target and estimate for each control
#         Plot showing expansion factor distribution
#
##############################################################################################################


### User Inputs [Read command line arguments]
popSimDir       <- "C:/Users/binny.paul.I-RSG/Documents/Projects/ODOT_PopSyn/PopulationSimTest/example_calm"
validationDir   <- "C:/Users/binny.paul.I-RSG/Documents/Projects/ODOT_PopSyn/PopulationSimTest/validation_calm"

scenario        <- "Base"						    
region          <- "CALM" 

geographyList   <- c("REGION", "PUMA", "TRACT", "TAZ")
plotGeographies <- c("REGION", "TRACT", "TAZ")

geogXWalk       <- read.csv(paste(popSimDir, "data/geo_cross_walk.csv", sep = "/"))
columnMap       <- read.csv(paste(validationDir, "columnMapPopSim_CALM.csv", sep = "/"))

seed_households <- read.csv(paste(popSimDir, "data/seed_households.csv", sep = "/"))
seed_col        <- c("PUMA", "hhnum", "WGTP")

expanded_hhid      <- read.csv(paste(popSimDir, "output/expanded_household_ids.csv", sep = "/"))
expanded_hhid_col  <- c("hh_id")

#   This is currently configured for 2 sub-seed geography
#   User should add more read lines when more geographies is involved
#   The nummber at the end of summary file name indicates the geographic index
#   Example, summary3 is the name for summary_TRACT which is the 3rd geography in the geography list
summary2           <- read.csv(paste(popSimDir, "output/summary_TAZ_PUMA.csv", sep = "/"))
summary3           <- read.csv(paste(popSimDir, "output/summary_TRACT.csv", sep = "/"))
summary4           <- read.csv(paste(popSimDir, "output/summary_TAZ.csv", sep = "/"))

summary2$meta_geog <- geogXWalk$REGION[match(summary2$id, geogXWalk$PUMA)]
summary1           <- summary2
summary1$geography <- geographyList[1]
summary1$id        <- summary1$meta_geog

### Install all required R packages to the user specified directory [Make sure the R library directory has write permissions for the user]
if (!"RODBC" %in% installed.packages()) install.packages("RODBC", repos='http://cran.us.r-project.org')
if (!"dplyr" %in% installed.packages()) install.packages("dplyr", repos='http://cran.us.r-project.org')
if (!"ggplot2" %in% installed.packages()) install.packages("ggplot2", repos='http://cran.us.r-project.org')
if (!"tidyr" %in% installed.packages()) install.packages("tidyr", repos='http://cran.us.r-project.org')
if (!"scales" %in% installed.packages()) install.packages("scales", repos='http://cran.us.r-project.org')
if (!"hydroGOF" %in% installed.packages()) install.packages("hydroGOF", repos='http://cran.us.r-project.org')

### Load libraries
R_PKGS <- c("RODBC", "dplyr", "ggplot2", "tidyr", "scales", "hydroGOF")
lib_sink <- suppressWarnings(suppressMessages(lapply(R_PKGS, library, character.only = TRUE)))

setwd(validationDir)

### Function to process each control
procControl <- function(geography, controlName, controlID, summaryID){
  

  #Fetching data
  geoIndex <- which(geographyList == geography)
  ev1 <- paste("sub_summary <- summary", geoIndex, sep = "")
  eval(parse(text = ev1))
  
  controls <- sub_summary[, c(which("id" == names(sub_summary)),which(controlID == names(sub_summary)))]
  synthesized <- sub_summary[, c(which("id" == names(sub_summary)),which(summaryID == names(sub_summary)))]
  colnames(controls) <- c("GEOGRAPHY", "CONTROL")
  colnames(synthesized) <- c("GEOGRAPHY", "SYNTHESIZED")
  
  # Meta controls are grouped by PUMAs, aggregation is required
  if(geoIndex==1){
    # aggregate control to meta geography
    controls <- controls %>%
      group_by(GEOGRAPHY) %>%
      summarise(CONTROL = sum(CONTROL)) %>%
      ungroup()
    
    # aggregate synthesized to meta geography
    synthesized <- synthesized %>%
      group_by(GEOGRAPHY) %>%
      summarise(SYNTHESIZED = sum(SYNTHESIZED)) %>%
      ungroup()
  }
  
  #Fetch and process each control for getting convergance statistics
  compareData <- left_join(controls, synthesized, by="GEOGRAPHY") %>%
    mutate(CONTROL = as.numeric(CONTROL)) %>%
    mutate(SYNTHESIZED = ifelse(is.na(SYNTHESIZED), 0, SYNTHESIZED)) %>%
    mutate(DIFFERENCE = SYNTHESIZED - CONTROL) %>%
    mutate(pcDIFFERENCE = ifelse(CONTROL > 0,(DIFFERENCE/CONTROL)*100,NA))
  
  #Calculate statistics
  Observed <- sum(compareData$CONTROL)
  Predicted <- sum(compareData$SYNTHESIZED)
  Difference <- Predicted - Observed
  pcDifference <- (Difference/Observed)*100
  N <- sum(compareData$CONTROL > 0)
  PRMSE <- (((sum((compareData$CONTROL - compareData$SYNTHESIZED)^2)/(sum(compareData$CONTROL > 0) - 1))^0.5)/sum(compareData$CONTROL))*sum(compareData$CONTROL > 0)*100
  meanPCDiff <- mean(compareData$pcDIFFERENCE, na.rm=TRUE)
  SDEV <- sd(compareData$pcDIFFERENCE, na.rm=TRUE)
  stats <- data.frame(controlName, geography, Observed, Predicted, Difference, pcDifference, N, PRMSE, meanPCDiff, SDEV)
  
  #Preparing data for difference frequency plot
  freqPlotData <- compareData %>%
    filter(CONTROL > 0) %>%
    group_by(DIFFERENCE) %>%
    summarise(FREQUENCY = n())
  
  if(geography %in% plotGeographies){
    #computing plotting parameters
    xaxisLimit <- max(abs(freqPlotData$DIFFERENCE)) + 10
    plotTitle <- paste("Frequency Plot: Syn - Control totals for", controlName, sep = " ")
    
    #Frequency Plot
    p1 <- ggplot(freqPlotData, aes(x=DIFFERENCE,y=FREQUENCY))+
      geom_point(colour="coral") +
      coord_cartesian(xlim = c(-xaxisLimit, xaxisLimit)) +
      geom_vline(xintercept=c(0), colour = "steelblue")+
      labs(title = plotTitle)
    ggsave(paste("plots/",controlID,".png",sep=""), width=9,height=6)
  }
  
  cat("\n Processed Control: ", controlName) 
  
  return(stats)
}

myRMSE <- function(FINALEXPANSIONS, AVERAGEEXPANSION, N){
  EXPECTED <- rep(AVERAGEEXPANSION,N)
  ACTUAL <- FINALEXPANSIONS
  return(rmse(ACTUAL, EXPECTED, na.rm=TRUE))
}

#Create plot directory
dir.create('plots', showWarnings = FALSE)

### Computing convergance statistics and write out results
stats <- apply(columnMap, 1, function(x) procControl(x["GEOGRAPHY"], x["NAME"], x["CONTROL"],x["SUMMARY"]))

stats <- do.call(rbind,stats)
write.csv(stats, paste(scenario, "PopulationSim stats.csv"), row.names = FALSE)

### Convergance plot
p2 <- ggplot(stats, aes(x = controlName, y=meanPCDiff)) +
  geom_point(shape = 15, colour = "steelblue", size = 2)+
  geom_errorbar(data = stats, aes(ymin=-SDEV,ymax=SDEV), width=0.2, colour = "steelblue") +
  scale_x_discrete(limits=rev(levels(stats$controlName))) + 
  geom_hline(yintercept=c(0)) +
  labs(x = NULL, y="Percentage Difference [+/- SDEV]", title = gsub("Region",region,"Region PopulationSim Controls Validation")) +
  coord_flip(ylim = c(-50, 50)) +
  theme_bw() +
  theme(plot.title=element_text(size=12, lineheight=.9, face="bold", vjust=1))

ggsave(file=paste(scenario, "PopulationSim Convergance-sdev.jpeg"), width=8,height=10)

### Convergance plot
p3 <- ggplot(stats, aes(x = controlName, y=meanPCDiff)) +
  geom_point(shape = 15, colour = "steelblue", size = 2)+
  geom_errorbar(data = stats, aes(ymin=-PRMSE,ymax=PRMSE), width=0.2, colour = "steelblue") +
  scale_x_discrete(limits=rev(levels(stats$controlName))) + 
  geom_hline(yintercept=c(0)) +
  labs(x = NULL, y="Percentage Difference [+/- PRMSE]", title = gsub("Region",region,"Region PopulationSim Controls Validation")) +
  coord_flip(ylim = c(-50, 50)) +
  theme_bw() +
  theme(plot.title=element_text(size=12, lineheight=.9, face="bold", vjust=1))

ggsave(file=paste(scenario, "PopulationSim Convergance-PRMSE.jpeg"), width=8,height=10)


### Uniformity Analysis
summary_hhid <- expanded_hhid %>% 
  mutate(FINALWEIGHT = 1) %>%
  select(FINALWEIGHT, expanded_hhid_col) %>%
  group_by(hh_id) %>%
  summarise(FINALWEIGHT = sum(FINALWEIGHT))
  
uniformity <- seed_households[seed_households$WGTP>0, seed_col] %>%
  left_join(summary_hhid, by = c("hhnum" = "hh_id")) %>%
  mutate(FINALWEIGHT = ifelse(is.na(FINALWEIGHT), 0, FINALWEIGHT)) %>%
  mutate(EXPANSIONFACTOR = FINALWEIGHT/WGTP) %>%
  mutate(EFBIN = cut(EXPANSIONFACTOR,seq(0,max(EXPANSIONFACTOR)+0.5,0.5),right=FALSE, include.lowest=FALSE))

uAnalysisPUMA <- group_by(uniformity, PUMA, EFBIN)

efPlotData <- summarise(uAnalysisPUMA, PC = n()) %>%
  mutate(PC=PC/sum(PC))

ggplot(efPlotData, aes(x=EFBIN, y=PC))  + 
  geom_bar(colour="black", fill="#DD8888", width=.7, stat="identity") + 
  guides(fill=FALSE) +
  xlab("RANGE OF EXPANSION FACTOR") + ylab("PERCENTAGE") +
  ggtitle("EXPANSION FACTOR DISTRIBUTION BY PUMA") + 
  facet_wrap(~PUMA, ncol=6) + 
  theme_bw()+
  theme(axis.title.x = element_text(face="bold"),
        axis.title.y = element_text(face="bold"),
        axis.text.x  = element_text(angle=90, size=5),
        axis.text.y  = element_text(size=5))  +
  scale_y_continuous(labels = percent_format())

ggsave("plots/EF-Distribution.png", width=15,height=10)

uAnalysisPUMA <- group_by(uniformity, PUMA)

uAnalysisPUMA <- summarize(uAnalysisPUMA
                           ,W = sum(WGTP)
                           ,Z = sum(FINALWEIGHT)
                           ,N = n()
                           ,EXP = Z/W
                           ,EXP_MIN = min(EXPANSIONFACTOR)
                           ,EXP_MAX = max(EXPANSIONFACTOR)
                           ,RMSE = myRMSE(EXPANSIONFACTOR, EXP, N))
write.csv(uAnalysisPUMA, "uniformity.csv", row.names=FALSE)


### Finish