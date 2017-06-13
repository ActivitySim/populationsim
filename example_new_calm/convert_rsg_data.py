# convert CALM popsim inputs to ActivitySim format
# Ben Stabler, ben.stabler@rsginc.com, 03/09/17
# https://github.com/RSGInc/populationsim/wiki/Software-Design
# low -> CALM TAZ System
# mid -> Census Tract
# meta -> PUMA / Region

import sys, os.path
import pandas as pd, numpy as np

pd.options.mode.chained_assignment = None  # turn off SettingWithCopyWarning

# settings
input_folder = "./data_rsg_raw/"
output_folder = "./data_rsg/"
output_datastore_fname = "./data_rsg/populationsim.h5"

if __name__ == "__main__":

    print("convert CALM popsim inputs to ActivitySim format")

    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("read input CSVs")

    seed_households = pd.read_csv(input_folder + "ss10hor.csv")
    seed_persons = pd.read_csv(input_folder + "ss10por.csv", low_memory=False)

    mid_control_data = pd.read_csv(input_folder + "CALMtractData.csv")
    low_control_data = pd.read_csv(input_folder + "CALMtazData.csv")
    osu_low_data = pd.read_csv(input_folder + "OSU.csv")

    gwalk = pd.read_csv(input_folder + "geographicCwalk.csv")

    print("seed household data processing")

    # delete records of Vacant Units and Group Quarters
    seed_households.index = seed_households.SERIALNO
    seed_households = seed_households[seed_households.NP != 0]  # Vacant Units
    seed_households = seed_households[seed_households.TYPE <= 1]  # Group Quarters

    # Setting the housing type using the Units in structure [BLD] attribute
    seed_households["HTYPE"] = 0
    seed_households.HTYPE[seed_households.BLD == 2] = 1  # Single-family
    seed_households.HTYPE[np.in1d(seed_households.BLD, [4, 5, 6, 7, 8, 9])] = 2  # Multi-family
    seed_households.HTYPE[np.in1d(seed_households.BLD, [1, 10])] = 3  # Mobile-home
    seed_households.HTYPE[seed_households.BLD == 3] = 4  # Duplex
    seed_households.HTYPE[
        np.invert(np.in1d(seed_households.BLD, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))] = 999  # Other

    # Setting number of workers based on Employment Status Recode [ESR] attribute in PUMS Person File
    num_workers = seed_persons[np.in1d(seed_persons.ESR, [1, 2, 4, 5])].SERIALNO.value_counts()
    num_workers.name = "NWESR"
    seed_households = seed_households.join(num_workers)
    seed_households.NWESR[pd.isnull(seed_households.NWESR)] = 0

    # Setting number of university students based on Grade level attending [SCHG] attribute in PUMS Person File
    nuniv_schg = seed_persons[np.in1d(seed_persons.SCHG, [6, 7])].SERIALNO.value_counts()
    nuniv_schg.name = "NUNIV_SCHG"
    seed_households = seed_households.join(nuniv_schg)
    seed_households.NUNIV_SCHG[pd.isnull(seed_households.NUNIV_SCHG)] = 0

    # Adjusting the household income to 2010 inflation adjusted dollars
    # 2010$s = (Reported income)*(Rolling reference factor for the year)*(Inflation adjustment) * inflation adj (2010 to 1995) MM
    seed_households["HHINCADJ"] = 999
    seed_households.HHINCADJ[seed_households.ADJINC == 1098342] = ((seed_households.HINCP[
                                                                        seed_households.ADJINC == 1098342] / 1.0) * 1.015675 * 1.08139 * 0.704342)
    seed_households.HHINCADJ[seed_households.ADJINC == 1069212] = ((seed_households.HINCP[
                                                                        seed_households.ADJINC == 1069212] / 1.0) * 1.016787 * 1.05156 * 0.704342)
    seed_households.HHINCADJ[seed_households.ADJINC == 1031272] = ((seed_households.HINCP[
                                                                        seed_households.ADJINC == 1031272] / 1.0) * 1.018389 * 1.01265 * 0.704342)
    seed_households.HHINCADJ[seed_households.ADJINC == 1015979] = ((seed_households.HINCP[
                                                                        seed_households.ADJINC == 1015979] / 1.0) * 1.999480 * 1.01651 * 0.704342)
    seed_households.HHINCADJ[seed_households.ADJINC == 1007624] = ((seed_households.HINCP[
                                                                        seed_households.ADJINC == 1007624] / 1.0) * 1.007624 * 1.00000 * 0.704342)

    #Setting age of head based on PUMS Person File
    agehoh = seed_persons[seed_persons.SPORDER==1].groupby(seed_persons.SERIALNO).AGEP.min()
    agehoh.name = "AGEHOH"
    seed_households = seed_households.join(agehoh)
    seed_households.AGEHOH[pd.isnull(seed_households.AGEHOH)] = 0 
    
    print("seed person data processing")

    # Deleting all GQ person records (i.e. have no household record)
    uniq_hhs = seed_households.SERIALNO.value_counts()
    seed_persons = seed_persons.join(uniq_hhs, on="SERIALNO", rsuffix="_RIGHT")
    seed_persons = seed_persons[pd.notnull(seed_persons.SERIALNO_RIGHT)]
    del seed_persons['SERIALNO_RIGHT']

    # Setting person weight as HH weight for use in the initial iteration of balancing
    seed_persons = seed_persons.join(seed_households["WGTP"], on="SERIALNO")

    # Determining employment status of the person based on employment status recode
    seed_persons["EMPLOYED"] = 0
    seed_persons.EMPLOYED[np.in1d(seed_persons.ESR, [1, 2, 4, 5])] = 1

    # Filtering out the SOC major group
    seed_persons["SOC"] = ''
    seed_persons.SOC[seed_persons.EMPLOYED == 0] = 0
    seed_persons.SOC[seed_persons.EMPLOYED != 0] = seed_persons.socp00[
        seed_persons.EMPLOYED != 0].map(lambda x: str(x).strip()[:2])
    seed_persons.SOC[seed_persons.SOC == 'N.'] = seed_persons.socp10[seed_persons.SOC == 'N.'].map(
        lambda x: str(x).strip()[:2])
    seed_persons.SOC = seed_persons.SOC.astype(str).astype(int)  # convert to int for later test

    # Setting the occupation category for the person based on SOC major group
    seed_persons["OCCP"] = 999
    seed_persons.OCCP[np.in1d(seed_persons.SOC, [11, 13, 15, 17, 27, 19, 39])] = 1
    seed_persons.OCCP[np.in1d(seed_persons.SOC, [21, 23, 25, 29, 31])] = 2
    seed_persons.OCCP[np.in1d(seed_persons.SOC, [35, 37])] = 3
    seed_persons.OCCP[np.in1d(seed_persons.SOC, [41, 43])] = 4
    seed_persons.OCCP[np.in1d(seed_persons.SOC, [45, 47, 49])] = 5
    seed_persons.OCCP[np.in1d(seed_persons.SOC, [51, 53])] = 6
    seed_persons.OCCP[np.in1d(seed_persons.SOC, [55])] = 7
    seed_persons.OCCP[np.in1d(seed_persons.SOC, [33])] = 8

    print("explode seed population for university students")

    # Setting the family type tag based on Number of persons in family [NPF] attribute
    seed_households["FAMTAG"] = 0
    seed_households.FAMTAG[pd.notnull(seed_households.NPF)] = 1

    # Setting person famTag status based on the HH famTag status
    seed_persons = seed_persons.join(seed_households["FAMTAG"], on="SERIALNO")

    # Flagging university student records to duplicate
    # Halving the weights for the households with university students
    seed_households["DUPCOUNT"] = 1
    seed_households.DUPCOUNT[seed_households.NUNIV_SCHG > 0] = 2
    seed_households.WGTP = np.round(seed_households.WGTP / seed_households.DUPCOUNT, 0)

    # Setting person records duplication flag [dupcount] based on the HH dupcount status
    seed_persons = seed_persons.join(seed_households["DUPCOUNT"], on="SERIALNO")

    # Duplicating the university student household records
    maxSERIALNO = max(seed_households.SERIALNO)
    extra_households = seed_households[seed_households.DUPCOUNT == 2]
    extra_households["NEWSERIALNO"] = range(maxSERIALNO + 1,
                                            maxSERIALNO + 1 + len(extra_households))

    # Duplicating the university student person records
    extra_persons = seed_persons[seed_persons.DUPCOUNT == 2]
    extra_persons = extra_persons.join(extra_households["NEWSERIALNO"], on="SERIALNO")
    extra_households = extra_households.set_index(extra_households.NEWSERIALNO, drop=False)
    extra_persons = extra_persons.set_index(extra_persons.NEWSERIALNO, drop=False)

    # Setting OSU indicator for half the university records
    extra_persons["OSUTAG"] = 0
    extra_persons.OSUTAG[(np.in1d(extra_persons.SCHG, [6, 7])) & (
    extra_persons.FAMTAG == 1)] = 1  # OSU student record in a family HH
    extra_persons.OSUTAG[(np.in1d(extra_persons.SCHG, [6, 7])) & (
    extra_persons.FAMTAG == 0)] = 2  # OSU student record in a non-family HH
    seed_persons["OSUTAG"] = 0

    # append university student records
    extra_households.SERIALNO = extra_households.NEWSERIALNO
    extra_persons.SERIALNO = extra_persons.NEWSERIALNO
    del extra_households["NEWSERIALNO"]
    del extra_persons["NEWSERIALNO"]
    seed_households = seed_households.append(extra_households)
    seed_persons = seed_persons.append(extra_persons)

    print("seed data cleaning for popsyn")

    # Generating household and person ID for use in PopSyn
    seed_households["HHNUM"] = range(1, len(seed_households) + 1)
    seed_persons = seed_persons.join(seed_households["HHNUM"], on="SERIALNO")

    # Converting NULLs to 0
    seed_households.WIF[pd.isnull(seed_households.WIF)] = 0
    seed_households.NWESR[pd.isnull(seed_households.NWESR)] = 0
    seed_persons.OCCP[pd.isnull(seed_persons.OCCP)] = 0
    seed_households.AGS[pd.isnull(seed_households.AGS)] = 0

    print("process low level control data")

    # rename fields
    low_control_data["HHSIZE1"] = low_control_data.HHS1BASE
    low_control_data["HHSIZE2"] = low_control_data.HHS2BASE
    low_control_data["HHSIZE3"] = low_control_data.HHS3BASE
    low_control_data["HHSIZE4"] = low_control_data.HHS4BASE

    low_control_data["HHAGE1"] = low_control_data.AGE1BASE
    low_control_data["HHAGE2"] = low_control_data.AGE2BASE
    low_control_data["HHAGE3"] = low_control_data.AGE3BASE
    low_control_data["HHAGE4"] = low_control_data.AGE4BASE

    low_control_data["HHINC1"] = low_control_data.HHI1BASE
    low_control_data["HHINC2"] = low_control_data.HHI2BASE
    low_control_data["HHINC3"] = low_control_data.HHI3BASE
    low_control_data["HHINC4"] = low_control_data.HHI4BASE

    del low_control_data["HHS1BASE"]
    del low_control_data["HHS2BASE"]
    del low_control_data["HHS3BASE"]
    del low_control_data["HHS4BASE"]

    del low_control_data["AGE1BASE"]
    del low_control_data["AGE2BASE"]
    del low_control_data["AGE3BASE"]
    del low_control_data["AGE4BASE"]

    del low_control_data["HHI1BASE"]
    del low_control_data["HHI2BASE"]
    del low_control_data["HHI3BASE"]
    del low_control_data["HHI4BASE"]

    # join osu fields
    osu_low_data = osu_low_data.set_index("TAZ")
    osu_low_data = osu_low_data.rename(columns={'family': 'OSUFAM', 'nonfamily': 'OSUNFAM'})
    low_control_data = low_control_data.set_index("TAZ")
    low_control_data = low_control_data.join(osu_low_data)

    # join geo data
    gwalk = gwalk.set_index("TAZ")
    gwalk = gwalk.rename(columns={'GEOID': 'TRACTGEOID'})
    low_control_data = low_control_data.join(gwalk)

    print("process mid level control data")

    # mid_control_data
    mid_control_data["SF"] = mid_control_data["OOHUDETACHED"] + mid_control_data["ROHUDETACHED"]
    mid_control_data["DUP"] = mid_control_data["OOHUATTACHED"] + mid_control_data["ROHUATTACHED"]
    mid_control_data["MF"] = mid_control_data[
        ["OOHUUB2", "OOHUUB3TO4", "OOHUUB5TO9", "OOHUUB10TO19", "OOHUUB20TO49", "OOHUUB50PLUS",
         "ROHUUB2", "ROHUUB3TO4", "ROHUUB5TO9", "ROHUUB10TO19", "ROHUUB20TO49",
         "ROHUUB50PLUS"]].sum(axis=1)
    mid_control_data["MH"] = mid_control_data[
        ["OOHUUBMH", "OOHUUBBOATRV", "ROHUUBMH", "ROHUUBBOATRV"]].sum(axis=1)
    mid_control_data = mid_control_data.set_index(
        [mid_control_data["COUNTY"], mid_control_data["TRACT"]])

    mid_control_data["PCHHWORK0"] = mid_control_data["HHWORK0"] / mid_control_data["TOTHH"]
    mid_control_data["PCHHWORK1"] = mid_control_data["HHWORK1"] / mid_control_data["TOTHH"]
    mid_control_data["PCHHWORK2"] = mid_control_data["HHWORK2"] / mid_control_data["TOTHH"]
    mid_control_data["PCHHWORK3"] = mid_control_data["HHWORK3"] / mid_control_data["TOTHH"]
    mid_control_data["PCSF"] = mid_control_data["SF"] / mid_control_data["TOTHH"]
    mid_control_data["PCDUP"] = mid_control_data["DUP"] / mid_control_data["TOTHH"]
    mid_control_data["PCMF"] = mid_control_data["MF"] / mid_control_data["TOTHH"]
    mid_control_data["PCMH"] = mid_control_data["MH"] / mid_control_data["TOTHH"]

    low_control_data_agg = pd.DataFrame()

    low_control_data_agg["TRACTGEOID"] = low_control_data.groupby("TRACTGEOID").TRACTGEOID.min()
    low_control_data_agg["TRACT"] = low_control_data.groupby("TRACTGEOID").TRACTCE.min()
    low_control_data_agg["COUNTY"] = low_control_data.groupby("TRACTGEOID").COUNTYFP.min()
    low_control_data_agg["HHBASE"] = low_control_data.groupby("TRACTGEOID").HHBASE.sum()
    low_control_data_agg["POPBASE"] = low_control_data.groupby("TRACTGEOID").POPBASE.sum()
    low_control_data_agg = low_control_data_agg.set_index(
        [low_control_data_agg["COUNTY"], low_control_data_agg["TRACT"]])

    mid_control_data = mid_control_data.join(low_control_data_agg["HHBASE"])
    mid_control_data = mid_control_data.join(low_control_data_agg["POPBASE"])
    mid_control_data = mid_control_data.join(low_control_data_agg["TRACTGEOID"])

    mid_control_data["HHWORK0"] = mid_control_data["PCHHWORK0"] * mid_control_data["HHBASE"]
    mid_control_data["HHWORK1"] = mid_control_data["PCHHWORK1"] * mid_control_data["HHBASE"]
    mid_control_data["HHWORK2"] = mid_control_data["PCHHWORK2"] * mid_control_data["HHBASE"]
    mid_control_data["HHWORK3"] = mid_control_data["PCHHWORK3"] * mid_control_data["HHBASE"]
    mid_control_data["SF"] = mid_control_data["PCSF"] * mid_control_data["HHBASE"]
    mid_control_data["DUP"] = mid_control_data["PCDUP"] * mid_control_data["HHBASE"]
    mid_control_data["MF"] = mid_control_data["PCMF"] * mid_control_data["HHBASE"]
    mid_control_data["MH"] = mid_control_data["PCMH"] * mid_control_data["HHBASE"]

    # check
    mid_control_data["HHW"] = mid_control_data["HHWORK0"] + mid_control_data["HHWORK1"] + \
                              mid_control_data["HHWORK2"] + mid_control_data["HHWORK3"]
    mid_control_data["HHUB"] = mid_control_data["SF"] + mid_control_data["DUP"] + mid_control_data[
        "MF"] + mid_control_data["MH"]
    mid_control_data["CHECKW"] = mid_control_data["HHW"] - mid_control_data["HHBASE"]
    mid_control_data["CHECKUB"] = mid_control_data["HHUB"] - mid_control_data["HHBASE"]
    mid_control_data["HHWORK0"] = mid_control_data["HHWORK0"] - mid_control_data["CHECKW"]
    mid_control_data["SF"] = mid_control_data["SF"] - mid_control_data["CHECKUB"]

    print("process meta level control data")

    meta_control_data = pd.DataFrame()

    meta_control_data["PUMSPOP"] = seed_persons.groupby("PUMA").WGTP.sum()

    meta_control_data["AGE1"] = seed_persons[
        (seed_persons.AGEP >= 0) & (seed_persons.AGEP <= 5)].groupby("PUMA").WGTP.sum()
    meta_control_data["AGE2"] = seed_persons[
        (seed_persons.AGEP >= 6) & (seed_persons.AGEP <= 12)].groupby("PUMA").WGTP.sum()
    meta_control_data["AGE3"] = seed_persons[
        (seed_persons.AGEP >= 13) & (seed_persons.AGEP <= 15)].groupby("PUMA").WGTP.sum()
    meta_control_data["AGE4"] = seed_persons[
        (seed_persons.AGEP >= 16) & (seed_persons.AGEP <= 17)].groupby("PUMA").WGTP.sum()
    meta_control_data["AGE5"] = seed_persons[
        (seed_persons.AGEP >= 18) & (seed_persons.AGEP <= 24)].groupby("PUMA").WGTP.sum()
    meta_control_data["AGE6"] = seed_persons[
        (seed_persons.AGEP >= 25) & (seed_persons.AGEP <= 34)].groupby("PUMA").WGTP.sum()
    meta_control_data["AGE7"] = seed_persons[
        (seed_persons.AGEP >= 35) & (seed_persons.AGEP <= 44)].groupby("PUMA").WGTP.sum()
    meta_control_data["AGE8"] = seed_persons[
        (seed_persons.AGEP >= 45) & (seed_persons.AGEP <= 54)].groupby("PUMA").WGTP.sum()
    meta_control_data["AGE9"] = seed_persons[
        (seed_persons.AGEP >= 55) & (seed_persons.AGEP <= 64)].groupby("PUMA").WGTP.sum()
    meta_control_data["AGE10"] = seed_persons[
        (seed_persons.AGEP >= 65) & (seed_persons.AGEP <= 74)].groupby("PUMA").WGTP.sum()
    meta_control_data["AGE11"] = seed_persons[
        (seed_persons.AGEP >= 75) & (seed_persons.AGEP <= 84)].groupby("PUMA").WGTP.sum()
    meta_control_data["AGE12"] = seed_persons[(seed_persons.AGEP >= 85)].groupby("PUMA").WGTP.sum()

    meta_control_data["OCCP1"] = seed_persons[seed_persons.OCCP == 1].groupby("PUMA").WGTP.sum()
    meta_control_data["OCCP2"] = seed_persons[seed_persons.OCCP == 2].groupby("PUMA").WGTP.sum()
    meta_control_data["OCCP3"] = seed_persons[seed_persons.OCCP == 3].groupby("PUMA").WGTP.sum()
    meta_control_data["OCCP4"] = seed_persons[seed_persons.OCCP == 4].groupby("PUMA").WGTP.sum()
    meta_control_data["OCCP5"] = seed_persons[seed_persons.OCCP == 5].groupby("PUMA").WGTP.sum()
    meta_control_data["OCCP6"] = seed_persons[seed_persons.OCCP == 6].groupby("PUMA").WGTP.sum()
    meta_control_data["OCCP7"] = seed_persons[seed_persons.OCCP == 7].groupby("PUMA").WGTP.sum()
    meta_control_data["OCCP8"] = seed_persons[seed_persons.OCCP == 8].groupby("PUMA").WGTP.sum()
    meta_control_data["OTH"] = seed_persons[seed_persons.OCCP == 999].groupby("PUMA").WGTP.sum()

    meta_control_data["PCAGE1"] = meta_control_data.AGE1 / meta_control_data.PUMSPOP
    meta_control_data["PCAGE2"] = meta_control_data.AGE2 / meta_control_data.PUMSPOP
    meta_control_data["PCAGE3"] = meta_control_data.AGE3 / meta_control_data.PUMSPOP
    meta_control_data["PCAGE4"] = meta_control_data.AGE4 / meta_control_data.PUMSPOP
    meta_control_data["PCAGE5"] = meta_control_data.AGE5 / meta_control_data.PUMSPOP
    meta_control_data["PCAGE6"] = meta_control_data.AGE6 / meta_control_data.PUMSPOP
    meta_control_data["PCAGE7"] = meta_control_data.AGE7 / meta_control_data.PUMSPOP
    meta_control_data["PCAGE8"] = meta_control_data.AGE8 / meta_control_data.PUMSPOP
    meta_control_data["PCAGE9"] = meta_control_data.AGE9 / meta_control_data.PUMSPOP
    meta_control_data["PCAGE10"] = meta_control_data.AGE10 / meta_control_data.PUMSPOP
    meta_control_data["PCAGE11"] = meta_control_data.AGE11 / meta_control_data.PUMSPOP
    meta_control_data["PCAGE12"] = meta_control_data.AGE12 / meta_control_data.PUMSPOP

    meta_control_data["PCOCCP1"] = meta_control_data.OCCP1 / meta_control_data.PUMSPOP
    meta_control_data["PCOCCP2"] = meta_control_data.OCCP2 / meta_control_data.PUMSPOP
    meta_control_data["PCOCCP3"] = meta_control_data.OCCP3 / meta_control_data.PUMSPOP
    meta_control_data["PCOCCP4"] = meta_control_data.OCCP4 / meta_control_data.PUMSPOP
    meta_control_data["PCOCCP5"] = meta_control_data.OCCP5 / meta_control_data.PUMSPOP
    meta_control_data["PCOCCP6"] = meta_control_data.OCCP6 / meta_control_data.PUMSPOP
    meta_control_data["PCOCCP7"] = meta_control_data.OCCP7 / meta_control_data.PUMSPOP
    meta_control_data["PCOCCP8"] = meta_control_data.OCCP8 / meta_control_data.PUMSPOP
    meta_control_data["PCOTH"] = meta_control_data.OTH / meta_control_data.PUMSPOP

    meta_control_data["POPBASE"] = low_control_data.groupby("PUMA").POPBASE.sum()
    meta_control_data["REGION"] = low_control_data.groupby("PUMA").REGION.min()

    meta_control_data["AGE1"] = np.round(meta_control_data["POPBASE"] * meta_control_data["PCAGE1"],
                                         0)
    meta_control_data["AGE2"] = np.round(meta_control_data["POPBASE"] * meta_control_data["PCAGE2"],
                                         0)
    meta_control_data["AGE3"] = np.round(meta_control_data["POPBASE"] * meta_control_data["PCAGE3"],
                                         0)
    meta_control_data["AGE4"] = np.round(meta_control_data["POPBASE"] * meta_control_data["PCAGE4"],
                                         0)
    meta_control_data["AGE5"] = np.round(meta_control_data["POPBASE"] * meta_control_data["PCAGE5"],
                                         0)
    meta_control_data["AGE6"] = np.round(meta_control_data["POPBASE"] * meta_control_data["PCAGE6"],
                                         0)
    meta_control_data["AGE7"] = np.round(meta_control_data["POPBASE"] * meta_control_data["PCAGE7"],
                                         0)
    meta_control_data["AGE8"] = np.round(meta_control_data["POPBASE"] * meta_control_data["PCAGE8"],
                                         0)
    meta_control_data["AGE9"] = np.round(meta_control_data["POPBASE"] * meta_control_data["PCAGE9"],
                                         0)
    meta_control_data["AGE10"] = np.round(
        meta_control_data["POPBASE"] * meta_control_data["PCAGE10"], 0)
    meta_control_data["AGE11"] = np.round(
        meta_control_data["POPBASE"] * meta_control_data["PCAGE11"], 0)
    meta_control_data["AGE12"] = np.round(
        meta_control_data["POPBASE"] * meta_control_data["PCAGE12"], 0)

    meta_control_data["OCCP1"] = np.round(
        meta_control_data["POPBASE"] * meta_control_data["PCOCCP1"], 0)
    meta_control_data["OCCP2"] = np.round(
        meta_control_data["POPBASE"] * meta_control_data["PCOCCP2"], 0)
    meta_control_data["OCCP3"] = np.round(
        meta_control_data["POPBASE"] * meta_control_data["PCOCCP3"], 0)
    meta_control_data["OCCP4"] = np.round(
        meta_control_data["POPBASE"] * meta_control_data["PCOCCP4"], 0)
    meta_control_data["OCCP5"] = np.round(
        meta_control_data["POPBASE"] * meta_control_data["PCOCCP5"], 0)
    meta_control_data["OCCP6"] = np.round(
        meta_control_data["POPBASE"] * meta_control_data["PCOCCP6"], 0)
    meta_control_data["OCCP7"] = np.round(
        meta_control_data["POPBASE"] * meta_control_data["PCOCCP7"], 0)
    meta_control_data["OCCP8"] = np.round(
        meta_control_data["POPBASE"] * meta_control_data["PCOCCP8"], 0)
    meta_control_data["NLF"] = np.round(meta_control_data["POPBASE"] * meta_control_data["PCOTH"],
                                        0)

    # only PUMAs that have zones
    meta_control_data = meta_control_data[meta_control_data.POPBASE > 0]

    print("write output HDF5 file")

    # create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # remove mixed type fields
    del seed_persons["RT"]
    del seed_persons["indp02"]
    del seed_persons["naicsp02"]
    del seed_persons["occp02"]
    del seed_persons["socp00"]
    del seed_persons["occp10"]
    del seed_persons["socp10"]
    del seed_persons["indp07"]
    del seed_persons["naicsp07"]

    # output hdf5 datastore
    seed_households.to_hdf(output_datastore_fname, "seed_households")
    seed_persons.to_hdf(output_datastore_fname, "seed_persons")
    low_control_data.to_hdf(output_datastore_fname, "low_control_data")
    mid_control_data.to_hdf(output_datastore_fname, "mid_control_data")
    meta_control_data.to_hdf(output_datastore_fname, "meta_control_data")
    gwalk.to_hdf(output_datastore_fname, "geo_cross_walk")

    print("write output CSVs")

    seed_households = seed_households[seed_households.PUMA==600]

    hh_ids = seed_households.SERIALNO.values
    seed_persons = seed_persons[seed_persons.SERIALNO.isin(hh_ids)]


    seed_households.to_csv(output_folder + "seed_households.csv", index=False)
    seed_persons.to_csv(output_folder + "seed_persons.csv", index=False)
    low_control_data.to_csv(output_folder + "low_control_data.csv", index=True)
    mid_control_data.to_csv(output_folder + "mid_control_data.csv", index=True)
    meta_control_data.to_csv(output_folder + "meta_control_data.csv", index=True)
    gwalk.to_csv(output_folder + "geo_cross_walk.csv", index=True)
