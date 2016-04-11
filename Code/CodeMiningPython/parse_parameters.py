"""Parses JSON parameter file for data cleaning and code mining."""

# Python imports.
import json
import numbers
import os
import sys

# User imports.
import parse_classes

# Globals
PYVERSION = sys.version_info[0]  # Determine major version number.


def main(fileParams):
    """Parse the user's parameter file.

    :param fileParams:  The user's parameter file.
    :type fileParams:   str
    :return :           The processed parameters.
    :rtype :            dict

    """

    parameters = {}  # Return value containing the processed parameters.
    errorsFound = []  # List recording all error messages to display.

    # Parse the JSON file of parameters.
    readParams = open(fileParams, 'r')
    parsedArgs = json.load(readParams)
    if PYVERSION == 2:
        parsedArgs = json_to_ascii(parsedArgs)  # Convert all unicode characters to ascii (only needed for Python < 3).
    readParams.close()

    #=====================================#
    # Parse the Data Cleaning Parameters. #
    #=====================================#
    if ("CleanerArgs" in parsedArgs) and parsedArgs["CleanerArgs"]:
        # Data cleaning arguments are present.
        isCleaning = True
        cleanerArgs = parsedArgs["CleanerArgs"]
        procCleanerArgs = {}  # The processed data cleaning arguments.

        #=======================#
        # Mandatory Parameters. #
        #=======================#
        # Check that the dirty data location exists.
        if "DirtyDataLocation" not in cleanerArgs:
            errorsFound.append("CLEANING: There must be a field called DirtyDataLocation in the CleanerArgs field entry.")
        elif not os.path.isfile(cleanerArgs["DirtyDataLocation"]):
            errorsFound.append("CLEANING: The dirty dataset location is not a file.")
        else:
            procCleanerArgs["DirtyDataLocation"] = cleanerArgs["DirtyDataLocation"]

        # Check that the location to write the cleaned data out to does not exist.
        isCleanDataLocValid = False
        if "CleanDataLocation" not in cleanerArgs:
            errorsFound.append("CLEANING: There must be a field called CleanDataLocation in the CleanerArgs field entry.")
        elif os.path.exists(cleanerArgs["CleanDataLocation"]):
            errorsFound.append("CLEANING: The location to save the cleaned dataset already exists.")
        else:
            isCleanDataLocValid = True
            procCleanerArgs["CleanDataLocation"] = cleanerArgs["CleanDataLocation"]

        #======================#
        # Optional Parameters. #
        #======================#
        procCleanerArgs["Delimiter"] = '\t'  # Default the delimiter to a tab.
        if "Delimiter" in cleanerArgs:
            # If a delimiter is specificed it must be a string.
            if not is_string(cleanerArgs["Delimiter"]):
                errorsFound.append("CLEANING: The parser delimiter must be a string.")
            else:
                procCleanerArgs["Delimiter"] = cleanerArgs["Delimiter"]

        procCleanerArgs["StripCommas"] = []  # Default to stripping commas from no columns.
        if "StripCommas" in cleanerArgs:
            # The list of columns to strip commas from must be contain only integers.
            if not all([isinstance(i, numbers.Integral) for i in cleanerArgs["StripCommas"]]):
                errorsFound.append("CLEANING: The parser columns to strip commas from must be a list of integers.")
            else:
                procCleanerArgs["StripCommas"] = cleanerArgs["StripCommas"]

        procCleanerArgs["RemoveCols"] = []  # Default to removing no columns.
        if "RemoveCols" in cleanerArgs:
            # The list of columns to remove must only contain integers (the column indices).
            if not all([isinstance(i, numbers.Integral) for i in cleanerArgs["RemoveCols"]]):
                errorsFound.append("CLEANING: The parser columns to remove must be a list of integers.")
            else:
                procCleanerArgs["RemoveCols"] = cleanerArgs["RemoveCols"]

        procCleanerArgs["Unbookend"] = {}  # Default to unbookending no columns.
        if "Unbookend" in cleanerArgs:
            for ind, i in enumerate(cleanerArgs["Unbookend"]):
                if not (("Column" in i) and ("String" in i) and ("IntOnly" in i)):
                    # Each object in "Unbookend" must contain these three names.
                    errorsFound.append("CLEANING: Entry number {0:d} in the list of parser columns to unbookend is "
                                       "missing information.".format(ind))
                elif (not isinstance(i["Column"], numbers.Integral)) or (not is_string(i["String"])) or \
                        (not isinstance(i["IntOnly"], bool)):
                    # Each object in "Unbookend" must contain these three fields formatted in this manner.
                    errorsFound.append("CLEANING: Entry number {0:d} in the list of parser columns to unbookend has "
                                       "incorrect types.".format(ind))
                else:
                    # This unbookend object (with index ind) is correctly formatted.
                    procCleanerArgs["Unbookend"][i["Column"]] = [i["String"], i["IntOnly"]]

        parameters["CleaningArgs"] = procCleanerArgs


    #===================================#
    # Parse the Code Mining Parameters. #
    #===================================#
    if ("MiningArgs" in parsedArgs) and parsedArgs["MiningArgs"]:
        # Code mining arguments are present.
        miningArgs = parsedArgs["MiningArgs"]
        procMiningArgs = {}  # The processed code mining arguments.

        #=======================#
        # Mandatory Parameters. #
        #=======================#
        # Check that the dataset location exists. The location is valid if it is a valid file, or it is a file
        # within the directory created from the data cleaning (and therefore doesn't exist yet).
        if "DataLocation" not in miningArgs:
            errorsFound.append("MINING: There must be a field called DataLocation in the MiningArgs field entry.")
        elif not os.path.isfile(miningArgs["DataLocation"]):
            if isCleaning and isCleanDataLocValid and (miningArgs["DataLocation"] == cleanerArgs["CleanDataLocation"]):
                # The dataset location doesn't exist currently, but will be created by the data cleaning.
                procMiningArgs["DataLocation"] = miningArgs["DataLocation"]
            else:
                # The dataset location doesn't exist currently, and will not be created by the data cleaning
                # (possibly due to errors in the cleaning parameters).
                errorsFound.append("MINING: The dataset for the mining does not exist, and won't be created by the "
                                   "data cleaning.")
        else:
            procMiningArgs["DataLocation"] = miningArgs["DataLocation"]

        # Check that the code mapping exists.
        if "CodeMapping" not in miningArgs:
            errorsFound.append("MINING: There must be a field called CodeMapping in the MiningArgs field entry.")
        elif not os.path.isfile(miningArgs["CodeMapping"]):
            errorsFound.append("MINING: The location given for the code mapping is not a file.")
        else:
            procMiningArgs["CodeMapping"] = miningArgs["CodeMapping"]

        # Check that the results directory is valid.
        if "ResultsLocation" not in miningArgs:
            errorsFound.append("MINING: There must be a field called ResultsLocation in the MiningArgs field entry.")
        elif os.path.exists(miningArgs["ResultsLocation"]) and not os.path.isdir(miningArgs["ResultsLocation"]):
            errorsFound.append("MINING: The code mining results directory location exists but is not a directory.")
        else:
            procMiningArgs["ResultsLocation"] = miningArgs["ResultsLocation"]

        # Check that the class definition is valid.
        if "Classes" not in miningArgs:
            errorsFound.append("MINING: There must be a field called Classes in the MiningArgs field entry.")
        else:
            classValidity = parse_classes.check_validity(miningArgs["Classes"])
            if not classValidity[0]:
                errorsFound.append(classValidity[1])
            else:
                procMiningArgs["Classes"] = miningArgs["Classes"]

        #======================#
        # Optional Parameters. #
        #======================#
        procMiningArgs["Lambda"] = [0.01]  # Default to only using a lambda value of 0.01.
        if "Lambda" in miningArgs:
            # The list of lambda values must contain only floats and integers.
            if not all([isinstance(i, numbers.Real) for i in miningArgs["Lambda"]]):
                errorsFound.append("MINING: The lambda values must be a list of floats.")
            else:
                procMiningArgs["Lambda"] = miningArgs["Lambda"]

        procMiningArgs["ElasticNetMixing"] = [0.15]  # Default to only using an alpha value of 0.1.
        if "ElasticNetMixing" in miningArgs:
            # The list of alpha values must contain only floats and integers.
            if not all([isinstance(i, numbers.Real) for i in miningArgs["Alpha"]]):
                errorsFound.append("MINING: The elastic net mixing values must be a list of floats.")
            else:
                procMiningArgs["Alpha"] = miningArgs["Alpha"]

        procMiningArgs["BatchSize"] = [500]  # Default to only using a batch size of 500.
        if "BatchSize" in miningArgs:
            # The list of batch sizes must contain only integers.
            if not all([isinstance(i, numbers.Integral) for i in miningArgs["BatchSize"]]):
                errorsFound.append("MINING: The batch size values must be a list of integers.")
            else:
                procMiningArgs["BatchSize"] = miningArgs["BatchSize"]

        procMiningArgs["MaxIter"] = [5]  # Default to only using a maximum iteration value of 10.
        if "MaxIter" in miningArgs:
            if not all([isinstance(i, numbers.Integral) for i in miningArgs["MaxIter"]]):
                errorsFound.append("MINING: The maximum iteration values must be a list of integers.")
            else:
                procMiningArgs["MaxIter"] = miningArgs["MaxIter"]

        procMiningArgs["CodeOccurrences"] = 0  # Default to accepting all codes.
        if "CodeOccurrences" in miningArgs:
            # The minimum number of patients a code needs to occur with must be an integer.
            if not isinstance(miningArgs["CodeOccurrences"], numbers.Integral):
                errorsFound.append("The minimum number of patients a code needs to occur with must be an integer.")
            else:
                procMiningArgs["CodeOccurrences"] = miningArgs["CodeOccurrences"]

        procMiningArgs["PatientOccurrences"] = 0  # Default to accepting all patients.
        if "PatientOccurrences" in miningArgs:
            # The minimum number of codes a patient needs to occur with must be an integer.
            if not isinstance(miningArgs["PatientOccurrences"], numbers.Integral):
                errorsFound.append("The minimum number of codes a patient needs to occur with must be an integer.")
            else:
                procMiningArgs["PatientOccurrences"] = miningArgs["PatientOccurrences"]

        procMiningArgs["CVFolds"] = 0  # Default to not using cross validation.
        if "CVFolds" in miningArgs:
            # The number of cross validation folds to use must be an integer.
            if not isinstance(miningArgs["CVFolds"], numbers.Integral):
                errorsFound.append("The number of cross validation folds must be an integer.")
            else:
                procMiningArgs["CVFolds"] = miningArgs["CVFolds"]

        procMiningArgs["DataNorm"] = 0  # Default to using no normalisation.
        if "DataNorm" in miningArgs:
            # The data normalisation value must be an integer.
            if not isinstance(miningArgs["DataNorm"], numbers.Integral):
                errorsFound.append("The data normalisation value must be an integer.")
            else:
                procMiningArgs["DataNorm"] = miningArgs["DataNorm"]

        procMiningArgs["DiscardThreshold"] = 0  # Default to not discarding any patients.
        if "DiscardThreshold" in miningArgs:
            # The discard threshold must be a float.
            if not isinstance(miningArgs["DiscardThreshold"], numbers.Real):
                errorsFound.append("The discard threshold must be a float.")
            else:
                procMiningArgs["DiscardThreshold"] = miningArgs["DiscardThreshold"]

        parameters["MiningArgs"] = procMiningArgs

    #=================#
    # Display Errors. #
    #=================#
    if errorsFound:
        print("\n\nThe following errors were encountered while parsing the input parameters:\n")
        print('\n'.join(errorsFound))
        sys.exit()

    return parameters


def is_string(objectToCheck):
    """Determine whether objectToCheck is a string in both Python 2.x and 3.x."""

    return isinstance(objectToCheck, str) if (PYVERSION == 3) else isinstance(objectToCheck, basestring)


def json_to_ascii(jsonObject):
    """Convert a JSON object from unicode to ascii strings.

    This will recurse through all levels of the JSON dictionary, and therefore may hit Python's recursion limit.
    To avoid this use object_hook in the json.load() function instead.

    """

    print(jsonObject)

    if isinstance(jsonObject, dict):
        # If the current part of the JSON oobject is a dictionary, then make all its keys and values ascii if needed.
        return dict([(json_to_ascii(key), json_to_ascii(value)) for key, value in jsonObject.iteritems()])
    elif isinstance(jsonObject, list):
        # If the current part of the JSON object is a list, then make all its elements ascii if needed.
        return [json_to_ascii(i) for i in jsonObject]
    elif isinstance(jsonObject, unicode):
        # If you've reached a unicode string convert it to ascii.
        return jsonObject.encode('utf-8')
    else:
        # You've reached a non-unicode terminus (e.g. an integer or null).
        return jsonObject