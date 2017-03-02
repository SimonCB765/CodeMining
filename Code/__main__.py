"""Code to run the code mining."""

# Python imports.
import argparse
import datetime
import json
import logging
import logging.config
import os
import shutil
import sys

# User imports.
from CodeMining import code_mining
import DataProcessing
importedJsonSchema = False
try:
    from Libraries.JsonschemaManipulation import Configuration
    import jsonschema
    importedJsonSchema = True
except ImportError:
    from CodeMining import Configuration

# Set Python 2 alterations.
PYVERSION = sys.version_info[1]
if PYVERSION < 3:
    FileNotFoundError = IOError  # FileNotFoundError does not exist in version 2.

# ====================== #
# Create Argument Parser #
# ====================== #
parser = argparse.ArgumentParser(description="Identify clinical codes for concepts in a data-driven manner.",
                                 epilog="For further information see the README.")

# Mandatory arguments.
parser.add_argument("input", help="The location of the file containing the input data. If data processing is being "
                                  "run, then this should contain an unprocessed dataset. If no processing is being "
                                  "run, then this should contain a processed dataset.")
parser.add_argument("config", help="The location of the JSON file containing the configuration parameters to use. For "
                                   "information on the parameters that can be used please see the README.")

# Optional arguments.
parser.add_argument("-c", "--coding",
                    help="The location of the file containing the mapping between codes and their descriptions. "
                         "Default: a file called Coding.tsv in the Data directory.",
                    type=str)
parser.add_argument("-n", "--noProcess",
                    action="store_true",
                    help="Whether the data should be prevented from being processed. Default: data will be processed.")
parser.add_argument("-o", "--output",
                    help="The location of the directory to save the output to. Default: a timestamped "
                         "subdirectory in the Results directory.",
                    type=str)
parser.add_argument("-p", "--processed",
                    help="The location of the file to save the processed data in. Only used if data processing is "
                         "enabled. Default: a file called X_Processed.tsv in the Data directory, where X is the name"
                         "of the input file.",
                    type=str)
parser.add_argument("-w", "--overwrite",
                    action="store_true",
                    help="Whether the output directory should be overwritten. Default: do not overwrite.")

# ============================ #
# Parse and Validate Arguments #
# ============================ #
args = parser.parse_args()
dirCode = os.path.dirname(os.path.join(os.getcwd(), __file__))  # Directory containing this file.
dirTop = os.path.abspath(os.path.join(dirCode, os.pardir))
dirResults = os.path.join(dirTop, "Results")
dirOutput = os.path.join(dirResults, "{:s}".format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
dirOutput = args.output if args.output else dirOutput
dirConfiguration = os.path.join(dirTop, "ConfigurationFiles")
dirData = os.path.join(dirTop, "Data")
isErrors = False  # Whether any errors were found.

# Create the output directory.
overwrite = args.overwrite
if overwrite:
    try:
        shutil.rmtree(dirOutput)
    except FileNotFoundError:
        # Can't remove the directory as it doesn't exist.
        pass
    os.makedirs(dirOutput)  # Attempt to make the output directory.
else:
    try:
        os.makedirs(dirOutput)  # Attempt to make the output directory.
    except FileExistsError as e:
        # Directory already exists so can't continue.
        print("\nCan't continue as the output directory location already exists and overwriting is not enabled.\n")
        sys.exit()

# Create the logger. In order to do this we need to overwrite the value in the configuration information that records
# the location of the file that the logs are written to.
fileLoggerConfig = os.path.join(dirTop, "ConfigurationFiles", "Loggers.json")
fileLogOutput = os.path.join(dirOutput, "Logs.log")
logConfigInfo = json.load(open(fileLoggerConfig, 'r'))
logConfigInfo["handlers"]["file"]["filename"] = fileLogOutput
logConfigInfo["handlers"]["file_timed"]["filename"] = fileLogOutput
logging.config.dictConfig(logConfigInfo)
logger = logging.getLogger("__main__")

# Create the configuration object.
config = Configuration.Configuration()

if importedJsonSchema:
    # The package (jsonschema) needed to import the configuration parameters is present.

    # Validate the configuration parameters and set defaults.
    fileSchema = os.path.join(dirConfiguration, "ParamSchema.json")
    if not os.path.isfile(args.config):
        logger.error("The supplied location of the configuration file is not a file.")
        isErrors = True
    else:
        # Set user configuration parameters validated against the base schema.
        try:
            config.set_from_json(args.config, fileSchema)
        except jsonschema.SchemaError as e:
            exceptionInfo = sys.exc_info()
            logger.error(
                "The schema is not a valid JSON schema. Please correct any changes made to the "
                "schema or download the original and save it at {:s}.\n{:s}".format(fileSchema, str(exceptionInfo[1]))
            )
            isErrors = True
        except jsonschema.ValidationError as e:
            exceptionInfo = sys.exc_info()
            logger.error(
                "The configuration file is not valid against the schema.\n{:s}".format(str(exceptionInfo[1]))
            )
            isErrors = True
        except jsonschema.RefResolutionError as e:
            exceptionInfo = sys.exc_info()
            logger.error(
                "The schema contains an invalid reference. Please correct any changes made to the "
                "schema or download the original and save it at {:s}.\n{:s}".format(fileSchema, str(exceptionInfo[1]))
            )
            isErrors = True
else:
    # The jsonschema package is not present, so create a standalone configuration object.

    # Set the configuration parameters.
    fileDefaultParams = os.path.join(dirConfiguration, "DefaultParams.json")
    config.set_from_json(fileDefaultParams)
    config.set_from_json(args.config)

    # Validate that the user supplied a case definitions entry.
    if "CaseDefinitions" not in config._configParams:
        logger.error("The input parameter file does not contain any case definitions.")
        isErrors = True

# Validate that only one case definition (at most) contains no codes and add the collector case definition if there
# is only one case defined.
caseDefs = config.get_param(["CaseDefinitions"])[1]
emptyCaseDefs = [i for i, j in caseDefs.items() if not j]
if len(emptyCaseDefs) > 1:
    logger.error("There were multiple case definitions defined without codes, only one can be defined in this manner.")
    isErrors = True
elif len(caseDefs) == 1 and emptyCaseDefs:
    logger.error("The only case definition contains no codes. There must be at least one case defined with codes.")
    isErrors = True
elif len(caseDefs) == 1 and not emptyCaseDefs:
    # There was only one case defined, so add a collector case.
    caseDefs["RemainderCases"] = []
    config.set_param(["CaseDefinitions"], caseDefs, overwrite=True)

# Validate the input location.
fileInputData = args.input
if not os.path.isfile(fileInputData):
    logger.error("The location containing the input data is not a file.")
    isErrors = True

# Validate the code description mapping file.
fileCodeMap = args.coding if args.coding else os.path.join(dirData, "Coding.tsv")
if not os.path.isfile(fileCodeMap):
    logger.error("The location containing the mapping from codes to descriptions is not a file.")
    isErrors = True

# Validate the location to store the processed data.
fileProcessedData = args.processed
if not fileProcessedData:
    # No location to save the processed data was provided, so create the file to hold the processed data.
    fileInputName = os.path.splitext(os.path.basename(os.path.normpath(fileInputData)))[0]
    fileProcessedData = os.path.join(dirData, "{:s}_Processed.tsv".format(fileInputName))
else:
    if not overwrite:
        # Check that the provided file location does not exist.
        if os.path.exists(fileProcessedData):
            logger.error("The location provided for the processed data file already exists and overwriting is not "
                         "enabled.")
            isErrors = True
    else:
        # Check that the directory that the file is meant to be in exists.
        containingDir = os.path.dirname(fileProcessedData)
        try:
            os.makedirs(containingDir)
        except FileExistsError as e:
            # Directory already exists.
            pass

# Display errors if any were found.
if isErrors:
    print("\nErrors were encountered while validating the input arguments. Please see the log file for details.\n")
    sys.exit()

# Record a copy of the parameters used.
with open(os.path.join(dirOutput, "Parameters.txt"), 'w') as fidParams:
    for i, j in config._configParams.items():
        fidParams.write("{:s}\t{:s}\n".format(i, str(j)))

# =================== #
# Run the Code Mining #
# =================== #
if not args.noProcess:
    # The dataset needs processing.
    logger.info("Now processing the input dataset.")

    # Process the data.
    inputDataFormat = config.get_param(["FileFormat"])[1]
    if inputDataFormat == "code_count":
        # Validate that the user supplied column indices for the patient IDs, codes and counts.
        isErrors = []
        if "PatientColumn" not in config._configParams:
            logger.error("The input parameter file does not contain an index for the patient column.")
            isErrors = True
        if "CodeColumn" not in config._configParams:
            logger.error("The input parameter file does not contain an index for the code column.")
            isErrors = True
        if "CountColumn" not in config._configParams:
            logger.error("The input parameter file does not contain an index for the code count column.")
            isErrors = True

        # Display errors if any were found.
        if isErrors:
            print(
                "\nErrors were encountered while validating the input arguments. Please see the log file for details.\n"
            )
            sys.exit()

        # Run the processing.
        DataProcessing.CodeCount.process_dataset.main(
            fileInputData, fileProcessedData, config._configParams["PatientColumn"], config._configParams["CodeColumn"],
            config._configParams["CountColumn"], config._configParams["Delimiter"],
            config._configParams["ColsToUnbookend"]
        )
    elif inputDataFormat == "journal_table":
        DataProcessing.JournalTable.process_table.main(fileInputData, fileProcessedData)
    fileInputData = fileProcessedData

# Perform the code mining.
logger.info("Now performing code mining.")
code_mining.main(fileInputData, fileCodeMap, dirOutput, config)
