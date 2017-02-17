"""Code to run the code mining."""

# Python imports.
import argparse
import logging
import logging.config
import os
import shutil
import sys

# User imports.





# ====================== #
# Create Argument Parser #
# ====================== #
parser = argparse.ArgumentParser(description="Analyse a collection of medical records.",
                                 epilog="For further information see the README.")

# Mandatory arguments.
parser.add_argument("input", help="The location of the input files. The specified location should be appropriate for "
                                  "the processing that will be carried out.")

# Optional arguments.
parser.add_argument("-c", "--config",
                    help="The location of the file containing the configuration parameters to use. "
                         "Default: a file called DefaultConfig.json in the ConfigurationFiles directory.",
                    type=str)
parser.add_argument("-e", "--encode",
                    default=None,
                    help="The encoding to convert strings in the JSON configuration file to. Default: no "
                         "conversion performed.",
                    type=str)
parser.add_argument("-n", "--noProcess",
                    action="store_true",
                    help="Whether the data should be prevented from being processed. Default: data can be processed.")
parser.add_argument("-o", "--output",
                    help="The location of the directory to save the sharded output to. Default: a top level "
                         "directory called Results.",
                    type=str)
parser.add_argument("-w", "--overwrite",
                    action="store_true",
                    help="Whether the output directory should be overwritten. Default: do not overwrite.")

# ============================ #
# Parse and Validate Arguments #
# ============================ #
args = parser.parse_args()
dirCurrent = os.path.dirname(os.path.join(os.getcwd(), __file__))  # Directory containing this file.
dirTop = os.path.abspath(os.path.join(dirCurrent, os.pardir))
dirOutput = os.path.join(dirTop, "Results")
dirOutput = args.output if args.output else dirOutput
dirOutputDataPrep = os.path.join(dirOutput, "DataProcessing")
fileDefaultConfig = os.path.join(dirTop, "ConfigurationFiles", "DefaultConfig.json")
fileConfigSchema = os.path.join(dirTop, "ConfigurationFiles", "ConfigurationSchema.json")
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
    os.makedirs(dirOutputDataPrep)  # Attempt to make the data preparation output directory.
else:
    try:
        os.makedirs(dirOutput)  # Attempt to make the output directory.
        os.makedirs(dirOutputDataPrep)  # Attempt to make the data preparation output directory.
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

# Validate the input location.
inputContent = args.input
if not os.path.exists(inputContent):
    logger.error("The input location does not exist.")
    isErrors = True

# Set default parameter values.
config = Configuration.Configuration()
try:
    if args.encode:
        config.set_from_json(fileDefaultConfig, fileConfigSchema, args.encode)
    else:
        config.set_from_json(fileDefaultConfig, fileConfigSchema)
except jsonschema.SchemaError as e:
    exceptionInfo = sys.exc_info()
    logger.error(
        "The configuration schema is not a valid JSON schema. Please correct any changes made to the "
        "schema or download the original and save it at {:s}.\n{:s}".format(fileConfigSchema, str(exceptionInfo[1]))
    )
    isErrors = True
except jsonschema.ValidationError as e:
    exceptionInfo = sys.exc_info()
    logger.error(
        "The default configuration file is not valid against the schema. Please correct any changes made to "
        "the file or download the original and save it at {:s}.\n{:s}".format(fileDefaultConfig, str(exceptionInfo[1]))
    )
    isErrors = True
except jsonschema.RefResolutionError as e:
    exceptionInfo = sys.exc_info()
    logger.error(
        "The configuration schema contains an invalid reference. Please correct any changes made to the "
        "schema or download the original and save it at {:s}.\n{:s}".format(fileConfigSchema, str(exceptionInfo[1]))
    )
    isErrors = True
except LookupError as e:
    logger.exception("Requested encoding {:s} to convert JSON strings to wasn't found.".format(args.encode))
    isErrors = True

# Validate and set any user supplied configuration parameters.
if args.config:
    if not os.path.isfile(args.config):
        logger.error("The supplied location of the configuration file is not a file.")
        isErrors = True
    else:
        try:
            if args.encode:
                config.set_from_json(args.config, fileConfigSchema, args.encode)
            else:
                config.set_from_json(args.config, fileConfigSchema)
        except jsonschema.ValidationError as e:
            exceptionInfo = sys.exc_info()
            logger.error(
                "The user provided configuration file is not valid against the schema.\n{:s}".format(
                    str(exceptionInfo[1]))
            )
            isErrors = True
        except LookupError as e:
            logger.exception("Requested encoding {:s} to convert JSON strings to wasn't found.".format(args.encode))
            isErrors = True

# Display errors if any were found.
if isErrors:
    print("\nErrors were encountered while validating the input arguments. Please see the log file for details.\n")
    sys.exit()

# =================== #
# Process the Dataset #
# =================== #
if not args.noProcess:
    # Processing of the data is to be performed.
    conversionToUse = config.get_param(["DataProcessing", "Converter"])[1]

    if conversionToUse == "JournalTable":
        # Convert the data from a journal table format to a flat file.

        if not os.path.isdir(inputContent):
            # The input was not a directory as needed.
            logger.error("The input location must be a directory for journal table conversion.")
            print("\nErrors were encountered prior to processing the journal table..\n")
            sys.exit()

        dirProcessedData = os.path.join(inputContent, "_ProcessedJournalTable_")
        if os.path.isdir(dirProcessedData):
            # The data has been processed previously.
            JournalTable.generate_datasets.main(dirProcessedData, dirOutputDataPrep, config)
        elif not os.path.exists(dirProcessedData):
            os.makedirs(dirProcessedData)
            JournalTable.process_table.main(inputContent, dirProcessedData)
            JournalTable.generate_datasets.main(dirProcessedData, dirOutputDataPrep, config)
        else:
            logger.error("The location to save the processed journal table data already exists and is not a directory.")
            print("\nErrors were encountered prior to processing the journal table..\n")
            sys.exit()

    else:
        # The converter specified is not valid.
        logger.error("The specified converter {:s} is not a valid converter choice.".format(conversionToUse))