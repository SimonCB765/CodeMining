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
from DataProcessing import JournalTable


# ====================== #
# Create Argument Parser #
# ====================== #
parser = argparse.ArgumentParser(description="Identify clinical codes for concepts in a data-driven manner.",
                                 epilog="For further information see the README.")

# Mandatory arguments.
parser.add_argument("input", help="The location of the file containing the input data. If data processing is being "
                                  "run, then this should contain an unprocessed dataset. If no processing is being "
                                  "run, then this should contain a processed dataset.")

# Optional arguments.
parser.add_argument("-n", "--noProcess",
                    action="store_true",
                    help="Whether the data should be prevented from being processed. Default: data will be processed.")
parser.add_argument("-o", "--output",
                    help="The location of the directory to save the output to. Default: a timestamped "
                         "subdirectory in the Results directory.",
                    type=str)
parser.add_argument("-p", "--processed",
                    help="The location of the directory to save the processed data to. Only used if data processing is "
                         "enabled. Default: a timestamped subdirectory in the Data directory.",
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

# Validate the input location.
inputContent = args.input
if not os.path.exists(inputContent):
    logger.error("The input location does not exist.")
    isErrors = True

# Display errors if any were found.
if isErrors:
    print("\nErrors were encountered while validating the input arguments. Please see the log file for details.\n")
    sys.exit()

# =================== #
# Run the Code Mining #
# =================== #
if not args.noProcess:
    # The dataset needs processing.
    logger.info("Now processing the input dataset.")

    # Create the directory to hold the processed data.
    fileInputName = os.path.splitext(os.path.basename(os.path.normpath(args.input)))[0]
    fileProcessedData = os.path.join(dirOutput, "{:s}_Processed.tsv".format(fileInputName))

    # Process the data.
    JournalTable.process_table.main(inputContent, fileProcessedData)
    inputContent = fileProcessedData

# Perform the code mining.
