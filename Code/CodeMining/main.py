# Python imports.
import argparse
import json
import os
import sys

# User imports.
import parse_and_clean
import select_data_subset


def main(args):
    """Run the code mining.

    :param args:    The list of arguments to the program.
    :type args:     list

    """

    #=========================#
    # Parse the user's input. #
    #=========================#
    parser = argparse.ArgumentParser(
        description="......................................",
        epilog="For further details and requirements please see the README.")
    parser.add_argument("dataset", help="Location of the dataset to generate the subset from.")
    parser.add_argument("outDir", help="Directory where the subset and dataset statistics should be recorded.")
    parser.add_argument("-p", "--parse", type=str, default=None, help="Location of the JSON format file containing the parser arguments.")
    parser.add_argument("-s", "--subset", type=str, default=None, help="Location of the JSON format file containing the subset arguments.")
    args = parser.parse_args()

    fileDataset = args.dataset  # File containing the input dataset to create the subset of.
    dirResults = args.outDir  # Location to save the generated subset.
    fileParserArgs = args.parse  # JSON file containing the arguments for the dataset parser/cleaner.
    fileSubsetArgs = args.subset  # JSON file containing the arguments for the subset generator.

    #========================================#
    # Process and validate the user's input. #
    #========================================#
    errorsFound = []  # List recording all error messages to display.

    if not os.path.isfile(fileDataset):
        # The input dataset must exist to continue.
        errorsFound.append("The dataset file does not exist.")
    if os.path.exists(dirResults):
        if not os.path.isdir(dirResults):
            # Results directory location exists but is not a directory.
            errorsFound.append("The output directory location exists but is not a directory.")
    else:
        # Create the results directory as the location is free.
        os.mkdir(dirResults)

    # Process the parser arguments (if supplied).
    if fileParserArgs:
        if not os.path.exists(fileParserArgs):
            # Parser arguments were supplied, but the location is invalid.
            errorsFound.append("The parser argument file location is not valid.")
        else:
            # Arguments supplied, so check them.
            readParserArgs = open(fileParserArgs, 'r')
            parserArgs = json.load(readParserArgs)
            readParserArgs.close()

            # Check that the delimiter is a string.
            if (("Delimiter" in parserArgs) and (not isinstance(parserArgs["Delimiter"], str))):
                errorsFound.append("The parser delimiter must be a string.")

            # Check that strip commas is a list of integers.
            if (("StripCommas" in parserArgs) and (not all([isinstance(i, int) for i in parserArgs["StripCommas"]]))):
                errorsFound.append("The parser columns to strip commas from must be a list of integers.")

            # Check that the list of columns to remove is a list of integers.
            if (("RemoveCols" in parserArgs) and (not all([isinstance(i, int) for i in parserArgs["RemoveCols"]]))):
                errorsFound.append("The parser columns to remove must be a list of integers.")

            # Check that the columns to unbookend argument is correctly formatted.
            if "Unbookend" in parserArgs:
                for i in parserArgs["Unbookend"]:
                    if not (("Column" in i) and ("String" in i) and ("IntOnly" in i)):
                        errorsFound.append("The parser columns to unbookend is incorrectly formatted. Please see the README for the correct formatting.")
                        break
                    elif ((not isinstance(i["Column"], int)) or (not isinstance(i["String"], str)) or (not isinstance(i["IntOnly"], bool))):
                        errorsFound.append("The parser columns to unbookend is incorrectly formatted. Please see the README for the correct formatting.")
                        break

    # Process the subset generator arguments (if supplied).
    if fileSubsetArgs:
        if not os.path.exists(fileSubsetArgs):
            # Subset arguments were supplied, but the location is invalid.
            errorsFound.append("The subset argument file location is not valid.")
        else:
            # Arguments supplied, so check them.
            readSubsetArgs = open(fileSubsetArgs, 'r')
            subsetArgs = json.load(readSubsetArgs)
            readSubsetArgs.close()

            # Check that the minimum number of patients a code needs to occur with is an integer.
            if (("MinCodeFrequency" in subsetArgs) and (not isinstance(subsetArgs["MinCodeFrequency"], int))):
                errorsFound.append("The minimum number of patients a code needs to occur with must be an integer.")

            # Check that the minimum number of frequently occurring codes a patient needs to occur with is an integer.
            if (("MinPatientFrequency" in subsetArgs) and (not isinstance(subsetArgs["MinPatientFrequency"], int))):
                errorsFound.append("The minimum number of frequently occurring codes a patient needs to occur with must be an integer.")

            # Check the class information.
            if "Classes" not in subsetArgs:
                errorsFound.append("No class information was supplied in the subset generation arguments.")
            else:
                classesWithoutCodes = [i for i in subsetArgs["Classes"] if not subsetArgs["Classes"][i]]  # Names of all classes without codes.
                if len(classesWithoutCodes) > 1:
                    # There can be at most one class without supplied codes.
                    errorsFound.append("The maximum number of classes without provided codes is 1. Classes without supplied codes were: {0:s}".format(','.join(classesWithoutCodes)))

    # Exit if errors were found.
    if errorsFound:
        print("\n\nThe following errors were encountered while parsing the input parameters:\n")
        print('\n'.join(errorsFound))
        sys.exit()

    # Parse and clean the dataset if required.
    if fileParserArgs:
        # Load the arguments for the parser.
        readParserArgs = open(fileParserArgs, 'r')
        parserArgs = json.load(readParserArgs)
        readParserArgs.close()

        # Process the arguments.
        delimiter = '\t' if ("Delimiter" not in parserArgs) else parserArgs["Delimiter"]
        colsToStripCommas = [] if ("StripCommas" not in parserArgs) else parserArgs["StripCommas"]
        colsToRemove = [] if ("RemoveCols" not in parserArgs) else parserArgs["RemoveCols"]
        colsToUnbookend = [] if ("Unbookend" not in parserArgs) else [[i["Column"], i["String"], i["IntOnly"]] for i in parserArgs["Unbookend"]]
        dirCleaned = dirResults + "/CleanedData"

        # Clean the data.
        fileDataset = parse_and_clean.parse_and_clean(fileDataset, dirCleaned, delimiter, colsToStripCommas, colsToRemove, colsToUnbookend)

    # Generate the subset if required.
    if fileSubsetArgs:
        # Load the arguments for the subset generation.
        readSubsetArgs = open(fileSubsetArgs, 'r')
        subsetArgs = json.load(readSubsetArgs)
        readSubsetArgs.close()

        # Process the arguments.
        minPatientsPerCode = 0 if ("MinCodeFrequency" not in subsetArgs) else subsetArgs["MinCodeFrequency"]
        codeDensity = 0 if ("MinPatientFrequency" not in subsetArgs) else subsetArgs["MinPatientFrequency"]
        isCollectorClass = False  # Whether there is a class with no codes that will be used to collect all examples not belonging to the other classes.
        collectorClass = None  # The class to use as a collector.
        classesWithoutCodes = [i for i in subsetArgs["Classes"] if not subsetArgs["Classes"][i]]  # Names of all classes without codes.
        if len(classesWithoutCodes) == 1:
            # There is one class that will contain all examples that do not belong to another class.
            isCollectorClass = True
            collectorClass = classesWithoutCodes[0]

        # Generate the subset.
        [fileDataset, ambiguousDataset] = select_data_subset.select_data_subset(fileDataset, subsetArgs["Classes"], dirResults, minPatientsPerCode,
                                                                                codeDensity, isCollectorClass, collectorClass)


if __name__ == '__main__':
    main(sys.argv[1:])