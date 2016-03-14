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
    parser.add_argument("classes", help="Location of the JSON format file containing the class definitions.")
    parser.add_argument("-c", "--codeDensity", type=int, default=0, help="Minimum number of unique FREQUENTLY OCCURRING codes required for a patient before they are kept in the subset.")
    parser.add_argument("-i", "--infrequent", type=int, default=0, help="Minimum number of patients that a code must appear in before it will be kept in the subset.")
    parser.add_argument("-o", "--outDir", default="Results", help="Directory where the subset and dataset statistics should be recorded.")
    parser.add_argument("-p", "--parse", type=str, default=None, help="Location of the JSON format file containing the parser arguments.")
    parser.add_argument("-s", "--subset", action="store_true", default=False, help="Whether a subset of the dataset should be generated.")
    args = parser.parse_args()

    fileDataset = args.dataset  # File containing the input dataset to create the subset of.
    fileClasses = args.classes  # JSON file containing the codes used to determine classes.
    codeDensity = args.codeDensity  # The minimum number of unique codes in a patient's record meeting the minPatientsPerCode criterion needed before they are recorded in the subset.
    minPatientsPerCode = args.infrequent  # The minimum number of patients that a code must be associated with before it is kept in the subset.
    dirResults = args.outDir  # Location to save the generated subset.
    fileParserArgs = args.parse  # JSON file containing the arguments for the dataset parser/cleaner.
    isSubsetNeeded = args.subset  # Whether a subset of the dataset will be generated.

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
    if fileParserArgs and not os.path.exists(fileParserArgs):
        # Parser arguments were supplied, but the location is invalid.
        errorsFound.append("The parser argument file location is not valid.")

    # Process the class information provided.
    isCollectorClass = False  # Whether there is a class with no codes that will be used to collect all examples not belonging to the other classes.
    collectorClass = None  # The class to use as a collector.
    if not os.path.isfile(fileClasses):
        # The mapping from codes to their descriptions must exist.
        errorsFound.append("The file of classes does not exist.")
    else:
        readClasses = open(fileClasses, 'r')
        jsonClasses = json.load(readClasses)
        readClasses.close()
        classesWithoutCodes = [i for i in jsonClasses if not jsonClasses[i]]  # Names of all classes without codes.

        # Determine whether there are any errors in the class file itself.
        if len(classesWithoutCodes) == 1:
            # There is one class that will contain all examples that do not belong to another class.
            isCollectorClass = True
            collectorClass = classesWithoutCodes[0]
        elif len(classesWithoutCodes) > 1:
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
        colsToUnbookend = [[i["Column"], i["String"], i["IntOnly"]] for i in parserArgs["Unbookend"]]
        dirCleaned = dirResults + "/CleanedData"

        # Clean the data.
        fileDataset = parse_and_clean.parse_and_clean(fileDataset, dirCleaned, delimiter=parserArgs["Delimiter"],
                                                      colsToStripCommas=parserArgs["StripCommas"],
                                                      colsToRemove=parserArgs["RemoveCols"],
                                                      colsToUnbookend=colsToUnbookend)

    # Generate the subset if required.
    if isSubsetNeeded:
        [fileDataset, ambiguousDataset] = select_data_subset.select_data_subset(fileDataset, jsonClasses, dirResults,
                                                                                minPatientsPerCode, codeDensity,
                                                                                isCollectorClass, collectorClass)


if __name__ == '__main__':
    main(sys.argv[1:])