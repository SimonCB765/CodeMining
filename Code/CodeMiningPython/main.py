"""Code to control the running of the entire code mining process."""

# Python imports.
import argparse
import sys

# User imports.
import CodeMiningPython.clean_data
import CodeMiningPython.code_mining
import CodeMiningPython.parse_parameters


def main(args):
    """Run the data cleaning and code mining.

    :param args:    The list of arguments to the program.
    :type args:     list

    """

    #=========================#
    # Parse the user's input. #
    #=========================#
    parser = argparse.ArgumentParser(
        description="................Perform clinical code identification using logistic regression................",
        epilog="Checking is performed to ensure that the required parameters are present in the parameter file,"
               " but no checking of correctness is performed. Please see the README for information on "
               "the parameters and the functionality.")
    parser.add_argument("params", help="Location of the parameter file.")
    args = parser.parse_args()
    fileParams = args.params  # File containing the parameters to use.
    parsedParameters = CodeMiningPython.parse_parameters.main(fileParams)

    #=================#
    # Clean the Data. #
    #=================#
    if "CleaningArgs" in parsedParameters:
        print("Now cleaning the data.")
        cleanerArgs = parsedParameters["CleaningArgs"]
        CodeMiningPython.clean_data.main(cleanerArgs["DirtyDataLocation"], cleanerArgs["CleanDataLocation"],
                                         cleanerArgs["CodeColumn"], cleanerArgs["Delimiter"],
                                         cleanerArgs["StripCommas"], cleanerArgs["RemoveCols"],
                                         cleanerArgs["Unbookend"])

    #======================#
    # Perform Code Mining. #
    #======================#
    if "MiningArgs" in parsedParameters:
        print("Now starting code mining.")
        miningArgs = parsedParameters["MiningArgs"]
        CodeMiningPython.code_mining.main(miningArgs["DataLocation"], miningArgs["CodeMapping"],
                                          miningArgs["ResultsLocation"], miningArgs["Classes"],
                                          miningArgs["Lambda"], miningArgs["ElasticNetMixing"],
                                          miningArgs["BatchSize"], miningArgs["MaxIter"], miningArgs["CodeOccurrences"],
                                          miningArgs["PatientOccurrences"], miningArgs["CVFolds"],
                                          miningArgs["DataNorm"], miningArgs["DiscardThreshold"])