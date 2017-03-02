"""Perform the code mining."""

# Python imports.
import logging
import os
import sys

# User imports.
from . import generate_dataset
from . import train_model

# 3rd party imports.
import numpy as np

# Globals.
LOGGER = logging.getLogger(__name__)


def main(fileDataset, fileCodeMapping, dirResults, config):
    """Perform the clinical code mining.

    :param fileDataset:         File location containing the data to use.
    :type fileDataset:          str
    :param fileCodeMapping:     File containing the mapping between codes and their descriptions.
    :type fileCodeMapping:      str
    :param dirResults:          Location of the directory in which the results will be saved.
    :type dirResults:           str
    :param config:              The JSON-like object containing the configuration parameters to use.
    :type config:               JsonschemaManipulation.Configuration

    """

    # Generate the mapping from codes to their descriptions. Each line in the file should contain two entries (a code
    # and its description) separated by a tab.
    mapCodeToDescr = {}
    with open(fileCodeMapping, 'r') as fidCodes:
        for line in fidCodes:
            chunks = (line.strip()).split('\t')
            mapCodeToDescr[chunks[0]] = chunks[1]

    # Generate the data matrix, a mapping of IDs of the ambiguous patients to the cases they appear to meet the
    # definitions of, two index mappings, the mapping from case names to defining codes and the case mapping.
    # The patient index map is a bidirectional mapping between patients and their row indices in the data matrix.
    # The code index map is a bidirectional mapping between codes and their column indices in the data matrix.
    # The case definition mapping records the codes that define each case.
    # The case mapping records which patients meet which case definition.
    LOGGER.info("Now generating dataset.")
    dataMatrix, ambiguousPatients, mapPatientIndices, mapCodeIndices, caseDefs, cases = generate_dataset.main(
        fileDataset, dirResults, mapCodeToDescr, config
    )

    # Check whether there are any cases with no patients (the ambiguous patient case does not need accounting for as
    # it is only in the case definition when there are ambiguous patients).
    noExampleCases = [i for i, j in cases.items() if len(j) == 0]
    if noExampleCases:
        LOGGER.error("The following cases have no unambiguous patients: {:s}".format(
            ','.join(noExampleCases))
        )
        print("\nErrors were encountered following case identification. Please see the log file for details.\n")
        sys.exit()

    # Map bidirectionally between case names and the integers used to represent them and determine the
    # integer case definition that each patient meets.
    # A value of NaN is used to indicate that a patient does not meet any case definition, and is therefore not used.
    dataClasses = np.empty(dataMatrix.shape[0])
    dataClasses.fill(np.nan)
    caseIntegerReps = {}
    for ind, (i, j) in enumerate(cases.items()):
        caseIntegerReps[ind] = i
        caseIntegerReps[i] = ind
        patientIndices = [k for k in j]
        dataClasses[patientIndices] = ind

    # Calculate masks for the patients and the codes. These will be used to select only those patients and codes
    # that are to be used for training/testing.
    patientsUsed = [k for i, j in cases.items() for k in j]
    patientMask = np.zeros(dataMatrix.shape[0], dtype=bool)
    patientMask[patientsUsed] = 1  # Set patients the meet a case definition to be used.
    codesUsedForCases = [k for i, j in caseDefs.items() for k in j]
    codeMask = np.ones(dataMatrix.shape[1], dtype=bool)
    codeMask[codesUsedForCases] = 0  # Mask out the codes used to calculate case membership.

    # Train the model.
    classifier = train_model.main(dataMatrix, dataClasses, dirResults, patientMask, codeMask, cases, config)

    # Record the coefficients of the trained model.
    fileCoefs = os.path.join(dirResults, "Coefficients.tsv")
    caseNames = sorted([i for i in caseDefs])
    with open(fileCoefs, 'w') as fidCoefs:
        # Write out the header.
        if len(caseNames) == 2:
            # If there are two classes, then there is only one coefficient per code.
            fidCoefs.write("Code\tDescription\tCoefficient\n")
        else:
            # If there are more than two classes, then there is one coefficient per class per code.
            fidCoefs.write("Code\tDescription\t{:s}\n".format('\t'.join(caseNames)))

        # Write out coefficients.
        for ind, i in enumerate(np.flatnonzero(codeMask)):
            code = mapCodeIndices[i]
            description = mapCodeToDescr.get(code, "Unknown Code")
            coefficient = classifier.coef_[:, ind]
            if len(caseNames) == 2:
                fidCoefs.write("{:s}\t{:s}\t{:1.4f}\n".format(code, description, coefficient[0]))
            else:
                fidCoefs.write("{:s}\t{:s}\t{:s}\n".format(
                    code, description, '\t'.join(["{:1.4f}".format(coefficient[caseIntegerReps[i]]) for i in caseNames])
                ))

    # Record the predictions of the ambiguous patients.
    fileAmbig = os.path.join(dirResults, "AmbiguousPatientPredictions.tsv")
    with open(fileAmbig, 'w') as fidAmbig:
        # Create the cutdown matrix of ambiguous patients.
        ambigPatientMask = np.zeros(dataMatrix.shape[0], dtype=bool)
        ambigPatientMask[sorted(ambiguousPatients)] = 1
        ambigDataMatrix = dataMatrix[:, codeMask]
        ambigDataMatrix = ambigDataMatrix[ambigPatientMask, :]

        if np.sum(ambigPatientMask) > 0:
            # There are some ambiguous patients.

            # Predict the cases the patients belong to.
            predictions = classifier.predict(ambigDataMatrix)
            posteriors = classifier.predict_proba(ambigDataMatrix)

            # Write out the predictions.
            fidAmbig.write("PatientID\tTrueCases\tPredictedCase\t{:s}_Posterior\n".format('_Posterior\t'.join(caseNames)))
            for i, j, k in zip(sorted(ambiguousPatients), predictions, posteriors):
                fidAmbig.write("{:s}\t{:s}\t{:s}\t{:s}\n".format(
                    mapPatientIndices[i], ','.join(sorted(ambiguousPatients[i])), caseIntegerReps[j],
                    '\t'.join(["{:1.4f}".format(k[caseIntegerReps[i]]) for i in caseNames])
                ))

    # Classify all patients not used for training the classifier (i.e. those meeting no case definition).
    fileNonTraining = os.path.join(dirResults, "NoCasePatientPredictions.tsv")
    with open(fileNonTraining, 'w') as fidNonTraining:
        # Create the cutdown matrix of patients not used for training.
        nonTrainingPatientMask = ~patientMask
        nonTrainDataMatrix = dataMatrix[:, codeMask]
        nonTrainDataMatrix = nonTrainDataMatrix[nonTrainingPatientMask, :]

        # Predict the cases the patients belong to.
        predictions = classifier.predict(nonTrainDataMatrix)
        posteriors = classifier.predict_proba(nonTrainDataMatrix)

        # Write out the predictions.
        fidNonTraining.write("PatientID\tPredictedCase\t{:s}_Posterior\n".format('_Posterior\t'.join(caseNames)))
        for i, j, k in zip(np.flatnonzero(nonTrainingPatientMask), predictions, posteriors):
            fidNonTraining.write("{:s}\t{:s}\t{:s}\n".format(
                mapPatientIndices[i], caseIntegerReps[j],
                '\t'.join(["{:1.4f}".format(k[caseIntegerReps[i]]) for i in caseNames])
            ))
