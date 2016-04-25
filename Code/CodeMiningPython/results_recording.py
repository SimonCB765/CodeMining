"""Auxiliary functions for writing out results."""


def record_ambiguous(firstClassifier, secondClassifier, ambiguousExamples, ambiguousMatrix, mapIntRepToClass,
                     mapIndToPatient, fileResults):
    """Record the predictions for the ambiguous examples made by the two models.

    :param firstClassifier:     The first model.
    :type firstClassifier:      sklearn.linear_model.SGDClassifier
    :param secondClassifier:    The second model.
    :type secondClassifier:     sklearn.linear_model.SGDClassifier
    :param ambiguousExamples:   The indices of the examples in the original dataset the are ambiguous.
    :type ambiguousExamples:    list
    :param ambiguousMatrix:     The dataset of ambiguous examples.
    :type ambiguousMatrix:      scipy sparse matrix
    :param mapIntRepToClass:    A mapping from the integer representing a class to the name of the class.
    :type mapIntRepToClass:     dict
    :param mapIndToPatient:
    :param fileResults:         The location where the predictions of the ambiguous examples should be written.
    :type fileResults:          str

    """

    with open(fileResults, 'w') as fidAmbig:
        # Write out the header.
        fidAmbig.write("PatientID\tFirstModelClass\t{0:s}\tSecondModelClass\t{1:s}\n".format(
            '\t'.join(["FirstModel_{0:s}_Post".format(mapIntRepToClass[i]) for i in mapIntRepToClass]),
            '\t'.join(["SecondModel_{0:s}_Post".format(mapIntRepToClass[i]) for i in mapIntRepToClass])))

        # Generate the predictions.
        firstModelAmbigPred = firstClassifier.predict(ambiguousMatrix)
        firstModelAmigPosts = firstClassifier.predict_proba(ambiguousMatrix)
        secondModelAmbigPred = secondClassifier.predict(ambiguousMatrix)
        secondModelAmigPosts = secondClassifier.predict_proba(ambiguousMatrix)

        # Write out the predictions.
        for ind, i in enumerate(ambiguousExamples):
            patientID = mapIndToPatient[i]
            fidAmbig.write("{0:s}\t{1:s}\t{2:s}\t{3:s}\t{4:s}\n".format(
                patientID, mapIntRepToClass[firstModelAmbigPred[ind]],
                '\t'.join(["{0:1.4f}".format(firstModelAmigPosts[ind, i]) for i in mapIntRepToClass]),
                mapIntRepToClass[secondModelAmbigPred[ind]],
                '\t'.join(["{0:1.4f}".format(secondModelAmigPosts[ind, i]) for i in mapIntRepToClass])))

def record_coefficients(firstClassifier, secondClassifier, mapIntRepToClass, indicesOfCodesUsed, mapCodeToInd,
                        mapCodeToDescr, fileResults):
    """Record the coefficients of the two models trained on the same codes.

    :param firstClassifier:     The first model.
    :type firstClassifier:      sklearn.linear_model.SGDClassifier
    :param secondClassifier:    The second model.
    :type secondClassifier:     sklearn.linear_model.SGDClassifier
    :param mapIntRepToClass:    A mapping from the integer representing a class to the name of the class.
    :type mapIntRepToClass:     dict
    :param indicesOfCodesUsed:  The indices of the codes used to train the models.
    :type indicesOfCodesUsed:   numpy array
    :param mapCodeToInd:        A mapping from codes to their indices in the data matrices used for training/testing.
    :type mapCodeToInd:         dict
    :param mapCodeToDescr:      A mapping from codes to their descriptions.
    :type mapCodeToDescr:       dict
    :param fileResults:         The location where the coefficients should be written.
    :type fileResults:          str

    """

    with open(fileResults, 'w') as fidCoefs:
        # Write out the header.
        if len(mapIntRepToClass) == 2:
            # If there are two classes, then there is only one coefficient per code.
            fidCoefs.write("Code\tDescription\tFirstModel\tSecondModel\n")
        else:
            # If there are more than two classes, then there is one coefficient per class per code.
            fidCoefs.write("Code\tDescription\t{0:s}\t{1:s}\n".format(
                '\t'.join(["FirstModel_{0:s}_Coef".format(mapIntRepToClass[i]) for i in mapIntRepToClass]),
                '\t'.join(["SecondModel_{0:s}_Coef".format(mapIntRepToClass[i]) for i in mapIntRepToClass])))

        # Reverse the mapping from codes to their indices in the data matrix. This is safe as each index is only
        # mapped to by one code, and you therefore won't have key clashes in the reverse mapping.
        mapIndToCode = {v : k for k, v in mapCodeToInd.items()}

        for ind, i in enumerate(indicesOfCodesUsed):
            code = mapIndToCode[i]
            firstModelCoefs = firstClassifier.coef_[:, ind]
            secondModelCoefs = secondClassifier.coef_[:, ind]
            if len(mapIntRepToClass) == 2:
                fidCoefs.write("{0:s}\t{1:s}\t{2:1.4f}\t{3:1.4f}\n".format(
                    code, mapCodeToDescr.get(code, "Unknown Code"), firstModelCoefs[0],
                    secondModelCoefs[0]))
            else:
                fidCoefs.write("{0:s}\t{1:s}\t{2:s}\t{3:s}\n".format(
                    code, mapCodeToDescr.get(code, "Unknown Code"),
                    '\t'.join(["{0:1.4f}".format(firstModelCoefs[i]) for i in mapIntRepToClass]),
                    '\t'.join(["{0:1.4f}".format(secondModelCoefs[i]) for i in mapIntRepToClass])))