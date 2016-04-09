"""Computes normalisations of matrix values."""

# 3rd party imports.
import numpy as np


def main(matrix, normMethod=0, normParam=None):
    """Normalise a matrix.

    The matrix is assumed to be ordered with examples as rows and variables as columns.

    :param matrix:          The matrix to normalise. Values in it are assumed to be appropriate for the chosen method.
    :type matrix:           scipy sparse matrix
    :param normMethod:      The code for the normalisation method to use. The acceptable values are:
                                0  - Leave the matrix as is. This is the default method.
                                1  - Convert the matrix to a binary representation. Each non-zero entry in the matrix
                                    is converted to a one.
                                2  - Convert the matrix to a tf-idf representation. Each non-zero entry in the matrix
                                    is converted to its tf-idf one.
                                3* - Make each column of the matrix have 0 mean.
                                4* - Make each column of the matrix have unit variance.
                                5* - Standardise each column (combine methods 3 and 4).
                                6  - Scale each column to be in the range [0, 1].
                                7  - Scale each column to have unit length using the Lp norm. Defaults to L2
                                    Euclidean norm. P is provided as the first value of normParam.
                                Asterisks indicate those methods that will cause a sparse matrix to become dense.
    :type normMethod:       int
    :param normParam:       The parameters (if any) needed for the normalisation method.
    :type normParam:        list
    :return :               The normalised matrix.
    :rtype :                The same type as the input matrix.

    """

    normalisationChoice = {
        1 : make_binary,
        2 : make_tf_idf,
        3 : make_zero_mean,
        4 : make_unit_variance,
        5 : make_standardised,
        6 : make_scaled,
        7 : make_lp_norm
    }

    normFunc = normalisationChoice.get(normMethod, lambda x, y :  x)
    return normFunc(matrix, normParam)


def make_binary(matrix, normParam=None):
    """Convert matrix to binary format."""

    return matrix != 0


def make_lp_norm(matrix, normParam=(2,)):
    """Scale each column to have unit length using the Lp norm.

    :param normParam:   The 'p' in Lp.
    :type normParam:    tuple with the first element and int

    """

    oldData = matrix.data.copy()  # Store a copy of the data in the matrix.
    matrix.data **= normParam[0]  # Elementwise power of the matrix.
    root = np.power(matrix.sum(axis=0), 1 / normParam[0])  # The pth root of the sum of each column's new values.
    matrix.data = oldData  # Restore the original data in the matrix.
    return matrix / root


def make_scaled(matrix, normParam=None):
    """Scale each column to be in the range [0, 1]."""

    minVals = matrix.min(axis=0).todense()
    maxVals = matrix.max(axis=0).todense()

    return (matrix - minVals) / (maxVals - minVals)


def make_standardised(matrix, normParam=None):
    """Standardise each column of matrix (zero mean and unit variance)."""

    return make_unit_variance(make_zero_mean(matrix))


def make_tf_idf(matrix, normParam=None):
    """Convert matrix to a tf-idf representation."""

    termsInEachDocument = matrix.sum(axis=1)  # Total sum of term counts for each example.
    termsInEachDocument = termsInEachDocument[:, None]  # Reshape to column vector.
    scaledTerms = matrix / termsInEachDocument  # Term counts for each example scaled by the example's total counts.
    docsTermOccursIn = np.sum(matrix != 0, axis=0)  # The number of example in which each term occurs.
    idf = np.log(matrix.shape[0] / docsTermOccursIn)  # The IDF for each term. The base of the logarithm doesn't matter.
    idf[idf == np.inf] = 0  # If a term doesn't occur in any documents it causes an inf from a divide by zero error.
    return np.dot(scaledTerms, np.diag(idf))


def make_unit_variance(matrix, normParam=None):
    """Make each column of matrix have unit variance."""

    return matrix / matrix.todense().std(axis=0, ddof=1)


def make_zero_mean(matrix, normParam=None):
    """Make each column have zero mean."""

    return matrix - matrix.mean(axis=0)