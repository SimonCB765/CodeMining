"""Computes normalisations of matrix values."""

# 3rd party imports.
import numpy as np
from scipy import sparse


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

    # Get and apply the desired normalisation function.
    # If the normalisation function requested doesn't exist, then perform no normalisation.
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
    """Convert matrix to a tf-idf representation.

    The matrix is interpreted with "documents" being the rows and "terms" being the columns.

    """

    nonZeroValues = matrix.nonzero()  # Get the indices of the non-zero entries in the matrix.

    # First get the sum of items in each row of the matrix.
    # This is analogous to the total number of terms in a document.
    termsInEachDocument = matrix.sum(axis=1)  # Total sum of term counts for each example.

    # Convert the total term counts to a list in order to simplify the scaling of the term counts.
    termsInEachDocument = [i[0] for i in termsInEachDocument.tolist()]
    
    # Taking the input matrix to be A, create a new sparse matrix B, such that B[i, j] == 0 if A[i, j] == 0,
    # and B[i, j] = (1 / termsInEachDocument[i]) if A[i, j] != 0. All non-zero entries in row i of matrix B
    # will therefore contain the inverse of the total number of terms in document i.
    scaledTerms = sparse.coo_matrix(([(1 / termsInEachDocument[i]) for i in nonZeroValues[0]],
        (nonZeroValues[0], nonZeroValues[1])))
    scaledTerms = sparse.csr_matrix(scaledTerms)

    # Scale the individual counts for each term in a document by the total number of terms in the document.
    # This equates to multiplying the matrix of inverse terms per document by the original data matrix.
    scaledTerms = matrix.multiply(scaledTerms)  # Term counts for each document scaled by the document's total counts.

    # Determine the number of documents in which each term occurs.
    # Need to index like [0] as an array made from a matrix becomes a 2 dimensional array.
    # In this case it would be a 1xN array, while we want a 1 dimensional array.
    docsTermOccursIn = np.array((matrix != 0).sum(axis=0))[0]

    # Calculate the idf for each term.
    # This is the log of the total number of documents divided by the number of documents containing the term.
    idf = np.log(matrix.shape[0] / docsTermOccursIn)  # The idf for each term. The logarithm base doesn't matter.
    idf[idf == np.inf] = 0  # If a term doesn't occur in any documents it causes an inf from a divide by zero error.
    
    # Taking the matrix of scaled terms to be A, create a sparse matrix B, such that B[i, j] == 0 if A[i, j] == 0,
    # and B[i, j] == idf[j] if A[i, j] != 0. All non-zero entries in column j of matrix B will therefore contain
    # the idf for term j.
    tfidfMat = sparse.coo_matrix(([idf[i] for i in nonZeroValues[1]], (nonZeroValues[0], nonZeroValues[1])))
    tfidfMat = sparse.csr_matrix(tfidfMat)
    
    # Calculate the idf.
    tfidfMat = scaledTerms.multiply(tfidfMat)

    return tfidfMat


def make_unit_variance(matrix, normParam=None):
    """Make each column of matrix have unit variance."""

    return matrix / matrix.todense().std(axis=0, ddof=1)


def make_zero_mean(matrix, normParam=None):
    """Make each column have zero mean."""

    return matrix - matrix.mean(axis=0)