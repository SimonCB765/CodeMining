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

    Conversion to tf-idf without scipy sparse matrices losing their sparsity at some point during the
    computation requires avoiding some vectorised operations, and therefore is slower than with
    dense matrices.

    """

    # Calculating tf-idf is fairly straightforward, but in order to prevent the sparse matrix becoming dense at
    # some point during the computation requires some convoluted and non-vectorised computations.

    # First get the sum of items in each row of the matrix. This is analogous to the total number of terms in a document.
    termsInEachDocument = matrix.sum(axis=1)  # Total sum of term counts for each example.

    # Convert the total term counts to a list in order to simplify the scaling of the term counts.
    termsInEachDocument = [i[0] for i in termsInEachDocument.tolist()]

    # Scale the individual counts for each term in a document by the total number of terms in the document.
    # This has to be (moderately) non-vectorised in order to prevent loss of sparsity, which makes it (slightly) slower.
    scaledTerms = matrix.copy()  # Term counts for each document scaled by the document's total counts.
    for ind, i in enumerate(termsInEachDocument):
        if i:
            # Prevent division by zero.

            # Although appearing more vectorised, the equation:
            # scaledTerms[ind, :] /= i
            # is substantially slower than than the one used. It also warns about alterations to the sparsity structure,
            # which I assume is due to the zero values in the sparse matrix being touched (although as the operation
            # being used is division, they won't change). I assume this is also the reason that it is slow.

            # Instead we will get the indices of the non-zero values in the row (i.e. the terms that this document
            # contains), and scale only those terms by the total number of terms in the document.
            nonzeroRowValues = scaledTerms[ind, :].nonzero()  # Get the indices of the non-zero values in the row.
            scaledTerms[ind, nonzeroRowValues[1]] /= i  # Scale the term counts in the input matrix's copy.

    # Determine the number of documents in which each term occurs.
    # Need to index like [0] as an array made from a matrix becomes a 2 dimensional array.
    # In this case it would be a 1xN array, while we want a 1 dimensional array.
    docsTermOccursIn = np.array((matrix != 0).sum(axis=0))[0]

    # Calculate the idf for each term.
    # This is the log of the total number of documents divided by the number of documents containing the term.
    idf = np.log(matrix.shape[0] / docsTermOccursIn)  # The idf for each term. The base of the logarithm doesn't matter.
    idf[idf == np.inf] = 0  # If a term doesn't occur in any documents it causes an inf from a divide by zero error.

    # Calculate the tf-idf.
    # This is the term frequency multiplied by the idf of the term.
    for ind, i in enumerate(idf):
        # Although appearing more vectorised, the equation:
        # scaledTerms[:, ind] *= i
        # is substantially slower than than the one used. It also warns about alterations to the sparsity structure,
        # which I assume is due to the zero values in the sparse matrix being touched (although as the operation
        # being used is division, they won't change). I assume this is also the reason that it is slow.

        # Instead we will get the indices of the non-zero values in the column (i.e. the documents that this term
        # occurs in), and multiply only those document's values or the term by the idf for the term.
        nonzeroColValues = scaledTerms[:, ind].nonzero()  # Get the indices of the non-zero values in the column.
        scaledTerms[nonzeroColValues[0], ind] *= i  # Multiply the scaled terms counts by the idf for the term.

    return scaledTerms


def make_unit_variance(matrix, normParam=None):
    """Make each column of matrix have unit variance."""

    return matrix / matrix.todense().std(axis=0, ddof=1)


def make_zero_mean(matrix, normParam=None):
    """Make each column have zero mean."""

    return matrix - matrix.mean(axis=0)