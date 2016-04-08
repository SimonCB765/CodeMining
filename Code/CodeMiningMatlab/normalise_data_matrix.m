function [dataMatrix] = normalise_data_matrix(dataMatrix, normMethod, normParam)
    % Normalise a matrix.
    %
    % The matrix is assumed to be organised with examples as rows and variables as columns.
    %
    % Keyword Arguments
    % dataMatrix - The matrix to normalise. The values in the matrix are assumed to be appropriate for the specified method.
    % normMethod - The method to use for normalisation. The acceptable values are:
    %                  0  - Leave the matrix as is. This is the default method.
    %                  1  - Convert the matrix to a binary representation. Each non-zero entry in the matrix is converted to a one.
    %                  2  - Convert the matrix to a tf-idf representation. Each non-zero entry in the matrix is converted to its tf-idf one.
    %                  3* - Make each column of the matrix have 0 mean.
    %                  4  - Make each column of the matrix have unit variance.
    %                  5* - Standardise each column (combine methods 3 and 4).
    %                  6  - Scale each column to be in the range [0, 1].
    %                  7  - Scale each column to have unit length using the Lp norm. Defaults to L2 Euclidean norm.
    %                       P is provided as the first value of normParam.
    %              Asterisks indicate those methods that will cause a sparse matrix to become dense.
    % normParam - Array of parameter values needed by some normalisation methods. Required values are detailed in the list above.

    if (nargin < 2) || isempty(normMethod)
        % Default the normalisation method to leave the matrix as is.
        normMethod = 0;
    end

    % Normalise the matrix.
    switch normMethod
        case 1
            % Make the matrix binary.
            dataMatrix = dataMatrix ~= 0;
        case 2
            % Convert each entry to a tf-idf value.
            documentTermCounts = sum(dataMatrix, 2);  % Total sum of term counts for each example.
            scaledTermCounts = bsxfun(@rdivide, dataMatrix, documentTermCounts);  % Term counts for each example scaled by the example's total term counts.
            docsTermOccursIn = sum(dataMatrix ~= 0);  % The number of documents in which each term occurs.
            idf = log(size(dataMatrix, 1) ./ docsTermOccursIn);  % The IDF for each term. The base of the logarithm doesn't matter.
            idf(isnan(idf)) = 0;  % If a term doesn't occur in any documents then it causes a NaN from a divide by zero error.
            dataMatrix = scaledTermCounts * diag(idf);
        case 3
            % Make each column have 0 mean.
            dataMatrix = bsxfun(@minus, dataMatrix, mean(dataMatrix));
        case 4
            % Make each column have unit variance.
            dataMatrix = dataMatrix / diag(std(dataMatrix));
        case 5
            % Standardise the data.
            dataMatrix = bsxfun(@minus, dataMatrix, mean(dataMatrix)) / diag(std(dataMatrix));
        case 6
            % Scale each column to have values in the rage [0, 1].
            minVals = min(dataMatrix);
            maxVals = max(dataMatrix);
            dataMatrix = bsxfun(@rdivide, bsxfun(@minus, dataMatrix, minVals), maxVals - minVals);
        case 7
            % Lp normalisation scaling.
            if (nargin < 3) || isempty(normParam) || (normParam(1) < 1)
                % Default to Euclidean norm.
                normParam = [2];
            end
            dataMatrix = bsxfun(@rdivide, dataMatrix, nthroot(sum(bsxfun(@power, abs(dataMatrix), normParam(1))), normParam(1)));
    end

end