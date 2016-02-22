function main_PLS(parameterFile)
    % Perform clinical code identification using partial least squares regression.
    %
    % Positive and negative examples for the model training will be determined based on the supplied codes.
    % This is achieved by taking any example that contains a positive code to be a positive example, and any example containing
    %     a negative code to be a negative example. Any examples that contain both positive and negative codes will not be used for
    %     training, but will have a separate disambiguation prediction performed at the end to see if their class can be determined
    %     using the final model.
    % The supplied codes can also be augmented with their child terms (through positiveChildren and negativeChildren).
    %     For example, if positiveCodes is C10E, then setting positiveChildren to true will result in C10E and all its child terms
    %     being used to determine positive examples.
    %
    % Keyword Arguments
    % inputData - The file containing the patient record dataset. Each line of the file should be formatted as:
    %                 PatentID\tCode\tOccurrences
    %             Occurrences records the number of times code Code appears in the record of the patient with ID PatientID.
    %             Both PatientID and Code are likely to appear on more than one line (as patients normally have more than one code associated them
    %             and codes are normally associated with more than one patient).
    % codeMapping - File containing a mapping between codes and their descriptions. Each line of the file should be formatted as:
    %                   Code\tDescription
    % positiveCodes - A list of codes that will be used to determine positive examples. The codes should be supplied as a comma
    %                 separated list, e.g. C10E,C10F. If the empty string is supplied, then it is taken to mean that all examples not
    %                 determined to be negative will be positive. Only one of positiveCodes and negativeCodes can be empty.
    % positiveChildren - True or false depending on whether child terms of the positiveCodes should be used in determining positive examples.
    % negativeCodes - A list of codes that will be used to determine negative examples. The codes should be supplied as a comma
    %                 separated list, e.g. C10E,C10F. If the empty string is supplied, then it is taken to mean that all examples not
    %                 determined to be positive will be negative. Only one of positiveCodes and negativeCodes can be empty.
    % negativeChildren - True or false depending on whether child terms of the negativeCodes should be used in determining negative examples.
    % foldsToUse - The number of folds to use for cross validation.
    % discardThreshold - The threshold at which examples from the training set will be discarded after training the initial model.
    %                    Positive examples will be discarded if their posterior probability is < discardThreshold, while negative examples will
    %                    be discarded if their posterior probability is > discardThreshold. This is symmetric as the posterior here indicates the probability
    %                    of the class of the example being positive. Therefore, keeping negative examples with posterior < discardThreshold is the same
    %                    as keeping all negative examples with a > discardThreshold probability of being negative.
    % maxComponents - The maximum number of components to use when determining the optimal number of components.
    % outputDir - The directory to save the results in (should be supplied without a trailing /). If outputDir is the empty string, then
    %             the results are saved in the directory the script is called from.

    % Setup default values for the input parameters.
    params = containers.Map();
    params('inputData') = '';  % Mandatory argument.
    params('codeMapping') = '';  % Mandatory argument.
    params('classes') = [];  % Mandatory argument.
    params('foldsToUse') = 10;  % Optional argument.
    params('discardThreshold') = 0.0;  % Optional argument.
    params('maxComponents') = 10;  % Optional argument.
    params('outputDir') = '/';  % Optional argument.

    % Setup the cell array to record error messages.
    parameterErrors = {};

    % Read the input parameters from the configuration file.
    fidParams = fopen(parameterFile, 'r');
    line = fgetl(fidParams);
    currentLineNumber = 1;
    while ischar(line)
        % Loop until there are no more lines in the file.
        split = strsplit(line, '\t');
        if (strcmp(split{1}, 'Class'))
            % If the line contains the values for a class parameter.
            if (numel(split) ~= 4)
                % There must be 4 entries on the line.
                parameterErrors(end + 1) = ['Class entry on line ' currentLineNumber ' has ' numel(splits) ' elements.'];
            else
                split{4} = iff(strcmpi(split{4}, 'true'), true, false);
                params('classes') = [params('classes'); split(2:end)];
            end
        else
            % If the line contains the value for a non-class parameter.
            params(split{1}) = split{2};
        end

        line = fgetl(fidParams);  % Get the next line from the file.
        currentLineNumber = currentLineNumber + 1;
    end
    fclose(fidParams);

    % Convert stored classes to more useful form.
    classData = params('classes');
    params('classNames') = classData(:, 1)';
    params('classCodes') = classData(:, 2)';
    params('classChildren') = classData(:, 3)';
    numberOfClasses = numel(params('classNames'));

    % Convert parameter values that have been read in from strings to the correct type.
    params('foldsToUse') = str2double(params('foldsToUse'));
    params('discardThreshold') = str2double(params('discardThreshold'));
    params('maxComponents') = str2double(params('maxComponents'));

    % Check that the input parameters are all the correct type, contain valid values and are present (if mandatory).
    % 1) check that the codes aren't both empty yadda yadda
    % 2) discardThreshold is between 0-1
    % 3) CV folds is an integer
    % 4) outputDir is set to '/' if called as the empty string.
    % 5) the folds, discard and components are numbers
    % 6) that there are at least two classes specified, and create a dummy one if not

    % Setup the RNG to ensure that results are reproducible.
    rng('default');

    % Setup the results directory.
    if (exist(outputDir, 'dir') ~= 7)
        % If the output directory does not exist, then create it.
        mkdir(outputDir);
    end

    % Create the mapping recording the description of each code.
    fidMapping = fopen(params('codeMapping'), 'r');
    mapCodesToDescriptions = textscan(fidMapping, '%s %s', 'Delimiter', '\t');
    fclose(fidMapping);
    mapCodesToDescriptions = containers.Map(mapCodesToDescriptions{1}, mapCodesToDescriptions{2});

    % Load the patient data from the input file. The result will be a 1x3 cell array, with the first entry being the patient IDs, the second
    % the codes and the third the counts.
    fidDataset = fopen(params('inputData'), 'r');
    data = textscan(fidDataset, '%d %s %d');
    fclose(fidDataset);

    % Determine unique patient IDs and codes, and create index mappings for each.
    uniquePatientIDs = unique(data{1});  % A list of the unique patient IDs in the dataset.
    patientIndexMap = containers.Map(uniquePatientIDs, 1:numel(uniquePatientIDs));  % A mapping of the patient IDs to their index in the uniquePatientIDs array.
    uniqueCodes = unique(data{2});  % A list of the unique codes in the dataset.
    codeIndexMap = containers.Map(uniqueCodes, 1:numel(uniqueCodes));  % A mapping of the codes to their index in the uniqueCodes array.

    % Create a sparse matrix of the data.
    % The matrix is created from three arrays: an array of row indices (sparseRows), an array of column indices (sparseCols) and an array of
    % values. The values will all be ones, as we are only interested in the presence or absence of an association between a patient and a code.
    % The matrix is created by saying that:
    % M = zeroes(numel(sparseRows), numel(sparseCols))
    % for i = 1:numel(sparseRows)
    %     M(sparseRows(i), sparseCols(i)) = 1
    % end
    % Conceptually a zero in an entry indicates that their is no association between the patient and a code, i.e. the patient does not have that code
    % in their medical history).
    sparseRows = cell2mat(values(patientIndexMap, num2cell(data{1})));  % Array of row indices corresponding to patient IDs.
    sparseCols = cell2mat(values(codeIndexMap, data{2}));  % Array of column indices corresponding to codes.
    dataMatrix = sparse(sparseRows, sparseCols, ones(numel(sparseCols), 1));

    % Determine the codes to use for partitioning the dataset into separate classes.
    % Any codes that are not recorded in the dataset will be ignored at this point.
    classCodes = cellfun(@(x) strsplit(x, ',')', params('classCodes'), 'UniformOutput', false);  % Codes used to determine membership of each class.
    classCodes = cellfun(@(x, y) iff(y, extract_child_codes(x, uniqueCodes), x), classCodes, params('classChildren'), 'UniformOutput', false);  % Extract children of the codes if needed.
    classCodes = cellfun(@(x) x(codeIndexMap.isKey(x)), classCodes, 'UniformOutput', false);  % Remove codes that don't exist in the dataset.
    classCodeIndices = cellfun(@(x) cell2mat(values(codeIndexMap, x)), classCodes, 'UniformOutput', false);  % Indices of the codes used.

    % Determine the examples containing the class specific codes.
    classExamples = cellfun(@(x) any(dataMatrix(:, x), 2), classCodeIndices, 'UniformOutput', false);  % Subsets of the dataset containing examples of each class.
    classExamples = cellfun(@(x) find(x), classExamples, 'UniformOutput', false);  % Indices of the examples in each class.

    % Determine the ambiguous examples.
    ambiguousExamples = [];  % The examples that occur in at least two of the classes.
    indexCombinations = nchoosek(1:numberOfClasses, 2);  % Get all pairs of index combinations. Each row contains a pair.
    for row = 1:numberOfClasses
        indexPair = indexCombinations(row, :);
        ambiguousExamples = union(ambiguousExamples, intersect(classExamples{indexPair(1)}, classExamples{indexPair(2)}));
    end

    % Generate the final sets of examples for each class by removing the ambiguous examples.
    classExamples = cellfun(@(x) setdiff(x, ambiguousExamples), classExamples, 'UniformOutput', false);
    numClassExamples = cellfun(@(x) numel(x), classExamples);

    % Select the codes that will be used for training.
    % These are codes that occur in > 50 patient records and were not used to partition the dataset into classes.
    % The removed codes either occur too infrequently to be reliable, or are able to perfectly separate the classes simply because they were chosen to separate them.
    codeOccurrences = sum(dataMatrix, 1);  % Sparse matrix recording the number of patients each code occurs in.
    indicesOfTrainingCodes = find(codeOccurrences > 50)';  % Column array of indices of the codes associated with over 50 patients.
    for i = 1:numberOfClasses
        % For each class, remove the codes used to determine the examples in the class.
        indicesOfTrainingCodes = setdiff(indicesOfTrainingCodes, classCodeIndices{i});
    end

    % Write out statistics about the codes.
    fidCodes = fopen([params('outputDir') '\CodeStatistics.tsv'], 'w');
    fprintf(fidCodes, 'Code\tDescription\tClass\tUsedForTraining\tTotalOccurences\tOccursInPositive\tOccursInNegative\n');
    for i = 1:numel(uniqueCodes)
        codeOfInterest = uniqueCodes{i};
        isCodePositive = any(strcmp(codeOfInterest, positiveCodes));  % If the code is being used to determine positive examples.
        isCodeNegative = any(strcmp(codeOfInterest, negativeCodes));  % If the code is being used to determine negative examples.
        codeClass = iff(isCodePositive, 'Positive', iff(isCodeNegative, 'Negative', 'Not_Used'));
        isCodeTraining = any(indicesOfTrainingCodes == i);  % Whether the code is to be used for training.
        patientsWithCode = dataMatrix(:, codeIndexMap(codeOfInterest));
        fprintf(fidCodes, '%s\t%s\t%s\t%s\t%d\t%d\t%d\n', codeOfInterest, query_dictionary(mapCodesToDescriptions, codeOfInterest), codeClass, ...
            isCodeTraining, nnz(patientsWithCode), nnz(patientsWithCode(positiveExamples, :)), nnz(patientsWithCode(negativeExamples, :)));
    end
    fclose(fidCodes);

end
