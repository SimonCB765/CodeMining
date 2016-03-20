function [classExamples, classOfExamples, indicesOfTrainingExamples] = main_mini_batch(fileParams)
    % Perform clinical code identification using logistic regression.
    %
    % Checking is performed to ensure that the required parameters are present in the parameter file,
    % but no checking of correctness is performed.
    %
    % Please see the README for information on the correct formatting and values for the parameter file.
    %
    % Keyword Arguments
    % fileParams - JSON format file containing function arguments.

    % Setup the RNG to ensure that results are reproducible.
    rng('default');

    % Determine the locations of the directories needed.
    dirScript = pwd;  % Directory containing this script.
    dirProject = fileparts(pwd);  % Directory containing all code, data and results.
    dirData = strcat(dirProject, '/Data');  % Top level data directory.
    dirResults = strcat(dirProject, '/Results');  % Top level results directory.
    dirLibrary = strcat(dirProject, '/Lib');  % Top level library directory.
    addpath(strcat(dirLibrary, '/JSONLab'));  % Make JSONLab available.
    addpath(strcat(dirLibrary, '/DETconf'));  % Make evaluation function available.

    % Parse and check the parameters.
    params = loadjson(fileParams);
    errorsFound = {};
    if ~isfield(params, 'DataLocation')
        errorsFound{size(errorsFound, 2) + 1} = 'A field called DataLocation must be present in the parameter file.';
    end
    if ~isfield(params, 'CodeMapping')
        errorsFound{size(errorsFound, 2) + 1} = 'A field called CodeMapping must be present in the parameter file.';
    end
    if ~isfield(params, 'ResultsLocation')
        errorsFound{size(errorsFound, 2) + 1} = 'A field called ResultsLocation must be present in the parameter file.';
    end
    if ~isfield(params, 'Classes')
        errorsFound{size(errorsFound, 2) + 1} = 'A field called Classes must be present in the parameter file.';
    end
    if ~isfield(params, 'Lambda')
        errorsFound{size(errorsFound, 2) + 1} = 'A field called Lambda must be present in the parameter file.';
    end
    if ~isfield(params, 'Alpha')
        errorsFound{size(errorsFound, 2) + 1} = 'A field called Alpha must be present in the parameter file.';
    end
    if size(errorsFound, 2) > 0
        % There was an error found, so display them and quit.
        errorMessage = strjoin(errorsFound, '\n');
        error(errorMessage);
    end

    % Load the simple non-class parameters.
    fileDataset = strcat(dirData, '/', params.DataLocation);  % Location of the dataset used for training.
    fileCodeMap = strcat(dirData, '/', params.CodeMapping);  % Location of the file containing the mapping between codes and descriptions.
    dirOutput = strcat(dirResults, '/', params.ResultsLocation);  % Location of the directory to write the results to.
    lambdaValues = params.Lambda;
    alphaValues = params.Alpha;
    codeOccurrences = 0;  % The number of patients each code must appear in to be used in the model.
    if isfield(params, 'CodeOccurrences')
        codeOccurrences = params.CodeOccurrences;
    end
    cvFolds = 0;
    if isfield(params, 'CVFolds')
        cvFolds = params.CVFolds;
        if cvFolds < 2
            % No way to use cross validation with less than two folds.
            cvFolds = 0;
        end
    end

    % Load the class parameters.
    % classCodes and classChildren will be structs with one element per class.
    % Each element in classCodes will contain a cell array recording the codes used to determine examples in the class.
    % Each element in classChildren will contain a cell array recording whether child codes are to be extracted.
    % For example, given JSON input like:
    %   "Classes" :
    %   {
    %   "Type1" : [{"Code" : "C10E", "GetChildren" : "Y"}, {"Code" : "44J3", "GetChildren" : "N"}],
    %   "Type2" : [{"Code" : "C10F", "GetChildren" : "Y"}],
    %   "Negative" : []
    %   }
    % The results structs would be:
    %   classCodes = Type1: {'C10E' '44J3'}
    %                Type2: {'C10F'}
    %                Negative: {}
    %   classChildren = Type1: {[1]  [0]}
    %                   Type2: {[1]}
    %                   Negative: {}
    classData = params.Classes;
    classNames = fieldnames(classData)';
    numberOfClasses = numel(classNames);
    for i = 1:numberOfClasses
        className = classNames{i};
        codes = {};
        children = {};
        currentClassData = classData.(className);
        for j = 1:numel(currentClassData)
            currentClassData{j};
            if isfield(currentClassData{j}, 'Code')
                codes{j} = currentClassData{j}.Code;
            else
                error('Entry %d for class %s does not have a code defined', j, className)
            end
            if isfield(currentClassData{j}, 'GetChildren')
                children{j} = lower(currentClassData{j}.GetChildren) == 'y';
            else
                children{j} = true;
            end
        end
        classCodes.(className) = codes;
        classChildren.(className) = children;
    end

    % Check whether there is a collector class.
    classesWithCodes = cellfun(@(x) numel(classCodes.(x)), classNames) == 0;
    numberNonCodeClasses = sum(classesWithCodes) == 1;
    if numberNonCodeClasses == 1
        % There is a class with no codes. This class will contain all examples not belonging to another class.
        isColectorClass = true;
        collectorClass = classNames(classesWithCodes);
    elseif numberNonCodeClasses > 1
        % There are too many classes without codes.
        error('Only one class can contain no codes');
    end

    % Setup the output directory.
    if (exist(dirOutput, 'dir') ~= 7)
        % If the output directory does not exist, then create it.
        mkdir(outputDir);
    end

    % Create the mapping recording the description of each code.
    fidMapping = fopen(fileCodeMap, 'r');
    mapCodesToDescriptions = textscan(fidMapping, '%s%s', 'Delimiter', '\t');
    fclose(fidMapping);
    mapCodesToDescriptions = containers.Map(mapCodesToDescriptions{1}, mapCodesToDescriptions{2});

    % Load the patient data from the input file. The result will be a 1x3 cell array,
    % with the first entry being the patient IDs, the second the codes and the third the counts.
    fidDataset = fopen(fileDataset, 'r');
    data = textscan(fidDataset, '%d%s%d', 'Delimiter', '\t');
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
    % Conceptually a zero in an entry indicates that there is no association between the patient and a code, i.e. the patient does not have that code
    % in their medical history).
    sparseRows = cell2mat(values(patientIndexMap, num2cell(data{1})));  % Array of row indices corresponding to patient IDs.
    sparseCols = cell2mat(values(codeIndexMap, data{2}));  % Array of column indices corresponding to codes.
    dataMatrix = sparse(sparseRows, sparseCols, ones(numel(sparseCols), 1));

    % Determine the codes to use for partitioning the dataset into separate classes.
    % Extract child codes if needed, and discard codes that are not recorded in the dataset.
    % The arrays are:
    %   classCodes - a 1 x numberOfClasses cell array of Nx1 cell arrays. Each cell contains a cell array recording the codes that define a class.
    %   classCodeIndices - a 1 x numberOfClasses cell array of Nx1 arrays. Each cell contains an array recording the indices of the codes
    %       that define a class.
    classCodes = cellfun(@(x) extract_child_codes(classCodes.(x), uniqueCodes, classChildren.(x)), classNames, 'UniformOutput', false);
    classCodes = cellfun(@(x) x(codeIndexMap.isKey(x)), classCodes, 'UniformOutput', false);  % Remove codes that don't exist in the dataset.
    classCodeIndices = cellfun(@(x) cell2mat(values(codeIndexMap, x)), classCodes, 'UniformOutput', false);  % Indices of the codes used.

    % Determine the examples containing the class specific codes.
    classExamples = cellfun(@(x) any(dataMatrix(:, x), 2), classCodeIndices, 'UniformOutput', false);  % Subsets of the dataset containing examples of each class.
    classExamples = cellfun(@(x) find(x), classExamples, 'UniformOutput', false);  % Indices of the examples in each class.

    % Determine the ambiguous examples.
    ambiguousExamples = [];  % The examples that occur in at least two of the classes.
    indexCombinations = nchoosek(1:numberOfClasses, 2);  % Get all pairs of index combinations. Each row contains a pair.
    for row = 1:size(indexCombinations, 1)
        indexPair = indexCombinations(row, :);
        ambiguousExamples = union(ambiguousExamples, intersect(classExamples{indexPair(1)}, classExamples{indexPair(2)}));
    end

    % Add the examples for the collector class.
    if isColectorClass
        usedExamples = vertcat(classExamples{:});  % Examples that belong to a class already.
        availableExamples = setdiff(1:numel(uniquePatientIDs), usedExamples)';  % Examples that don't belong to a class.
        classExamples{find(strcmp(classNames, collectorClass))} = availableExamples;
    end

    % Select the codes that will be used for training.
    % These codes occur in enough patients and were not used to partition the dataset into classes.
    patientsPerCode = sum(dataMatrix, 1);  % Sparse matrix recording the number of patients each code occurs in.
    indicesOfTrainingCodes = find(patientsPerCode > codeOccurrences)';  % Column array of indices of the codes associated with over 50 patients.
    for i = 1:numberOfClasses
        % For each class, remove the indices of the codes used to determine the examples in the class from the array of training code indices.
        indicesOfTrainingCodes = setdiff(indicesOfTrainingCodes, classCodeIndices{i});
    end

    % Write out statistics about the codes.
    fidCodes = fopen(strcat(dirOutput, '/CodeStatistics.tsv'), 'w');
    header = 'Code\tDescription\tClass\tUsedForTraining\tTotalOccurences';
    for i = 1:numberOfClasses
        header = strcat(header, sprintf('\tOccurencesIn_%s', classNames{i}));
    end
    header = strcat(header, '\n');
    fprintf(fidCodes, header);
    for i = 1:numel(uniqueCodes)
        codeOfInterest = uniqueCodes{i};
        codeDescription = query_dictionary(mapCodesToDescriptions, codeOfInterest);  % Some code descriptions contain % signs that need escaping.
        codeDescription = strrep(codeDescription, '%', '%%');
        codeClass = cellfun(@(x) ~isempty(find(strcmp(x, codeOfInterest), 1)), classCodes);
        codeClass = classNames(codeClass);
        if (isempty(codeClass))
            codeClass = '-';
        else
            codeClass = codeClass{1};
        end
        isCodeTraining = iff(any(indicesOfTrainingCodes == i), 'Y', 'N');  % Whether the code is to be used for training.
        patientsWithCode = dataMatrix(:, codeIndexMap(codeOfInterest));  % Data slice of the patients that are associated with the code.
        output = sprintf('%s\t%s\t%s\t%s\t%d', codeOfInterest, codeDescription, codeClass, isCodeTraining, nnz(patientsWithCode));
        for j = 1:numberOfClasses
            output = strcat(output, sprintf('\t%d', nnz(patientsWithCode(classExamples{j}, :))));
        end
        output = strcat(output, '\n');
        fprintf(fidCodes, output);
    end
    fclose(fidCodes);

    % Generate the final sets of examples for each class by removing the ambiguous examples.
    % Perform this after writing out the code information so that the written out information contains code occurrence counts before
    % ambiguous examples are removed.
    classExamples = cellfun(@(x) setdiff(x, ambiguousExamples), classExamples, 'UniformOutput', false);
    numClassExamples = cellfun(@(x) numel(x), classExamples);

    % Check whether any classes have no examples.
    missingClasses = classNames(numClassExamples == 0);
    if (~isempty(missingClasses))
        % There are missing classes.
        errors = {}
        for i = 1:numel(missingClasses)
            errors{end+1} = sprintf('Class %s contains no examples following the removal of ambiguous examples.', missingClasses{i})
        end
        errorMessage = strjoin(errors, '\n');
        error(errorMessage);
    end

    % Create arrays that record the class of each example and the indices of the training examples.
    classOfExamples = zeros(size(dataMatrix, 1), 1);
    for i = 1:numberOfClasses
        classOfExamples(classExamples{i}) = i;
    end
    indicesOfTrainingExamples = cell2mat(classExamples');

    % Train the model.
    if cvFolds == 0
        % Training if cross validation is not being used.
        % Determine the model's training matrix and target array.
        trainingMatrix = dataMatrix(indicesOfTrainingExamples, indicesOfTrainingCodes);  % Subset of the dataset containing only examples and codes to use for training.
        trainingTarget = classOfExamples(indicesOfTrainingExamples);
    else
        % Training if cross validation is being used.
        % Create cross validation partitions.
        cvPartitionArray = classOfExamples(indicesOfTrainingExamples);
        crossValPartition = cvpartition(cvPartitionArray, 'KFold', params('foldsToUse'));

        % Perform the training.
        for i = 1:crossValPartition.NumTestSets
            % Determine the model's training matrix and target array.
            trainingMatrix = dataMatrix(crossValPartition.training(1), indicesOfTrainingCodes);  % Subset of the dataset containing only examples and codes to use for training.
            trainingTarget = classOfExamples(crossValPartition.training(1));
        end
    end
end