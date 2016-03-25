function [regLogModel, predictions, trainingTarget, recordOfDescent] = main_mini_batch(fileParams)
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
    rng(0, 'twister');

    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % Check Input Parameters %
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    % Load the parameters.
    dirProject = fileparts(pwd);  % Directory containing all code, data and results.
    dirLibrary = strcat(dirProject, '/Lib');  % Top level library directory.
    addpath(strcat(dirLibrary, '/JSONLab'));  % Make JSONLab available.
    addpath(strcat(dirLibrary, '/DETconf'));  % Make evaluation function available.
    params = loadjson(fileParams);

    % Check parameters that need to be present.
    errorsFound = {};
    if ~isfield(params, 'DataLocation') && ~isfield(params, 'WorkspaceLocation')
        errorsFound{size(errorsFound, 2) + 1} = 'A field called DataLocation or WorkplaceLocation must be present in the parameter file.';
    end
    if isfield(params, 'DataLocation') && (exist(params.DataLocation, 'file') ~= 2)
        errorsFound{size(errorsFound, 2) + 1} = 'A dataset location was provided, but the file does not exist.';
    end
    if ~isfield(params, 'CodeMapping')
        errorsFound{size(errorsFound, 2) + 1} = 'A field called CodeMapping must be present in the parameter file.';
    elseif exist(params.CodeMapping, 'file') ~= 2
        errorsFound{size(errorsFound, 2) + 1} = 'The location provided for the code mapping file does not contain a file.';
    end
    if ~isfield(params, 'ResultsLocation')
        errorsFound{size(errorsFound, 2) + 1} = 'A field called ResultsLocation must be present in the parameter file.';
    end
    if ~isfield(params, 'Classes')
        errorsFound{size(errorsFound, 2) + 1} = 'A field called Classes must be present in the parameter file.';
    end
    if size(errorsFound, 2) > 0
        % There was an error found, so display them and quit.
        errorMessage = strjoin(errorsFound, '\n');
        error(errorMessage);
    end

    % Determine whether the data will be loaded from a .mat file or by parsing a text file, and whether the created matrix should be saved.
    isSaveWorkspace = isfield(params, 'WorkspaceLocation');  % Whether the workspace should be saved if it does not already exist.
    isLoadWorkspace = isSaveWorkspace && (exist(params.WorkspaceLocation, 'file') == 2);  % Whether the data should be loaded from a .mat file.

    % Load simple non-class parameters.
    fileCodeMap = params.CodeMapping;  % Location of the file containing the mapping between codes and descriptions.
    dirOutput = params.ResultsLocation;  % Location of the directory to write the results to.
    lambdaValues = [0.01];  % Regularisation parameter values.
    if isfield(params, 'Lambda')
        lambdaValues = params.Lambda;
    end
    alphaValues = [0.01];  % Gradient descent learning rates.
    if isfield(params, 'Alpha')
        alphaValues = params.Alpha;
    end
    batchSizeValues = [100];  % Batch sizes for the mini batch.
    if isfield(params, 'BatchSize')
        batchSizeValues = params.BatchSize;
    end
    maxIterValues = [10];  % Maximum numbers of mini batch iterations.
    if isfield(params, 'MaxIter')
        maxIterValues = params.MaxIter;
    end
    codeOccurrences = 0;  % The number of patients each code must appear in to be used in the model.
    if isfield(params, 'CodeOccurrences')
        codeOccurrences = params.CodeOccurrences;
    end
    cvFolds = 0;  % Number of cross validation folds to use.
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
    isCollectorClass = false;
    if numberNonCodeClasses == 1
        % There is a class with no codes. This class will contain all examples not belonging to another class.
        isCollectorClass = true;
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

    %%%%%%%%%%%%%%%%%%%%%%
    % Create Data Matrix %
    %%%%%%%%%%%%%%%%%%%%%%
    if isLoadWorkspace
        % Load the workspace.
        load(params.WorkspaceLocation);
    else
        % Load the patient data from the input file. The result will be a 1x3 cell array,
        % with the first entry being the patient IDs, the second the codes and the third the counts.
        fileDataset = params.DataLocation;  % Location of the dataset used for training.
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
%%%%%
        dataMatrix = sparse(sparseRows, sparseCols, ones(numel(sparseCols), 1));
%        dataMatrix = sparse(sparseRows, sparseCols, double(data{3}));
%%%%%%
        % Save the loaded data.
        if isSaveWorkspace
            save(params.WorkspaceLocation, 'uniquePatientIDs', 'patientIndexMap', 'uniqueCodes', 'codeIndexMap', 'dataMatrix');
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Determine Codes and Examples to Use %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    if isCollectorClass
        usedExamples = vertcat(classExamples{:});  % Examples that belong to a class already.
        availableExamples = setdiff(1:numel(uniquePatientIDs), usedExamples)';  % Examples that don't belong to a class.
        classExamples{strcmp(classNames, collectorClass)} = availableExamples;
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
        errors = {};
        for i = 1:numel(missingClasses)
            errors{end+1} = sprintf('Class %s contains no examples following the removal of ambiguous examples.', missingClasses{i});
        end
        errorMessage = strjoin(errors, '\n');
        error(errorMessage);
    end

    % Remove the codes used to determine class membership from the dataset.
    % Need to transpose the class code indices cell array as there is only a guarantee that they have the same number of columns,
    % while a 1xN cell array would need all cells to have the same number of rows before cell2mat can be used.
    dataMatrix(:, cell2mat(classCodeIndices')) = 0;

    % Create an array that records the class of each example and one that records the indices of the training examples.
    classOfExamples = zeros(size(dataMatrix, 1), 1);
    for i = 1:numberOfClasses
        classOfExamples(classExamples{i}) = i;
    end
    indicesOfTrainingExamples = cell2mat(classExamples');

    % Determine the model's training matrix and target array.
    trainingMatrix = dataMatrix(indicesOfTrainingExamples, indicesOfTrainingCodes);  % Data subset containing only examples and codes to use for training.
    trainingTarget = classOfExamples(indicesOfTrainingExamples);

    %%%%%%%%%%%%%%%%%%%
    % Train the Model %
    %%%%%%%%%%%%%%%%%%%
    if cvFolds == 0
        % Training if cross validation is not being used.
        for numIter = 1:numel(maxIterValues)
            for bSize = 1:numel(batchSizeValues)
                for aVal = 1:numel(alphaValues)
                    for lVal = 1:numel(lambdaValues)
                        % Create model, train model and make predictions on the training set.
                        regLogModel = RegMultinomialLogistic(alphaValues(aVal), batchSizeValues(bSize), lambdaValues(lVal), maxIterValues(numIter));
                        recordOfDescent = regLogModel.train(trainingMatrix, trainingTarget, []);
                        predictions = regLogModel.test(trainingMatrix);
                    end
                end
            end
        end
    else
        % Training if cross validation is being used.

        % Create cross validation partitions.
        crossValPartitions = cvpartition(trainingTarget, 'KFold', cvFolds);

        for numIter = 1:numel(maxIterValues)
            for bSize = 1:numel(batchSizeValues)
                for aVal = 1:numel(alphaValues)
                    for lVal = 1:numel(lambdaValues)
                        % Generate the record of the class prediction for each example in the full training set.
                        predictions = zeros(numel(trainingTarget), numberOfClasses);

                        % Perform the training.
                        for i = 1:crossValPartitions.NumTestSets
                            % Determine the model's training matrix and target array.
                            cvTrainingMatrix = trainingMatrix(crossValPartitions.training(i), :);  % Subset of training examples in this fold.
                            cvTrainingTarget = trainingTarget(crossValPartitions.training(i));

                            % Create the model.
                            regLogModel = RegMultinomialLogistic(alphaValues(aVal), batchSizeValues(bSize), lambdaValues(lVal), maxIterValues(numIter));

                            % Train the model.
                            recordOfDescent = regLogModel.train(cvTrainingMatrix, cvTrainingTarget, []);

                            % Generate the predictions for this test fold.
                            cvTestMatrix = trainingMatrix(crossValPartitions.test(i), :);
                            cvPredictions = regLogModel.test(cvTestMatrix);
                            predictions(crossValPartitions.test(i), :) = cvPredictions;
                        end
                    end
                end
            end
        end

end