function main_mini_batch(fileParams)
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
    dirProject = fileparts(fileparts(pwd));  % Directory containing all code, data and results.
    dirLibrary = strcat(dirProject, '/Lib');  % Top level library directory.
    addpath(strcat(dirLibrary, '/JSONlab'));  % Make JSONLab available.
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
    end
    dataNorm = 0;  % Normalisation mode to use.
    if isfield(params, 'DataNorm')
        dataNorm = params.DataNorm;
    end
    discardThreshold = 0;  % Threshold to use when determining which examples to discard before training the second model.
    if isfield(params, 'DiscardThreshold')
        discardThreshold = params.DiscardThreshold;
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
        mkdir(dirOutput);
    end

    % Add the current directory to the Python path if needed.
    if count(py.sys.path, '') == 0
        insert(py.sys.path, int32(0), '');
    end

    % Create the mapping recording the description of each code. The results will be a struct with the two following entries:
    %   Codes contains the codes.
    %   Descriptions contains the description of each code.
    % Each entry is an 1xN cell array.
    codeData = py.matlab_file_parser.codelist_parser(fileCodeMap);
    codeData = struct(codeData);  % Convert py.dict to Matlab struct.
    codeData = structfun(@(x) cell(x), codeData, 'UniformOutput', false);  % Convert each entry from py.list to Matlab cell array.
    codeData.Codes = cellfun(@(x) char(x), codeData.Codes, 'UniformOutput', false);  % Convert the codes from py.str to Matlab char.
    codeData.Descriptions = cellfun(@(x) char(x), codeData.Descriptions, 'UniformOutput', false);  % Convert the descriptions from py.str to Matlab char.
    mapCodesToDescriptions = containers.Map(codeData.Codes, codeData.Descriptions);

    %%%%%%%%%%%%%%%%%%%%%%
    % Create Data Matrix %
    %%%%%%%%%%%%%%%%%%%%%%
    if isLoadWorkspace
        % Load the workspace.
        load(params.WorkspaceLocation);
    else
        % Load the patient data from the input file. The results will be a struct with the three following entries:
        %   IDs contains the patientIDs.
        %   Codes contains the codes.
        %   Counts contains the number of times the code occurred with the given patient.
        % Each entry is an 1xN cell array.
        fileDataset = params.DataLocation;
        data = py.matlab_file_parser.dataset_parser(fileDataset);
        data = struct(data);  % Convert py.dict to Matlab struct.
        data = structfun(@(x) cell(x), data, 'UniformOutput', false);  % Convert each entry from py.list to Matlab cell array.
        data.Codes = cellfun(@(x) char(x), data.Codes, 'UniformOutput', false);  % Convert the codes from py.str to Matlab char.

        % Determine unique patient IDs and codes, and create index mappings for each.
        uniquePatientIDs = unique(cell2mat(data.IDs));  % A list of the unique patient IDs in the dataset.
        patientIndexMap = containers.Map(uniquePatientIDs, 1:numel(uniquePatientIDs));  % A mapping of the patient IDs to their index in the uniquePatientIDs array.
        uniqueCodes = unique(data.Codes);  % A list of the unique codes in the dataset.
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
        sparseRows = cell2mat(values(patientIndexMap, data.IDs));  % Array of row indices corresponding to patient IDs.
        sparseCols = cell2mat(values(codeIndexMap, data.Codes));  % Array of column indices corresponding to codes.
        dataMatrix = sparse(sparseRows, sparseCols, double(cell2mat(data.Counts)));  % Sparse matrices need doubles, and double() can't take a cell array.

        % Save the loaded data.
        if isSaveWorkspace
            save(params.WorkspaceLocation, 'uniquePatientIDs', 'patientIndexMap', 'uniqueCodes', 'codeIndexMap', 'dataMatrix');
        end
    end

    % Normalise the data matrix.
    dataMatrix = normalise_data_matrix(dataMatrix, dataNorm);

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
    indicesOfTrainingCodes = find(patientsPerCode >= codeOccurrences)';  % Column array of indices of the codes associated with over 50 patients.
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
        codeDescription = query_dictionary(mapCodesToDescriptions, codeOfInterest);
        codeDescription = strrep(codeDescription, '%', '%%');  % Some code descriptions contain % signs that need escaping.
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

    % Create an array that records the class of each example and one that records the indices of the training examples.
    classOfExamples = zeros(size(dataMatrix, 1), 1);
    for i = 1:numberOfClasses
        classOfExamples(classExamples{i}) = i;
    end
    indicesOfTrainingExamples = cell2mat(classExamples');

    % Determine the model's training matrix and target array.
    trainingMatrix = dataMatrix(indicesOfTrainingExamples, indicesOfTrainingCodes);  % Data subset containing only examples and codes to use for training.
    trainingTarget = classOfExamples(indicesOfTrainingExamples);

    % Convert the classes in the training target array into a matrix with the same number of columns as there are classes.
    % Each column will correspond to a single class. Given trainingTarget, the array of classes, the matrix of classes (targetMatrix)
    % will be organised so that targetMatrix(i, 1) == 1 if trainingTarget(i) contains the first class.
    % Otherwise targetMatrix(i, 1) == 0. For example:
    %   target == [1 2 3 2 1]
    %                  [1 0 0]
    %                  [0 1 0]
    %   targetMatrix = [0 0 1]
    %                  [0 1 0]
    %                  [1 0 0]
    targetMatrix = logical(zeros(numel(trainingTarget), numberOfClasses));
    for i = 1:numberOfClasses
        targetMatrix(trainingTarget == i, i) = 1;
    end

    %%%%%%%%%%%%%%%%%%%
    % Train the Model %
    %%%%%%%%%%%%%%%%%%%
    if cvFolds == 0
        % Training if cross validation is not being used.
        fidPerformanceFirst = fopen(strcat(dirOutput, '/Performance_FirstModel.tsv'), 'w');
        fidPerformanceSecond = fopen(strcat(dirOutput, '/Performance_SecondModel.tsv'), 'w');
        fprintf(fidPerformanceFirst, 'NumIterations\tBatchSize\tAlpha\tLambda\tFinalGMean\tDescent\n');
        fprintf(fidPerformanceSecond, 'NumIterations\tBatchSize\tAlpha\tLambda\tFinalGMean\tDescent\n');
        for numIter = 1:numel(maxIterValues)
            for bSize = 1:numel(batchSizeValues)
                for aVal = 1:numel(alphaValues)
                    for lVal = 1:numel(lambdaValues)
                        % Display current test.
                        disp(sprintf('Now running - Iterations=%d    Batch Size=%d    Alpha=%1.4f    Lambda=%1.4f    Time=%s', ...
                            maxIterValues(numIter), batchSizeValues(bSize), alphaValues(aVal), lambdaValues(lVal), datestr(now)));

                        % Create, train and evaluate the first model on the training set.
                        regLogModelFirst = RegMultinomialLogistic(alphaValues(aVal), batchSizeValues(bSize), lambdaValues(lVal), maxIterValues(numIter));
                        recordOfDescentFirst = regLogModelFirst.train(trainingMatrix, trainingTarget, []);
                        predictionsFirst = regLogModelFirst.test(trainingMatrix);
                        performanceFirst = regLogModelFirst.calculate_performance(predictionsFirst, trainingTarget, [0.5]);

                        % Record information about the performance of the first model.
                        descentStringFirst = sprintf('%1.4f,', recordOfDescentFirst);
                        descentStringFirst = descentStringFirst(1:end-1);  % Strip off the final comma that was added.
                        fprintf(fidPerformanceFirst, '%d\t%d\t%1.4f\t%1.4f\t%1.4f\t%s\n', maxIterValues(numIter), batchSizeValues(bSize), ...
                            alphaValues(aVal), lambdaValues(lVal), performanceFirst.gMean, descentStringFirst);

                        % Remove the observations with posterior probabilities worse than the discard threshold.
                        % Also determine the new indices of the original training examples in the cut down set of good examples.
                        % This is calculated by taking the cumulative sum of a binary array indicating the good examples.
                        % Taking only those indices that indicate good examples (i.e. are not a 0 in goodExamples), then the cumsum
                        % indicates the new index of those examples in the cut down training set.
                        %                [0]             [1]
                        %                [1]             [1]
                        % goodExamples = [0]    cumsum = [1]
                        %                [1]             [2]
                        %                [1]             [3]
                        goodExamples = predictionsFirst(targetMatrix) >= discardThreshold;
                        newIndices = cumsum(goodExamples);
                        trainingMatrixSecond = trainingMatrix(goodExamples, :);
                        trainingTargetSecond = trainingTarget(goodExamples);
                        if numel(unique(trainingTargetSecond)) < numberOfClasses
                            % The prediction errors of the first model have caused all examples of (at least) one class to be removed from the dataset.
                            % Skip training the second model and writing out predictions/coefficients.
                            disp('WARNING: All examples of one class have been removed for having poor predictions.')
                            fprintf(fidPerformanceSecond, '%d\t%d\t%1.4f\t%1.4f\t-\t-\n', maxIterValues(numIter), batchSizeValues(bSize), ...
                                alphaValues(aVal), lambdaValues(lVal));
                        else
                            % Create, train and evaluate the second model using the cut down training set.
                            % The performance evaluation is actually performed on the whole training set rather than the cut down one.
                            regLogModelSecond = RegMultinomialLogistic(alphaValues(aVal), batchSizeValues(bSize), lambdaValues(lVal), maxIterValues(numIter));
                            recordOfDescentSecond = regLogModelSecond.train(trainingMatrixSecond, trainingTargetSecond, []);
                            predictionsSecond = regLogModelSecond.test(trainingMatrix);
                            performanceSecond = regLogModelSecond.calculate_performance(predictionsSecond, trainingTarget, [0.5]);

                            % Record information about the second model.
                            descentStringSecond = sprintf('%1.4f,', recordOfDescentSecond);
                            descentStringSecond = descentStringSecond(1:end-1);  % Strip off the final comma that was added.
                            fprintf(fidPerformanceSecond, '%d\t%d\t%1.4f\t%1.4f\t%1.4f\t%s\n', maxIterValues(numIter), batchSizeValues(bSize), ...
                                alphaValues(aVal), lambdaValues(lVal), performanceSecond.gMean, descentStringSecond);

                            % Record the model posteriors and predictions.
                            fidPredictions = fopen(strcat(dirOutput, '/Predictions.tsv'), 'w');
                            headerFirstModel = '\tFirstModel_MaxProbClass';
                            headerSecondModel = '\tSecondModel_MaxProbClass';
                            for i = 1:numberOfClasses
                                headerFirstModel = strcat(headerFirstModel, sprintf('\tFirstModel_%s_Posterior', classNames{i}));
                                headerSecondModel = strcat(headerSecondModel, sprintf('\tSecondModel_%s_Posterior', classNames{i}));
                            end
                            header = strcat('PatientID\tClass', headerFirstModel, '\tDiscarded', headerSecondModel, '\n');
                            fprintf(fidPredictions, header);
                            for i = 1:numel(indicesOfTrainingExamples)
                                patientID = uniquePatientIDs(indicesOfTrainingExamples(i));  % Get the ID of the patient.
                                patientClass = classNames{trainingTarget(i)};  % Get the actual class of the patient.
                                firstModelPredClass = classNames{performanceFirst.maxProb(i)};  % Get the predicted class of the patient.
                                firstModelPosteriors = predictionsFirst(i, :);  % Get the posteriors for the patient.
                                firstModelPosteriors = sprintf('%1.4f\t', firstModelPosteriors);
                                firstModelPosteriors = firstModelPosteriors(1:end-1);  % Strip off the final tab that was added.
                                discarded = iff(goodExamples(i), 'N', 'Y');  % Whether the patient was not used for training the second model.
                                secondModelPredClass = classNames{performanceSecond.maxProb(i)};  % Get the predicted class of the patient.
                                secondModelPosteriors = predictionsSecond(i, :);  % Get the posteriors for the patient.
                                secondModelPosteriors = sprintf('%1.4f\t', secondModelPosteriors);
                                secondModelPosteriors = secondModelPosteriors(1:end-1);  % Strip off the final tab that was added.
                                fprintf(fidPredictions, '%d\t%s\t%s\t%s\t%s\t%s\t%s\n', patientID, patientClass, firstModelPredClass, firstModelPosteriors, ...
                                    discarded, secondModelPredClass, secondModelPosteriors);
                            end
                            fclose(fidPredictions);

                            % Record the model coefficients.
                            fidCoefficients = fopen(strcat(dirOutput, '/Coefficients.tsv'), 'w');
                            headerFirstModel = '';
                            headerSecondModel = '';
                            for i = 1:numberOfClasses
                                headerFirstModel = strcat(headerFirstModel, sprintf('\tFirstModel_%s_Coefficient', classNames{i}));
                                headerSecondModel = strcat(headerSecondModel, sprintf('\tSecondModel_%s_Coefficient', classNames{i}));
                            end
                            header = strcat('Code\tDescription', headerFirstModel, headerSecondModel, '\n');
                            fprintf(fidCoefficients, header);
                            for i = 1:numel(uniqueCodes)
                                code = uniqueCodes{i};  % Get the code.
                                codeDescription = query_dictionary(mapCodesToDescriptions, code);  % Get the code's description.
                                codeDescription = strrep(codeDescription, '%', '%%');  % Some code descriptions contain % signs that need escaping.
                                codeUsedForTraining = i == indicesOfTrainingCodes;  % Boolean array indicating whether the code was used for training.
                                if sum(codeUsedForTraining) ~= 0
                                    % The code was used for training, so it has coefficients.
                                    codeIndex = find(codeUsedForTraining);
                                    codeIndex = codeIndex(1);  % Get the index of the code in terms of the coefficients.
                                    coefficientsFirst = regLogModelFirst.coefficients(codeIndex + 1, :);  % Add one as the coefficients have the bias term as the first entry.
                                    coefficientsSecond = regLogModelSecond.coefficients(codeIndex + 1, :);  % Add one as the coefficients have the bias term as the first entry.
                                    coefficients = sprintf('%1.4f\t', [coefficientsFirst coefficientsSecond]);
                                else
                                    % The code wasn't used for training, so has no coefficients.
                                    coefficients = repmat(sprintf('0\t'), 1, numberOfClasses * 2);
                                end
                                coefficients = coefficients(1:end-1);  % Strip off the final tab that was added.
                                fprintf(fidCoefficients, '%s\t%s\t%s\n', code, codeDescription, coefficients);
                            end
                            fclose(fidCoefficients);

                            % Record the predictions of the ambiguous examples using the models.
                            ambigData = dataMatrix(ambiguousExamples, indicesOfTrainingCodes);  % Data subset containing only ambiguous examples.
                            ambigPredsFirst = regLogModelFirst.test(ambigData);
                            ambigPerfFirst = regLogModelFirst.calculate_performance(ambigPredsFirst, [], []);
                            ambigPredsSecond = regLogModelSecond.test(ambigData);
                            ambigPerfSecond = regLogModelSecond.calculate_performance(ambigPredsSecond, [], []);
                            fidAmbigPredictions = fopen(strcat(dirOutput, '/AmbiguousPredictions.tsv'), 'w');
                            headerFirstModel = '\tFirstModel_MaxProbClass';
                            headerSecondModel = '\tSecondModel_MaxProbClass';
                            for i = 1:numberOfClasses
                                headerFirstModel = strcat(headerFirstModel, sprintf('\tFirstModel_%s_Posterior', classNames{i}));
                                headerSecondModel = strcat(headerSecondModel, sprintf('\tSecondModel_%s_Posterior', classNames{i}));
                            end
                            header = strcat('PatientID', headerFirstModel, '\tDiscarded', headerSecondModel, '\n');
                            fprintf(fidAmbigPredictions, header);
                            for i = 1:numel(ambiguousExamples)
                                patientID = ambiguousExamples(i);  % Get the ID of the patient.
                                firstModelPredClass = classNames{ambigPerfFirst.maxProb(i)};  % Get the predicted class of the patient.
                                firstModelPosteriors = ambigPredsFirst(i, :);  % Get the posteriors for the patient.
                                firstModelPosteriors = sprintf('%1.4f\t', firstModelPosteriors);
                                firstModelPosteriors = firstModelPosteriors(1:end-1);  % Strip off the final tab that was added.
                                secondModelPredClass = classNames{ambigPerfSecond.maxProb(i)};  % Get the predicted class of the patient.
                                secondModelPosteriors = ambigPredsSecond(i, :);  % Get the posteriors for the patient.
                                secondModelPosteriors = sprintf('%1.4f\t', secondModelPosteriors);
                                secondModelPosteriors = secondModelPosteriors(1:end-1);  % Strip off the final tab that was added.
                                fprintf(fidAmbigPredictions, '%d\t%s\t%s\t%s\t%s\n', patientID, firstModelPredClass, firstModelPosteriors, ...
                                    secondModelPredClass, secondModelPosteriors);
                            end
                            fclose(fidAmbigPredictions);
                        end
                    end
                end
            end
        end
        fclose(fidPerformanceFirst);
        fclose(fidPerformanceSecond);
    else
        % Training if cross validation is being used.
        fidPerformance = fopen(strcat(dirOutput, '/CVPerformance.tsv'), 'w');
        foldHeader = sprintf('Fold%d\t', 1:cvFolds);
        foldHeader = foldHeader(1:end-1);  % Strip off the final tab that was added.
        fprintf(fidPerformance, 'NumIterations\tBatchSize\tAlpha\tLambda\t%s\tMaxProbError\n', foldHeader);
        crossValPartitions = cvpartition(trainingTarget, 'KFold', cvFolds);  % Create cross validation partitions.
        for numIter = 1:numel(maxIterValues)
            for bSize = 1:numel(batchSizeValues)
                for aVal = 1:numel(alphaValues)
                    for lVal = 1:numel(lambdaValues)
                        % Display current test.
                        disp(sprintf('Now running - Iterations=%d    Batch Size=%d    Alpha=%1.4f    Lambda=%1.4f    Time=%s', ...
                            maxIterValues(numIter), batchSizeValues(bSize), alphaValues(aVal), lambdaValues(lVal), datestr(now)));

                        % Generate the record of the class prediction for each example in the full training set.
                        predictions = zeros(numel(trainingTarget), numberOfClasses);
                        maxProbClass = zeros(numel(trainingTarget), 1);
                        fprintf(fidPerformance, '%d\t%d\t%1.4f\t%1.4f', maxIterValues(numIter), batchSizeValues(bSize), alphaValues(aVal), lambdaValues(lVal));

                        % Perform the training.
                        for i = 1:crossValPartitions.NumTestSets
                            % Determine the model's training/testing matrix and target/testing array.
                            cvTrainingMatrix = trainingMatrix(crossValPartitions.training(i), :);
                            cvTestMatrix = trainingMatrix(crossValPartitions.test(i), :);
                            cvTrainingTarget = trainingTarget(crossValPartitions.training(i));
                            cvTestTarget = trainingTarget(crossValPartitions.test(i));

                            % Create, train and evaluate the model.
                            regLogModel = RegMultinomialLogistic(alphaValues(aVal), batchSizeValues(bSize), lambdaValues(lVal), maxIterValues(numIter));
                            recordOfDescent = regLogModel.train(cvTrainingMatrix, cvTrainingTarget, []);
                            cvPredictions = regLogModel.test(cvTestMatrix);
                            performance = regLogModel.calculate_performance(cvPredictions, cvTestTarget, [0.5]);

                            % Record the model's performance.
                            predictions(crossValPartitions.test(i), :) = cvPredictions;
                            maxProbClass(crossValPartitions.test(i)) = performance.maxProb;
                            fprintf(fidPerformance, '\t%1.4f', performance.gMean);
                        end

                        % Calculate the maximum probability error.
                        maxProbError = sum(trainingTarget ~= maxProbClass) / numel(trainingTarget);
                        fprintf(fidPerformance, '\t%1.4f', maxProbError);
                        fprintf(fidPerformance, '\n');
                    end
                end
            end
        end
        fclose(fidPerformance);

end