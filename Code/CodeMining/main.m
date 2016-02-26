% Reset the workspace.
clear all
clc
close all
rng('default')  % Ensure results are reproducible.

%% Setup globals and flags.

foldsToUse = 10;  % The number of CV folds to use.

% Load the mapping of Read V2 codes (and some EMIS vendor codes) to their descriptions descriptions.
load coding

% Flag to indicate which comparison is being done. If set to true, then the comparison is between people with type 1 and
% type 2 diabetes, else it is between people with diabetes and people without.
% Also determine the threshold at which recodings will be deemed to be successful. Anything <= recodingCutoff OR >= (1 - recodingCutoff) is considered strong enough evidence for recoding.
isType1Type2 = true;
recodingCutoff = 0.1;

% Setup the results directory.
if isType1Type2
    subFolder = 'Type1VsType2';
else
    subFolder = 'DiabetesVsNonDiabetes';
end
if (exist(subFolder, 'dir') == 7)
    % If the directory exists then delete it before creating it.
    rmdir(subFolder, 's');
end
mkdir(subFolder);

%% Process the patient dataset.

% Load the data on the patients. Each row in the file contains:
% PatentID\tCode\tOccurences
% Therefore, each line records the number of times a specific Read V2 code occurs in a specific patient's medical record.
% Both patient IDs and Read codes are likely to appear on more then one line (as patients have more than one code associated
% them and codes are associated with more than one patient).
% data.id is an array containing the patient IDs.
% data.key is an array containing the Read codes.
% data.count is an array containing the association counts.
%[data.id, data.key, data.counts] = textread('CutDownPatientData.tsv', '%d %s %d');
[data.id, data.key, data.counts] = textread('PatientData.tsv', '%d %s %d');

% Determine unique patient IDs and codes, and create index mappings for each.
uniqueCodes = unique(data.key);  % A list of the unique Read codes in the dataset.
codeIndexMap = containers.Map(uniqueCodes, 1:numel(uniqueCodes));  % Map the Read codes to their index in the uniqueCodes array.
uniquePatientIDs = unique(data.id);  % A list of all the patient IDs in the dataset.
patientIndexMap = containers.Map(uniquePatientIDs, 1:numel(uniquePatientIDs));  % Map the patient IDs to their index in the uniquePatientIDs array.

% Create the sparse matrix.
% The matrix is created from three arrays: an array of row indices (sparseRows), an array of column indices (sparseCols) and an array of
% values (data.counts). The matrix is created by saying that the entry in the matrix M[sparseRows[i], sparseCols[i]] = data.counts[i].
% sparseRows contains one value for each entry in the dataset file, with each value being the index of a patient ID such that sparseRows[i] = the
%     index of the patient ID appearing in the ith row of the dataset file.
% sparseCols contains one value for each entry in the dataset file, with each value being the index of a Read code such that sparseCows[i] = the
%     index of the code appearing in the ith row of the dataset file.
% Thus, this sparse matrix will have a row for each patient ID (10,000 rows) and one column for each Read code in the dataset.
sparseRows = cell2mat(values(patientIndexMap, num2cell(data.id)));
sparseCols = cell2mat(values(codeIndexMap, data.key));
dataMatrix = sparse(sparseRows, sparseCols, data.counts, numel(uniquePatientIDs), numel(uniqueCodes));

% Determine Read codes (and their indices) for the codes that indicate diabetes. These should be manually examined to make sure that
% no child codes that don't truly indicate diabetes have been included.
type1DiabetesCodes = uniqueCodes(cellfun(@(x) ~isempty(regexp(x, '^C10E')), uniqueCodes));  % Codes that begin with C10E.
type2DiabetesCodes = uniqueCodes(cellfun(@(x) ~isempty(regexp(x, '^C10F')), uniqueCodes));  % Codes that begin with C10F.
diabetesCodes = uniqueCodes(cellfun(@(x) ~isempty(regexp(x, '^C10')), uniqueCodes));  % Codes that begin with C10.
type1DiabetesIndices = cell2mat(values(codeIndexMap, type1DiabetesCodes));  % The indices of the type 1 diabetes codes.
type2DiabetesIndices = cell2mat(values(codeIndexMap, type2DiabetesCodes));  % The indices of the type 2 diabetes codes.
diabetesIndices = cell2mat(values(codeIndexMap, diabetesCodes));  % The indices of the diabetes codes.

% Write out the codes that will be used to indicate diabetes.
fid = fopen([subFolder '\DiabetesCodesUsed.tsv'], 'w');
fprintf(fid, 'Type\tCode\tDescription\tPatientsAssociatedWith\tOccurencesWithC10E\tOccurencesWithC10F\tOccurrencesWithBoth\n');
for i = 1:numel(diabetesCodes)
    codeOfInterest = diabetesCodes{i};
    occurrences = nnz(dataMatrix(:, codeIndexMap(codeOfInterest)));
    occurrencesWithC10E = nnz((dataMatrix(:, codeIndexMap('C10E')) > 0) & (dataMatrix(:, codeIndexMap(codeOfInterest)) > 0));
    occurrencesWithC10F = nnz((dataMatrix(:, codeIndexMap('C10F')) > 0) & (dataMatrix(:, codeIndexMap(codeOfInterest)) > 0));
    occurrencesWithBoth = nnz((dataMatrix(:, codeIndexMap('C10E')) > 0) & (dataMatrix(:, codeIndexMap('C10F')) > 0) & (dataMatrix(:, codeIndexMap(codeOfInterest)) > 0));
    type1Code = sum(strcmp(codeOfInterest, type1DiabetesCodes)) == 1;
    type2Code = sum(strcmp(codeOfInterest, type2DiabetesCodes)) == 1;
    diabetesType = iff(type1Code, '1', iff(type2Code, '2', '-'));
    fprintf(fid, '%s\t%s\t%s\t%d\t%d\t%d\t%d\n', diabetesType, codeOfInterest, query_dictionary(coding, codeOfInterest), occurrences, occurrencesWithC10E, occurrencesWithC10F, occurrencesWithBoth);
end
fclose(fid);

% Generate the positive and negative examples.
% Start by creating a boolean matrix with one row per patient and one column per code being used to indicate the disease.
% Next flatten this matrix columnwise into a single (column) array using logical OR (provided by any(, 2))
% This enables simpler indexing when finding the positive and negative examples.
if (isType1Type2)
    patientsWithType1 = full(dataMatrix(:, type1DiabetesIndices) > 0);  % Boolean matrix. A 1 indicates that the patient is associated with the type 1 code.
    patientsWithType1 = any(patientsWithType1, 2);  % Boolean array. A 1 indicates that the patient has type 1 diabetes.
    patientsWithType2 = full(dataMatrix(:, type2DiabetesIndices) > 0);  % Boolean matrix. A 1 indicates that the patient is associated with the type 2 code.
    patientsWithType2 = any(patientsWithType2, 2);  % Boolean array. A 1 indicates that the patient has type 2 diabetes.

    patientsWithBothTypes = find(patientsWithType1 & patientsWithType2);  % Indices of patients with both type 1 and type 2 diabetes.
    positiveExamples = setdiff(find(patientsWithType1), find(patientsWithType2));  % Indices of patients with ONLY type 1 diabetes.
    negativeExamples = setdiff(find(patientsWithType2), find(patientsWithType1));  % Indices of patients with ONLY type 2 diabetes.
    
    % Remove the codes used to determine type 1 and type 2 diabetes from the dataset to prevent their use as predictor variables.
    % This can be achieved by setting all counts of them to 0, as they will then not be picked up as occuring in enough people to warrant use.
    dataMatrix(:, [type1DiabetesIndices; type2DiabetesIndices]) = 0;
else
    diabeticPatients = full(dataMatrix(:, diabetesIndices) > 0);  % Boolean matrix. A 1 indicates that the patient is associated with the diabetes code.
    diabeticPatients = any(diabeticPatients, 2);  % Boolean array. A 1 indicates that the patient has diabetes.
    
    positiveExamples = find(diabeticPatients);  % Indices of patients with either diabetes.
    negativeExamples = setdiff((1:numel(uniquePatientIDs))', positiveExamples);  % Indices of patients without diabetes.
    
    % Remove the codes used to determine diabetes from the dataset to prevent their use as predictor variables.
    % This can be achieved by setting all counts of them to 0, as they will then not be picked up as occuring in enough people to warrant use.
    dataMatrix(:, diabetesIndices) = 0;
end
numPositiveExamples = numel(positiveExamples);
numNegativeExamples = numel(negativeExamples);

% Select only those codes that occur in more than 50 patient records.
% As dataMatrix contains counts, we need > 0 to make a boolean matrix that just records presence/absence before we sum.
codePresence = dataMatrix > 0;  % Convert sparse matrix from recording counts of associations between a patient and a given code to simply recording presence/absence of association.
codeOccurrences = full(sum(codePresence));  % Matrix of counts of the number of different patients each code is associated with.
indicesOfCommonCodes = find(codeOccurrences > 50);  % Array of indices of the codes associated with over 50 patients.

% Calculate the entropy of the codes with enough associations.
% I believe this is calculated as the entropy between the histograms of counts from the two classes for a given code.
% entropyReordering is a reordering of the indices of the entries in indicesOfCommonCodes by descending entropy.
% entropy is the entropy of the common codes ordered according to their indices position in indicesOfCommonCodes.
sparseSubsetCommonCodes = dataMatrix([positiveExamples' negativeExamples'], indicesOfCommonCodes);  % Subset of the sparse matrix containing only the examples being used and the codes with enough associations.
sparseSubsetTarget = [ones(numPositiveExamples, 1); zeros(numNegativeExamples, 1)];  % Target array for the sparse subset. 1s for the positive examples and 0s for negative examples.
[entropyReordering, entropy] = rankfeatures(sparseSubsetCommonCodes', sparseSubsetTarget, 'Criterion', 'entropy');  % Calculate entropy for the codes. Transpose the sparse subset to calculate entropy for codes not examples.

% Write out the entropy for each code that occurs in greater than 50 patient records.
sortedEntropy = entropy(entropyReordering);
codeIndicesSortedByEntropy = indicesOfCommonCodes(entropyReordering);
codesSortedByEntropy = uniqueCodes(codeIndicesSortedByEntropy);
fid = fopen([subFolder '\Entropy.tsv'], 'w');
fprintf(fid, 'Code\tDescription\tEntropy\tPositivesOccursWith\tNegativesOccursWith\n');
for i = 1:numel(entropy)
    codeIndex = codeIndicesSortedByEntropy(i);
    codeOfInterest = codesSortedByEntropy{i};
    codeEntropy = sortedEntropy(i);
    positiveOccurences = numel(find(dataMatrix(positiveExamples, codeIndex)));
    negativeOccurences = numel(find(dataMatrix(negativeExamples, codeIndex)));
    fprintf(fid, '%s\t%s\t%1.4f\t%d\t%d\n', codeOfInterest, query_dictionary(coding, codeOfInterest), codeEntropy, positiveOccurences, negativeOccurences);
end
fclose(fid);

% Find the codes with infinite entropy, and therefore the ones that are only present in one class.
indicesOfInfEntropyCodes = indicesOfCommonCodes(entropy == inf);  % The indices of the codes with infinite entropy.
numberOfInfCodes = numel(indicesOfInfEntropyCodes);  % Number of codes with infinite entropy.
infiniteEntropyCodes = uniqueCodes(indicesOfInfEntropyCodes);  % The names of the codes with infinite entropy.

% Determine which class the infinite entropy codes appear in.
positiveInfEntropy = dataMatrix(positiveExamples, indicesOfInfEntropyCodes);  % Sparse matrix containing all positive examples associated with an infinite entropy code.
[infPosPatientIndices, infPosCodeIndices] = find(positiveInfEntropy);  % Get the indices of the elements in the sparse matrix;
indicesCodesOnlyInPositive = indicesOfInfEntropyCodes(unique(infPosCodeIndices));  % Indices of the codes that are only associated with positive examples.
codesOnlyInPositive = uniqueCodes(indicesCodesOnlyInPositive);  % Codes that are only associated with positive examples.
indicesCodesOnlyInNegative = setdiff(indicesOfInfEntropyCodes, indicesCodesOnlyInPositive);  % Indices of the codes that are only associated with negative examples.
codesOnlyInNegative = uniqueCodes(indicesCodesOnlyInNegative);  % Codes that are only associated with negative examples.

% Write out the infinite entropy codes and statistics about them.
fid = fopen([subFolder '\InfiniteEntropyCodes.tsv'], 'w');
fprintf(fid, 'Class\tCode\tDescription\tPatientsAssociatedWith\n');
for i = 1:numel(codesOnlyInPositive)
    codeOfInterest = codesOnlyInPositive{i};
    occurrences = nnz(dataMatrix(:, codeIndexMap(codeOfInterest)));
    fprintf(fid, 'Positive\t%s\t%s\t%d\n', codeOfInterest, query_dictionary(coding, codeOfInterest), occurrences);
end
for i = 1:numel(codesOnlyInNegative)
    codeOfInterest = codesOnlyInNegative{i};
    occurrences = nnz(dataMatrix(:, codeIndexMap(codeOfInterest)));
    fprintf(fid, 'Negative\t%s\t%s\t%d\n', codeOfInterest, query_dictionary(coding, codeOfInterest), occurrences);
end
fclose(fid);

% Select the final subset of codes to use in training.
% This is achieved by taking those codes associated with over 50 patients, and removing any codes with infinite entropy or where the entropy is
% less than 1 (equivalently where log(entropy) < 0).
indicesOfTrainingCodes = indicesOfCommonCodes((entropy ~= inf) & (entropy >= 1));  % Indices of the codes to use for training.
trainingCodes = uniqueCodes(indicesOfTrainingCodes);  % Names of the codes to use for training.
fid = fopen([subFolder '\TrainingCodes.tsv'], 'w');
fprintf(fid, 'Code\tDescription\n');
for i = 1:numel(trainingCodes)
    codeOfInterest = trainingCodes{i};
    fprintf(fid, '%s\t%s\n', codeOfInterest, query_dictionary(coding, codeOfInterest));
end
fclose(fid);

%% Perform the initial training process.

% Determine the training set and target array.
initialTrainingMatrix = dataMatrix([positiveExamples; negativeExamples], indicesOfTrainingCodes);  % Subset of dataset containing only patients and codes to use for training.
initialTrainingTarget = [ones(numPositiveExamples, 1); zeros(numNegativeExamples, 1)];  % Class target array for training.

% Create the cross validation partition.
initialCrossValPartition = cvpartition(initialTrainingTarget, 'KFold', foldsToUse);

% Train the initial model using all positive and negative examples.
% Make the training set record binary associations (rather than counts) as that is what we are interested in.
display(['Training initial model - ' datestr(now)]);
tic
[initialModelCoefficients, initialModelFitInfo] = lassoglm(initialTrainingMatrix > 0, initialTrainingTarget, 'binomial', 'NumLambda', 25, 'Alpha', 0.9, 'LambdaRatio', 1e-4, 'CV', initialCrossValPartition);
toc

% Determine the non-zero coefficients and the codes they correspond to.
indexOfLambdaToUse = initialModelFitInfo.Index1SE;  % Index of largest lambda value with deviance within one standard error of the minimum.
modelCoefficients = initialModelCoefficients(:, indexOfLambdaToUse);  % Coefficients of the model with the chosen value of lambda.
[unused, coefficientReordering] = sort(abs(modelCoefficients), 'descend');  % Sort coefficients so that 0 value coefficients are at the bottom.
sortedCoefficientIndices = indicesOfTrainingCodes(coefficientReordering);  % Training code indices sorted by absolute coefficient.
sortedCoefficientCodes = uniqueCodes(sortedCoefficientIndices);  % Codes ordered by absolute coefficient value.
fid = fopen([subFolder '\InitialModelCoefficients.tsv'], 'w');
fprintf(fid, 'Coefficient\tCode\tDescription\n');
for i = 1:numel(sortedCoefficientCodes)
    codeOfInterest = sortedCoefficientCodes{i};
    fprintf(fid, '%1.4f\t%s\t%s\n', modelCoefficients(coefficientReordering(i)), codeOfInterest, query_dictionary(coding, codeOfInterest));
end
fclose(fid);

% Compute the posterior probabilities for the training examples.
modelIntercept = initialModelFitInfo.Intercept(indexOfLambdaToUse);  % Determine the model's intercept.
initialModelCoefsWithIntercept = [modelIntercept; modelCoefficients];  % Add the intercept to the coefficients.
posteriorProbabilities = glmval(initialModelCoefsWithIntercept, initialTrainingMatrix, 'logit');  % Calculate posterior probabilities.
fid = fopen([subFolder '\InitialModelPosteriors.tsv'], 'w');
fprintf(fid, 'Class\tPosterior\tPatientID\n');
for i = 1:numPositiveExamples
    posterior = posteriorProbabilities(i);  % Positive examples come first, so posterior is indexed by i.
    fprintf(fid, 'Positive\t%1.4f\t%d\n', posterior, uniquePatientIDs(positiveExamples(i)));
end
for i = 1:numNegativeExamples
    posterior = posteriorProbabilities(numPositiveExamples + i);  % Negative examples come last, so need to add number of positive examples to index.
    fprintf(fid, 'Negative\t%1.4f\t%d\n', posterior, uniquePatientIDs(negativeExamples(i)));
end
fclose(fid);

% Determine training examples to use in training the final model. For positive examples, this will be any example with a posterior probability
% greater than 0.2, and for negative examples any example with posterior less than 0.2. This is symmetric as the posterior indicates probability
% of the class of the example being positive. Therefore, keeping negative examples with posterior < 0.2 is the same as keeping all negative
% examples with greater than a 0.8 probability of being negative.
finalPositiveExamples = positiveExamples(posteriorProbabilities(1:numPositiveExamples) > 0.2);  % Patient indices of positive examples used to train the final model.
finalNegativeExamples = negativeExamples(posteriorProbabilities(numPositiveExamples + 1:numPositiveExamples + numNegativeExamples) < 0.2);  % Patient indices of negative examples used to train the final model.
numFinalPositiveExamples = numel(finalPositiveExamples);
numFinalNegativeExamples = numel(finalNegativeExamples);

%% Record a medical history for each patient that is being discarded from the training set.

% Determine the discarded training examples.
discardedPosExamples = positiveExamples(posteriorProbabilities(1:numPositiveExamples) <= 0.2);
discardedNegExamples = negativeExamples(posteriorProbabilities(numPositiveExamples + 1:numPositiveExamples + numNegativeExamples) >= 0.2);
discardedExamples = [discardedPosExamples; discardedNegExamples];

% Create the directory to hold the patient records.
patientDir = [subFolder '\DiscardedTrainingExamples'];
if (exist(patientDir, 'dir') == 7)
    % If the directory exists then delete it before creating it.
    rmdir(patientDir, 's');
end
mkdir(patientDir);

for i = 1:numel(discardedExamples)
    if i <= numel(discardedPosExamples)
        predictedClass = 'PredAsType_2';
    else
        predictedClass = 'PredAsType_1';
    end
    patientID = uniquePatientIDs(discardedExamples(i));
    patientRecord = dataMatrix(discardedExamples(i), :);  % Get the subset of the dataset for the specific patient.
    [unused, codeIndices, codeCounts] = find(patientRecord);  % Get the indices and counts of the codes related to the patient.
    fid = fopen([patientDir '\Patient_' num2str(patientID) '_' predictedClass '.tsv'], 'w');
    fprintf(fid, 'Code\tDescription\tCount\n');
    for j = 1:numel(codeIndices)
        codeOfInterest = cell2mat(uniqueCodes(codeIndices(j)));
        fprintf(fid, '%s\t%s\t%d\n', codeOfInterest, query_dictionary(coding, codeOfInterest), codeCounts(j));
    end
    fclose(fid);
end

%% Train the second model.

% Determine the training set and target array.
finalTrainingMatrix = dataMatrix([finalPositiveExamples; finalNegativeExamples], indicesOfTrainingCodes);  % Subset of dataset containing only patients and codes to use for training.
finalTrainingTarget = [ones(numFinalPositiveExamples, 1); zeros(numFinalNegativeExamples, 1)];  % Class target array for training.

% Create the cross validation partition.
finalCrossValPartition = cvpartition(finalTrainingTarget, 'KFold', foldsToUse);

% Train the final model using the subset of positive and negative examples.
% Make the training set record binary associations (rather than counts) as that is what we are interested in.
display(['Training final model - ' datestr(now)]);
tic
[finalModelCoefficients, finalModelFitInfo] = lassoglm(finalTrainingMatrix > 0, finalTrainingTarget, 'binomial', 'NumLambda', 25, 'Alpha', 0.9, 'LambdaRatio', 1e-4, 'CV', finalCrossValPartition);
toc

% Determine the non-zero coefficients and the codes they correspond to.
indexOfLambdaToUse = finalModelFitInfo.Index1SE;  % Index of largest lambda value with deviance within one standard error of the minimum.
modelCoefficients = finalModelCoefficients(:, indexOfLambdaToUse);  % Coefficients of the model with the chosen value of lambda.
[unused, coefficientReordering] = sort(abs(modelCoefficients), 'descend');  % Sort coefficients so that 0 value coefficients are at the bottom.
sortedCoefficientIndices = indicesOfTrainingCodes(coefficientReordering);  % Training code indices sorted by absolute coefficient.
sortedCoefficientCodes = uniqueCodes(sortedCoefficientIndices);  % Codes ordered by absolute coefficient value.
fid = fopen([subFolder '\FinalModelCoefficients.tsv'], 'w');
fprintf(fid, 'Coefficient\tCode\tDescription\n');
for i = 1:numel(sortedCoefficientCodes)
    codeOfInterest = sortedCoefficientCodes{i};
    fprintf(fid, '%1.4f\t%s\t%s\n', modelCoefficients(coefficientReordering(i)), codeOfInterest, query_dictionary(coding, codeOfInterest));
end
fclose(fid);

% Compute the posterior probabilities for the training examples.
modelIntercept = finalModelFitInfo.Intercept(indexOfLambdaToUse);  % Determine the model's intercept.
finalModelCoefsWithIntercept = [modelIntercept; modelCoefficients];  % Add the intercept to the coefficients.
posteriorProbabilities = glmval(finalModelCoefsWithIntercept, finalTrainingMatrix, 'logit');  % Calculate posterior probabilities.
fid = fopen([subFolder '\FinalModelPosteriors.tsv'], 'w');
fprintf(fid, 'Class\tPosterior\tPatientID\n');
for i = 1:numFinalPositiveExamples
    posterior = posteriorProbabilities(i);  % Positive examples come first, so posterior is indexed by i.
    fprintf(fid, 'Positive\t%1.4f\t%d\n', posterior, uniquePatientIDs(finalPositiveExamples(i)));
end
for i = 1:numFinalNegativeExamples
    posterior = posteriorProbabilities(numFinalPositiveExamples + i);  % Negative examples come last, so need to add number of positive examples to index.
    fprintf(fid, 'Negative\t%1.4f\t%d\n', posterior, uniquePatientIDs(finalNegativeExamples(i)));
end
fclose(fid);

%% Estimate the performance of the models.

thresholdsToUse = (0:100) / 100;  % The posterior thresholds to use to generate the ROC curve.

% Optimal lambda values calculated when training the initial and final models.
initialLambda = initialModelFitInfo.Lambda(initialModelFitInfo.Index1SE);
finalLambda = finalModelFitInfo.Lambda(finalModelFitInfo.Index1SE);

% Create arrays of class names for the training examples used in each model.
initialModelClassArray = cellstr([repmat('positive', numPositiveExamples, 1); repmat('negative', numNegativeExamples, 1)]);
finalModelClassArray = cellstr([repmat('positive', numFinalPositiveExamples, 1); repmat('negative', numFinalNegativeExamples, 1)]);

% Open results files.
initFid = fopen([subFolder '\InitialModelCVResults.tsv'], 'w');
fprintf(initFid, ['ClassificationThreshold\t' num2str(thresholdsToUse, '\t%.2f') '\n']);
finalFid = fopen([subFolder '\FinalModelCVResults.tsv'], 'w');
fprintf(finalFid, ['ClassificationThreshold\t' num2str(thresholdsToUse, '\t%.2f') '\n']);

% Generate predictions.
for i = 1:foldsToUse
    display(['Cross validation fold ' num2str(i) ' - ' datestr(now)]);
    tic
    
    % Train the models.
    intialTrainingFold = initialCrossValPartition.training(i);  % Binary array indicating examples used for training the initial model.
    [initialModelCoefficientsCV, initialModelFitInfoCV] = lassoglm(initialTrainingMatrix(intialTrainingFold, :) > 0, initialTrainingTarget(intialTrainingFold, :), 'binomial', 'Lambda', initialLambda, 'Alpha', 0.9);
    finalTrainingFold = finalCrossValPartition.training(i);  % Binary array indicating examples used for training the final model.
    [finalModelCoefficientsCV, finalModelFitInfoCV] = lassoglm(finalTrainingMatrix(finalTrainingFold, :) > 0, finalTrainingTarget(finalTrainingFold, :), 'binomial', 'Lambda', finalLambda, 'Alpha', 0.9);

    % Test the models.
    initialTestFold = initialCrossValPartition.test(i);  % Binary array indicating examples used for testing the initial model.
    initialModelCoefsWithInterceptCV = [initialModelFitInfoCV.Intercept; initialModelCoefficientsCV];  % Add the intercept to the coefficients.
    initialModelPostProbsCV = glmval(initialModelCoefsWithInterceptCV, initialTrainingMatrix(initialTestFold, :), 'logit');  % Calculate posterior probabilities.
    initialModelPostProbsCV = num2cell(initialModelPostProbsCV);
    finalTestFold = finalCrossValPartition.test(i);  % Binary array indicating examples used for testing the final model.
    finalModelCoefsWithInterceptCV = [finalModelFitInfoCV.Intercept; finalModelCoefficientsCV];  % Add the intercept to the coefficients.
    finalModelPostProbsCV = glmval(finalModelCoefsWithInterceptCV, finalTrainingMatrix(finalTestFold, :), 'logit');  % Calculate posterior probabilities.
    finalModelPostProbsCV = num2cell(finalModelPostProbsCV);
    
    fprintf(initFid, 'Fold%d\t', i);
    fprintf(finalFid, 'Fold%d\t', i);
    for j = 1:numel(thresholdsToUse)
        % Calcualte stats for the initial model.
        initModelTrueClasses = initialModelClassArray(initialTestFold);
        initModelPredClasses = cellfun(@(x) iff(x < thresholdsToUse(j), 'negative', 'positive'), initialModelPostProbsCV, 'UniformOutput', false);
        initModelCorrect = initModelTrueClasses(strcmp(initModelTrueClasses, initModelPredClasses));
        initModelNumCorrect = numel(initModelCorrect);
        initModelTP = sum(ismember(initModelCorrect, 'positive'));
        initModelTN = initModelNumCorrect - initModelTP;
        initModelWrong = initModelPredClasses(~strcmp(initModelTrueClasses, initModelPredClasses));
        initModelNumWrong = numel(initModelWrong);
        initModelFP = sum(ismember(initModelWrong, 'positive'));
        initModelFN = initModelNumWrong - initModelFP;
        fprintf(initFid, '%d,%d,%d,%d\t', initModelTP, initModelFP, initModelTN, initModelFN);
        
        % Calculate stats for the final model.
        finalModelTrueClasses = finalModelClassArray(finalTestFold);
        finalModelPredClasses = cellfun(@(x) iff(x < thresholdsToUse(j), 'negative', 'positive'), finalModelPostProbsCV, 'UniformOutput', false);
        finalModelCorrect = finalModelTrueClasses(strcmp(finalModelTrueClasses, finalModelPredClasses));
        finalModelNumCorrect = numel(finalModelCorrect);
        finalModelTP = sum(ismember(finalModelCorrect, 'positive'));
        finalModelTN = finalModelNumCorrect - finalModelTP;
        finalModelWrong = finalModelPredClasses(~strcmp(finalModelTrueClasses, finalModelPredClasses));
        finalModelNumWrong = numel(finalModelWrong);
        finalModelFP = sum(ismember(finalModelWrong, 'positive'));
        finalModelFN = finalModelNumWrong - finalModelFP;
        fprintf(finalFid, '%d,%d,%d,%d\t', finalModelTP, finalModelFP, finalModelTN, finalModelFN);
    end
    fprintf(initFid, '\n');
    fprintf(finalFid, '\n');
    
    toc
end

% Close results files.
fclose(initFid);
fclose(finalFid);

%% Record basic stats about the trained models and the dataset.

% Record some statistics about the trained models.
fid = fopen([subFolder '\ModelStats.txt'], 'w');
fprintf(fid, 'CV folds used - %d\n', foldsToUse);
fprintf(fid, 'Initial model lambda - %d\n', initialModelFitInfo.Lambda(initialModelFitInfo.Index1SE));
fprintf(fid, 'Initial model non-zero coefficients - %d\n', initialModelFitInfo.DF(initialModelFitInfo.Index1SE));
fprintf(fid, 'Initial model deviance - %d\n', initialModelFitInfo.Lambda1SE);
fprintf(fid, 'Initial model deviance standard error - %d\n', initialModelFitInfo.SE(initialModelFitInfo.Index1SE));
fprintf(fid, 'Final model lambda - %d\n', finalModelFitInfo.Lambda(finalModelFitInfo.Index1SE));
fprintf(fid, 'Final model non-zero coefficients - %d\n', finalModelFitInfo.DF(finalModelFitInfo.Index1SE));
fprintf(fid, 'Final model deviance - %d\n', finalModelFitInfo.Lambda1SE);
fprintf(fid, 'Final model deviance standard error - %d\n', finalModelFitInfo.SE(finalModelFitInfo.Index1SE));
fclose(fid);

% Record some statistics about the data.
fid = fopen([subFolder '\DatasetStats.txt'], 'w');
fprintf(fid, 'Initial positive examples - %d\n', numPositiveExamples);
fprintf(fid, 'Initial negative examples - %d\n', numNegativeExamples);
fprintf(fid, 'Final positive examples - %d\n', numFinalPositiveExamples);
fprintf(fid, 'Final negative examples - %d\n', numFinalNegativeExamples);
if (isType1Type2)
    fprintf(fid, 'Examples meeting both criteria - %d', numel(patientsWithBothTypes));
else
    fprintf(fid, 'Examples meeting both criteria - 0');
end
fclose(fid);

%% Record classifications for the ambiguous patients (only needed when comparing type 1 and type 2 patients).

if (isType1Type2)
    % Determine the posterior probabilities of the examples that are marked as both type 1 and type 2.
    examplesToRecode = patientsWithBothTypes;
    bothTypesMatrix = dataMatrix(examplesToRecode, indicesOfTrainingCodes);  % Subset of dataset containing patients with both types of diabetes and codes used for training.
    initialModelPostProbs = glmval(initialModelCoefsWithIntercept, bothTypesMatrix, 'logit');  % Calculate posterior probabilities using the initial model.
    finalModelPostProbs = glmval(finalModelCoefsWithIntercept, bothTypesMatrix, 'logit');  % Calculate posterior probabilities using the final model.

    % Record predictions for ambiguous patients.
    fid = fopen([subFolder '\AmbiguousPatientPosteriors.tsv'], 'w');
    fprintf(fid, 'InitialModelPosterior\tFinalModelPosterior\tPatientID\n');
    for i = 1:numel(examplesToRecode)
        initialModelPosterior = initialModelPostProbs(i);
        finalModelPosterior = finalModelPostProbs(i);
        fprintf(fid, '%1.4f\t%1.4f\t%d\n', initialModelPosterior, finalModelPosterior, uniquePatientIDs(examplesToRecode(i)));
    end
    fclose(fid);

    % Create the directory to hold the patient records.
    patientDir = [subFolder '\DisambiguatedPatients'];
    if (exist(patientDir, 'dir') == 7)
        % If the directory exists then delete it before creating it.
        rmdir(patientDir, 's');
    end
    mkdir(patientDir);
    fid = fopen([patientDir '\_RecodingCutoffUsed.txt'], 'w');
    fprintf(fid, 'Examples with posterior <= %.2f or >= %.2f are considered to be conclusively disambiguated.', recodingCutoff, 1 - recodingCutoff);
    fclose(fid);
    
    % Create a medical history for each ambiguos patient that has had their diabetes type status predicted conclusively.
    for i = 1:numel(examplesToRecode)
        predictedType1 = (finalModelPostProbs(i) >= (1 - recodingCutoff));
        predictedType2 = (finalModelPostProbs(i) <= recodingCutoff);
        if (predictedType2 || predictedType1)
            if predictedType1
                predictedClass = 'Type1';
            else
                predictedClass = 'Type2';
            end
            patientID = uniquePatientIDs(examplesToRecode(i));
            patientRecord = dataMatrix(examplesToRecode(i), :);  % Get the subset of the dataset for the specific patient.
            [unused, codeIndices, codeCounts] = find(patientRecord);  % Get the indices and counts of the codes related to the patient.
            fid = fopen([patientDir '\Patient_' num2str(patientID) '_' predictedClass '.tsv'], 'w');
            fprintf(fid, 'Code\tDescription\tCount\n');
            for j = 1:numel(codeIndices)
                codeOfInterest = cell2mat(uniqueCodes(codeIndices(j)));
                fprintf(fid, '%s\t%s\t%d\n', codeOfInterest, query_dictionary(coding, codeOfInterest), codeCounts(j));
            end
            fclose(fid);
        end
    end
end
