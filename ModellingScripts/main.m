% Reset the workspace.
clear all
clc
close all
rng('default')  % Ensure results are reproduceable.

% Load the mapping of Read V2 codes (and some EMIS vendor codes) to their descriptions descriptions.
load coding

% Flag to indicate which comparison is being done. If set to true, then the comparison is between people with type 1 and
% type 2 diabetes, else it is between people with diabetes and people without.
% Also determine the threshold at which recodings will be deemed to be successful. Anything <= recodingCutoff OR >= (1 - recodingCutoff) is considered strong enough evidence for recoding.
isType1Type2 = true;
recodingCutoff = 0;

% Load the data on the patients. Each row in the file contains:
% PatentID\tCode\tOccurences
% Therefore, each line records the number of times a specific Read V2 code occurs in a specific patient's medical record.
% Both patient IDs and Read codes are likely to appear on more then one line (as patients have more than one code associated
% them and codes are associated with more than one patient).
% data.id is an array containing the patient IDs.
% data.key is an array containing the Read codes.
% data.count is an array containing the association counts.
[data.id, data.key, data.counts] = textread('CutDownPatientData.tsv', '%d %s %d');

% Determine unique patient IDs and codes, and create index mappings for each.
uniqueCodes = unique(data.key);  % A list of the unique Read codes in the dataset.
codeIndexMap = containers.Map(uniqueCodes, 1:numel(uniqueCodes));  % Map the Read codes to their index in the uniqueCodes array.
uniquePatientIDs = unique(data.id);  % A list of all the patient IDs in the dataset.
patientIndexMap = containers.Map(uniquePatientIDs, 1:numel(uniquePatientIDs));  % Map the patient IDs to their index in the uniquePatientIDs array.

% Create the sparse matrix.
% The matrix is created from three arrays: an array of row indices (sparseRows), an array of column indices (sparseCols) and an array of
% values (data.counts). The matrix is created by saying that the entry in the matrix M[sparseRows[i], sparseCols[i]] = data.counts[i].
% This sparse matrix will have a row for each patient ID (10,000 rows) and one column for each Read code in the dataset.

% For sparseRows, first generate a cell array with each cell containing a patient
% ID. The ordering of the IDs in the array is the same as in the dataset
% file. Next create a second cell array of the same length by mapping each
% patient ID to its index (1-10000). Then convert this to a normal array.
% The sparseRows array therefore contains one value for each entry in the dataset
% file, with each value in the array being an index of a patient ID (valued
% between 1 and 10,000).
sparseRows = cell2mat(values(patientIndexMap, num2cell(data.id)));
% For sparseCols, generate a cell array containing the index of each code
% (with the array ordered the same as the dataset file). Next convert this
% to a normal array. The sparseCols array therefore contains one value for each entry in the dataset
% file, with each value in the array being an index of a code.
sparseCols = cell2mat(values(codeIndexMap, data.key));
dataMatrix = sparse(sparseRows, sparseCols, data.counts, numel(uniquePatientIDs), numel(uniqueCodes));

% Determine Read codes (and their indices) for the codes that indicate type 1 or type 2 diabetes. These should be manually examined to make sure that
% no child codes that don't truly indicate diabetes have been included.
type1DiabetesCodes = uniqueCodes(cellfun(@(x) ~isempty(regexp(x, '^C10E')), uniqueCodes));  % Codes that begin with C10E.
type2DiabetesCodes = uniqueCodes(cellfun(@(x) ~isempty(regexp(x, '^C10F')), uniqueCodes));  % Codes that begin with C10F.
type1DiabetesIndices = cell2mat(values(codeIndexMap, type1DiabetesCodes));  % The indices of the type 1 diabetes codes.
type2DiabetesIndices = cell2mat(values(codeIndexMap, type2DiabetesCodes));  % The indices of the type 2 diabetes codes.

% Write out the codes that will be used to indicate type 1 and type 2 diabetes.
fid = fopen('DiabetesCodesUsed.tsv', 'w');
fprintf(fid, 'Type\tCode\tDescription\tPatientsAssociatedWith\tOccurencesWithoutItsParent\tOccurencesWithOppositeParent\n');
for i = 1:numel(type1DiabetesCodes)
    codeOfInterest = type1DiabetesCodes{i};
    occurences = nnz(dataMatrix(:, codeIndexMap(codeOfInterest)));
    occurrencesWithoutC10E = (dataMatrix(:, codeIndexMap('C10E')) == 0) & (dataMatrix(:, codeIndexMap(codeOfInterest)) > 0);
    occurrencesWithC10F = (dataMatrix(:, codeIndexMap('C10F')) > 0) & (dataMatrix(:, codeIndexMap(codeOfInterest)) > 0);
    fprintf(fid, '1\t%s\t%s\t%d\t%d\t%d\n', codeOfInterest, query_dictionary(coding, codeOfInterest), occurences, nnz(occurrencesWithoutC10E), nnz(occurrencesWithC10F));
end
for i = 1:numel(type2DiabetesCodes)
    codeOfInterest = type2DiabetesCodes{i};
    occurences = nnz(dataMatrix(:, codeIndexMap(codeOfInterest)));
    occurrencesWithC10E = (dataMatrix(:, codeIndexMap('C10E')) > 0) & (dataMatrix(:, codeIndexMap(codeOfInterest)) > 0);
    occurrencesWithoutC10F = (dataMatrix(:, codeIndexMap('C10F')) == 0) & (dataMatrix(:, codeIndexMap(codeOfInterest)) > 0);
    fprintf(fid, '2\t%s\t%s\t%d\t%d\t%d\n', codeOfInterest, query_dictionary(coding, codeOfInterest), occurences, nnz(occurrencesWithoutC10F), nnz(occurrencesWithC10F));
end
fclose(fid);

% Determine the patients that have a form of diabetes.
% For each of type 1 and type 2 diabetes, first create a boolean matrix with one row per patient and one column per code that indicates that disease.
% Next flatten this matrix columnwise into a single (column) array using logical OR (provided by any(, 2))
% This enables simpler indexing when finding the positive and negative examples.
patientsWithType1 = full(dataMatrix(:, type1DiabetesIndices) > 0);  % Boolean matrix. A 1 indicates that the patient is associated with the type 1 code.
patientsWithType1 = any(patientsWithType1, 2);  % Boolean array. A 1 indicates that the patient has type 1 diabetes.
patientsWithType2 = full(dataMatrix(:, type2DiabetesIndices) > 0);  % Boolean matrix. A 1 indicates that the patient is associated with the type 2 code.
patientsWithType2 = any(patientsWithType2, 2);  % Boolean array. A 1 indicates that the patient has type 2 diabetes.

% Generate the positive and negative examples.
if (isType1Type2)
    patientsWithBothTypes = patientsWithType1 & patientsWithType2;  % Boolean array. A 1 indicates that the patient has type 1 and type 2 diabetes.
    positiveExamples = setdiff(find(patientsWithType1), find(patientsWithType2));  % Indices of patients with ONLY type 1 diabetes.
    negativeExamples = setdiff(find(patientsWithType2), find(patientsWithType1));  % Indices of patients with ONLY type 2 diabetes.
else
    positiveExamples = find(patientsWithType1 | patientsWithType2);  % Indices of patients with either type 1 or type 2 diabetes.
    negativeExamples = setdiff((1:numel(uniquePatientIDs))', positiveExamples);  % Indices of patients without diabetes.
end
numPositiveExamples = numel(positiveExamples);
numNegativeExamples = numel(negativeExamples);

% Select only those codes that occur in more than 50 patient records.
% As dataMatrix contains counts, we need > 0 to make a boolean matrix that just records presence/absence before we sum.
codePresence = dataMatrix > 0;  % Convert sparse matrix from recording counts of associations between a patient and a given code to simply recording presence/absence of association.
codeOccurences = full(sum(codePresence));  % Matrix of counts of the number of different patients each code is associated with.
indicesOfCommonCodes = find(codeOccurences > 50);  % Array of indices of the codes associated with over 50 patients.

% Calculate the entropy of the codes with enough associations.
% I believe this is calculated as the entropy between the histograms of counts from the two classes for a given code.
% entropyReordering is a reordering of the indices of the entries in indicesOfCommonCodes by descending entropy.
% entropy is the entropy of the common codes ordered according to their indices position in indicesOfCommonCodes.
sparseSubsetCommonCodes = dataMatrix([positiveExamples' negativeExamples'], indicesOfCommonCodes);  % Subset of the sparse matrix containing only the examples being used and the codes with enough associations.
sparseSubsetTarget = [ones(numPositiveExamples, 1); zeros(numNegativeExamples, 1)];  % Target array for the sparse subset. 1s for the positive examples and 0s for negative examples.
[entropyReordering, entropy] = rankfeatures(sparseSubsetCommonCodes', sparseSubsetTarget, 'Criterion', 'entropy');  % Calculate entropy for the codes. Transpose the sparse subset to calculate entropy for codes not examples.

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
fid = fopen('InfiniteEntropyCodes.tsv', 'w');
fprintf(fid, 'Class\tCode\tDescription\tPatientsAssociatedWith\n');
for i = 1:numel(codesOnlyInPositive)
    codeOfInterest = codesOnlyInPositive{i};
    occurences = nnz(dataMatrix(:, codeIndexMap(codeOfInterest)));
    fprintf(fid, 'Positive\t%s\t%s\t%d\n', codeOfInterest, query_dictionary(coding, codeOfInterest), occurences);
end
for i = 1:numel(codesOnlyInNegative)
    codeOfInterest = codesOnlyInNegative{i};
    occurences = nnz(dataMatrix(:, codeIndexMap(codeOfInterest)));
    fprintf(fid, 'Negative\t%s\t%s\t%d\n', codeOfInterest, query_dictionary(coding, codeOfInterest), occurences);
end
fclose(fid);

% Select the final subset of codes to use in training.
% This is achieved by taking those codes associated with over 50 patients, and removing any codes with infinite entropy or where the entropy is
% less than 1 (equivalently where log(entropy) < 0).
indicesOfTrainingCodes = indicesOfCommonCodes((entropy ~= inf) & (entropy >= 1));  % Indices of the codes to use for training.
trainingCodes = uniqueCodes(indicesOfTrainingCodes);  % Names of the codes to use for training.
fid = fopen('TrainingCodes.tsv', 'w');
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

% Train the initial model using all positive and negative examples.
% Make the training set record binary associations (rather than counts) as that is what we are interested in.
tic
[initialModelCoefficients, initialModelFitInfo] = lassoglm(initialTrainingMatrix > 0, initialTrainingTarget, 'binomial', 'NumLambda', 25, 'Alpha', 0.9, 'LambdaRatio', 1e-4, 'CV', 2);
toc

% Determine the non-zero coefficients and the codes they correspond to.
indexOfLambdaToUse = initialModelFitInfo.Index1SE;  % Index of largest lambda value with deviance within one standard error of the minimum.
modelCoefficients = initialModelCoefficients(:, indexOfLambdaToUse);  % Coefficients of the model with the chosen value of lambda.
[sortedCoefficients, coefficientReordering] = sort(abs(modelCoefficients), 'descend');  % Sort coefficients so that 0 value coefficients are at the bottom.
nonZeroCoefficients = modelCoefficients(coefficientReordering(sortedCoefficients ~= 0));  % Non-zero coefficients ordered by absolute value.
nonZeroCoefCodeIndices = indicesOfTrainingCodes(coefficientReordering(sortedCoefficients ~= 0));  % Indices of codes with non-zero coefficients ordered by absolute coefficient value.
nonZeroCoefCodes = uniqueCodes(nonZeroCoefCodeIndices);  % Codes with non-zero coefficients ordered by absolute coefficient value.
fid = fopen('InitialModelCoefficients.tsv', 'w');
fprintf(fid, 'Coefficient\tCode\tDescription\n');
for i = 1:numel(nonZeroCoefCodes)
    codeOfInterest = nonZeroCoefCodes{i};
    fprintf(fid, '%1.4f\t%s\t%s\n', nonZeroCoefficients(i), codeOfInterest, query_dictionary(coding, codeOfInterest));
end
fclose(fid);

% Compute the posterior probabilities for the training examples.
modelIntercept = initialModelFitInfo.Intercept(indexOfLambdaToUse);  % Determine the model's intercept.
initialModelCoefsWithIntercept = [modelIntercept; modelCoefficients];  % Add the intercept to the coefficients.
posteriorProbabilities = glmval(initialModelCoefsWithIntercept, initialTrainingMatrix, 'logit');  % Calculate posterior probabilities.
fid = fopen('InitialModelPosteriors.tsv', 'w');
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

%% Train the second model.

% Determine the training set and target array.
finalTrainingMatrix = dataMatrix([finalPositiveExamples; finalNegativeExamples], indicesOfTrainingCodes);  % Subset of dataset containing only patients and codes to use for training.
finalTrainingTarget = [ones(numFinalPositiveExamples, 1); zeros(numFinalNegativeExamples, 1)];  % Class target array for training.

% Train the final model using the subset of positive and negative examples.
% Make the training set record binary associations (rather than counts) as that is what we are interested in.
tic
[finalModelCoefficients, finalModelFitInfo] = lassoglm(finalTrainingMatrix > 0, finalTrainingTarget, 'binomial', 'NumLambda', 25, 'Alpha', 0.9, 'LambdaRatio', 1e-4, 'CV', 2);
toc

% Determine the non-zero coefficients and the codes they correspond to.
indexOfLambdaToUse = finalModelFitInfo.Index1SE;  % Index of largest lambda value with deviance within one standard error of the minimum.
modelCoefficients = finalModelCoefficients(:, indexOfLambdaToUse);  % Coefficients of the model with the chosen value of lambda.
[sortedCoefficients, coefficientReordering] = sort(abs(modelCoefficients), 'descend');  % Sort coefficients so that 0 value coefficients are at the bottom.
nonZeroCoefficients = modelCoefficients(coefficientReordering(sortedCoefficients ~= 0));  % Non-zero coefficients ordered by absolute value.
nonZeroCoefCodeIndices = indicesOfTrainingCodes(coefficientReordering(sortedCoefficients ~= 0));  % Indices of codes with non-zero coefficients ordered by absolute coefficient value.
nonZeroCoefCodes = uniqueCodes(nonZeroCoefCodeIndices);  % Codes with non-zero coefficients ordered by absolute coefficient value.
fid = fopen('FinalModelCoefficients.tsv', 'w');
fprintf(fid, 'Coefficient\tCode\tDescription\n');
for i = 1:numel(nonZeroCoefCodes)
    codeOfInterest = nonZeroCoefCodes{i};
    fprintf(fid, '%1.4f\t%s\t%s\n', nonZeroCoefficients(i), codeOfInterest, query_dictionary(coding, codeOfInterest));
end
fclose(fid);

% Compute the posterior probabilities for the training examples.
modelIntercept = finalModelFitInfo.Intercept(indexOfLambdaToUse);  % Determine the model's intercept.
finalModelCoefsWithIntercept = [modelIntercept; modelCoefficients];  % Add the intercept to the coefficients.
posteriorProbabilities = glmval(finalModelCoefsWithIntercept, finalTrainingMatrix, 'logit');  % Calculate posterior probabilities.
fid = fopen('FinalModelPosteriors.tsv', 'w');
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

%% Generate information about model performance, dataset statistics and recodings.

% Record some statistics about the model performance.
fid = fopen('ModelPerformance.txt', 'w');
fprintf(fid, 'Initial model lambda - %d\n', initialModelFitInfo.Lambda);
fprintf(fid, 'Initial model non-zero coefficients - %d\n', initialModelFitInfo.DF(initialModelFitInfo.Index1SE));
fprintf(fid, 'Initial model deviance - %d\n', initialModelFitInfo.Lambda1SE);
fprintf(fid, 'Initial model deviance standard error - %d\n', initialModelFitInfo.SE(initialModelFitInfo.Index1SE));
fprintf(fid, 'Final model lambda - %d\n', finalModelFitInfo.Lambda);
fprintf(fid, 'Final model non-zero coefficients - %d\n', finalModelFitInfo.DF(finalModelFitInfo.Index1SE));
fprintf(fid, 'Final model deviance - %d\n', finalModelFitInfo.Lambda1SE);
fprintf(fid, 'Final model deviance standard error - %d\n', finalModelFitInfo.SE(finalModelFitInfo.Index1SE));
fclose(fid);

% Record some statistics about the data.
fid = fopen('DatasetStats.txt', 'w');
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


% Calculate posteriors for examples that may need recoding.
if (isType1Type2)
    % Determine the posterior probabilities of the examples that are marked as both type 1 and type 2.
    examplesToRecode = patientsWithBothTypes;
    bothTypesMatrix = dataMatrix(examplesToRecode, indicesOfTrainingCodes);  % Subset of dataset containing patients with both types of diabetes and codes used for training.
    initialModelPostProbs = glmval(initialModelCoefsWithIntercept, bothTypesMatrix, 'logit');  % Calculate posterior probabilities using the initial model.
    finalModelPostProbs = glmval(finalModelCoefsWithIntercept, bothTypesMatrix, 'logit');  % Calculate posterior probabilities using the final model.
else
    % Determine the posteriors for the non-diabetic examples.
    examplesToRecode = negativeExamples;
    nonDiabeticMatrix = dataMatrix(examplesToRecode, indicesOfTrainingCodes);  % Subset of dataset containing non-diabetic patients and codes used for training.
    initialModelPostProbs = glmval(initialModelCoefsWithIntercept, nonDiabeticMatrix, 'logit');  % Calculate posterior probabilities using the initial model.
    finalModelPostProbs = glmval(finalModelCoefsWithIntercept, nonDiabeticMatrix, 'logit');  % Calculate posterior probabilities using the final model.
end
fid = fopen('RecodingPosteriors.tsv', 'w');
fprintf(fid, 'InitialModelPosterior\tFinalModelPosterior\tPatientID\n');
for i = 1:numel(examplesToRecode)
    initialModelPosterior = initialModelPostProbs(i);
    finalModelPosterior = finalModelPostProbs(i);
    fprintf(fid, '%1.4f\t%1.4f\t%d\n', initialModelPosterior, finalModelPosterior, uniquePatientIDs(examplesToRecode(i)));
end
fclose(fid);

% Create a medical history for each patient that has had their diabetes type status determined conclusively.
% This is not needed when differentiating diabetes patients from non-diabetes patients.
if (isType1Type2)
    % Create the directory to hold the patient records.
    patientDir = 'PatientRecords';
    if (exists(patientDir, 'dir') == 7)
        % If the directory exists then delete it before creating it.
        rmdir(patientDir);
    end
    mkdir(patientDir);
    
    % Create the record for each patient.
    for i = 1:numel(examplesToRecode)
        if ((finalModelPosterior(i) <= recodingCutoff) | (finalModelPosterior(i) >= (1- recodingCutoff)))
            patientID = uniquePatientIDs(examplesToRecode(i);
            patientRecord = dataMatrix(patientID, :);  % Get the subset of the dataset for the specific patient.
            fid = fopen([patientDir '/Patient_' patientID '.tsv'], 'w');
            fprintf(fid, 'Code\tCount\n');
            fclose(fid);
        end
    end
end