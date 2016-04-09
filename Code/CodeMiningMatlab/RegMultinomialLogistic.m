classdef RegMultinomialLogistic < handle
    % Class defining the methods used for performing regularised multinomial logistic regression.

    properties (SetAccess = private)
        alpha = 0.01;  % Gradient descent learning rate.
        batchSize = 100;  % Size of the mini-batch.
        coefficients = [];  % Matrix of coefficients (one column per class) with an intercept as the first term.
        lambda = 0.01;  % Regularisation parameter.
        maxIter = 10;  % Maximum number of iterations.
        targetClasses = [];  % The classes sorted in the order of the columns in coefficients.
                             % E.g. targetClasses(1) corresponds to the coefficients in coefficients(:, 1).
    end
    methods (Access = private)
        function [predictions] = calc_logistic(obj, data)
            % Calculate the logistic output.
            predictions = 1./ (1 + exp(-data * obj.coefficients));
        end
    end
    methods
        function [obj] = RegMultinomialLogistic(alpha, batchSize, lambda, maxIter)
            % Constructor.
            if nargin < 1
                alpha = [];
            end
            if nargin < 2
                batchSize = [];
            end
            if nargin < 3
                lambda = [];
            end
            if nargin < 4
                maxIter = [];
            end
            obj.set_parameters(alpha, batchSize, lambda, maxIter);
        end

        function [performance] = calculate_performance(obj, predictions, target, thresholds)
            % Determine the performance of a classifier.
            %
            % Keyword Arguments
            % predictions - NxK matrix of predictions (N examples K classes).
            % target - Nx1 column vector of target values. Target values should be integers, with the smallest integer
            %          corresponding to the first column in predictions, and the largest the last.
            % thresholds - Array of T thresholds to calculate performance metrics for. All values must be in the range [0, 1].
            %
            % Return Values
            % performance - A struct containing 6 fields:
            %                   performance.maxProb - Nx1 array of the maximum probability classification for each of the N examples.
            %                                         Determined as the class with the model with the highest prediction value.
            %                   performance.thresholds - The thresholds used.
            %                   performance.sensitivities - TxK matrix of the sensitivity of each model at each threshold.
            %                   performance.specificities - TxK matrix of the specificity of each model at each threshold.
            %                   performance.modelGMeans - TxK matrix of the G-Mean of each model at each threshold.
            %                   performance.gMean - The G-Mean of the maximum probability classifications.
            %               The maxProb field is always populated. The other fields will be empty unless a threshold is supplied.

            % Determine the maximum probability classifications.
            [~, column] = max(predictions, [], 2);
            maxProbPred = obj.targetClasses(column);
            performance.maxProb = maxProbPred;

            if (nargin < 3) || isempty(thresholds)
                % Skip calculating all but the maximum probability classification.
                performance.thresholds = [];
                performance.sensitivities = [];
                performance.specificities = [];
                performance.modelGMeans = [];
                performance.AUC = [];
                performance.gMean = [];
            else
                % Convert the target classes into a matrix with the same number of columns as there are classes.
                % Each column will correspond to a single class. Given target, the array of classes, the matrix of classes (targetMatrix)
                % will be organised so that targetMatrix(i, 1) == 1 if target(i) contains the first class.
                % Otherwise targetMatrix(i, 1) == 0. For example:
                %   target == [1 2 3 2 1]
                %                  [1 0 0]
                %                  [0 1 0]
                %   targetMatrix = [0 0 1]
                %                  [0 1 0]
                %                  [1 0 0]
                targetMatrix = zeros(numel(target), numel(obj.targetClasses));
                for i = 1:numel(obj.targetClasses)
                    targetMatrix(target == obj.targetClasses(i), i) = 1;
                end

                % Calculate metrics for each threshold.
                sensitivities = zeros(numel(thresholds), numel(obj.targetClasses));
                specificities = zeros(numel(thresholds), numel(obj.targetClasses));
                modelGMeans = zeros(numel(thresholds), numel(obj.targetClasses));
                for i = 1:numel(thresholds)
                    % Calculate for each model which examples would be classed as positive and which negative if you were
                    % using the current threshold.
                    calculatedClasses = predictions >= thresholds(i);

                    % Determine base measures.
                    truePositives = sum(targetMatrix & calculatedClasses) / size(predictions, 1);
                    falsePositives = sum(~targetMatrix & calculatedClasses) / size(predictions, 1);
                    trueNegatives = sum(~targetMatrix & ~calculatedClasses) / size(predictions, 1);
                    falseNegatives = sum(targetMatrix & ~calculatedClasses) / size(predictions, 1);

                    % Determine composite measures.
                    sensitivity = truePositives ./ (truePositives + falsePositives);
                    sensitivity(isnan(sensitivity)) = 0;
                    sensitivities(i, :) = sensitivity;
                    specificity = trueNegatives ./ (trueNegatives + falseNegatives);
                    specificity(isnan(specificity)) = 0;
                    specificities(i, :) = specificity;
                    gMean = sqrt(sensitivity .* specificity);
                    modelGMeans(i, :) = gMean;
                end
                performance.thresholds = thresholds;
                performance.sensitivities = sensitivities;
                performance.specificities = specificities;
                performance.modelGMeans = modelGMeans;

                % Determine the G-Mean for the set of models using the maximum probability classifications.
                numClassPredictions = arrayfun(@(x) sum(maxProbPred == x), obj.targetClasses);  % The number of predictions for each class (both correct and not).
                correctPredictions = maxProbPred == target;  % Boolean array indicating examples correctly predicted.
                classesOfCorrectPreds = target(correctPredictions);  % Array containing the class of each example predicted correctly.
                numCorrectPredictions = arrayfun(@(x) sum(classesOfCorrectPreds == x), obj.targetClasses);  % Number of examples from each class predicted correctly.
                classSensitivity = numCorrectPredictions ./ numClassPredictions;  % The sensitivity of the maximum probability classification for each class.
                gMean = nthroot(prod(classSensitivity), numel(obj.targetClasses));  % The G-Mean of the maximum probability classification.
                gMean = iff(isnan(gMean), 0, gMean);  % Convert NaNs to 0s.
                performance.gMean = gMean;
            end
        end

        function reset_coefficients(obj)
            % Resets the coefficients.
            obj.coefficients = [];
        end

        function set_parameters(obj, alpha, batchSize, lambda, maxIter)
            % Convenience function to set parameters.
            if nargin >= 1 && ~isempty(alpha)
                obj.alpha = alpha;
            end
            if nargin >= 2 && ~isempty(batchSize)
                obj.batchSize = batchSize;
            end
            if nargin >= 3 && ~isempty(lambda)
                obj.lambda = lambda;
            end
            if nargin >= 4 && ~isempty(maxIter)
                obj.maxIter = maxIter;
            end
        end

        function [predictions] = test(obj, testMatrix)
            % Generate predictions for examples in testMatrix.
            %
            % Keyword Arguments
            % testMatrix - A NxM matrix of test examples. If there are C coefficients, then M + 1 == C.
            %
            % Return Values
            % predictions - The posterior predictions for each of the K logistic regression models trained.

            numVariables = size(testMatrix, 2);
            if numVariables + 1 ~= size(obj.coefficients, 1),
                error('The test matrix has %d variables, but there are %d coefficients (including the bias).', ...
                    numVariables, size(obj.coefficients, 1));
            end;
            testMatrix = [ones(size(testMatrix, 1), 1) testMatrix];  % Add the bias.
            predictions = obj.calc_logistic(testMatrix);
        end

        function [recordOfDescent] = train(obj, trainingMatrix, target, recordOfDescent)
            % Train the logistic regression model using a mini-batch gradient descent approach.
            %
            % For multiple classes, the approach taken is a one vs all approach.
            %
            % Keyword Arguments
            % trainingMatrix - An NxM matrix containing the training examples.
            % target - An Nx1 vector of classes. The classes should be ordered so that target(i) contains the class of example trainingMatrix(i).
            % recordOfDescent - An array recording the maximum probability classification error at the start of each batch.
            %
            % Return Values
            % recordOfDescent - The input array with the error for each batch tested during this call to train added to it.

            % Check parameters and initialise variables.
            if size(trainingMatrix, 1) ~= numel(target),
                error('The training matrix contains %d examples, while the target vector has %d.', size(trainingMatrix, 1), numel(target));
            end;
            if nargin < 4
                recordOfDescent = [];
            end

            % Set up the coefficients.
            uniqueClasses = unique(target);
            if ~isempty(obj.coefficients)
                % Training has been performed before.
                % Check that the training matrix provided has the correct number of variables.
                if (size(trainingMatrix, 2) + 1) ~= size(obj.coefficients, 1)
                    error('The training matrix contains %d variables, while there are %d coefficients already initialised (including the bias).', ...
                        size(trainingMatrix, 2) - 1, size(obj.coefficients, 1));
                end

                % Handle any classes in target that are not in obj.targetClasses.
                missingClasses = setdiff(uniqueClasses, obj.targetClasses);  % Classes in target but not in obj.targetClasses.
                missingClasses = sprintf('%d ', missingClasses);
                error('The following classes were previously used to train the models, but are not in the current training set: %s', missingClasses);
            else
                % Training has not been performed before, so perform basic initialisation.
                obj.targetClasses = uniqueClasses;
                obj.coefficients = zeros(1 + size(trainingMatrix, 2), numel(uniqueClasses));
            end

            % Convert the classes into a matrix with the same number of columns as there are classes.
            % Each column will correspond to a single class. Given target, the array of classes, the matrix of classes (targetMatrix)
            % will be organised so that targetMatrix(i, 1) == 1 if target(i) contains the first class.
            % Otherwise targetMatrix(i, 1) == 0. For example:
            %   target == [1 2 3 2 1]
            %                  [1 0 0]
            %                  [0 1 0]
            %   targetMatrix = [0 0 1]
            %                  [0 1 0]
            %                  [1 0 0]
            targetMatrix = zeros(numel(target), numel(obj.targetClasses));
            for i = 1:numel(obj.targetClasses)
                targetMatrix(target == obj.targetClasses(i), i) = 1;
            end

            % Perform the training.
            currentIter = 1;
            numberCoefficients = size(obj.coefficients, 1);
            while (currentIter <= obj.maxIter)
                % Generate the data permutation. The test sets of cvpartition(target, 'KFold', k) will give indices of the examples to
                % include in each mini-batch. The batches will also be stratified.
                numBatches = ceil(numel(target) / obj.batchSize);
                if numBatches < 2
                    % If the number of batches to create is less than two, then you can't use cvpartition.
                    % Basically mini-batch isn't needed, and you want to do full batch gradient descent.
                    partitions.NumTestSets = 1;
                    partitions.test = @(x) logical(ones(numel(target), 1));
                else
                    partitions = cvpartition(target, 'KFold', numBatches);
                end

                for i = 1:partitions.NumTestSets
                    % Determine training batch data, target classes and number of examples for this batch.
                    miniBatchSize = sum(partitions.test(i));
                    miniBatchTrain = trainingMatrix(partitions.test(i), :);
                    miniBatchTrain = [ones(miniBatchSize, 1) miniBatchTrain];  % Add the bias term.
                    miniBatchTarget = targetMatrix(partitions.test(i), :);

                    % Calculate the logistic output (predictions for each training example).
                    predictions = obj.calc_logistic(miniBatchTrain);
                    predictionErrors = predictions - miniBatchTarget;  % Difference between predicted value and actual class value.

                    % Determine the maximum probability error at the start of this batch.
                    miniBatchClasses = target(partitions.test(i));
                    performance = obj.calculate_performance(predictions, miniBatchClasses, []);
                    maxProbError = sum(miniBatchClasses ~= performance.maxProb) / miniBatchSize;
                    recordOfDescent = [recordOfDescent maxProbError];

                    % Update the coefficients.
                    % This process is somewhat complicated to account for the possibility of multiple classes.
                    % Assume we have five examples in the batch, that there are three feature and three classes in the dataset.
                    % Taking the first batch in the first iteration (so that all coefficients are 0), we may have a situation like:
                    %                                                [1 0 0]               [0.5 0.5 0.5]                    [-0.5 0.5 0.5]
                    %                      [0 0 0]                   [0 1 0]               [0.5 0.5 0.5]                    [0.5 -0.5 0.5]
                    %   obj.coefficients = [0 0 0] miniBatchTarget = [0 1 0] predictions = [0.5 0.5 0.5] predictionErrors = [0.5 -0.5 0.5]
                    %                      [0 0 0]                   [0 0 1]               [0.5 0.5 0.5]                    [0.5 0.5 -0.5]
                    %                                                [1 0 0]               [0.5 0.5 0.5]                    [-0.5 0.5 0.5]
                    % For logistic regression gradient descent we need to multiply the error for each example i by every coefficient j.
                    % This means that we need to multiple each column in predictionErrors by miniBatchTrain.
                    % We therefore need to iterate through the classes, replicate the prediction errors for the one vs all model for the class
                    % and multiply the mini batch by the replicated errors.
                    % We can then use the result of this to update the class' one vs all model coefficients.
                    for j = 1:numel(obj.targetClasses)
                        nonRegularisedTerm = bsxfun(@times, miniBatchTrain, predictionErrors(:, j));  % Non-regularised portion of the gradient descent equation.
                        regularisationTerm = (obj.lambda / miniBatchSize) * obj.coefficients(:, j);  % Regularised portion of the gradient descent equation.
                        obj.coefficients(:, j) = obj.coefficients(:, j) - (obj.alpha * ((sum(nonRegularisedTerm) / miniBatchSize)' + regularisationTerm));
                    end
                end

                % Update the iteration number.
                currentIter = currentIter + 1;
            end
        end
    end

end