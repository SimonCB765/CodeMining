classdef RegMultinomialLogistic < handle
    % Class defining the methods used for performing regularised multinomial logistic regression.

    properties (SetAccess = private)
        coefficients = [];  % Matrix of coefficients (one column per class) with an intercept as the first term.
        targetClasses = [];  % The classes sorted in the order of the columns in coefficients.
                             % E.g. targetClasses(1) corresponds to the coefficients in coefficients(:, 1).
        calc_logistic = @(data) 1./ (1 + exp(-data * weights));  % Calculate predicted values for logistic regression.
    end
    properties
        alpha = 0.01;
        batchSize = 500;
        lambda = 1;
        maxIter = 10;
    end
    methods
        function reset_coefficients(obj)
            % Resets the coefficients so the model can be retrained.
            obj.coefficients = [];
        end

        function set_parameters(obj, alpha, batchSize, lambda, maxIter)
            % Convenience function to set all parameters at once.
            obj.alpha = alpha;
            obj.batchSize = batchSize;
            obj.lambda = lambda;
            obj.maxIter = maxIter;
        end

        function [predictions] = test(obj, testMatrix)
            % Generate predictions for examples in testMatrix.
            %
            % Keyword Arguments
            % testMatrix - A NxM matrix of test examples. If there are C coefficients, then M + 1 == C.

            numVariables = size(testMatrix, 1);
            if numVariables ~= numel(obj.coefficients),
                error('The test matrix has %d variables, but there are %d coefficients (including the bias).', ...
                    numVariables, numel(obj.coefficients));
            end;
            predictions = obj.calc_logistic(testMatrix);
        end

        function train(obj, trainingMatrix, target, referenceClass, recordDescent)
            % Train the logistic regression model using a mini batch gradient descent approach.
            %
            % For multiple classes, the approach taken is a one vs all approach.
            %
            % Keyword Arguments
            % trainingMatrix - An NxM matrix containing the training examples.
            % target - An Nx1 vector of classes.
            % referenceClass - Class against which the logistic regression(s) will be performed.
            %                  Defaults to a class with largest value (e.g. 3 if the classes are 1, 2 and 3).
            % recordDescent - Whether the gradient descent should be recorded. Defaults to not recording.

            % Check parameters and initialise variables.
            if size(trainingMatrix, 1) ~= numel(target),
                error('The training matrix and target vector have different numbers of examples.');
            end;
            if isempty(obj.coefficients)
                % Create the matrix of coefficients. One column per class.
                obj.coefficients = zeros(1 + size(trainingMatrix, 2), numel(unique(target)));
            end
            if nargin < 3 || isempty(referenceClass)
                referenceClass = max(unique(target));
            end
            if nargin < 4 || isempty(recordDescent)
                recordDescent = false;
            end
        end
    end

end


function weights = reg_logistic_train(train_data, target, weights, alpha, lambda, batch_size, max_iter, debug)

% stochastic training
reorder = randperm(size(train_data,1));
train_data(reorder,:)= train_data;
target(reorder) = target;

% Now training

%construct the batch indexes
batch_begin = 1:batch_size:size(train_data, 1);
batch_end = batch_begin-1;
batch_end(1)=[];
batch_end = [batch_end, size(train_data, 1)];

b_size = batch_end-batch_begin+1; %this is the mini-batch size

m = numel(weights); % number of dimensions

%fprintf(1,'Has %d mini batches cyling up to %d times.\n', numel(b_size), max_iter);

error_ = zeros(numel(b_size) * max_iter, 1);

done = 0;
iter = 0;
iter_b = 0;
while(~done)

  iter=iter+1;

  for b = 1:numel(batch_begin), % num of targets to train

    iter_b = iter_b+1;

    dat = [ones(b_size(b), 1) train_data(batch_begin(b):batch_end(b),:)]; %create the data matrix and add the bias
    t = target(batch_begin(b):batch_end(b)); %target
    h = 1 ./ ( 1+ exp(- dat * weights')); %logistic output

    %fprintf(1,'.'); %check convergence before updating the weights
    %error_(iter_b,:) = [mean(h<0.5 == t), wer(h(t==0),h(t==1))];
    %subplot(1,2,1); bar(error_(:,1));
    %subplot(1,2,2); bar(error_(:,2));
    if debug,
      error_(iter_b) = mean(h<0.5 == t);
      plot(error_, 'x-');
      %drawnow;
    end;
    %fprintf(1,'Classification error: %1.2f , EER = %1.2f\n',));
    %end;

    %implementation following the equation https://class.coursera.org/ml-005/lecture/42
    d_weight = repmat((h-t), 1,m) .* dat ;
    reg_term = lambda/b_size(b) * weights; %regularisation term

    weights = weights - alpha * ( sum(d_weight) / b_size(b) + reg_term);
  end;

  %fprintf(1,'\n# of false accept %d, # of false reject', false_accept, false_reject);
  %mistake = false_accept + false_reject;

  if (iter >= max_iter),
    done = 1;
  end;
end;