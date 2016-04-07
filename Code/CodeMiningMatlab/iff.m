function result = iff(condition, trueResult, falseResult)
    % Implements a ternary if statement (similar to ?: in other languages).
    %
    % The difference here is that both trueResult and falseResult must be valid values.
    % There is no short circuiting, as both values are passed into this function (and must therefore exist).
    %
    % Keyword Arguments
    % condition - Boolean condition evaluating to true or false.
    % trueResult - Value to return if the condition is true.
    % falseResult - Value to return if the condition is false.

    if condition
        result = trueResult;
    else
        result = falseResult;
    end

end
