function result = iff(condition,trueResult,falseResult)
% Implements a ternary if statement (similar to ?: in other languages).

if condition
    result = trueResult;
else
    result = falseResult;
end