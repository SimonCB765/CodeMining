function [codeList] = extract_child_codes(inputCodes, codesToCheck, findChildren)
    % Extract child codes of an array of input codes.
    %
    % All codes in inputCodes will be returned, and optionally their child codes.
    % Look through the inputCodes array of codes, and extract any codes in codesToCheck that begin with a codes in inputCodes.
    % If findChildren is not set for a code in inputCodes, then just return the code in inputCodes.
    %
    % For example, if inputCodes is {'C10' 'B10'} and findChildren is {true false}, then all codes beginning with C10 will
    % be returned (e.g. C10F, C10E). In addition, the code B10 will be returned, but not its children.
    %
    % Keyword Arguments
    % inputCodes - Cell array of the codes whose children are to be found.
    % codesToCheck - Cell array of the codes that need to be checked for childhood.
    % findChildren - Cell array indicating which codes should have their child codes extracted.

    % Check inputs.
    if nargin < 3
        % Default to finding children for all input codes.
        findChildren = repmat({1}, 1, numel(inputCodes));
    end
    if numel(findChildren) < numel(inputCodes)
        % Pad the findChildren array to ensure it is as long as the inputCodes array.
        findChildren(end + 1 : numel(inputCodes)) = {1};
    end

    % Create the regular expression for matching each code in inputCodes.
    % If a code is not having its children extracted, then it gets a $ added to its regular expression.
    % For example, inputCodes is {'C10' 'B10'} and findChildren is {true false}, then
    % codeStrMatches = {'^C10' '^B10$'}
    codeStrMatches = cellfun(@(x, y) strcat('^', x, iff(y, '', '$')), inputCodes, findChildren, 'UniformOutput', false);

    codeList = cell2mat(cellfun(@(x) check_code_is_child(x), codesToCheck, 'UniformOutput', false));
    codeList = codesToCheck(codeList);

    function [isChild] = check_code_is_child(codeBeingChecked)
        % Check whether codeBeingChecked is a child code of any of the input codes.

        % Preallocate cell array that will be used to contain the multiple copies of the code being checked.
        codeBeingCheckedCopies = cell(1, numel(inputCodes));
        codeBeingCheckedCopies(:) = {codeBeingChecked};

        % Check whether the code is a match for any of the codes in the input list.
        % A code is a child if at least one pattern matches (i.e. if not all matches are empty).
        codeMatches = regexp(codeBeingCheckedCopies, codeStrMatches);
        isChild = ~all(cellfun(@isempty, codeMatches));  % Check if there is any match at all.
    end

end
