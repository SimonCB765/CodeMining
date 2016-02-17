function [codeList] = extract_child_codes(inputCodes, codesToCheck)
    % Extract all codes in the codesToCheck array that contain atleast one code in inputCodes as their starting characters.
    % This will also return all codes in inputCodes that are found in codesToCheck.
    %
    % For example, if the input codes are {'C10', 'B10'}, then the function will return all codes from codesToCheck
    % that begin with C10 or B10 (e.g. C10F, B103, B10G, etc.).
    %
    % Keyword Arguments
    % inputCodes - Cell array of the codes whose children are to be found.
    % codesToCheck - Cell array of the codes that need to be checked for childhood.

    codeList = cellfun(@(x) check_code_is_child(x), codesToCheck);
    codeList = codesToCheck(codeList);

    function [isChild] = check_code_is_child(codeBeingChecked)
        % Check whether codeBeingChecked is a child code of any of the input codes.
        
        % Preallocate cell array that will be used to contain the multiple copies of the code being checked.
        codeBeingCheckedCopies = cell(1, numel(inputCodes));
        codeBeingCheckedCopies(:) = {codeBeingChecked};
        
        % Check whether the code is a match for any of the codes in the input list.
        % A code is a child if at least one pattern matches (i.e. if not all matches are empty).
        codeMatches = regexp(codeBeingCheckedCopies, strcat({'^'}, inputCodes));
        isChild = ~all(cellfun(@isempty, codeMatches), 2);  % Check if there is any match at all.
    end

end
