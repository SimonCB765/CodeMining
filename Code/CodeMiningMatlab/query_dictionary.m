function [codeDescription] = query_dictionary(codeMap, code)
    % Safely access the mapping from codes to descriptions.
    % The code makes the assumption that all valid clinical codes have a description in the mapping, and therefore that
    % any missing codes are for vendor specific drugs.
    %
    % Keyword Arguments
    % codeMap - containers.Map mapping each code to its description.
    % code - The code to extract the description for.

    if isKey(codeMap, code)
        % If the code is in the map, then get its description.
        codeDescription = codeMap(code);
    else
        % If the code is not in the map, then it is an unknown vendor specific drug.
        codeDescription = 'Unknown vendor specific drug';
    end

end
