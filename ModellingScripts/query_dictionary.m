function [codeDescription] = query_dictionary(codeMap, code)

if isKey(codeMap, code)
    codeDescription = codeMap(code);
else
    codeDescription = 'Unknown vendor specific drug';
end
