function qry_dictionary(coding, codelist, relevance, fid)

if nargin<3,
  relevance=[];
end;

if nargin<4,
  fid=1; % print to the secreen
end;


for i=1:numel(codelist),
  if isKey(coding,codelist{i}),
    if isempty(relevance),
      fprintf(fid,'%s, %s\n', codelist{i}, coding(codelist{i}));
    else
      fprintf(fid,'%1.3f, %s, %s\n', relevance(i), codelist{i}, coding(codelist{i}));
    end;
  else
    if isempty(relevance),
      fprintf(fid,'%s, Unknown (vendor-specific drug)\n', codelist{i});
    else
      fprintf(fid,'%1.3f, %s, Unknown (vendor-specific drug)\n', relevance(i), codelist{i});
    end;
  end;
end;
