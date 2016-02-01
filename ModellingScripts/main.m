% Reset the workspace.
clear all
clc
close all

% Load the ?????????????????????????????????????Read V2 codes ?????????????????????? descriptions.
load coding

% Load the data on the patients. Each row in the file contains:
% PatentID\tCode\tOccurences
% Therefore, each line records the number of times a specific Read V2 code occurs in a specific patient's medical record.
% Both patient IDs and Read codes are likely to appear on more then one line (as patients have more than one code associated
% them and codes are associated with more than one patient).
% data.id is an array containing the patient IDs.
% data.key is an array containing the Read codes.
% data.count is an array containing the association counts.
[data.id, data.key, data.count] = textread('PatientData.csv', '%d %s %d');

% Determine unique patient IDs and codes, and create index mappings for each.
uniqueCodes = unique(data.key);  % A list of the unique Read codes in the dataset.
codeIndexMap = containers.Map(uniqueCodes, 1:numel(uniqueCodes));  % Map the Read codes to their index in the uniqueCodes array.
uniquePatientIDs = unique(data.id);  % A list of all the patient IDs in the dataset.
patientIndexMap = containers.Map(uniquePatientIDs, 1:numel(uniquePatientIDs));  % Map the patient IDs to their index in the uniquePatientIDs array.





%% create the sparse matrix
row=cell2mat(values(patientIndexMap, num2cell(data.id)));
col=cell2mat(values(codeIndexMap, data.key));
mat=sparse(row, col, data.count, numel(keys(patientIndexMap)), numel(keys(codeIndexMap)));




%% positive examples are those who don't have diabetes -- either 1 or 2
pos=(full(mat(:, codeIndexMap('C10E')) ==0) & full(mat(:, codeIndexMap('C10F')) ==0));
pos=find(pos);


%neg = setdiff( find(mat(:, codeIndexMap('C10F'))==0), find(mat(:, codeIndexMap('C10E'))==0));
neg = setdiff(1:numel(uniquePatientIDs), pos)';

%% Find the codes that is most densely populated and sort them
% I could have done so using SQL
code_count=full(sum(mat>0));
[count, inx]=sort(code_count,'descend');
nCodes = find(count>50, 1, 'last' );
{uniqueCodes{inx(1:nCodes)}}'
uniqueCodes_idx=inx(1:nCodes);
full(sum(mat(:,uniqueCodes_idx)>0))

%% plot frequency count
subplot(3,1,1);
set(gca,'fontsize',16);
bar(count(1:nCodes));
xlabel('Code index');
ylabel('Count');
axis tight;
fname = 'freq_count';
%print('-depsc2',['Pictures/main_mysql_local_host_v2__' fname '.eps']);
%print('-dpng',['Pictures/main_mysql_local_host_v2__' fname '.png']);

%% rank features using all data
X=mat([pos' neg'],uniqueCodes_idx);
Y = [ones(numel(pos),1); zeros(numel(neg),1)];
[IDX2, Z] = rankfeatures(X', Y, 'Criterion', 'entropy');
{uniqueCodes{uniqueCodes_idx(IDX2(1:100))}}


%% display the matrix X
clf;
set(gca,'fontsize',16);
imagesc(X>0);
%colorbar;
fname = 'sparse_matrix_w_freq_count_binary';
%print('-depsc2',['Pictures/main_mysql_local_host_v2__' fname '.eps']);
%print('-dpng',['Pictures/main_mysql_local_host_v2__' fname '.png']);

%% plot relative entropy
%clf;
%subplot(2,1,1);
set(gca,'fontsize',16);
bar(log(Z(IDX2)))
axis tight;
ylabel('Relative entropy (log)');
xlabel('Codes re-ordered by relative entropy');
axis tight;
%%
fname = 'feature_relevance';
print('-depsc2',['Pictures/main_mysql_local_host_v2__' fname '.eps']);
print('-dpng',['Pictures/main_mysql_local_host_v2__' fname '.png']);

%%
clf;
set(gca,'fontsize',16);
imagesc(X(:, IDX2)>0)
xlabel('Codes re-ordered by relative entropy');
ylabel('Patient index');
fname = 'matrix_RE';
%print('-depsc2',['Pictures/main_mysql_local_host_v2__' fname '.eps']);
%print('-dpng',['Pictures/main_mysql_local_host_v2__' fname '.png']);

%% Find the # of var with Z=Inf
InfZ_index=max(find(Z(IDX2) == inf))

%% visualize the dense X matrix with column ordered by entropy
clf;
set(gca,'fontsize',16);
imagesc( X([1:2000], IDX2(1:InfZ_index * 2))>0);
xlabel('Codes re-ordered by relative entropy');
ylabel('Patient index (1-230 C10E)');
fname = 'matrix_RE_zoom_in';
%print('-depsc2',['Pictures/main_mysql_local_host_v2__' fname '.eps']);
%print('-dpng',['Pictures/main_mysql_local_host_v2__' fname '.png']);




%% check which column is empty or is full
col=sum((X(1:numel(pos), 1:InfZ_index))>0);
features_present = find(col>0);
features_absent = find(col==0);

display('Features present');
{uniqueCodes{uniqueCodes_idx(IDX2(features_present))}}

display('Features absent');
{uniqueCodes{uniqueCodes_idx(IDX2(features_absent))}}

%% print the Inf code lists
codeInf = {uniqueCodes{uniqueCodes_idx(IDX2(1:InfZ_index))}};
qry_dictionary(coding, codeInf)

%% train logistic regression with only discriminative features
nCodes_discrim = sum( log(Z(IDX2(InfZ_index+1:nCodes))) >=0);

chosen_features = InfZ_index + [1:nCodes_discrim];
bar(Z( IDX2( chosen_features)))

%% Let's use lasso glm with parallel computing
% See for a tutorial:
% http://www.mathworks.co.uk/help/stats/lasso-regularization-of-generalized-linear-models.html

%opt = statset('UseParallel',true);
opt = statset('UseParallel','always');
matlabpool open 2
%%
K=2;
indices = crossvalind('Kfold', numel(neg), K);
%%
K=1;
Y = [ ones(numel(pos),1); zeros(sum(indices==K),1)];
X=mat([pos' neg(indices==K)'],uniqueCodes_idx(IDX2(chosen_features)));


%%
rng('default') % for reproducibility
tic
[B,S] = lassoglm(X>0, Y,'binomial','NumLambda',25, ...
  'Alpha',0.9,'LambdaRatio',1e-4,'CV',5,'Options',opt);
toc

%%
lassoPlot(B,S,'PlotType','CV');

%%
lassoPlot(B,S,'PlotType','Lambda','XScale','log')

% The right (green) vertical dashed line represents the Lambda providing the smallest cross-validated deviance. The left (blue) dashed line has the minimal deviance plus no more than one standard deviation. This blue line has many fewer predictors:

%%
[S.DF(S.Index1SE) S.DF(S.IndexMinDeviance)]
size(B)
%%
indx = S.Index1SE;
B0 = B(:,indx);
nonzeros = sum(B0 ~= 0)

cnst = S.Intercept(indx);
B1 = [cnst;B0];
%%
yfit = glmval(B1,X,'logit');

hist(Y - yfit) % plot residuals
title('Residuals from lassoglm model')

%%
bar(B1)
%%
bar(yfit)
%%
[coefs, index] = sort(abs(B0),'descend');
final_uniqueCodes = {uniqueCodes{uniqueCodes_idx(IDX2(chosen_features(index)))}}
qry_dictionary(coding, final_uniqueCodes, B0(index));
%%
fid = fopen('non-diabtes_uniqueCodes.csv', 'w');
qry_dictionary(coding, final_uniqueCodes, B0(index), fid);
fclose(fid);

%%
fid = fopen('inf_entropy_uniqueCodes.csv', 'w');
qry_dictionary(coding, codeInf, 1:numel(codeInf),fid );
fclose(fid);