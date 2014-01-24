function metrics = synthImportMetrics(base_filename)
%function metrics = synthImportMetrics(base_filename)
%
% @description
%  
%   Imports metrics from a file specified by base_filename.  
%   Expects these files:
%     base_filename.hdr -- ascii list of unscaled metric names
%     base_filename.val -- ascii list of unscaled metric values
%
% @arguments
% 
%   base_filename -- string -- 
%
% @return
% 
%  metrics -- structure with attributes 'header' and 'data':
%    data -- 2d array where each row is a different ind, and
%            columns correspond to data of: ID, GENETIC_AGE, FEASIBLE, metric1,
%            metric2, metric3, ..., metricN.
%    header -- header{1}{1} has string 'IND_ID GENETIC_AGE FEASIBLE...'
%
% @exceptions
%
% @notes
%
%

tmpvals = load([base_filename '.val']);

% we assume that there are as many header entries as there are values
formatstring='';
for i=1:size(tmpvals,2)
    formatstring=[formatstring '%s'];
end

% read in the headers
fid = fopen([base_filename '.hdr'],'r');
hdr = textscan(fid,formatstring,'delimiter',',');
fclose(fid);

metrics = [];
metrics.header = hdr;
metrics.data = tmpvals;
