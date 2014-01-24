function objectives = synthImportObjectives(objectives_base_filename, metrics_base_filename)
%function objectives = synthImportObjectives(objectives_base_filename, metrics_base_filename)
%
% @description
%  
%   Imports list of objectives from a file specified by base_filename.  
%   Expects these files:
%     base_filename.hdr -- ascii list of unscaled metric names
%
% @arguments
% 
%   objectives_base_filename -- string -- 
%   metrics_base_filename -- string -- 
%
% @return
% 
%  objectives -- just like synthImportMetrics, except just objectives data
%
% @exceptions
%
% @notes
%
%


% read in objective vars file
fid = fopen([objectives_base_filename '.hdr'],'r');
[s,count] = fscanf(fid, '%s', inf);
fclose(fid);

objective_vars = {};
remain = s;
while length(remain) > 0
  [token, remain] = strtok(remain,',');
  objective_vars = [objective_vars token];
end
    
% we construct objectives.data as a subset of the metrics data. To
% do that, we need to align the metrics' names with the objectives
metrics = synthImportMetrics(metrics_base_filename); 
all_metric_vars = metrics.header;
all_metric_X = metrics.data'; % [metric][sample]

objective_I = []; %these are the indices into metrics which are objectives
for i = 1:length(all_metric_vars)
    found_match = 0;
    for j = 1:length(objective_vars)
        if strcmp(objective_vars{j}, all_metric_vars{i})
            found_match = 1;
            break
        end
    end
    if found_match
        objective_I = [objective_I i];
    end
end
objective_X = all_metric_X(objective_I, :);

objectives = [];
objectives.header = objective_vars;
objectives.data = objective_X';

