function topos = synthImportTopos(base_filename)
%function topos = synthImportTopos(base_filename)
%
% @description
%  
%   Imports topos from a file specified by base_filename.  
%   Expects these files:
%     base_filename.hdr -- ascii list of unscaled topology variable names
%     base_filename.val -- ascii list of unscaled topology variable values
%
% @arguments
% 
%   base_filename -- string -- 
%
% @return
% 
%  topos -- structure with attributes 'header' and 'data':
%    data -- 2d array where each row is a different ind, and
%            columns correspond to data of: topo_var1_value, topo_var2_value,... 
%    header -- header{1}{1} has string 'topo_var1_name topo_var2_name ...'
%    strings -- 1d cell array -- summary strings, with one entry per ind
%
% @exceptions
%
% @notes
%
%

%the functionality is _exactly_ like importing metrics, so just use that!!
topos = synthImportMetrics(base_filename);

all_topo_X = topos.data'; % [topovar index][sample index]
topos.strings = {}; %{sample index} 
for sample_i = 1:size(all_topo_X, 2)
    s = '';
    for topovar_i = 2:size(all_topo_X, 1) %ignore 1st entry of ind ID
        choice_val = all_topo_X(topovar_i, sample_i);
        if choice_val == -1
            s = [s 'X'];
        else
            s = [s num2str(choice_val)];
        end
    end
    topos.strings{sample_i} = s;
end