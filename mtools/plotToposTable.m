function [] = plotToposTable(topos)
%plotToposTable(topos)
% Plots a table that maps ind IDs to topologies
% where 'topos' can come from: synthImportTopos([base '_topos']);

fig = 1;
table_width = 550;
table_height = 400;
row_height = 11;
gFont.size=8;

[num_rows, num_columns] = size(topos.data);

cell_data = num2cell(topos.data);
for row_i = 1:num_rows
    s = sprintf('%012.f', cell_data{row_i, 1});
    cell_data{row_i, 1} = s;
end    

columninfo.titles = {}; %IND_ID var1 var2 ... varN
for column_i = 1:num_columns
    varname = topos.header{column_i};
    varname = strrep(varname, '_', '');
    varname = strrep(varname, 'stage', 'stg'); %compress
    varname = strrep(varname, 'chosenpartindex', 'chosenidx'); %compress
    varname = strrep(varname, 'cascode', 'casc');%compress
    columninfo.titles = [columninfo.titles varname];%compress
end

columninfo.formats = {'%020d'};
for column_i = 1:num_columns-1
    columninfo.formats = [columninfo.formats {'%d'}];
end

columninfo.weight = ones(1, num_columns);
columninfo.multipliers = ones(1, num_columns);
columninfo.isEditable = zeros(1, num_columns);
columninfo.isNumeric = [0 ones(1, num_columns - 1)];
columninfo.withCheck = 0; % optional to put checkboxes along left side

gFont.name='Helvetica';
tbl = axes('units', 'pixels','position', [1 1 table_width table_height]);

mltable(fig, tbl, 'CreateTable', columninfo, row_height, cell_data, gFont);



