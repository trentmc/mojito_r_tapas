%play.m

%==========================================================================
%set target data
basedir = '/users/micas/tmcconag/novelty_results/';

%%pick one of the following
%base = [basedir 'results_three_objs/base'];
%base= [basedir 'merge4-results-p43_20071115_204811/base'];
%base= [basedir 'merge4-62-results-p63_20071115_211316/base'];
base= [basedir 'merge4-62-results-p63_20071117_032910/base']; %final results

%==========================================================================
%load data

%(not used, so turned off)
%metrics = synthImportMetrics([base '_metrics']); 
%all_metric_vars = metrics.header; %{metric index}
%all_metric_X = metrics.data'; % [metric index][sample index]

topos = synthImportTopos([base '_topos']);
all_topo_vars = topos.header; % {topo var index}
all_topo_X = topos.data'; % [topovar index][sample index]
all_topo_strings = topos.strings; %{sample index}

topovar_index_of_stage_choice = 2; %HACK
onestage_sample_I = find(all_topo_X(topovar_index_of_stage_choice,:) == 0);
twostage_sample_I = find(all_topo_X(topovar_index_of_stage_choice,:) == 1);

colors = [];
for sample_i = 1:size(all_topo_X, 2)
    if all_topo_X(topovar_index_of_stage_choice, sample_i) == 0
        colors = [colors 'r'];
    else
        colors = [colors 'b'];
    end
end

%(not used, so turned off)
%[scaled, unscaled] = synthImportPoints([base '_points']);  
%all_unscaled_vars = unscaled.header; % {var index}
%all_unscaled_X = unscaled.data'; % [var index][sample index]

%'objective_X' has _only_ metric_value data for objectives; not ind_ID etc
objectives = synthImportObjectives([base '_objectives'], [base '_metrics']);
objective_vars = objectives.header; % {objective index}
objective_X = objectives.data'; % [objective index][sample index]
for i = 1:length(objective_vars)
    objective_vars{i} = strrep(objective_vars{i}, '_', ' ');
end

%'notposs_X' has _only_ metric value data for objectives; not ind_ID etc
notposs = synthImportMetrics([base '_notposs']);
notposs_X = notposs.data'; % [objective index][sample index]

objective_X01 = objective_X; %like objective_X, but range is [0,1] per row
for var_i = 1:size(objective_X,1)
    mn = min(objective_X(var_i, :));
    mx = max(objective_X(var_i, :));
    objective_X01(var_i,:) = (objective_X(var_i,:) - mn) / (mx - mn);
end

%classification data
X = [objective_X notposs_X]'; %[sample index][var index]
class_labels = all_topo_strings; %[sample_index]
for i = 1:size(notposs_X, 2)
    class_labels = [class_labels 'not possible'];
end
class_label_per_index = unique(class_labels); %[class #] : class_label
class_indices = []; %[sample index]
for sample_i = 1:length(class_labels)
    %determine the class_index of sample_i
    for class_index = 1:length(class_label_per_index)
        if strcmp(class_label_per_index{class_index}, class_labels{sample_i})
            break
        end
    end
    class_indices = [class_indices class_index];
end

%==========================================================================
%start plotting here

%maybe plot a table of ind IDs <=> topology values
if 0
    plotToposTable(topos)
end

%maybe plot pareto front as 3d scatterplot
if 0
    figure(2);
    plot3(objective_X(1,:), objective_X(2,:), objective_X(3,:), 'ro');
            
    xlabel(objective_vars{1});
    ylabel(objective_vars{2});
    zlabel(objective_vars{3});
    grid on;
end


%maybe plot pareto front with convex-hull approach
if 0
    figure(3);
    plot3dParetoFront(objective_X(1,:), objective_X(2,:), objective_X(3,:), ...
        objective_vars);
end

%maybe do 2d scatterplot grid
%objective_X = objectives.data'; % [objective index][sample index]
if 1
    figure(4);
    font_size = 11; %10 is default
    font_weight = 'bold'; %'normal' or 'bold'
    
    %affects relative size of plotted area vs. whitespace surrounding
    %outer_position = [-.2 -.2 1.4 1.4]; %[left bottom width height, all normalized] default [0 0 1 1]
    outer_position = [0 0 1 1]; %[left bottom width height, all normalized] default [0 0 1 1]
    
    %h = plotmatrix(objective_X','o');
    %set(h, 'MarkerSize','3');
    
    %linetypes = 'ox+*sdv^<>phox+';
    num_objs = size(objective_X, 1);
    subplot_i = 0;
    
    %hack so that we don't have to have "*10^9" on the slewrate axis
    objective_vars2 = objective_vars;
    objective_vars2{5} = 'slewrate / 10^9'; %
    objective_X2 = objective_X;
    objective_X2(5,:) = objective_X2(5,:) / 1e9; %
        
    for row_i = 1:num_objs
        for col_j = 1:num_objs
            subplot_i = subplot_i + 1;
            h = subplot(num_objs, num_objs, subplot_i);
            if row_i == col_j
                hist(objective_X2(row_i,:), 20);
            else
                hold off;
                plot(objective_X2(col_j,onestage_sample_I), objective_X2(row_i,onestage_sample_I), 'ks', 'MarkerSize', 5);
                hold on;
                plot(objective_X2(col_j,twostage_sample_I), objective_X2(row_i,twostage_sample_I), 'k+', 'MarkerSize', 5);
                hold off;              
            end
            
            %set y-label and y-tick marks (on 1st column, with exceptions)
            if (col_j == 1 && row_i == 1)
                ylabel(objective_vars2{row_i}, 'FontSize', [font_size], 'FontWeight', font_weight);
                set(h, 'YTickLabel', '');
            elseif (col_j == 2 && row_i ==1)
                ; 
            elseif (col_j == 1)
                ylabel(objective_vars2{row_i}, 'FontSize', [font_size], 'FontWeight', font_weight);
            else
                set(h, 'YTickLabel', '');
            end
                        
            %set x-label and x-tick marks
            if (row_i == num_objs && col_j == num_objs)
                xlabel(objective_vars2{col_j}, 'FontSize', [font_size], 'FontWeight', font_weight);
                set(h, 'XTickLabel', '');
            elseif (row_i == (num_objs-1) && col_j == num_objs)
                ;
            elseif (row_i == num_objs)
                xlabel(objective_vars2{col_j}, 'FontSize', [font_size], 'FontWeight', font_weight);
            else
                set(h, 'XTickLabel', '');
            end
            
            set(h, 'FontSize', [font_size], 'FontWeight', font_weight);
            %set(h, 'OuterPosition', outer_position);
            pos = get(h, 'Position');
            dw = 0.01;
            dh = 0.015;
            newpos = [pos(1)-dw, pos(2)-dh, pos(3)+dw*2, pos(4)+dh*2];
            set(h, 'Position', newpos);
        end
    end
end



%maybe build tree
if 0
    min_n_per_node = 10; %1 for full fit, 10 for more compact tree
    split_criterion = 'gdi'; %one of: gdi, twoing, deviance    

    tree = classregtree(X, class_labels, 'names', objective_vars, ...
        'splitmin', min_n_per_node, 'splitcriterion', split_criterion);
    
    %determine cost vs. pruning level (via cross-validation), and plot it
    [c,s,n,best_level] = test(tree,'cross',X, class_labels);
    [mincost,minloc] = min(c);
    figure(5);
    plot(n,c,'b-o',...
        n(best_level+1),c(best_level+1),'bs',...
        n,(mincost+s(minloc))*ones(size(n)),'k--');
    xlabel('Tree size (number of terminal nodes)');
    
    %maybe prune based on level
    if 0
        pruned_tree = prune(tree,'level', best_level);
        
    %else prune based on target number of leaf nodes
    else
        pruned_tree = prune(tree,'nodes', 15);
    end
    
    %view the final tree
    view(pruned_tree);
    
end

%maybe do hierarchal clustering and display with dendrogram
if 0
    Y = pdist(objective_X01', 'cityblock');
    Z = linkage(Y, 'single'); %generates a hierarchical cluster tree
    p = 0; %set to 60 or larger to have fewer leaves; set to 0 to have all leaves
    figure(6);
    [H,T] = dendrogram(Z, p, 'labels', all_topo_strings, ...
        'orientation', 'left', 'colorthreshold', 'default');
    set(H,'LineWidth',2);
    
    max_num_clusters = 10;
    cluster_index_per_sample = cluster(Z,'maxclust', max_num_clusters);
    for cluster_i = 1:max(cluster_index_per_sample)
        I = find(cluster_index_per_sample == cluster_i);
        s = [sprintf('Cluster: %3d design(s).  ', length(I))];
        for obj_i = 1:length(objective_vars)
            s = [s sprintf('%s=[%s,%s], ', ...
                objective_vars{obj_i}, ...
                coststr(min(objective_X(obj_i,I))), ...
                coststr(max(objective_X(obj_i,I))) ...
                )];
        end
        s = [s '\n'];
        fprintf(1, s);
    end
end


%maybe plot radar (spider) plot
if 0
    figure(7);
    spiderplot2(objective_X, max(objective_X')', min(objective_X')', colors', objective_vars);
end
