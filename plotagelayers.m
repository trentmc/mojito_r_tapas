%  load('tmp') % tmp is generated with the parsecosts.py

age_gap = 10
max_nb_age_layers = 10

nb_gens = size(tmp, 1) - 1;
nb_layers = 1 + floor(nb_gens / age_gap);

gen_ids = tmp(1:end-1,1);
costs = tmp(1:end-1, 2:end);

figure
hold on
colors = {};


colors{end+1} = 'b.-';
colors{end+1} = 'g.-';
colors{end+1} = 'c.-';
colors{end+1} = 'm.-';
colors{end+1} = 'r.-';
colors{end+1} = 'b.--';
colors{end+1} = 'g.--';
colors{end+1} = 'c.--';
colors{end+1} = 'm.--';
colors{end+1} = 'r.--';

age_layers = {}
for i=1:nb_layers
  layer.starts_at = (i-1) * age_gap + 1;
  layer.id = i;
  layer.xvals = [];
  layer.yvals = [];
  age_layers{i} = layer;
end

for gen=1:length(gen_ids)
    base_age_layer = max(0, floor(((gen-1) / age_gap) - max_nb_age_layers + 1));
    % the first max_nb_age_layers layers
    for i=1:max_nb_age_layers
      cost = costs(gen, i);
      if cost < 0
        age_layers{i+base_age_layer}.xvals = [age_layers{i+base_age_layer}.xvals gen_ids(gen)];
        age_layers{i+base_age_layer}.yvals = [age_layers{i+base_age_layer}.yvals cost];
      end
    end
end

%      layer_starts_at = min(find(costs(:, i) < 0.0));
%  
%      xvals = gen_ids(layer_starts_at:end);
%      yvals = costs(layer_starts_at:end, i);

for i=1:nb_layers
    layer = age_layers{i};
    plot(layer.xvals, layer.yvals, colors{mod(i, length(colors))+1});
    plot(gen_ids, costs(:, 1), 'k*');
end
grid
