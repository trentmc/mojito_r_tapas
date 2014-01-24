function [] = plot_nmse(base_dir)
%function [] = plot_nmse(base_dir)
%
% 1. Loads in files:
%   $base_dir/best_waveforms_an0_env0_sw0.txt
%   $base_dir/best_waveforms_reference_an0.txt
%
% 2. Plots yhat vs. DC_in, and y vs. DC_in
%
  
eval(['load ' base_dir '/best_waveforms_reference_an0.txt;']);
eval(['load ' base_dir '/best_waveforms_an0_env0_sw0.txt;']);

y = best_waveforms_reference_an0(1,:);
dc_in = best_waveforms_an0_env0_sw0(1,:);
yhat = best_waveforms_an0_env0_sw0(2,:);

plot(dc_in, y, 'b', dc_in, yhat, 'r');
xlabel('Input DC voltage');
ylabel('Output DC voltage');
