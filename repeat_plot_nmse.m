function [] = repeat_plot_nmse(base_dir, interval)
%function [] = repeat_plot_nmse(base_dir, interval)
%
% Calls plot_nmse(base_dir) every 'interval' seconds.
  

  
while 1
  plot_nmse(base_dir);
  title(ctime(time()));
  pause(interval)
end
