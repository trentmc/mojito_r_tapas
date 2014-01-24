function [] = interpPlot(x, y, z, axis_labels)
%function [] = interpPlot(x, y, z, axis_labels)
%
%interpolation-based surface plot

gridsize = 200;
[XI, YI] = meshgrid(min(x):(max(x)-min(x))/gridsize:max(x), min(y):(max(y)-min(y))/gridsize:max(y));

ZI = griddata(x,y,z, XI,YI); %interpolates


%Plot the gridded data along with the nonuniform data points used to generate it:
mesh(XI,YI,ZI), hold
plot3(x,y,z,'o'), hold off

xlabel(axis_labels{1});
ylabel(axis_labels{2});
zlabel(axis_labels{3});