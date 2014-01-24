function [] = plot3dParetoFront(x, y, z, axis_labels)
%function [] = plot3dParetoFront(x, y, z, axis_labels))
%
%plot a 3d pareto front, by approximating it as a convex hull

X = [x; y; z]'; %num_samples x num_variables
K = convhulln(X);
h = trisurf(K,x,y,z, 'FaceColor','white', 'MarkerFaceColor', 'black', 'Marker', 'o');

xlabel(axis_labels{1});
ylabel(axis_labels{2});
zlabel(axis_labels{3});