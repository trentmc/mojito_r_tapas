function s = coststr(x)
%function s = coststr(x)
%
%Pretty-print a scalar value 'x'

if (abs(x) > 0.1) & (abs(x) < 1000)
    s = sprintf('%.2f', x);
else
    s = sprintf('%.2e', x);
end

