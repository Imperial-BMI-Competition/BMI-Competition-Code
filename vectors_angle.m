function [dot_product] = vectors_angle(a,b)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
x1 = a(1); y1 = a(2);
x2 = b(1); y2 = b(2);
dot_product = atan2d(x1*y2-y1*x2,x1*x2+y1*y2);
end

