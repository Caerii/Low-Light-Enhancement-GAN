function [g,lE]=gsolve(Z,B,l)

% Given a set of pixel values observed for several pixels in several
% images with different exposure times, this function returns the
% imaging systems response function g as well as the log film irradiance
% values for the observed pixels.
%
% Assumes:
%
%  Zmin = 0
%  Zmax = 255
%
% Arguments:
%
%  Z(i,j) is the pixel values of pixel location number i in image j
%  B(j)   is the log delta t, or log shutter speed, for image j
%  l      is lamdba, the constant that determines the amount of smoothness
% 
% Returns:
%
%  g(z)   is the log exposure corresponding to pixel value z
%  lE(i)  is the log film irradiance at pixel location i

n = 256;
w = ones(n,1)/n;
m = size(Z,1);
p = size(Z,2);

A = sparse([], [], [], m*p+n+1,n+m, m*p*2 + n*3);
b = zeros(size(A,1),1);

k = 1;              %% Include the data-fitting equations
for i=1:m
  for j=1:p
    wij = w(Z(i,j)+1);
    A(k,Z(i,j)+1) = wij; A(k,n+i) = -wij; b(k,1) = wij * B(j);
    k=k+1;
  end
end

A(k,129) = 1;       %% Fix the curve by setting its middle value to 0
k=k+1;

for i=1:n-2         %% Include the smoothness equations
  A(k,i)=l*w(i+1); A(k,i+1)=-2*l*w(i+1); A(k,i+2)=l*w(i+1);
  k=k+1;
end

x = A\b;            %% Solve the system using SVD

g = x(1:n);
lE = x(n+1:size(x,1));
