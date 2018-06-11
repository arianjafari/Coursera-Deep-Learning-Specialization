function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
size_z = size(z);
m= size_z(1,1);
n= size_z(1,2);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

for i=1:m
	for j=1:n
		g(i,j)=1/(1+exp(-z(i,j)));
end


% =============================================================

end
