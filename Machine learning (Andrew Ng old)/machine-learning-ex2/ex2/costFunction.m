function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
size_X = size(X);
n = size_X(1,2);

% You need to return the following variables correctly 


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

z = zeros(m,1);
for i=1:m

	z(i)=theta'*X(i,:)';
end
z;
size_z=size(z);
g = sigmoid(z);
size(g);	

J = 0;
grad = zeros(size(theta));


for i=1:m
	J= J + (-y(i)*log(g(i))-(1-y(i))*log(1-g(i)));
end

J= J/m;

for j=1:n

	grad(j) = 0;
	for i=1:m
		grad(j) = grad(j)+(g(i)-y(i)).*X(i,j);
	end

	grad(j)=grad(j)/m;

end





% =============================================================

end
