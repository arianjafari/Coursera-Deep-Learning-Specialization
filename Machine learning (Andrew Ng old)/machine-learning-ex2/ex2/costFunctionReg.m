function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


z = zeros(m,1);
for i=1:m

	z(i)=theta'*X(i,:)';
end
z;
size_z=size(z);
g = sigmoid(z);
size(g);	
J = 0; 
J1 = 0;
J2 = 0;
grad = zeros(size(theta));


for i=1:m
	J1= J1 + (-y(i)*log(g(i))-(1-y(i))*log(1-g(i)));
end

for j=2:n
	J2 = J2 + theta(j)^2;
end

J= J1/m+ (0.5*lambda/m)*J2;

grad(1) = 0;
	for i=1:m
		grad(1) = grad(1)+(g(i)-y(i)).*X(i,1);
	end

	grad(1)=grad(1)/m;


for j=2:n

	grad(j) = 0;
	for i=1:m
		grad(j) = grad(j)+(g(i)-y(i)).*X(i,j);
	end

	grad(j)=grad(j)/m+(lambda/m)*theta(j);

end




% =============================================================

end
