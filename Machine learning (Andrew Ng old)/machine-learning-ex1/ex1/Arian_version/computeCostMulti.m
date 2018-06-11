function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
data = load('ex1data2.txt');
y = data(:,3);
m = length(y); % number of training examples
X = data(:,1:2);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


[X_norm, mu, sigma] = featureNormalize2(X);

X_new_norm = [ones(m,1), X_norm(:,1:2)];

theta = zeros(3,1);

% You need to return the following variables correctly 
J = 0;
for i=1:m
	J=J+(theta'*X_new_norm(i,:)'-y(i)).^2;
end
J= J/(2*m);


% =========================================================================

end


function [X_norm, mu, sigma] = featureNormalize2(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

m = length(X(:,1));
mean_X = mean(X);
std_X = std(X);


% You need to set these values correctly
X_norm = zeros(m,2);

if(std_X(1,1) ~= 0.0)
	X_norm(:,1) = (X(:,1)-mean_X(1,1))/std_X(1,1);
else
	X_norm(:,1) = (X(:,1)-mean_X(1,1));
endif
if(std_X(1,2) ~= 0.0)	 
	X_norm(:,2) = (X(:,2)-mean_X(1,2))/std_X(1,2);
else
	X_norm(:,2) = (X(:,2)-mean_X(1,2));
endif
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));


end