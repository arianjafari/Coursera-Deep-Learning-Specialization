function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

data = load('ex1data2.txt');
y = data(:,3);
%m = length(y); % number of training examples
X = data(:,1:2);
m = length(X(:,1));


% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
% 

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


% ============================================================

end
