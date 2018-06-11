function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
% Initialize some useful values
data = load('ex1data2.txt');
y = data(:,3);
m = length(y); % number of training examples
X = data(:,1:2);
NOF = size(X,2);
[X_norm, mu, sigma] = featureNormalize1(X);

X_new_norm = [ones(m,1), X_norm(:,1:2)];

theta = zeros(3,1);
alpha = 0.01;
num_iters = 1500;



    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    for j=1:NOF
        der(j)=0;
        for i=1:m

            der(j) = der(j)+(theta'*X_new_norm(i,:)'-y(i)).*X_new_norm(i,j);
        end
        der(j)=-alpha/m*der(j);
    end

    for j=1:NOF 
        theta(j,1)=theta(j,1)+der(j);
    end
    
    % ============================================================

    % Save the cost J in every iteration    
J_history(iter) = computeCostMulti1(X, y, theta);

end  

end

function [X_norm, mu, sigma] = featureNormalize1(X)
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

% ============================================================

end

function J = computeCostMulti1(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

m = length(y); % number of training examples

[X_norm, mu, sigma] = featureNormalize1(X);

X_new_norm = [ones(m,1), X_norm(:,1:2)];

%theta = zeros(3,1);

% You need to return the following variables correctly 
J = 0;
for i=1:m
    J=J+(theta'*X_new_norm(i,:)'-y(i)).^2;
end
J= J/(2*m);

% =========================================================================

end


