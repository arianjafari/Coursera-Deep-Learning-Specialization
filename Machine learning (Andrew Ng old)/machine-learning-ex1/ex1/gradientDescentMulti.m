function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
% Initialize some useful values
m = length(y); % number of training examples
NOF = size(X,2);
[X_norm, mu, sigma] = featureNormalize(X(:,2:end));

X = [ones(m,1), X_norm(:,1:end)];




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

            der(j) = der(j)+(theta'*X(i,:)'-y(i)).*X(i,j);
        end
        der(j)=-alpha/m*der(j);
    end

    for j=1:NOF 
        theta(j,1)=theta(j,1)+der(j);
    end
    
    % ============================================================

    % Save the cost J in every iteration

J_history(iter) = computeCostMulti(X, y, theta);

end  

end