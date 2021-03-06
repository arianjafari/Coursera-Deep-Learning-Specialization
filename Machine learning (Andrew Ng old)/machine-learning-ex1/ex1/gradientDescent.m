function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values

   % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

m = length(y); % number of training examples

J_history = zeros(num_iters, 1);


for iter = 1:num_iters

    for j=1:2
        der(j)=0;
        for i=1:m

            der(j) = der(j)+(theta'*X(i,:)'-y(i)).*X(i,j);
        end
        der(j)=-alpha/m*der(j);
    end

    for j=1:2 
        theta(j,1)=theta(j,1)+der(j);
    end
    


 
    % ============================================================

    % Save the cost J in every iteration    
J_history(iter) = computeCost(X, y, theta);

end  

end
