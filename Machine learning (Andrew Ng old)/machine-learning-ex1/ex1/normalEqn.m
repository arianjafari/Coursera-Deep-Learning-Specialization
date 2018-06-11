function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);



% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%
y;
X;
X';
XTX = X'*X;
pinv(XTX);
theta = pinv(XTX)*X'*y;
% ---------------------- Sample Solution ----------------------




% -------------------------------------------------------------


% ============================================================

end
