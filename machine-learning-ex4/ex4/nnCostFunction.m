function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Calculate outputs of second layer
a2 = sigmoid(X*Theta1');

% Add a column of ones to the a2 outputs
a2 = [ones(m, 1) a2];

% Calculate outputs of third layer
a3 = sigmoid(a2*Theta2');

% Assign h_theta to be the outputs from the final layer
h_theta = a3;

% Recode y
y_new = zeros(m, num_labels);
for i = 1:m
  y_new(i, y(i)) = 1;
endfor

% Calculate the unregularised cost
Junreg = 0;
for i = 1:m
  % Cost for each example
  Junregi = -y_new(i,:)*log(h_theta(i,:))' - (1 - y_new(i,:))*log(1 - h_theta(i,:))';
  % Sum the costs
  Junreg = Junreg + Junregi;
endfor
% Divide by number of examples summed
Junreg = Junreg/m;


% Calculate the regularised cost
Jreg = 0;
Theta1_reg = [zeros(size(Theta1)(1), 1), Theta1(:,2:end)];
Theta2_reg = [zeros(size(Theta2)(1), 1), Theta2(:,2:end)];

Theta_sum1 = 0;
for j = 1:size(Theta1)(1)
  Theta1_slice = Theta1_reg(j, :);
  Theta_sum1 = Theta_sum1 + Theta1_slice*Theta1_slice';
endfor
Theta_sum2 = 0;
for j = 1:size(Theta2)(1)
  Theta2_slice = Theta2_reg(j, :);
  Theta_sum2 = Theta_sum2 + Theta2_slice*Theta2_slice';
endfor
Jreg = lambda/(2*m)*(Theta_sum1 + Theta_sum2);

% Add together the costs
J = Junreg + Jreg;


% Backpropagation
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for t = 1:m
  % Take a slice of the X data
  xt = X(t, :);
  a1 = xt;

  % Calculate outputs of second layer
  z2 = xt*Theta1';
  a2 = sigmoid(z2);

  % Add a column of ones to the a2 outputs
  a2 = [1 a2];

  % Calculate outputs of third layer
  z3 = a2*Theta2';
  a3 = sigmoid(z3);

  delta3 = a3 - y_new(t,:);
  
  delta2 = delta3*Theta2.*[1 sigmoidGradient(z2)];
  
  delta2 = delta2(2:end);

  Delta1 = Delta1 + delta2'*a1;
  Delta2 = Delta2 + delta3'*a2;
endfor

Theta1_grad = 1/m*(Delta1 + lambda*Theta1_reg);
Theta2_grad = 1/m*(Delta2 + lambda*Theta2_reg);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
