function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Calculate outputs of second layer
a2 = sigmoid(X*Theta1');

% Add a column of ones to the a2 outputs
a2 = [ones(m, 1) a2];

% Calculate outputs of third layer
a3 = sigmoid(a2*Theta2');

% Assign h_theta to be the output from the final layer
h_theta = a3;

% Find maximums of each row
[M, iM] = max(h_theta, [], 2);

p = iM;

% =========================================================================


end
