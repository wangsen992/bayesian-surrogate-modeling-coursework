clear;
load('Problem3b1.mat');

% Obtain the non-zero values and their row and column indices
[i_t, j_t, s_t] = find(Xtrain);

% Calculate the class prior MLE
num_class = max(ytrain);
Nc = zeros(1,num_class);
for i = 1:length(ytrain)
    Nc(ytrain(i)) = Nc(ytrain(i)) + 1;
end

% Calculate the number of counts for N_jc
Njc = zeros(size(Xtrain,2), num_class);
for i = 1:length(j_t)
    Njc(j_t(i), ytrain(i_t(i))) = Njc(j_t(i), ytrain(i_t(i))) + 1;
end

% calculate parameters
pi_c = Nc/length(ytrain);
theta_jc = Njc./Nc;

% Perform test
p_yc = zeros(length(ytest), num_class);
for i = 1:length(ytest)
    for c = 1:num_class
        p_x = 1;
        for j = 1:size(Xtest,2)
            p_x = p_x * theta_jc(j,c)^Xtest(i,j) * (1-theta_jc(j,c))^(1-Xtest(i,j)); 
        end
        p_yc(i,c) = pi_c(c) * p_x;
    end
end
[m ,y_output] = max(p_yc,[],2);

% Calculate the misclassification rate
Nerr = 0;
for i = 1:length(y_output)
    if y_output(i) ~= ytest(i)
        Nerr = Nerr + 1;
    end
end
err = Nerr/length(y_output)
       