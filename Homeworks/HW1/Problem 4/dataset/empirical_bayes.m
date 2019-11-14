clear;

load('Problem4.mat')
N = data.n; y = data.y;y0 = N-y;
input = cat(1, y, N-y).';

a = polya_fit_simple(input);
alpha = a(1); beta = a(2);

theta = zeros(length(N), 1);
theta_map = zeros(length(N), 1);
variance = zeros(length(N),1);

for i = 1:length(N)
    lambda = (alpha+beta)/(alpha+beta+N(i));
    m1 = alpha/(alpha+beta);
    theta(i) = lambda * m1 + (1-lambda)*y(i)/N(i);
    theta_map(i) = (y(i) + alpha -1)/(N(i) + alpha + beta - 2);
    variance(i) = (y(i)+alpha)*(y0(i)+beta)/(alpha+beta+N(i))^2/(alpha+beta+N(i)+1);
end

%%
figure;
subplot(5,1,1);
bar(1:length(theta), N);
title("Populations");
subplot(5,1,2);
bar(1:length(theta), y);
title("Number of cancers");
subplot(5,1,3);
bar(1:length(theta), y./N);
title("MLE estimate of \theta_i");
subplot(5,1,4);
bar(1:length(theta), theta_map);
title("MAP estimate of \theta_i");
subplot(5,1,5);
bar(1:length(theta), theta);
title("Posterior Mean of \theta_i");

