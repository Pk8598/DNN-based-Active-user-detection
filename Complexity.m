clear all;
lambd = 18;
L     = [47,239];
T     = [40,8];
N     = L-1;
N_fft = 512;

%% Average all user success probability
close all;
N_p    = lambd:3000;
for n = 1:length(N_p)
    all_su(n) = exp(-lambd)*(1+lambd/N_p(n))^N_p(n);
end
figure;
plot(N_p,all_su);
title('Average all user success probability with lambda = 18');
xlabel('Number of preambles');
ylabel('Average all user success probability');

%% Poisson PMF
x   = 0:50;
PMF = poisspdf(x,lambd);
figure;
scatter(x,PMF,'Marker',"o");
title('Poisson with mean 18');
xlabel('Number of active users');
ylabel('Probability');
grid on;

%% CDF of number of users in any TO
figure;
k = 0:lambd;
for t = 1:length(T)
    cdf = binocdf(k,lambd,1/T(t));
    stairs(k,cdf,'DisplayName',sprintf('L = %d, T = %d',L(t),T(t)));
    xlabel('Number of users in a TO');
    ylabel('Cumulative probability');
    hold on;
    K(t) = find(cdf>=0.99,1,'first')-1;
end
hold off;
legend show;

%% Complexity 
for i = 1:length(L)
    Conv = N(i)*(6*L(i) + (34/9)*N_fft*log2(N_fft));
    DNN  = 32*L(i)^2 + 4*L(i)*N(i) + K(i)*(6*L(i) + (34/9)*N_fft*log2(N_fft));
    factor(i) = Conv/DNN;
end
