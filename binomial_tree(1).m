
function []=binomial_tree(S0)
sigma=0.5;
N=20;
r=0.1;
T= 0.5;
K=10;
delta_t=T/(N);
u=exp(sigma*sqrt(delta_t));
d=exp(-sigma*sqrt(delta_t));
p=(exp(r*delta_t)-d)/(u-d);

S=zeros(N+1,N+1);
delta_t=T/(N);
v=zeros(N+1,N+1);
for n=N+1:-1:1
for i=1:n 
    S(n,i)=S0*u^(i-1)*d^(n-i);
end
end
plot((0:N)*delta_t,S,'*','LineWidth', 5);
Smc(1)=S0;
for j=1:N
if rand<p
    Smc(j+1)=Smc(j)*u;
else
    Smc(j+1)=Smc(j)*d;
end
end
hold on
plot((0:N)*delta_t,Smc,'LineWidth', 5);
title 'Binomial Tree'
end