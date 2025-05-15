function [prix]=Option_Eu_S0_fixe(S0)
sigma=0.5;
N=9;
r=0.05;
T=1;
K=100;
S=zeros(N+1,N+1);
delta_t=T/(N);
u=exp(sigma*sqrt(delta_t)); %Convergence vers le modele de Black et Scholes
d=exp(-sigma*sqrt(delta_t));
q=(exp(r*delta_t)-d)/(u-d);
v=zeros(N+1,N+1);
for n=1:N+1
for i=1:n
S(n,i)=u^(i-1)*d^(n-i)*S0;
v(N+1,i)=max(S(N+1,i)-K,0);
end
end

for n=N:-1:1
for i=1:n 
v(n,i)=exp(-r*delta_t)*(q*v(n+1,i+1)+(1-q)*v(n+1,i));
delta(n,i)=(v(n+1,i+1)-v(n+1,i))/(S(n+1,i+1)-S(n+1,i));
end
end

prix=v(1,1);

disp('Stock');
disp(S')
disp('V, Option');
disp(v');
disp('Delta');
disp(delta');
end
