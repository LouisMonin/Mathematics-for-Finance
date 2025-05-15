function [prix]=Option_Eu_S0_fixe_Bin(S0)
N=3;
r=0.05;
%T=N;
K=100;
S=zeros(N+1,N+1);
u=1.1;      
d=0.9;      
q=(1+r-d)/(u-d);
v=zeros(N+1,N+1);
for n=1:N+1
for i=1:n
S(n,i)=u^(i-1)*d^(n-i)*S0;
v(N+1,i)=max(S(N+1,i)-K,0);
end
end

for n=N:-1:1
for i=1:n 
v(n,i)=(q*v(n+1,i+1)+(1-q)*v(n+1,i))/(1+r);
delta(n,i)=(v(n+1,i+1)-v(n+1,i))/(S(n+1,i+1)-S(n+1,i));
end
end

prix=v(1,1);

disp('Stock S');
disp(S')
disp('Option V');
disp(v');
disp('Delta');
disp(delta');
end
