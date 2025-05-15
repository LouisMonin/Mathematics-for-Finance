
function[prix_option] = Option_Eu_graphe()
prix_option=zeros(1,401);
K=100;   
    function[g]=payoff(S,K)
        g=max(S-K,0);
    end


for j = 1 : 401
S0(j)=0.5*(j-1);
prix_option(j)=Option_Eu_S0_fixe(S0(j));
end
figure
plot(S0, prix_option,'LineWidth',2)
hold on;

plot(S0,payoff(S0,K),'r','LineWidth',2)

xlabel 'prix de S0'
ylabel 'prix de loption'
title 'Prix de loption Européenne'
end
