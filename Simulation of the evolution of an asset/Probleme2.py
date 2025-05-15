#Problème 2 : Prix de l'option ASSET or NOTHING
#via Monte-Carlo

#Question 1

import numpy as np
import matplotlib.pyplot as plt

# Paramètres
K = 10          
T = 0.5         
r = 0.1        
sigma = 0.5     
M = 1000      
S0_range = np.linspace(0, 40, 100)  

# Fonction pour calculer le prix de l'option Asset or Nothing par Monte-Carlo
def monte_carlo_asset_or_nothing(S0, K, T, r, sigma, M):
    Z = np.random.standard_normal(M)  
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)  
    payoff = np.where(ST < K, ST, 0)  
    return np.exp(-r * T) * np.mean(payoff)  

# Calcul des prix pour différents S0
V_mc = [monte_carlo_asset_or_nothing(S0, K, T, r, sigma, M) for S0 in S0_range]

# Tracer le graphique
plt.figure(figsize=(10, 6))
plt.plot(S0_range, V_mc, label="Monte-Carlo Asset or Nothing (t=0)", color='blue', linewidth=2)

plt.title("Prix de l'option Asset or Nothing par Monte-Carlo à t=0")
plt.xlabel("Prix initial du sous-jacent $S_0$")
plt.ylabel("Valeur de l'option $V(0, S_0)$")
plt.legend()
plt.grid()
plt.show()


#Question 2

import numpy as np

# Paramètres
K = 10          
T = 0.5         
r = 0.1         
sigma = 0.5     
M = 1000
t_partial = T / 3  
S_t = 6            

# Fonction pour Monte-Carlo avec horizon réduit
def monte_carlo_partial_asset_or_nothing(S, K, t, T, r, sigma, M):
    Z = np.random.standard_normal(M)  
    remaining_time = T - t            
    ST = S * np.exp((r - 0.5 * sigma**2) * remaining_time + sigma * np.sqrt(remaining_time) * Z)  
    payoff = np.where(ST < K, ST, 0)  
    return np.exp(-r * remaining_time) * np.mean(payoff)  

# Calcul du prix de l'option à t=T/3 pour S=6
V_partial = monte_carlo_partial_asset_or_nothing(S_t, K, t_partial, T, r, sigma, M)
print(f"Le prix de l'option ASSET OR NOTHING à t=T/3 pour S=6 est : {V_partial:.4f}")

#Question 3

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Paramètres
K = 10        
T = 0.5
r = 0.4        
sigma = 0.5    
M = 1000      

# Fonction Monte-Carlo pour Asset or Nothing
def monte_carlo_partial_asset_or_nothing(S, K, t, T, r, sigma, M):
    Z = np.random.standard_normal(M)
    remaining_time = T - t
    ST = S * np.exp((r - 0.5 * sigma**2) * remaining_time + sigma * np.sqrt(remaining_time) * Z)
    payoff = np.where(ST < K, ST, 0)
    return np.exp(-r * remaining_time) * np.mean(payoff)

# Discrétisation du temps et des prix S
time_steps = np.linspace(0, T, 20)  # Discrétisation de l'intervalle [0, T]
S_values = np.linspace(0, 40, 50)  
V_surface = np.zeros((len(time_steps), len(S_values)))

# Calcul de la surface des prix
for i, t in enumerate(time_steps):
    for j, S in enumerate(S_values):
        V_surface[i, j] = monte_carlo_partial_asset_or_nothing(S, K, t, T, r, sigma, M)

# Tracer la surface
T_grid, S_grid = np.meshgrid(time_steps, S_values)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T_grid, S_grid, V_surface.T, cmap='viridis')

ax.set_title('Surface des prix V(t, S) pour Asset or Nothing par Monte-Carlo')
ax.set_xlabel('Temps t')
ax.set_ylabel('Prix du sous-jacent S')
ax.set_zlabel('Valeur de l\'option V')
plt.show()



