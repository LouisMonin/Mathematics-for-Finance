#Problème 4 : Prix de l'option barrière via
#Monte-Carlo. Down and Out Asset or Nothing option

#Question 1

import numpy as np
import matplotlib.pyplot as plt

# Paramètres
Nmc = 1000   
L = 20       
T = 0.5      
r = 0.1     
sigma = 0.5  
K = 10       
B1 = 4       
N = 100      
dt = T / N  

# Fonction pour simuler le prix d'une option Down and Out Asset or Nothing
def monte_carlo_down_and_out(S0, K, B1, T, r, sigma, Nmc):
    dt = T / N
    payoff = []
    for _ in range(Nmc):
        S = S0
        knocked_out = False
        for _ in range(N):
            Z = np.random.normal()
            S *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
            if S <= B1:  
                knocked_out = True
                break
        if not knocked_out:
            payoff.append(S if S < K else 0)  
    return np.exp(-r * T) * np.mean(payoff)

# Gamme de valeurs pour S0
S0_range = np.linspace(0, L, 100)
V_down_and_out = [monte_carlo_down_and_out(S0, K, B1, T, r, sigma, Nmc) for S0 in S0_range]

# Code pour tracer le graphique
plt.figure(figsize=(8, 6))
plt.plot(S0_range, V_down_and_out, label='Down-and-Out Asset or Nothing (t=0)', color='blue')
plt.title('Prix de l\'option Down-and-Out Asset or Nothing à t=0')
plt.xlabel('Prix initial du sous-jacent $S_0$')
plt.ylabel('Valeur de l\'option $V(0, S_0)$')
plt.legend()
plt.grid()
plt.show()

#Question 2

from mpl_toolkits.mplot3d import Axes3D

# Fonction pour calculer la surface des prix
def monte_carlo_surface_down_and_out(S_values, t_values, K, B1, T, r, sigma, Nmc):
    dt = T / N
    V_surface = np.zeros((len(t_values), len(S_values)))

    for i, t in enumerate(t_values):
        remaining_time = T - t
        for j, S in enumerate(S_values):
            payoff = []
            for _ in range(Nmc):
                S_path = S
                knocked_out = False
                for _ in range(int(remaining_time / dt)):
                    Z = np.random.normal()
                    S_path *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
                    if S_path <= B1:  
                        knocked_out = True
                        break
                if not knocked_out:
                    payoff.append(S_path if S_path < K else 0)  
            V_surface[i, j] = np.exp(-r * remaining_time) * np.mean(payoff)
    return V_surface

# Discrétisation pour le graphique 3D
S_values = np.linspace(0, L, 50)
t_values = np.linspace(0, T, 20)
V_surface = monte_carlo_surface_down_and_out(S_values, t_values, K, B1, T, r, sigma, Nmc)

# Grille pour le tracé
T_grid, S_grid = np.meshgrid(t_values, S_values)

# Tracé de la surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T_grid, S_grid, V_surface.T, cmap='viridis')

ax.set_title('Surface des prix Down-and-Out Asset or Nothing')
ax.set_xlabel('Temps $t$')
ax.set_ylabel('Prix du sous-jacent $S$')
ax.set_zlabel('Valeur de l\'option $V(t, S)$')
plt.show()
