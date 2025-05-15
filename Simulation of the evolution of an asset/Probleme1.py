# Problème 1 : Résolution par la méthode aux 
# Différences Finies de l'équation de Black et Scholes
# pour l'option ASSET or NOTHING

# La fonction Pay-Off de l'option ASSET or NOTHING 

import numpy as np
import matplotlib.pyplot as plt

def pay_off_asset(S, K):
    payoff = np.piecewise(S, 
                          [S < K, S >= K],
                          [lambda S: S, 0])
    return payoff

# Paramètres
K = 10  # Strike price, je fixe 10 comme on a l'habitude de le faire en classe
S = np.linspace(0, 40, 500)  # Prix du sous-jacent ST entre 0 et 40, pareil ce sont des valeurs que je choisis

# Calcul du Pay-off
payoff = pay_off_asset(S, K)

# Tracé du graphique
plt.figure(figsize=(8, 6))
plt.plot(S, payoff, color='blue', linewidth=2)
plt.title('Pay-off Asset')
plt.xlabel('Prix ST')
plt.ylabel('Pay-off')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(K, color='gray', linestyle='--', linewidth=0.8, label=f'K={K}')
plt.legend()
plt.grid()
plt.show()

# Partie I Condition aux limites de Dirichlet

# Question 1) 

import numpy as np
import matplotlib.pyplot as plt

# Paramètres
L = 30      
T = 0.5     
r = 0.1     
sigma = 0.5 
K = 10      

N = 99      # Discrétisation de l'intervalle, [O,L]
M = 4999    # Discrétisation de l'intervalle, [0,T]

dS = L / N
dt = T / M

S = np.linspace(0, L, N+1)
t = np.linspace(0, T, M+1)

# Fonction de Pay-off Asset or Nothing
def pay_off_asset(S, K):
    return np.where(S < K, S, 0)

# Résolution de l'équation de Black-Scholes avec Euler explicite
def solve_black_scholes():
    V = np.zeros((M+1, N+1))
    V[-1, :] = pay_off_asset(S, K)  # Condition finale
    V[:, 0] = 0                     # Condition aux limites pour S=0
    V[:, -1] = 0                    # Condition aux limites pour S=L

    # Calcul par méthode d'Euler explicite
    for n in range(M-1, -1, -1):
        for i in range(1, N):
            dV_dS = (V[n+1, i+1] - V[n+1, i-1]) / (2 * dS)
            d2V_dS2 = (V[n+1, i+1] - 2 * V[n+1, i] + V[n+1, i-1]) / (dS**2)
            V[n, i] = V[n+1, i] + dt * (
                r * S[i] * dV_dS
                + 0.5 * sigma**2 * S[i]**2 * d2V_dS2
                - r * V[n+1, i]
            )
    return V

#Résolution
V = solve_black_scholes()

#Code pour affcher le graphe 
plt.figure(figsize=(8, 6))
plt.plot(S, V[0, :], label='V(t=0, S)', color='blue', linewidth=2)
plt.plot(S, V[-1, :], label='V(t=T, S)', color='red', linestyle='--', linewidth=2)
plt.title('Prix de l\'option Asset or Nothing: V(t, S) à t=0 et t=T')
plt.xlabel('Prix du sous-jacent S')
plt.ylabel('Valeur de l\'option V')
plt.legend()
plt.grid()
plt.show()

#Question 2 : Tracer la surface des prix V(t,S) en 3 dimensions

#On importe les bibliothèques nécessaires
from mpl_toolkits.mplot3d import Axes3D

T_grid, S_grid = np.meshgrid(t, S, indexing='ij')  

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(S_grid, T_grid, V, cmap='viridis', edgecolor='none')
ax.set_title('Surface des prix V(t, S)')
ax.set_xlabel('Prix du sous-jacent S')
ax.set_ylabel('Temps t')
ax.set_zlabel('Valeur de l\'option V')
plt.show()


# Question 3 : Calculer de V(t = T/3, S = 6)

t_index = int(M * (1/3))
s_index = np.argmin(np.abs(S - 6))

V_T3_S6 = V[t_index, s_index]
print(f"Valeur de l'option à t = T/3 et S = 6 : {V_T3_S6:.4f}")

#On obtient "Valeur de l'option à t = T/3 et S = 6 : 5.5919"

#Partie II : Volatilité est locale

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paramètres
L = 30      
T = 0.5     
r = 0.1     
sigma = 0.5 
K = 10     

N = 99      # Discrétisation de l'intervalle, [O,L]
M = 4999    # Discrétisation de l'intervalle, [0,T]

dS = L / N
dt = T / M

S = np.linspace(0, L, N+1)
t = np.linspace(0, T, M+1)

# Fonction de volatilité locale
def sigma_locale(t):
    if t < T/2:
        return sigma * (1 + np.sin(np.pi * (T - t) / (2 * T)))
    else:
        return sigma

# Fonction de Pay-off Asset or Nothing
def pay_off_asset(S, K):
    return np.where(S < K, S, 0)

# Résolution de l'équation de Black-Scholes avec volatilité locale
def solve_black_scholes_local_volatility():
    V = np.zeros((M+1, N+1))
    V[-1, :] = pay_off_asset(S, K)  
    V[:, 0] = 0                     
    V[:, -1] = 0

    # Calcul par méthode d'Euler explicite
    for n in range(M-1, -1, -1):
        for i in range(1, N):
            sigma_t = sigma_locale(t[n])  
            dV_dS = (V[n+1, i+1] - V[n+1, i-1]) / (2 * dS)
            d2V_dS2 = (V[n+1, i+1] - 2 * V[n+1, i] + V[n+1, i-1]) / (dS**2)
            V[n, i] = V[n+1, i] + dt * (
                r * S[i] * dV_dS
                + 0.5 * sigma_t**2 * S[i]**2 * d2V_dS2
                - r * V[n+1, i]
            )
    return V

# Résolution avec volatilité locale
V_local = solve_black_scholes_local_volatility()

# Comparaison avec la volatilité constante (Partie I)
V_constant = solve_black_scholes()

# Question 6 : Comparaison des surfaces 2D
plt.figure(figsize=(10, 6))
plt.plot(S, V_constant[0, :], label='Volatilité constante (t=0)', color='blue', linewidth=2)
plt.plot(S, V_local[0, :], label='Volatilité locale (t=0)', color='green', linestyle='--', linewidth=2)
plt.plot(S, V_constant[-1, :], label='Volatilité constante (t=T)', color='red', linewidth=2)
plt.plot(S, V_local[-1, :], label='Volatilité locale (t=T)', color='orange', linestyle='--', linewidth=2)
plt.title('Comparaison des prix de l\'option Asset or Nothing (Volatilité constante vs locale)')
plt.xlabel('Prix du sous-jacent S')
plt.ylabel('Valeur de l\'option V')
plt.legend()
plt.grid()
plt.show()

