#Problème 3 : Prix de l'option "Asset or Nothing"
#dans le modèle de Ho-Lee

#Question 1

import numpy as np
import matplotlib.pyplot as plt

# Paramètres
r0 = 0.1
omega = 0.3
T = 0.5
sigma = 0.5
gamma = 0.2
K = 10  
Nmc = 1000
dt = 0.001
N = int(T / dt)


# Temps discrétisé
t = np.linspace(0, T, N+1)

# Calcul de a(t)
def a(t):
    return np.exp(gamma * (T - t)) / T

# Simulation de r_t
def simulate_r(Nmc, N, dt, r0, omega, t):
    r_trajectories = np.zeros((Nmc, N+1))
    r_trajectories[:, 0] = r0
    for i in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt), Nmc)
        r_trajectories[:, i] = r_trajectories[:, i-1] + a(t[i-1]) * dt + omega * dW
    return r_trajectories

# Générer les trajectoires
r_trajectories = simulate_r(Nmc, N, dt, r0, omega, t)

# Tracer les trajectoires de r_t
plt.figure(figsize=(10, 6))
plt.plot(t, r_trajectories[:100].T, alpha=0.5)  
plt.title('Trajectoires de $r_t$ (taux d\'intérêt)')
plt.xlabel('Temps $t$')
plt.ylabel('Taux d\'intérêt $r_t$')
plt.grid()
plt.show()

#Question 2

# Paramètres pour S_t
sigma = 0.5
S0 = 10 

# Simulation de S_t
def simulate_S(Nmc, N, dt, S0, r_trajectories, sigma):
    S_trajectories = np.zeros((Nmc, N+1))
    S_trajectories[:, 0] = S0
    for i in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt), Nmc)
        S_trajectories[:, i] = S_trajectories[:, i-1] * (
            1 + r_trajectories[:, i-1] * dt + sigma * dW
        )
    return S_trajectories

# Générer les trajectoires
S_trajectories = simulate_S(Nmc, N, dt, S0, r_trajectories, sigma)

# Tracer les trajectoires de S_t
plt.figure(figsize=(10, 6))
plt.plot(t, S_trajectories[:100].T, alpha=0.5)  
plt.title('Trajectoires de $S_t$ (Prix de l\'actif)')
plt.xlabel('Temps $t$')
plt.ylabel('Prix $S_t$')
plt.grid()
plt.show()

#Question 3

# Payoff pour Asset or Nothing
def payoff_asset_or_nothing(ST, K):
    return np.where(ST < K, ST, 0)

# Monte Carlo pour V(0, S0)
def monte_carlo_asset_or_nothing(S0, K, T, r_trajectories, S_trajectories, dt):
    discount_factors = np.exp(-np.sum(r_trajectories[:, :-1], axis=1) * dt)
    payoff = payoff_asset_or_nothing(S_trajectories[:, -1], K)
    return np.mean(discount_factors * payoff)

# Calcul du prix pour une gamme de S0
S0_range = np.linspace(0, 40, 50)  
V_mc = []

for S0 in S0_range:
    S_trajectories = simulate_S(Nmc, N, dt, S0, r_trajectories, sigma)
    V_mc.append(monte_carlo_asset_or_nothing(S0, K, T, r_trajectories, S_trajectories, dt))

# Tracer le prix de l'option
plt.figure(figsize=(10, 6))
plt.plot(S0_range, V_mc, label='Asset or Nothing (Modèle Ho-Lee)', color='blue')
plt.title('Prix de l\'option Asset or Nothing à t=0 (Modèle Ho-Lee)')
plt.xlabel('Prix initial $S_0$')
plt.ylabel('Valeur de l\'option $V(0, S_0)$')
plt.legend()
plt.grid()
plt.show()


