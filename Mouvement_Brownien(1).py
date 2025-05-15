from numpy.random import rand
import matplotlib.pyplot as plt
import random
from math import sqrt
import matplotlib.colors as plt_col

def fonction(k,delta_t):
    W=[0]
    for i in range (1,k):
        W.append(W[i-1]+sqrt(delta_t)*random.random())
    plt.plot(W)
    plt.show()
fonction(1000, 2/100)

def fonction2(k,n,Nmc,delta_t):
    W=[0]
    memory = 0
    for j in range (1,Nmc):
        for i in range (k, n-1):
            memory = W[i-1]+sqrt(delta_t)*random.random()
            W.append(memory)
        plt.plot(W)
        plt.show()
        
fonction2(100, 300, 2000, 0.1)

