import random as np
import matplotlib.pyplot as plt

def fonction(x):
    init =[]
    memory = 0
    n=100
    for i in range(n):
        if np.random() <1/2:
            x=1
        else:
            x=-1
        memory =memory + x
        init.append(memory)
    plt.plot(init)
