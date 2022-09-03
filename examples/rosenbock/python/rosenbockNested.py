import numpy as np
import matplotlib.pyplot
import bilby


def rosenBockLikelihood(c, a, mu, n1, n2, n , b):
    LL= 0
    LL -= a * (c[0]-mu)**2
    
    for i in np.arange(n2):
        for j in np.arange(n1)[1:]:
            if(i == 1):
                LL -=b[j * (n1-1) + i] * (c[j*(n1-1)+i] - c[0]*c[0])**2
            else:
                LL-= b[j*(n1-1) +i] * (c[(n1-1)+i] - c[j*(n1-1) + i -1] * c[j*(n1-1) + i -1])**2
    return LL


