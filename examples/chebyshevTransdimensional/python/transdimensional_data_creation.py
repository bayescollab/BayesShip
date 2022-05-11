import numpy as np
import matplotlib.pyplot as plt
import os

def signal_model( t, alphas,T):
    yti = np.sum([alphas[x] * (t/T)**x for x in np.arange(len(alphas))])
    return yti 

beta = 2
N = 100
sigma = 1
dt = 1
M = 3
run_id = "{}_{}_{}_{}_{}".format(beta,M,sigma,N,dt)
os.mkdir("data/{}/".format(run_id))

time = np.linspace(0,N*dt,N)
noise = np.random.normal(0, sigma, N)
alphas = np.random.normal(0, beta,M)
signal = np.asarray([signal_model(t, alphas,N*dt) for t in time])
plt.plot(time,signal)
plt.plot(time,noise+signal)
plt.show()
plt.close()
np.savetxt("data/{}/clean_data_transdimensional.csv".format(run_id),signal,delimiter=',')
np.savetxt("data/{}/full_data_transdimensional.csv".format(run_id),signal+noise,delimiter=',')


