import numpy as np
import matplotlib.pyplot as plt
from corner import corner


cov = np.array([ [ 2,1.5,0],
    [ 1.5,2,0],
    [ 0,0,2]])

mean = np.array([1,-50,20])

data1 = np.random.multivariate_normal(mean=mean,cov=cov,size=10000)
#data1 = np.random.multivariate_normal(mean=mean,cov=cov,size=5000)
#data2 = np.random.multivariate_normal(mean=-1*mean,cov=cov,size=5000)
#data = np.insert(data1,len(data1),data2,axis=0)
data = data1
np.savetxt("data/DiffEv_testing_true_data.csv",data,delimiter=',')
corner(data)
plt.savefig("plots/DiffEv_testing_truth.pdf")
plt.close()
