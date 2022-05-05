import numpy as np
import matplotlib.pyplot as plt
from corner import corner


cov = np.array([ [ 1,.2,.4],
    [ .2,5,.6],
    [ .4,.6,20]])

mean = np.array([1,-50,20])

data1 = np.random.multivariate_normal(mean=mean,cov=cov,size=5000)
data2 = np.random.multivariate_normal(mean=-1*mean,cov=cov,size=5000)
data = np.insert(data1,len(data1),data2,axis=0)
np.savetxt("data/KDE_testing_true_data.csv",data,delimiter=',')
corner(data)
plt.savefig("plots/KDE_testing_truth.pdf")
plt.close()
