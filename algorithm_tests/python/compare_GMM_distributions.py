import numpy as np 
import matplotlib.pyplot as plt
from corner import corner, overplot_lines

dataTrue = np.loadtxt("data/GMM_testing_true_data.csv",delimiter=',')
dataResampled = np.loadtxt("data/GMM_testing_resampled.csv",delimiter=',')


from scipy.stats import gaussian_kde

fig = corner(dataTrue, weights = np.ones(len(dataTrue))/len(dataTrue))
fig = corner(dataResampled, fig = fig, weights = np.ones(len(dataResampled))/len(dataResampled),color='blue')
overplot_lines(fig, dataTrue[-2], color="C1")
plt.savefig("plots/GMM_comparison.pdf")
plt.close()


#plt.scatter(dataResampled[:,1],dataResampled[:,2])
#plt.scatter(dataResampled[:,0],dataResampled[:,2])
#plt.savefig("plots/DiffEv_comparison_scatter.pdf")
#plt.close()
