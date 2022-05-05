import numpy as np 
import matplotlib.pyplot as plt
from corner import corner

dataTrue = np.loadtxt("data/KDE_testing_true_data.csv",delimiter=',')
dataResampled = np.loadtxt("data/KDE_testing_resampled.csv",delimiter=',')


from scipy.stats import gaussian_kde

kde = gaussian_kde(dataTrue.T)
datascipy = kde.resample(len(dataResampled)).T


fig = corner(dataTrue, weights = np.ones(len(dataTrue))/len(dataTrue))
fig = corner(dataResampled, fig = fig, weights = np.ones(len(dataResampled))/len(dataResampled),color='blue')
fig = corner(datascipy, fig = fig, weights = np.ones(len(datascipy))/len(datascipy),color='green')
plt.savefig("plots/KDE_comparison.pdf")
plt.close()

