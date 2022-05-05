import numpy as np
from corner import corner 
import matplotlib.pyplot as plt

#data = np.loadtxt("data/multiModalLikelihoodTestData.csv",delimiter=',')
#corner(data)
#plt.savefig("plots/multiModalLikelihoodCorner.pdf")
#plt.close()
#
#fig, ax = plt.subplots(nrows=2, ncols = 1, figsize=[5,5])
#ax[0].plot(data[:,0])
#ax[1].plot(data[:,1])
#plt.tight_layout()
#plt.savefig("plots/multiModalLikelihoodTrace.pdf")
#plt.close()

#data = np.loadtxt("data/multiModalLikelihoodTestDataHot.csv",delimiter=',')
#corner(data)
#plt.savefig("plots/multiModalLikelihoodCornerHot.pdf")
#plt.close()

#fig, ax = plt.subplots(nrows=2, ncols = 1, figsize=[5,5])
#ax[0].plot(data[:,0])
#ax[1].plot(data[:,1])
#plt.tight_layout()
#plt.savefig("plots/multiModalLikelihoodTraceHot.pdf")
#plt.close()

import h5py
dataFile = h5py.File("data/multiModalLikelihoodTest_output.hdf5",'r')
first=True
data = None
for chain in dataFile["MCMC_OUTPUT"].keys():
    if "CHAIN" in chain:
        if first:
            data = np.array(dataFile["MCMC_OUTPUT"][chain])
            first = False
        else:
            data = np.insert(data,len(data), np.array(dataFile["MCMC_OUTPUT"][chain]),axis=0)
fig = corner(data)
plt.savefig("plots/multiModalLikelihoodHdf5.pdf")
plt.close()

dataFile = h5py.File("data/multiModalLikelihoodTestPrior_output.hdf5",'r')
first=True
data = None
for chain in dataFile["MCMC_OUTPUT"].keys():
    if "CHAIN" in chain:
        if first:
            data = np.array(dataFile["MCMC_OUTPUT"][chain])
            first = False
        else:
            data = np.insert(data,len(data), np.array(dataFile["MCMC_OUTPUT"][chain]),axis=0)
#plt.hist(data,bins=50,histtype='stepfilled')
fig = corner(data)
plt.savefig("plots/multiModalLikelihoodTest_prior.pdf")
plt.close()
