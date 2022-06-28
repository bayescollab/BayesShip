import numpy as np
import matplotlib.pyplot as plt
from bayesshippy.mcmcRoutines import MCMC_unpack_file

#dataSets = [np.loadtxt("data/gaussianLikelihoodTestData.csv"),np.loadtxt("data/gaussianLikelihoodTestDataHot.csv")]
#names = ["plots/gaussianLikelihoodTestPlot.pdf","plots/gaussianLikelihoodTestPlotHot.pdf"]
#names2 = ["plots/gaussianLikelihoodTestTrace.pdf","plots/gaussianLikelihoodTestTraceHot.pdf"]
#
#ct=0
#for data in dataSets:
#    if(ct ==0):
#        print("STD of data: ",np.std(data))
#        print("STD error percentage: ",abs(np.std(data)-1))
#        print("Mean of data: ",np.mean(data))
#        print("Mean error / STD of data: ",abs(np.mean(data))/np.std(data))
#        print("1/sqrt(N): ",1./np.sqrt(len(data)))
#
#    plt.hist(data,density=True, bins=100,histtype='stepfilled')
#    if(ct == 0 ):
#        x = np.linspace(np.amin(data),np.amax(data),500)
#        plt.plot(x, np.exp(-(x)**2/2 ) / np.sqrt(2*np.pi))
#    plt.axvline(np.std(data),color='red')
#    plt.axvline(-np.std(data),color='red')
#    plt.savefig(names[ct])
#    plt.close()
#
#    plt.plot(data)
#    plt.savefig(names2[ct])
#    plt.close()
#    ct+=1


data = MCMC_unpack_file("data/gaussianLikelihoodTest_output.hdf5")
plt.hist(data,bins=50,histtype='stepfilled',density=True,alpha=.8)

x = np.linspace(np.amin(data),np.amax(data),500)
plt.plot(x, np.exp(-(x)**2/2 ) / np.sqrt(2*np.pi))
plt.axvline(np.std(data),color='red')
plt.axvline(-np.std(data),color='red')
plt.savefig("plots/gaussian_HDF5_data.pdf")
plt.close()

data = MCMC_unpack_file("data/gaussianLikelihoodTestPrior_output.hdf5")
plt.hist(data,bins=50,histtype='stepfilled')
plt.savefig("plots/gaussian_HDF5_data_prior.pdf")
plt.close()

