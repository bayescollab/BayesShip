import numpy as np
from corner import corner 
import matplotlib.pyplot as plt
import bayesshippy.mcmcRoutines as mcmc


run = mcmc.MCMCOutput("data/multiModalLikelihoodTest_output.hdf5")
dataObj = run.unpackMCMCData(betaID = 0,sizeCap=2e4)
print(len(dataObj["data"]))
fig = corner(dataObj["data"])
plt.savefig("plots/multiModalLikelihoodHdf5.pdf")
plt.close()
