import matplotlib.pyplot as plt 
import numpy as np
import bilby
from multiprocessing import Pool

def cheb_fn(P,coeff,x ):
    return np.sum(coeff[:P] * np.cos(np.arange(P)*np.arccos(x)))

def cheb_fn_vec(P,coeff,x ):
    return np.matmul(coeff[:P], np.cos(np.outer(np.arange(P),np.arccos(x))))
    #return np.sum(coeff[:P] * np.cos(np.arange(P)*np.arccos(x)))

class chebLikelihood(bilby.Likelihood):
    def __init__(self, P,data):
        self.keys = ["x{}".format(i) for i in np.arange(P)]
        self.parameters = {k: None for k in self.keys}
        self.P = P
        self.data = data
        self._marginalized_parameters = []

    def log_likelihood(self):
        coeff = []
        for i in np.arange(self.P):
            coeff.append(self.parameters["x{}".format(i)])
        sigma = self.parameters["sigma"]
        dn = 2./(len(self.data)-1)
        recon_signal = np.ones(len(self.data),dtype=np.double)
        #for x in np.arange(len(self.data)):
        #    recon_signal[x] = cheb_fn(self.P, coeff,-1 + x*dn);
        xvec = np.linspace(-1,1,len(self.data))
        recon_signal = cheb_fn_vec(self.P, coeff, xvec)
        #print(recon_signal)
        #recon_signal= cheb_fn(self.P, coeff,-1 + np.arange(len(self.data))*dn)
        #ll = 0
        #for x in np.arange(len(self.data)):
        #    ll -= (self.data[x]-recon_signal[x])**2/(2. * sigma*sigma)
        #ll-= (len(self.data)/2.) * np.log(2.*np.pi * sigma * sigma)
        ll = -1*np.sum(np.power(self.data - recon_signal,2) )/(2.* sigma * sigma)- (len(self.data)/2.) * np.log(2.*np.pi * sigma * sigma)
        return ll

beta = 2
M = 3
sigma = 1
t = 100
dt = 1
run_id = "{}_{}_{}_{}_{}".format(beta,M,sigma,t,dt)
outdir = 'data/transdimensionalChebyshev/'+run_id+"/bilby/"
#data = np.loadtxt("data/full_data_transdimensional_5_5_1_100.csv",delimiter=',')
data = np.loadtxt("data/transdimensionalChebyshev/{}/full_data_transdimensional.csv".format(run_id),delimiter=',')

###############################
#likelihoodTest = chebLikelihood(3, data)
#likelihoodTest.parameters = {"sigma":.32,"x0":-3.2,"x1":2.7,"x2":3.3}
#print(likelihoodTest.log_likelihood())
#exit()
###############################


#for P in np.arange(1,10):
#for P in [3]:
for P in [3,4,5]:
    print(P)
    label = "{}".format(P)
    likelihood = chebLikelihood(P, data)
    priors = {}
    priors['sigma'] = bilby.core.prior.Uniform(.01,10,'sigma')
    for x in np.arange(P):
        priors["x{}".format(x)]  = bilby.core.prior.Uniform(-10,10,'x{}'.format(x))
    result = bilby.run_sampler(likelihood=likelihood, priors = priors, outdir = outdir, label = label,clean=True,nlive=1000, npool=8, nact=10,dlogz=.1 )
    
    print("Evidence: ",result.log_evidence)
    np.savetxt(outdir+"log_evidence_{}.txt".format(P),np.array([result.log_evidence]))
