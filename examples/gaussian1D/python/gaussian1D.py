import numpy as np
from bayesshippy import bayesshipSamplerpy as bss


class likelihoodSlow(bss.probabilityFn):
    def __init__(self):
        bss.probabilityFn.__init__(self)
    def eval(self, posInfo, b):
        pos = bss.doubleArray(posInfo.dimension)
        pos = pos.frompointer(posInfo.parameters)
        return -.5 * pos[0]*pos[0]
class priorSlow(bss.probabilityFn):
    def __init__(self):
        bss.probabilityFn.__init__(self)
    def eval(self, posInfo, b):
        pos = bss.doubleArray(posInfo.dimension)
        pos = pos.frompointer(posInfo.parameters)
        if(pos[0] < -10 or pos[0] > 10):
            return bss.limitInf
        else:
            return 1.

class likelihood(bss.probabilityFn):
    def __init__(self):
        bss.probabilityFn.__init__(self)
    def eval(self, posInfo, b):
        param = posInfo.getParameter(0)
        return -.5 * param*param
class prior(bss.probabilityFn):
    def __init__(self):
        bss.probabilityFn.__init__(self)
    def eval(self, posInfo, b):
        param = posInfo.getParameter(0)
        if(param < -10 or param > 10):
            return bss.limitInf
        else:
            return 1.

ll = likelihood()
lp = prior()
s = bss.bayesshipSampler(ll,lp)
s.initialPosition = bss.positionInfo(1,False)
s.initialPosition.setElement(10.,0)
print("InitialPosition ll",ll.eval(s.initialPosition,1))
print("InitialPosition lp",lp.eval(s.initialPosition,1))
s.maxDim = 1
s.ignoreExistingCheckpoint = True
s.independentSamples=3000
s.burnIterations=5000

s.priorIterations = 0
s.burnPriorIterations = 0
s.writePriorData = False

s.threadPool = True
s.threads = 4

s.outputDir = "data/python/"
s.outputFileMoniker="gaussianLikelihoodTestPY"

s.ensembleN = 4
s.ensembleSize = 5

s.sample()
