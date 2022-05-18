%module(directors="1") bayesshipSamplerpy
%{
#include "bayesship/bayesshipSampler.h"
#include "bayesship/dataUtilities.h"
#include "bayesship/utilities.h"
%}

%include "carrays.i"
%array_class(double, doubleArray);

/* turn on director wrapping Callback */
%feature("director") probabilityFn;

%include "bayesship/dataUtilities.h"
%include "bayesship/utilities.h"
%include "bayesship/bayesshipSampler.h"

