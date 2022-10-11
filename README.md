# The Bayesian Spaceship, sailing through parameter and model spaces 

This software performs statistical inference through standard Markov chain Monte Carlo (MCMC) and Reversible Jump MCMC (RJMCMC) algorithms. 

## Methodology 

The workhorse algorithm for this software is Parallel Tempering (PT), which works by running multiple, independent Markov chains in parallel, then allowing two chains to swap positions in parameter and/or model space through "swaps". This method has been shown to be highly effective for multi-modal distributions and for RJMCMC applications. The computational tax of running multiple chains is typically drastically made up for in the efficiency of drawing independent samples.

## Technical Details of the implementation

The software is written in c++, with a swig wrapped version also available in Python (bayesshippy). The compilation/installation is completed using CMake. The "clang" is unsupported at the moment, so the compiler must be gcc (g++).

