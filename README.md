# Mixed-Bandlimited-Time-Encoding
This repository reproduces the figure in the paper "Encoding and Decoding Mixed Bandlimited Signals Using Spiking Integrate-and-Fire Neurons" and provides further examples for multi-signal multi-channel encoding and decoding.


[![DOI](https://zenodo.org/badge/216613096.svg)](https://zenodo.org/badge/latestdoi/216613096)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/karenadam/Mixed-Bandlimited-Time-Encoding/master?filepath=Code%2FMulti-Signal%20Multi-Channel%20Encoding.ipynb)

# Abstract
Conventional sampling focuses on encoding and decoding bandlimited signals by recording signal amplitudes at known time points. Alternately, sampling can be approached using biologically-inspired schemes. Among these are integrate-and-fire time encoding machines (IF-TEMs). They behave like simplified versions of spiking neurons and encode their input using spike times rather than amplitudes.
Moreover, when multiple of these neurons jointly process a set of mixed signals, they form one layer in a feedforward spiking neural network. In this paper, we investigate the encoding and decoding potential of such a layer.
We propose a setup to sample a set of bandlimited signals, by mixing them and sampling the result using different IF-TEMs. We provide conditions for perfect recovery of the set of signals from the samples in the noiseless case, and suggest an algorithm to perform the reconstruction.

https://arxiv.org/abs/1910.09413
