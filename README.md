# spikeFRInder: Spike inference algorithm using frequency-domain FRI framework

This repository contains the code for the methods for estimating spikes in a stream of decaying exponentials.

### Basic Model

The model assumes the input signal to come from a stream of dirac delta functions convolved with a decaying exponential. Thus, the signal in continuous-time is of the form

<img src="figures/model_form.png" alt="model form" width="400">

This signal is then corrupted with noise and sampled to create the input to the method. The method estimates the amplitudes and locations of the delta functions. In other words, it finds

.. math::
	a_{k}, t_{k} \text{for} k = {1, ..., K}


