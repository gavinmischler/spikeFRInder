"""
########################################

These functions originally comes from https://github.com/lucastheis/c2s
We changed the robust linear regression method to use sklearn since we were unable to
download the cmt package from the authors.

########################################


Tools for the prediction of spike trains from calcium traces.
This module contains functions for predicting spikes from fluorescence traces obtained
from two-photon calcium images. Data should be stored as a list of dictionaries, where
each dictionary corresponds to a cell or recording. Each dictionary has to contain at least
the entries C{calcium} and C{fps}, which correspond to the recorded fluorescence trace and
its sampling rate in frames per second.
	>>> data = [
	>>>	{'calcium': [[0., 0., 0., 0.]],     'fps': 10.4},
	>>>	{'calcium': [[0., 0., 0., 0., 0.]], 'fps': 12.1}]
The data here is only used to illustrate the format. Each calcium trace is expected to
be given as a 1xT array, where T is the number of recorded frames. After importing the
module,
	>>> import c2s
we can use L{preprocess<c2s.preprocess>} to normalize the calcium traces and
C{predict<c2s.predict>} to predict firing rates:
	>>> data = c2s.preprocess(data)
	>>> data = c2s.predict(data)
The predictions for the i-th cell can be accessed via:
	>>> data[i]['predictions']
Simultaneously recorded spikes can be stored either as binned traces
	>>> data = [
	>>>	{'calcium': [[0., 0., 0., 0.]],     'spikes': [[0, 1, 0, 2]],    'fps': 10.4},
	>>>	{'calcium': [[0., 0., 0., 0., 0.]], 'spikes': [[0, 0, 3, 1, 0]], 'fps': 12.1}]
or, preferably, as spike times in milliseconds:
	>>> data = [
	>>>	{'calcium': [[0., 0., 0., 0.]],     'spike_times': [[15.1, 35.2, 38.1]],      'fps': 10.4},
	>>>	{'calcium': [[0., 0., 0., 0., 0.]], 'spike_times': [[24.2, 28.4 32.7, 40.2]], 'fps': 12.1}]
The preprocessing function will automatically compute the other format of the spike trains if one
of them is given. Using the method L{train<c2s.train>}, we can train a model to predict spikes from
fluorescence traces
	>>> data = c2s.preprocess(data)
	>>> results = c2s.train(data)
and then use it to make predictions:
	>>> data = c2s.predict(data, results)
It is important that the data used for training undergoes the same preprocessing as the data
used when making predictions.
@undocumented: optimize_predictions
@undocumented: robust_linear_regression
@undocumented: percentile_filter
@undocumented: downsample
@undocumented: responses
@undocumented: generate_inputs_and_outputs
@undocumented: DEFAULT_MODEL
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Lucas Theis <lucas@theis.io>'
__docformat__ = 'epytext'
__version__ = '0.1.0dev'

import sys
from copy import copy, deepcopy
from base64 import b64decode
from warnings import warn
from pickle import load, loads
from numpy.random import seed
from numpy import percentile, asarray, arange, zeros, where, repeat, sort, cov, mean, std, ceil
from numpy import vstack, hstack, argmin, ones, convolve, log, linspace, min, max, square, sum, diff
from numpy import corrcoef, array, eye, dot, empty, seterr, isnan, any, zeros_like
from numpy.random import rand
from scipy.signal import resample
from scipy.stats import poisson
from scipy.stats.mstats import gmean
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.io import loadmat
try:
	from cmt.models import MCGSM, STM, Poisson
	from cmt.nonlinear import ExponentialFunction, BlobNonlinearity
	from cmt.tools import generate_data_from_image, extract_windows
	from cmt.transforms import PCATransform
	from cmt.utils import random_select
except ImportError:
	from sklearn.linear_model import HuberRegressor
	warn('Install conditional modeling toolkit (https://github.com/lucastheis/cmt) for full functionality.')
#from .experiment import Experiment

try:
	from roc import roc
except:
	pass

PYTHON3 = sys.version.startswith('3.')


def load_data(filepath):
	"""
	Loads data in either pickle or MATLAB format.
	@type  filepath: string
	@param filepath: path to dataset
	@rtype: list
	@return: list of dictionaries containing the data
	"""

	if filepath.lower().endswith('.mat'):
		data = []
		data_mat = loadmat(filepath)

		if 'data' in data_mat:
			data_mat = data_mat['data'].ravel()

			for entry_mat in data_mat:
				entry = {}

				for key in entry_mat.dtype.names:
					entry[key] = entry_mat[key][0, 0]

				for key in ['calcium', 'spikes', 'spike_times']:
					if key in entry:
						entry[key] = entry[key].reshape(1, entry[key].size)
				if 'fps' in entry:
					entry['fps'] = float(entry['fps'])
				if 'cell_num' in entry:
					entry['cell_num'] = int(entry['cell_num'])

				data.append(entry)

		elif 'predictions' in data_mat:
			for predictions in data_mat['predictions'].ravel():
				data.append({'predictions': predictions.reshape(1, predictions.size)})

		return data

	if filepath.lower().endswith('.xpck'):
		experiment = Experiment(filepath)
		if 'data' in experiment.results:
			return experiment['data']
		if 'predictions' in experiment.results:
			data = []
			for predictions in experiment['predictions']:
				data.append({'predictions': predictions.reshape(1, predictions.size)})
			return data
		return []

	try:
		with open(filepath) as handle:
			return load(handle)
	except UnicodeDecodeError:
		# Open files saved with Python 2 in Python 3
		with open(filepath, 'rb') as handle:
			return load(handle, encoding='latin1')


def preprocess(data, fps=100., old_fps=60, filter=None, verbosity=0, fps_threshold=.1):
	"""
	Normalize calcium traces and spike trains.
	This function does three things:
		1. Remove any linear trends using robust linear regression.
		2. Normalize the range of the calcium trace by the 5th and 80th percentile.
		3. Change the sampling rate of the calcium trace and spike train.
	If C{filter} is set, the first step is replaced by estimating and removing a baseline using
	a percentile filter (40 seconds seems like a good value for the percentile filter).
	@type  data: list
	@param data: list of dictionaries containing calcium/fluorescence traces
	@type  fps: float
	@param fps: desired sampling rate of signals
	@type  filter: float/none
	@param filter: percentile filter length in seconds
	@type  filter: float/None
	@param filter: number of seconds used in percentile filter
	@type  verbosity: int
	@param verbosity: if positive, print messages indicating progress
	@type  fps_threshold: float
	@param fps_threshold: only resample if sampling rate differs more than this
	@rtype: list
	@return: list of preprocessed recordings
	"""

	seed(42)

	data = deepcopy(data)

	for k in range(len(data)):
		if verbosity > 0:
			print('Preprocessing calcium trace {0}...'.format(k))

		data[k]['fps'] = float(data[k]['fps'])

		if filter is None:
			# remove any linear trends
			# x = arange(data[k]['calcium'].size)
			# a, b = robust_linear_regression(x, data[k]['calcium'])

			# data[k]['calcium'] = data[k]['calcium'] - (a * x + b)

			# using LinearRegression from sklearn
			X_temp = arange(0, len(data[k]['calcium'])).reshape(-1,1)
			model = HuberRegressor()
			model.fit(X_temp, data[k]['calcium'])
			# calculate trend
			trend = model.predict(X_temp)
			# detrend
			data[k]['calcium'] = data[k]['calcium'] - trend

		else:
			data[k]['calcium'] = data[k]['calcium'] - \
				percentile_filter(data[k]['calcium'], window_length=int(data[k]['fps'] * filter), perc=5)

		# normalize dispersion
		calcium05 = percentile(data[k]['calcium'], 5)
		calcium80 = percentile(data[k]['calcium'], 80)

		if calcium80 - calcium05 > 0.:
			data[k]['calcium'] = ((data[k]['calcium'] - calcium05) / float(calcium80 - calcium05)).reshape((len(data[k]['calcium']),))

		# compute spike times if binned spikes are given
		if 'spikes' in data[k] and 'spike_times' not in data[k]:
			spikes = asarray(data[k]['spikes'].ravel(), dtype='uint16')

			# compute spike times in milliseconds
			spike_times = where(spikes > 0)[0]
			spike_times = repeat(spike_times, spikes[spike_times])
			spike_times = (spike_times + rand(*spike_times.shape)) * (1000. / data[k]['fps'])

			data[k]['spike_times'] = sort(spike_times).reshape(1, -1)

		# normalize sampling rate
		if fps is not None and fps > 0. and abs(data[k]['fps'] - fps) > fps_threshold:
			# number of samples after update of sampling rate
			num_samples = int(float(data[k]['calcium'].size) * fps / data[k]['fps'] + .5)

			if num_samples != data[k]['calcium'].size:
				# factor by which number of samples will actually be changed
				factor = num_samples / float(data[k]['calcium'].size)

				# resample calcium signal
				data[k]['calcium'] = resample(data[k]['calcium'].ravel(), num_samples).reshape(1, -1)
				data[k]['fps'] = data[k]['fps'] * factor
		else:
			# don't change sampling rate
			num_samples = data[k]['calcium'].size

		# compute binned spike trains if missing
		if 'spike_times' in data[k] and ('spikes' not in data[k] or num_samples != data[k]['spikes'].size):
			# spike times in bins
			spike_times = asarray(data[k]['spike_times'] * (data[k]['fps'] / 1000.), dtype=int).ravel()
			spike_times = spike_times[spike_times < num_samples]
			spike_times = spike_times[spike_times >= 0]

			# create binned spike train
			data[k]['spikes'] = zeros([1, num_samples], dtype='uint16')
			for t in spike_times:
				data[k]['spikes'][0, t] += 1
	
		# make sure spike trains are row vectors
		if 'spikes' in data[k]:
			data[k]['spike_times'] = data[k]['spike_times'].reshape(-1,) #data[k]['spike_times'].reshape(1, -1)
			data[k]['spikes'] = data[k]['spikes'].reshape(-1,) #data[k]['spikes'].reshape(1, -1)

		# added by Gavin
		data[k]['calcium'] = data[k]['calcium'].reshape(-1,)
		data[k]['spike_count'] = int(sum(data[k]['spikes']))

	return data


def robust_linear_regression(x, y, num_scales=3, max_iter=1000):
	"""
	Performs linear regression with Gaussian scale mixture residuals.
	$$y = ax + b + \\varepsilon,$$
	where $\\varepsilon$ is assumed to be Gaussian scale mixture distributed.
	@type  x: array_like
	@param x: list of one-dimensional inputs
	@type  y: array_like
	@param y: list of one-dimensional outputs
	@type  num_scales: int
	@param num_scales: number of Gaussian scale mixture components
	@type  max_iter: int
	@param max_iter: number of optimization steps in parameter search
	@rtype: tuple
	@return: slope and y-intercept
	"""

	x = asarray(x).reshape(1, -1)
	y = asarray(y).reshape(1, -1)

	# preprocess inputs
	m = mean(x)
	s = std(x)

	x = (x - m) / s

	# preprocess outputs using simple linear regression
	C = cov(x, y)
	a = C[0, 1] / C[0, 0]
	b = mean(y) - a * mean(x)

	y = y - (a * x + b)

	# robust linear regression
	model = MCGSM(
		dim_in=1,
		dim_out=1,
		num_components=1,
		num_scales=num_scales,
		num_features=0)

	model.initialize(x, y)
	model.train(x, y, parameters={
		'train_means': True,
		'max_iter': max_iter})

	a = (a + float(model.predictors[0])) / s
	b = (b + float(model.means)) - a * m

	return a, b


def percentile_filter(x, window_length, perc=5):
	"""
	For each point in a signal, computes a percentile from a window surrounding it.
	@type  window_length: int
	@param window_length: length of window in bins
	@type  perc: int
	@param perc: which percentile to compute
	@rtype: ndarray
	@return: array of the same size as C{x} containing the percentiles
	"""

	shape = x.shape
	x = x.ravel()
	y = empty(x.size)
	d = window_length // 2 + 1

	for t in range(x.size):
		fr = max([t - d + 1, 0])
		to = t + d
		y[t] = percentile(x[fr:to], perc)

	return y.reshape(shape)


def downsample(signal, factor):
	"""
	Downsample signal by averaging neighboring values.
	@type  signal: array_like
	@param signal: one-dimensional signal to be downsampled
	@type  factor: int
	@param factor: this many neighboring values are averaged
	@rtype: ndarray
	@return: downsampled signal
	"""

	if factor < 2:
		return asarray(signal)
	return convolve(asarray(signal).ravel(), ones(factor), 'valid')[::factor]
