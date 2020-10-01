"""
Functions to reproduce figure 3 from the paper. The data can be downloaded
from http://crcns.org/data-sets/methods/cai-1. We took the data
and initially converted it to .txt files which are read in by this script.


This script can be run either from scratch, or from a stored file.

>>> python3 crossval.py train <new_pickle_filename_to_create.p>

>>> python3 crossval.py test <existing_pickle_filename_to_use.p>

Author: Gavin Mischler
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from scipy.io import savemat, loadmat
import pickle

from c2s_preprocess_modified import preprocess, downsample

from spikeFRInder import sliding_window_predict


TRAIN_TEST_STATE = sys.argv[1]
PICKLE_FILENAME = sys.argv[2]

path_to_data = 'data/'

# The following 19 recordings were used for analysis. They were preprocessed in nearly the same way as
# done by the preprocess function in https://github.com/lucastheis/c2s/blob/master/c2s/c2s.py

# This preprocessing results in a list called `data_preprocessed` of length 19, where each item
# in the list is a dict containing at least the following fields
# - 'calcium': preprocessed calcium signal
# - 'fps': always 100
# - 'spikes': spike rate, same shape as 'calcium'
# - 'spike_count': number of spikes in the full signal


recordings = ['20120502_cell1_001','20120502_cell1_002','20120502_cell1_004',
              '20120521_cell2_002','20120521_cell2_003',
              '20120521_cell4_003','20120521_cell4_004','20120521_cell4_005','20120521_cell4_007',
              '20120521_cell5_003','20120521_cell5_005','20120521_cell5_006','20120521_cell5_007',
              '20120521_cell7_001','20120521_cell7_002','20120521_cell7_003',
              '20120521_cell10_002','20120521_cell10_003','20120521_cell10_004']


data = []
data_processed = []


for filename in recordings:

    # filename = 'data_' + filename
    fname = path_to_data + 'signal_data_' + filename + '.txt'
    signal_array = np.loadtxt(fname, delimiter=',')
    time = signal_array[:,0]
    signal = signal_array[:,1]

    fname = path_to_data + 'spikes_data_' + filename + '.txt'
    spikes_array = np.loadtxt(fname, delimiter=',')
    Fs = spikes_array[0,0]
    spikes = spikes_array[1:,0]
    spike_indices = spikes_array[1:,1].astype(int)
    # spike_indices = (Fs * spikes).astype(int)
    spikes_binary_full = np.zeros_like(time)
    spikes_binary_full[spike_indices] = 1

    # Preprocess the signal
    data_dict = {'calcium': signal,
                 'fps': Fs,
                 'spikes': spikes_binary_full}

    data.append(data_dict)

data_preprocessed = preprocess(data, old_fps=Fs, filter=None)

#=========================================================
# Compute average spikes in all other strips
#=========================================================

total_spikes = 0
spike_counts = []
for i, strip_data in enumerate(data_preprocessed):
    total_spikes += strip_data['spike_count']
    spike_counts.append(strip_data['spike_count'])

average_spikes_per_strip = []
for i in range(len(spike_counts)):
    trace_removed = np.delete(spike_counts, i)
    average_spikes_per_strip.append(int(round(np.mean(trace_removed))))


#=========================================================
# Helper functions
#=========================================================

def percentile_threshold_likelihood(likelihood, thresh_percentile, set_above_to_1=False):
    thresh = np.percentile(likelihood, thresh_percentile)
    likelihood_new = likelihood.copy()
    likelihood_new[likelihood_new<=thresh] = 0
    if set_above_to_1:
        likelihood_new[likelihood_new>thresh] = 1
    return likelihood_new

def downsample(signal, factor):
    """
    Taken from https://github.com/lucastheis/c2s
    
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
    return np.convolve(np.asarray(signal).ravel(), np.ones(factor), 'valid')[::factor]

# put this in FRI_helpers later
from sklearn.metrics import roc_auc_score
def get_auc_notbinary(spikes, predictions):
    spikes[spikes>1] = 1
    return roc_auc_score(spikes, predictions)


sigma_tests = np.linspace(0.5, 7.5, 8)
threshold_tests = np.linspace(0, 90, 16)

#=========================================================
# Cross validation training
#=========================================================

if TRAIN_TEST_STATE == 'train':

    crossval_corr = np.zeros((len(sigma_tests), len(threshold_tests), len(data_preprocessed)))

    pred_results = []

    for i, strip_data in enumerate(data_preprocessed):
        print('Processing {}/{}'.format(i+1, len(data_preprocessed)))
        
        spikes_data = data_preprocessed[i]['spikes']
        
        output = sliding_window_predict(strip_data['calcium'], Fs=strip_data['fps'],
                                          K=average_spikes_per_strip[i],
                                          window_lengths=[301, 601, 801, 1101],
                                          jump_size=30,
                                          OF=4,
                                          smoothing_sigma=None)
        
        pred_out = {}
        pred_out['likelihood'] = output

        pred_results.append(pred_out)

        for sigma_idx, sigma in enumerate(sigma_tests):
            smoothed_likelihood = gaussian_filter1d(output, sigma=sigma)
            
            for thresh_idx, thresh in enumerate(threshold_tests):
                thresh_likelihood = percentile_threshold_likelihood(smoothed_likelihood, thresh_percentile=thresh, set_above_to_1=False)
                
                crossval_corr[sigma_idx, thresh_idx, i] = np.corrcoef(thresh_likelihood, spikes_data)[0,1]




    #=========================================================
    # Cross validation testing
    #=========================================================

    for i, strip_data in enumerate(data_preprocessed):
        print('Processing {}/{}'.format(i, len(data_preprocessed)))
        
        spikes_data = data_preprocessed[i]['spikes']
        
        # remove strip i from crossval array
        crossval_without_i = np.delete(crossval_corr, i, axis=2)
        
        # average over training strips then get max index
        crossval_corr_averaged = np.mean(crossval_without_i, axis=2)
        max_idx = np.unravel_index(crossval_corr_averaged.argmax(), crossval_corr_averaged.shape)
        sigma = sigma_tests[max_idx[0]]
        thresh = threshold_tests[max_idx[1]]
        
        print('Using parameters: {}'.format((sigma, thresh)))
        
        pred_out = pred_results[i]
        likelihood = pred_out['likelihood']
        
        smoothed_likelihood = gaussian_filter1d(likelihood, sigma=sigma)
        smoothed_likelihood = percentile_threshold_likelihood(smoothed_likelihood, thresh_percentile=thresh, set_above_to_1=False)
        
        pred_out['smoothed_likelihood'] = smoothed_likelihood # after thresholding
        pred_out['sigma'] = sigma
        pred_out['thresh'] = thresh
        
        pred_results[i] = pred_out
        

    # save results
    pickle.dump(pred_results, open(PICKLE_FILENAME, 'wb'))

else:
    # Read in the file already created
    pred_results = pickle.load(open(PICKLE_FILENAME, 'rb'))


#=========================================================
# Create matrices storing average data for plotting
#=========================================================

bin_widths = np.arange(10, 110, 10)

corr_array = np.zeros((len(pred_results), len(bin_widths)))
corr_array_foopsi = np.zeros_like(corr_array)
corr_array_raw = np.zeros_like(corr_array)

auc_array = np.zeros((len(pred_results), len(bin_widths)))
auc_array_foopsi = np.zeros_like(auc_array)
auc_array_raw = np.zeros_like(auc_array)


# compute correlations for our method
for i, (outputs, filename) in enumerate(zip(pred_results, recordings)):
    

    thresh = outputs['thresh']
    sigma = outputs['sigma']
    likelihood_counts = outputs['smoothed_likelihood']
    likelihood_counts = outputs['likelihood']
    likelihood_counts = gaussian_filter1d(likelihood_counts, sigma=sigma)
    likelihood_counts = percentile_threshold_likelihood(likelihood_counts, thresh_percentile=thresh, set_above_to_1=False)

    print((thresh, sigma))

    spikes_data = data_preprocessed[i]['spikes']
    calcium_data = data_preprocessed[i]['calcium']
    

    for k, bin_width in enumerate(bin_widths):
        
        if bin_width != 10:
            factor = int(bin_width / 10)
            downsampled_spikes = downsample(spikes_data, factor)
            downsampled_raw = downsample(calcium_data, factor)
            downsampled_likelihood = downsample(likelihood_counts, factor)

        else: 
            downsampled_spikes = spikes_data.copy()
            downsampled_raw = calcium_data.copy()
            downsampled_likelihood = likelihood_counts.copy()
    
        # correlation
        corr_array[i,k] = np.corrcoef(downsampled_likelihood, downsampled_spikes)[0,1]
        corr_array_raw[i,k] = np.corrcoef(downsampled_raw, downsampled_spikes)[0,1]
        
        # AUC
        auc_array[i,k] = get_auc_notbinary(downsampled_spikes, downsampled_likelihood)
        auc_array_raw[i,k] = get_auc_notbinary(downsampled_spikes, downsampled_raw)



#=========================================================
# Create figures
#=========================================================

AXIS_SIZE = 18
TITLE_SIZE = 22
TICK_SIZE = 15

avg_corr = np.mean(corr_array, axis=0)
std_corr = np.std(corr_array, axis=0)
se_corr = std_corr / np.sqrt(corr_array.shape[0])

avg_corr_raw = np.mean(corr_array_raw, axis=0)
std_corr_raw = np.std(corr_array_raw, axis=0)
se_corr_raw = std_corr_raw / np.sqrt(corr_array_raw.shape[0])



avg_auc = np.mean(auc_array, axis=0)
std_auc = np.std(auc_array, axis=0)
se_auc = std_auc / np.sqrt(auc_array.shape[0])

avg_auc_raw = np.mean(auc_array_raw, axis=0)
std_auc_raw = np.std(auc_array_raw, axis=0)
se_auc_raw = std_auc_raw / np.sqrt(auc_array_raw.shape[0])


fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].errorbar(range(len(bin_widths)), avg_corr, color='g', yerr=2*se_corr, label='Proposed')
ax[0].errorbar(range(len(bin_widths)), avg_corr_raw, color='#1f77b4', yerr=2*se_corr_raw, label='Calcium Signal')
# ax.errorbar(range(len(bin_widths)), avg_corr_Lzero, color='c', yerr=std_corr_Lzero, label='L-0')
# ax.errorbar(range(len(bin_widths)), avg_corr_Lzero_con, color='pink', yerr=std_corr_Lzero_con, label='L-0 con.')
ax[0].set_xticks(range(len(bin_widths)))
ax[0].set_xticklabels(bin_widths, fontsize=TICK_SIZE)
ax[0].set_xlabel('Bin Width [ms]', fontsize=AXIS_SIZE)
ax[0].set_title('Correlation', fontsize=TITLE_SIZE)
ax[0].set_ylim([0, 0.6])
# ax[0].set_yticklabels(fontsize=TICK_SIZE)
ax[0].tick_params(labelsize=AXIS_SIZE)
ax[0].grid()
ax[0].legend(fontsize=15, loc='upper left')
# plt.show()

# fig, ax = plt.subplots()
ax[1].errorbar(range(len(bin_widths)), avg_auc, color='g', yerr=2*se_auc, label='proposed')
ax[1].errorbar(range(len(bin_widths)), avg_auc_raw, color='#1f77b4', yerr=2*se_auc_raw, label='raw calcium')
# ax.errorbar(range(len(bin_widths)), avg_auc_Lzero, color='c', yerr=std_auc_Lzero, label='L-0')
# ax.errorbar(range(len(bin_widths)), avg_auc_Lzero_con, color='pink', yerr=std_auc_Lzero_con, label='L-0 con.')
ax[1].set_xticks(range(len(bin_widths)))
ax[1].set_xticklabels(bin_widths, fontsize=TICK_SIZE)
ax[1].set_xlabel('Bin Width [ms]', fontsize=AXIS_SIZE)
ax[1].set_title('AUC', fontsize=TITLE_SIZE)
ax[1].set_ylim([0, 1])
# ax[1].set_yticklabels(fontsize=TICK_SIZE)
ax[1].grid()
ax[1].tick_params(labelsize=AXIS_SIZE)
# ax.legend()
plt.tight_layout()
plt.savefig('corr_and_auc_plot.png', dpi=400)
plt.show()
