import os
import sys
import random
import traceback
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from scipy.stats import zscore
from sklearn import preprocessing
from scipy import signal

# import wfdb
import mne

import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

from ecgdetectors import Detectors
# https://github.com/berndporr/py-ecg-detectors

# ann2label = {
#     "Sleep stage W": 0,
#     "Sleep stage 1": 1,
#     "Sleep stage 2": 2,
#     "Sleep stage 3": 3,
#     "Sleep stage 4": 3,
#     "Sleep stage R": 4,
#     "Sleep stage ?": 5,
#     "Movement time": 5
# }


def load_eeg_channel(edf_file, ch_name="FP1", fs_target=50, log=print):
    try:
        raw = mne.io.read_raw_edf(edf_file, preload=False)
        ch_idx = -1        
        for cname in raw.info.get('ch_names'):
            ch_idx += 1
            if cname.upper().find(ch_name.upper()) > -1:
                ch_name = cname
                break
        hz = mne.pick_info(raw.info, [ch_idx], verbose=False)['sfreq']
        hz = int(hz)
        log(f"Hz:{hz}, channel_names:{raw.info.get('ch_names')}")
        raw.pick_channels([ch_name])
        recording = raw.get_data().flatten()

    except:
        log(f"Error reading {edf_file}, caused by - {traceback.format_exc()}")
        return
    # Down sample to 50Hz
    down_factor = hz // fs_target
    target_samples = len(recording) // down_factor
    recording = signal.resample(recording, target_samples)
    return recording


class EEGDataset():
    r"""EEG multi-channel dataset. Read only a single channel.
    """
    def __init__(
            self, data_directory, hz=100, seg_sec=10, ch_names=["FP1"], 
            log=print, filter_records=[]
    ):
        self.data_directory = data_directory
        self.hz = hz
        self.seg_sec = seg_sec
        self.seg_dim = self.hz * self.seg_sec
        self.log = log
        self.ch_names = ch_names
        if len(self.ch_names) > 1:
            raise Exception("More than 1 channel not supported.")
        self.filter_records = filter_records
        self.record_names = []
        self.record_wise_segments = {}
        self.segments = []
        self.seg_labels = []
        self.log(f"data:{data_directory}, hz:{hz}, seg_sec:{seg_sec}, ")
        self._initialise()
        self.indexes = [i for i in range(len(self.segments))]
        np.random.shuffle(self.indexes)

    def _initialise(self):
        count_file = 0
        for f in os.listdir(self.data_directory):
            # Let's consider only EC edf files.
            if not f.endswith("EC.edf"):
                continue
            rec_name = f[:-4]
            if len(self.filter_records) > 0 and not rec_name in self.filter_records:
                continue

            self.log(f"Loading {rec_name}...")
            self.record_names.append(rec_name)
            if self.record_wise_segments.get(rec_name) is None:
                    self.record_wise_segments[rec_name] = []
            
            signals = load_eeg_channel(
                f"{self.data_directory}/{f}", ch_name=self.ch_names[0], 
                fs_target=self.hz, log=self.log)
            self.log(signals.shape)
            signal_single_chan = signals[:]

            clz_label_dist = {}
            seg_count = 0
            start = 0
            while start + self.seg_dim < len(signal_single_chan):
                seg = signal_single_chan[start : start + self.seg_dim]
                seg = zscore(seg)  # normalisation
                seg = np.expand_dims(seg, axis=1)  # (n_samp,) -> (n_samp, 1)
                label = 1 if rec_name.startswith("MDD") else 0
                
                # record distribution
                if clz_label_dist.get(label) is None:
                    clz_label_dist[label] = 0
                clz_label_dist[label] += 1
                
                self.segments.append(seg)
                self.seg_labels.append(label)
                self.record_wise_segments[rec_name].append(len(self.segments)-1)

                start += self.seg_dim
                seg_count += 1
            self.log(
                f"... n_seg:{seg_count}, clz_lbl_dist:{clz_label_dist}")
            count_file += 1
        # sample distribution
        # 
        self.indexes = range(len(self.segments))
        _dist = np.unique(
            [self.seg_labels[i] for i in self.indexes], return_counts=True)
        self.log(f"Total files:{count_file}, n_seg:{len(self.segments)}, distribution:{_dist}")


class PartialDataset(Dataset):
    r"""Generate dataset from a parent dataset and indexes."""

    def __init__(
        self, dataset=None, seg_index=None, test=False, shuffle=False, as_np=False, log=print
    ):
        r"""Instantiate dataset from parent dataset and indexes."""
        self.memory_ds = dataset
        self.indexes = seg_index[:]
        self.test = test
        self.shuffle = shuffle
        self.as_np = as_np
        self.log = log
        self.label_idx_dict = {}

    def on_epoch_end(self):
        r"""End of epoch."""
        if self.shuffle and not self.test:
            np.random.shuffle(self.indexes)

    def __len__(self):
        r"""Dataset length."""
        return len(self.indexes)

    def __getitem__(self, idx):
        r"""Find and return item."""
        ID = self.indexes[idx]
        # trainX = np.array(self.memory_ds.segments[ID])
        trainX = self.memory_ds.segments[ID]
        trainY = self.memory_ds.seg_labels[ID]

        if self.as_np:
            return trainX, trainY

        X_tensor = Variable(torch.from_numpy(trainX)).type(torch.FloatTensor)
        # print(f"X_tensor before: {X_tensor.size()}")
        r"numpy array shape: (n_samp, n_chan), reshape it to (n_chan, n-samp)."
        # Segment in (1, n_samp) form, still need below line?
        X_tensor = X_tensor.reshape(X_tensor.size()[1], -1)
        Y_tensor = trainY
        if torch.any(torch.isnan(X_tensor)):
            X_tensor = torch.nan_to_num(X_tensor)
        return X_tensor, Y_tensor
    

def test_ecg_datasource():
    datasource = EEGDataset(
        data_directory="data/4244171",
        hz=100, seg_sec=30,
        filter_records=['H S1 EC']
    )
    p_ds = PartialDataset(
        dataset=datasource, 
        seg_index=datasource.record_wise_segments[datasource.record_names[0]], 
        as_np=True)
    sleep_seg_count = 0
    for i in range(5):
        seg, lbl = p_ds[i]
        print(f"partial-ds, seg:{seg.shape}, label:{lbl}")
        # plt.ylim((0, 1.2))
        plt.plot(range(seg.shape[0]), seg[:, 0])
        plt.show()


def main():
    test_ecg_datasource()


if __name__ == "__main__":
    try:
        main()
    except:
        # traceback.print_exc(file=sys.stdout)
        print(traceback.format_exc())