#coding: utf-8
from __future__ import division
import os.path

import numpy as np
import pylab as pl
from sklearn import svm, cross_validation

_curdir = os.path.dirname(__file__)

def preprocess():
    """
    Input datasets:
        fsample --> si riferisce solamente alla frequenza di campionamento (300 Hz);
        label --> associa ad ogni canale un'etichetta per identificarlo spazialmente:
        'MLx' (left), 'MRx' (right) e 'MZx' (zenith) con x=C,F,O,P,T per central,
        frontal, occipital, parietal and temporal;
        lf_freq --> matrice [127 x 274 x 376] = [n trial con task left x n canali x
        densità spettrale] (matrice che più ti interessa, da questa ti ricavi la densità
        media nei vari intervalli: 2-4 4-8 8-13 ecc);
        sample_freq --> asse delle frequenze (in Hz) su cui è calcolata la densità
        spettrale di potenza;
        time --> asse dei tempi (in secondi) su cui era acquisito il segnale meg.
    Output dataset:
        left --> matrice (127, 274, 7) = (trial left x channel x  μ_i),
        con  μ_i la media per i vari intervalli di frequenza
        label --> associa ad ogni canale un'etichetta per identificarlo
        spazialmente

    """
    left = np.load(os.path.join(_curdir, 'freq_left_gradient.npy.npz'))
    right = np.load(os.path.join(_curdir, 'freq_right_gradient.npy.npz'))

    dataset = dict()
    dataset['label'] = left['label']  # = right['label']
    for direction, trials in ('left', left['lf_freq']), ('right', right['rg_freq']):
        samples, channels, freqs = trials.shape
        step = freqs // 6
        features = [[[np.mean(trials[trial][channel][start:start+step])
                     for start in range(0, freqs, step)]
                    for channel in range(channels)]
                   for trial in range(samples)]
        dataset[direction] = features

    np.savez_compressed(os.path.join(_curdir, 'dataset'), **dataset)


# dataset = np.load(os.path.join(_curdir, 'dataset.npz'))
# labels = dataset['label']
# trials, channels, freqs = dataset['left'].shape  # ~ dataset['right'].shape
# # set up first classifier
# fst = {channel : svm.SVC() for channel in dataset['label']}
# fst['global'] = svm.SVC()
#
# kfold = cross_validation.KFold(trials-trials//10, n_folds=10)
# # set up second classifier
# snd = svm.SVC(loss='l1', penality='l1')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Machine Learning MEG project')
    parser.add_argument('-p', '--preprocess',
                        action='store_true',
                        dest='preprocess',
                        help='Preprocess datased given by CIMeC',
    )
    args = parser.parse_args()
    if args.preprocess:
        preprocess()
    else:
        parser.print_help()
