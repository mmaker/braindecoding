#coding: utf-8
from __future__ import division
import os.path
import warnings
import itertools

import numpy as np
import pylab as pl
from sklearn import svm, cross_validation

_curdir = os.path.dirname(__file__)

# hide shitty gtk+ warning messages.
warnings.simplefilter("ignore")


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

    print 'dataset preprocessed, saving..'
    np.savez_compressed(os.path.join(_curdir, 'dataset'), **dataset)


def plot():
    """
    Plot dataset in a pretty way.
    """
    dataset = np.load(os.path.join(_curdir, 'dataset.npz'))
    labels = dataset['label']
    trials, channels, freqs = map(xrange, dataset['left'].shape)


def learn(c=None, kernel='rbf', **kwargs):
    dataset = np.load(os.path.join(_curdir, 'dataset.npz'))
    labels = dataset['label']
    dataset = ([(x, -1) for x in dataset['left']] +
               [(x, +1) for x in dataset['right']])
    trials = len(dataset)
    if c is None:
        c = trials//2
    channels, freqs = map(xrange, dataset[0][0].shape)  # ~dataset[0][n]
    # set up first classifier, composed of one classifier per each channel, plus
    # one global.
    fst = [svm.SVC(kernel=kernel, C=c, **kwargs)
           for channel in range(len(channels)+1)]
    # set up second classifier
    snd = svm.LinearSVC(penalty='l1', dual=False)
    globalshape = len(freqs) * len(channels)

    # notation: 'o' stands for 'outer', 'i' for inner
    ofold = cross_validation.KFold(n=trials, k=10)
    for otrains, otests in ofold:
        otraining = [dataset[x] for x in otrains]
        otesting = np.array([dataset[x] for x in otests], object).T
        ifold = cross_validation.KFold(n=len(otraining), k=10)
        for itrains, itests in ifold:
            itraining = np.array([otraining[x] for x in itrains], object).T
            itesting = np.array([otraining[x] for x in itests], object).T

            input, output = itraining
            for channel in channels:
                fst[channel].fit(input[channel], output)
                print 'training channel {} ({}/{})  \r'.format(
                        labels[channel], channel, len(channels)),
            fst[-1].fit(np.reshape(input, globalshape), output)

            # print accuracy
            accuracy = sum(fst[channel].predict(input[channel]) == output
                           for channel in channels )
        print '\n accuracy: {}%'.format(
            accuracy[0] * 100 / (2* len(channels)*len(testing)),
            len(accuracy)
        )
        # train second dataset
        # snd.train(
        #[fst[channel].predict(dataset['left', test, i]) for i, channel in enumerate(labels)],


def tune():
    """
    Learn, using specific constraints.
    """



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Machine Learning MEG project')
    parser.add_argument('-p', '--preprocess',
                        action='store_true',
                        dest='preprocess',
                        help='Preprocess datased given by CIMeC',
    )
    parser.add_argument('-d', '--plot',
                        action='store_true',
                        dest='plot',
                        help='plot the dataset'
    )
    parser.add_argument('-l', '--learn',
                        action='store_true',
                        dest='learn',
                        help='learn using the stacked classifier'
    )
    args = parser.parse_args()
    if args.preprocess:
        preprocess()
    if args.plot:
        plot()
    if args.learn:
        learn()
    if not any(vars(args).values()):
        parser.print_help()
