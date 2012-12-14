#coding: utf-8
from __future__ import division
from multiprocessing import Pool
import os.path
import warnings
import itertools
import signal
import sys

import numpy as np
import pylab as pl
from sklearn import svm, cross_validation


__author__ = 'Michele Orru`'
__email__ = 'maker@python.it'
__license__ = """ DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
"""


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


def learn(parameters=None):
    dataset = np.load(os.path.join(_curdir, 'dataset.npz'))
    labels = dataset['label']
    left = np.array([[x, -1] for x in dataset['left']])
    right = np.array([[x, +1] for x in dataset['right']])
    dataset = np.concatenate((left, right))

    trials = dataset.shape[0]
    channels, freqs = dataset[0][0].shape  # ~dataset[0][n]
    # assert all(dataset[trial][0].shape == (channels, freqs)
    #            for trial in xrange(trials))

    # set up first classifier, composed of one classifier per each channel, plus
    # one global.
    if parameters is None:
        parameters = dict(kernel='rbf',
                          C=trials//2,
        )
    fst = [svm.SVC(**parameters)
           for channel in xrange(channels+1)]
    # set up second classifier
    snd = svm.LinearSVC(penalty='l1', dual=False)
    globalshape = freqs * channels

    # notation: 'o' stands for 'outer', 'i' for inner
    ofold = cross_validation.KFold(n=trials, k=10)
    for otraining, otesting in ofold:
        ifold = cross_validation.KFold(n=len(otraining), k=10)
        for itrains, itests in ifold:
            itraining = otraining[itrains]
            itesting = otraining[itests]

            input, output = dataset[itraining].T
            # fuck you, numpy.
            input = np.array([x for x in input])
            for channel in xrange(channels):
                fst[channel].fit(input[:, channel], output)
                print 'training channel {} ({}/{})  \r'.format(
                        labels[channel], channel, channels),
            fst[-1].fit(np.reshape(input, (-1, globalshape), order='F'), output)

            ## wtf am i doing here. ##
            input, output = dataset[itesting].T
            input = np.array([x for  x in input])
            accuracy = np.mean([sum(fst[channel].predict(input[:, channel]) == output)
                                for channel in range(channels)])
            print '\n accuracy: {}/{}'.format(
                accuracy, len(input),
            )

        # train second dataset
        input, output = dataset[otraining].T
        input = np.array([x for x in input])
        input = np.array([fst[channel].predict(input[:, channel])
                          for channel in range(channels)]).T
        snd.fit(input, output)

    # save somewhere, classifiers
    np.savez_compressed(
        os.path.join(_curdir, 'learners-{}'.format(str(parameters))),
        first=fst,
        second=snd,
    )


def tune(processes=None):
    """
    Learn, using specific constraints.
    """
    # exit immediately with a KeyboardInterrupt
    signal.signal(signal.SIGINT, sys.exit)

    pool = Pool(processes)
    parameters = []
    parameters.extend(dict(kernel='rbf', C=c)
                      for c in np.arange(10, 100, 0.5))
    parameters.extend(dict(kernel='poly', C=c, gamma=gamma)
                      for c in np.arange(10, 100, 0.5)
                      for gamma in np.arange(0, 50, 0.5))
    pool.map_async(learn, parameters).get()


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
    parser.add_argument('-t', '--tune',
                        nargs='?',
                        default=2,
                        type=int,
                        dest='tune',
                        metavar='PROCESSES',
                        help='sart up a pool'
    )
    args = parser.parse_args()
    if args.preprocess:
        preprocess()
    if args.plot:
        plot()
    if args.learn:
        learn()
    if args.tune:
        tune(args.tune)
    if not any(vars(args).values()):
        parser.print_help()
