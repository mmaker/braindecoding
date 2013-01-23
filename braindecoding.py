#!/usr/bin/env python
#coding: utf-8
from __future__ import division, with_statement
from multiprocessing import Pool
from copy import deepcopy
import os.path
import warnings
import itertools
import signal
import sys

import numpy as np
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3

from sklearn import svm, cross_validation
from sklearn.metrics import zero_one_score as accuracy

__author__ = 'Michele Orru`'
__email__ = 'maker@python.it'
__license__ = """ DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Michele Orru` <maker@python.it>

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
    [Danilo Benozzo]
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
        left --> matrice (127, 274, 7), (±1) = (trial x channel x  μ), output
        con  μ la media per i vari intervalli di frequenza
        label --> associa ad ogni canale un'etichetta per identificarlo
        spazialmente

    """
    left = np.load(os.path.join(_curdir, 'freq_left_gradient.npy.npz'))
    right = np.load(os.path.join(_curdir, 'freq_right_gradient.npy.npz'))

    dataset = []
    for trials, direction in (left['lf_freq'], -1), (right['rg_freq'], +1):
        samples, channels, freqs = trials.shape
        step = freqs // 6
        for sample in xrange(samples):
            trial = np.array(
                [[np.mean(trials[sample][channel][start:start+step])
                  for start in range(0, freqs, step)]
                 for channel in xrange(channels)]
            )
            dataset.append((trial, direction))

    print 'dataset preprocessed, saving..'
    np.savez_compressed(os.path.join(_curdir, 'dataset'),
                        dataset=np.array(dataset, dtype=object, copy=False),
                        label=left['label']    #=right['label']
    )


def plot(filter='rbf', outputlog='output.log'):
    """
    Plot output logs in a pretty way.
    """
    with open(outputlog) as f:
        lines = [line.strip().split(',', 1) for line in f
                 if 'BEST' in line and filter in line]

    # template:
    # "BEST: 63.33%, parameters: {'kernel': 'poly', 'C': 96.0, 'gamma': 49.5}"
    lines = [(float(x.split(':')[1][:-1]), eval(y.split(':',1)[1]))
             for x, y in lines]

    if filter == 'rbf':
        x, y = np.array([(fst, snd['C']) for fst, snd in lines]).T
        pl.plot(x, y)
        pl.ylabel('C')
        pl.xlabel('accuracy')
    elif filter == 'poly':
        x, y, z = np.array(
           [(float(fst), float(snd['C']), float(snd['gamma']))
            for fst, snd in lines]
        ).T

        fig = pl.figure()
        ax = p3.Axes3D(fig)
        ax.plot_wireframe(z,y,x)
        ax.set_zlabel('acc')
        ax.set_ylabel('C')
        ax.set_xlabel(u'ɣ')

    pl.title('%s performance graph on file %s' % (filter, outputlog))
    pl.show()

def learn(parameters=None):
    parameters = parameters or dict(kernel='rbf',
                                    C=100)
    npz = np.load(os.path.join(_curdir, 'dataset.npz'))
    labels = npz['label']
    dataset = npz['dataset']
    best = type('best', (object, ), dict(fst=None, snd=None, acc=0))

    trials = len(dataset)
    channels, freqs = dataset[0][0].shape  # ~dataset[n][0]
    globalshape = freqs * channels

    assert all(dataset[trial][0].shape == (channels, freqs)
               for trial in xrange(trials))

    # set up first classifier, composed of one classifier per each channel, plus
    # one global.
    fst = [svm.SVC(**parameters)
           for channel in xrange(channels+1)]
    # set up second classifier
    snd = svm.LinearSVC(penalty='l1', dual=False)

    # shuffle dataset at start
    np.random.shuffle(dataset)

    # notation: 'o' stands for 'outer', 'i' for inner
    ofold = cross_validation.KFold(n=trials, k=10)
    for training, testing in ofold:
        # select features from the dataset
        training = dataset[training]
        testing = dataset[testing]

        ifold = cross_validation.KFold(n=len(training), k=10)
        testing_input, testing_output = testing.T
        testing_input = np.array([x for x in testing_input])

        for training1, training2 in ifold:
            training1 = training[training1]
            training2 = training[training2]

            input, output = training1.T
            # fuck you, numpy.
            input = np.array([x for x in input])
            for channel in xrange(channels):
                fst[channel].fit(input[:, channel], output)
                # print 'training channel {} ({}/{})  \r'.format(
                #         labels[channel], channel, channels),
            fst[-1].fit(np.reshape(input, (-1, globalshape), order='F'), output)

            # train second classifier, using predicted data from other
            # classifiers
            input, output = training2.T
            input = np.array([x for x in input])
            input = np.array([fst[channel].predict(input[:, channel])
                              for channel in xrange(channels)]).T
            snd.fit(input, output)

            # evaluate learner's accuracy. If an acceptable one is found, quit
            # immediately. Otherwise, compare with the best.
            predicted = np.array([fst[channel].predict(testing_input[:, channel])
                                  for channel in xrange(channels)]).T
            predicted = [snd.predict(x) for x in predicted]
            acc = accuracy(testing_output, predicted)
            if acc > best.acc:
                best.fst = deepcopy(fst)
                best.snd = deepcopy(snd)
                best.acc = acc
            if acc > .75:
                # save somewhere, classifiers
                np.savez_compressed(
                  os.path.join(_curdir,
                      'learners-{}-acc{}'.format(str(parameters), int(acc*100))),
                  first=fst,
                  second=snd,
                )
                print 'WHOA: {}%, parameters: {}'.format(round(acc*100, 2), str(parameters))
    else:
        print 'BEST: {}%, parameters: {}'.format(round(best.acc*100, 2), str(parameters))
        return best

def tune(processes=None):
    """
    Start a pool of processes, each one computing a different learn() functions.
    """
    # exit immediately with a KeyboardInterrupt
    signal.signal(signal.SIGINT, sys.exit)

    pool = Pool(processes)
    parameters = []
    parameters.extend(dict(kernel='rbf', C=c)
                      for c in np.arange(50, 100, 0.5))
    parameters.extend(dict(kernel='poly', C=c, gamma=gamma)
                      for c in np.arange(50, 100, 0.5)
                      for gamma in np.arange(10, 30, 0.5))
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
                        nargs='*',
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
                        default=False,
                        type=int,
                        dest='tune',
                        metavar='PROCESSES',
                        help='sart up a pool'
    )
    args = parser.parse_args()
    if args.preprocess:
        preprocess()
    if args.plot is not None:
        plot(*args.plot)
    if args.learn:
        learn()
    if args.tune:
        tune(args.tune)
    if all(x is None for x in vars(args).values()):
        parser.print_help()
