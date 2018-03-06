import pandas as pd
import numpy as np
import sys
import os
import time
import collections
from random import random
import seaborn as sns
import matplotlib.pyplot as plt

from utils import *


class ScoreStats:

    def __init__(self, name, score_fn, plot_marker='+'):
        self.name = name
        self.score_fn = score_fn
        self.plot_marker = plot_marker

    def run(self, n_iter=None, max_time=None, display_every=100):
        self.scores = []
        self.score_times = []
        self.avg_scores = []
        start_time = time.time()

        i = 1
        while True:
            score = self.score_fn()
            elapsed = time.time()-start_time

            self.scores.append(score)
            self.avg_scores.append(np.array(self.scores).mean())
            self.score_times.append(elapsed)

            if i % display_every == 0:
                if n_iter is not None:
                    print 'iteration', i, '/', str(n_iter)
                if max_time is not None:
                    print 'iteration', i, ', elapsed:', elapsed, 'seconds'

            i += 1

            if n_iter is not None and i > n_iter:
                break
            if max_time is not None and time.time()-start_time > max_time:
                break

        self.elapsed_time = time.time() - start_time
        self.elapsed_time_per_score = self.elapsed_time/len(self.scores)
        self.std = np.std(self.scores)
        self.mean = np.array(self.scores).mean()

    def distplot(self, ax):
        sns.distplot(self.scores, hist=False, rug=False,
                     label=str(self.name), ax=ax)

    def cumul_average_plot(self, ax):
        ax.plot(self.score_times, self.avg_scores, label=str(self.name))

    def plot_std_mean(self, ax):
        ax.plot(self.std, self.mean, self.plot_marker, label=str(self.name))

    def plot_time_std(self, ax):
        ax.plot(self.elapsed_time, self.std,
                self.plot_marker, label=str(self.name))

    def plot_bench_time_mean(self, ax):
        bt95samples = self.samples_for_ci_precision_float(95, 0.001)
        bt95time = self.elapsed_time_per_score*bt95samples
        ax.plot(bt95time, self.mean, self.plot_marker, label=str(self.name))

    def summary(self):
        print 'Elaped:', self.elapsed_time, 'seconds,', self.elapsed_time_per_score, '/score'
        print 'standard deviation:', self.std
        print 'Mean score:', self.mean

        bt95samples = self.samples_for_ci_precision_float(95, 0.001)
        bt95time = self.elapsed_time_per_score*bt95samples
        print 'Scoring time to achieve 95% accuracy at 0.001 scoring precision', bt95time, 'seconds (', np.ceil(bt95samples), 'samples)'

        bt98samples = self.samples_for_ci_precision_float(98, 0.001)
        bt98time = self.elapsed_time_per_score*bt98samples
        print 'Scoring time to achieve 98% accuracy at 0.001 scoring precision', bt98time, 'seconds (', np.ceil(bt98samples), 'samples)'

    def samples_for_ci_precision_float(self, ci, precision):
        # map confidence interval desired in % -> z values
        Zs = {
            80: 1.28,
            90: 1.645,
            95: 1.96,
            98: 2.33,
            99: 2.58
        }
        orderedZs = collections.OrderedDict(
            sorted(Zs.items(), key=lambda t: t[0]))

        standard_deviation = self.std

        z = Zs[ci]
        return np.power(standard_deviation * z / precision, 2)


def plot_scores_statistics(scores_stats):

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.suptitle("Scoring statistics", fontsize=10)

    ax = plt.subplot("221")
    ax.set_title("Distributions")
    for ss in scores_stats:
        ss.distplot(ax)

    ax = plt.subplot("222")
    ax.set_title("cumul average")
    for ss in scores_stats:
        ss.cumul_average_plot(ax)

    ax = plt.subplot("223")
    ax.set_title("x=std / y=mean")
    for ss in scores_stats:
        ss.plot_std_mean(ax)

    ax = plt.subplot("224")
    ax.set_title("x=benchtime / y=mean")
    for ss in scores_stats:
        ss.plot_bench_time_mean(ax)

    #plt.legend(loc='upper right')
    plt.show()
