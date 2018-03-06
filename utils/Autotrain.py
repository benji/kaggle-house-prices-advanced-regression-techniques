import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from utils import *


class Autotrain:
    '''
    This class will find the higher scoring variant or set of variants.
    You provide
    - variants: the variants (any kind of object)
    - score_variant_fn: how to score a variant
    - goal: 'min' or 'max' (whether to minimize or maximize the score)

    Important note: it is preferrable that the scoring method give stable scores.
    For example, if you shuffle your data prior to scoring, set the seed to a fixed number
    to get consistent results.
    '''

    def __init__(self, verbose=False, stop_if_no_progress=True):
        self.verbose = verbose
        self.stop_if_no_progress = stop_if_no_progress

    def find_single_best(self, variants, score_variant_fn=None, goal='min', tqdm=False):
        '''
        Finds the one variant that yields the best score
        '''
        return self.__find_single_best_with_score_fn(variants, score_variant_fn, goal, use_tqdm=tqdm)

    def find_multiple_best(self, variants=None, variants_fn=None, score_variants_fn=None, goal='min', on_validated_variant_fn=None, on_round_finish_fn=None, tqdm=True):
        '''
        Finds the list of variants that yields the best score.
        Here a variant is a list of variants.
        The algorithm proceeds iteratively by finding one by one the columns which yield
        the best score of all the columns when added to the list of validated variants.
        The algorithm stops when the score stop improving.

        variants_fn: if empty, the algorithm will simply remove the selected variant from the pool for the next round.
        You can implement this method if you need to remove more.
        '''
        validated_variants = []
        best_score = None

        if variants_fn is None:
            available_variants = list(variants)  # copy

        round_i = 1
        while True:
            if variants_fn is not None:
                available_variants = variants_fn(validated_variants)

            if len(available_variants) == 0:
                print 'No more variants to evaluate.'
                break

            mulitple_variants = [validated_variants +
                                 [v] for v in available_variants]

            mulitple_variant, score = self.__find_single_best_with_score_fn(
                mulitple_variants, score_variants_fn, goal, use_tqdm=tqdm)

            improved = self.is_better_score(best_score, score, goal)

            print 'Round', round_i, 'score:', score, 'improved:', improved, 'variants:', mulitple_variant

            if self.stop_if_no_progress and not improved:
                break

            if improved:
                best_score = score

            elected_variant = mulitple_variant[-1]
            validated_variants.append(elected_variant)

            if on_validated_variant_fn is not None:
                on_validated_variant_fn(elected_variant)

            if on_round_finish_fn is not None:
                on_round_finish_fn(validated_variants)

            if variants_fn is None:
                available_variants.remove(mulitple_variant[-1])

            round_i += 1

        # final
        print 'Final score:', best_score, ' variants:', validated_variants

        return validated_variants, best_score

    def is_better_score(self, ref_score, new_score, goal):
        if ref_score is None:
            return True
        elif goal == 'min' and new_score < ref_score:
            return True
        elif goal == 'max' and new_score > ref_score:
            return True
        else:
            return False

    def __find_single_best_with_score_fn(self, variants, score_variant_fn, goal, use_tqdm=False):
        '''Finds the one variant that yields the best score'''
        best_variant = None
        best_score = None

        for v in tqdm(variants, disable=not use_tqdm):
            score = score_variant_fn(v)

            if self.verbose:
                print 'score:', score, 'variant:', v

            if self.is_better_score(best_score, score, goal):
                best_score = score
                best_variant = v

        return best_variant, best_score
