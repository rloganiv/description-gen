"""N-Gram language model implementation"""

from __future__ import division, print_function

import argparse
import numpy as np
from collections import Counter, defaultdict
from nltk.tokenize.treebank import TreebankWordTokenizer

 
tokenizer = TreebankWordTokenizer()


class NGramLanguageModel(object):

    def __init__(self,
                 n,
                 alpha=0,
                 interpolation_weights=None,
                 backoff=1e-6,
                 start_token='<SOS>',
                 end_token='<EOS>'):
        """Initializes the n-gram language model.

        Args:
            n: Size of n-gram.
            alpha: Constant used in Laplace smoothing.
            interpolation_weights: Weights used to interpolate lower-order
                n-gram probabilities.
            backoff: Replacement value for zero probability.
            start_token: Token used to pad starts of sentences.
            end_token: Token used to end sentences.
        """
        assert n >= 1, "n must be a positive integer"
        self.n = n
        self.alpha = alpha
        self._lbackoff = np.log2(backoff)
        self._start_token = start_token
        self._end_token = end_token
        self._counts = defaultdict(Counter)


    def fit(self, x):
        """Fits the n-gram model. Note: This will reset existing counts.

        Args:
            x: Any finite iterable which produces sentences, where a sentence
                is any finite iterable which produces tokens. For example:
                    
                    [['First', 'Sentence'], ['Second', 'Sentence']]
        """
        self._counts = defaultdict(Counter)
        self.update(x)

    def update(self, x):
        """Updates the n-gram model counts.
        
        Args:
            x: Any finite iterable which produces sentences, where a sentence
                is any finite iterable which produces tokens. For example:
                    
                    [['First', 'Sentence'], ['Second', 'Sentence']]
        """
        for sentence in x:
            sentence = self._pad(sentence)
            sentence_length = len(sentence)
            # Loop over subsequences from sentence
            for i in range(self.n - 1, sentence_length + 1):
                # Loop over smaller n-gram sizes
                for j in range(0, self.n + 1):
                    # Edge case - do not want to include <SOS> in n-grams
                    if i < j:
                        continue
                    ngram = [tuple(sentence[i-j:i])]
                    self._counts[j].update(ngram)

    def perplexity(self, x):
        """Measures the perplexity of a dataset.

        Args:
            x: Any finite iterable which produces sentences, where a sentence
                is any finite iterable which produces tokens. For example:
                    
                    [['First', 'Sentence'], ['Second', 'Sentence']]
        """
        sum = 0
        total = 0
        for sentence in x:
            sentence = self._pad(sentence)
            prev = []
            for i, word in enumerate(sentence):
                if i >= self.n - 1:
                    cp = self.conditional_log2_probability(word, prev)
                    sum += cp
                    total += 1
                prev.append(word)
        exponent = - sum / total
        out = 2 ** exponent
        return out

    def conditional_log2_probability(self, word, prev):
        """Computes the log probability p(word | prev) with backoff.

        Args:
            word: Word to get probability of.
            prev: Previous words in sequence to condition on.

        Returns:
            log2( p (word | prev))
        """
        n = self.n
        v = self._counts[0][()]
        ngram = tuple([*prev[-(n-1):], word])

        def _recursion(ngram, i):
            try:
                top_count = self._counts[i][ngram[(n-i):]]
            except KeyError:
                top_count = 0
            try:
                bottom_count = self._counts[i-1][ngram[(n-i):-1]]
            except KeyError:
                bottom_count = 0
            if i == 1:
                numerator = top_count + self.alpha
            else:
                np, dp = _recursion(ngram, i-1)
                numerator = top_count + self.alpha * v * np / dp
            denominator = bottom_count + self.alpha * v
            return numerator, denominator

        try:
            numerator, denominator = _recursion(ngram, self.n)
        except ZeroDivisionError:
            out = self._lbackoff
        else:
            if numerator == 0:
                out = self._lbackoff
            else:
                out = np.log2(numerator) - np.log2(denominator)

        return out

    def _pad(self, sentence):
        """Applies the relevant padding to a sentence.

        Args:
            sentence: Sentence to be padded.
        """
        out = [self._start_token] * (self.n - 1)
        out += sentence 
        out += [self._end_token]
        return out

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        assert value >= 0, "alpha must be a non-negative real number"
        self._alpha = value

