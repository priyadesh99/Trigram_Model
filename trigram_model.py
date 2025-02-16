import sys
from collections import defaultdict
import math
import random
import os
import os.path
import numpy as np
"""
COMS W4705 - Natural Language Processing - Spring 2025
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile,'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1
    """
    # pad the sequence with n-1 START tokns, then iterate and get ngrams
    sequence = ['START'] * (n-1) + sequence + ['STOP']

    n_grams = []
    for i in range(len(sequence) - n + 1):
      n_grams.append(tuple(sequence[i:i+n]))

    return n_grams


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        self.total_unigrams = 0


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts.
        """

        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        for sentence in corpus:
            for unigram in get_ngrams(sentence, 1):
                self.unigramcounts[unigram] += 1

            for bigram in get_ngrams(sentence, 2):
                self.bigramcounts[bigram] += 1

            for trigram in get_ngrams(sentence, 3):
                self.trigramcounts[trigram] += 1

                if trigram[:2] == ('START', 'START'):
                    # to compute trigram probs of type ('START', 'START', 'xxx')
                    self.bigramcounts[('START', 'START')] += 1
       
        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if self.bigramcounts[(trigram[0], trigram[1])] > 0:
            return self.trigramcounts[trigram]/self.bigramcounts[trigram[:2]]

        return 1/len(self.lexicon)


    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if self.unigramcounts[(bigram[0],)] > 0:
            return self.bigramcounts[bigram]/self.unigramcounts[bigram[:1]]

        return 1/len(self.lexicon)

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once,
        # store in the TrigramModel instance, and then re-use it.

        if self.total_unigrams == 0:
            self.total_unigrams = sum(self.unigramcounts.values()) - self.unigramcounts[('START',)] + self.unigramcounts[('STOP',)]

        return self.unigramcounts[unigram] / (self.total_unigrams)
    
    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        current_trigram = (None, 'START', 'START')
        sentence = list()
        i = 0

        while current_trigram[2] != 'STOP' and i < t:
            word_1 = current_trigram[1]
            word_2 = current_trigram[2]
           
            candidates = [trigram for trigram in self.trigramcounts.keys() if trigram[:2] == (word_1, word_2)]
            probabilities = [self.raw_trigram_probability(trigram) for trigram in candidates]

            generated_word = np.random.choice([candidate[2] for candidate in candidates], 1, p=probabilities)[0]
            current_trigram = (word_1, word_2, generated_word)
            sentence.append(generated_word)
            i += 1

        return sentence    


    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation).
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
   
        p1 = self.raw_unigram_probability(trigram[2:])
        p2 = self.raw_bigram_probability(trigram[1:])
        p3 = self.raw_trigram_probability(trigram)

        return lambda1 * p1 + lambda2 * p2 + lambda3 * p3

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        log_prob = 0.0

        for trigram in trigrams:
          prob = self.smoothed_trigram_probability(trigram)
          log_prob += math.log2(prob) 
        
        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6)
        Returns the log probability of an entire sequence.
        """
        total_log_probs = 0.0
        total_words = 0

        for sentence in corpus:
          total_log_probs += self.sentence_logprob(sentence)
          total_words += len(sentence)

        return 2 ** (-total_log_probs / total_words)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
        if pp1 < pp2:
            correct += 1  
        total += 1

    for f in os.listdir(testdir2):
        pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))

        if pp2 < pp1:
            correct += 1  # Model 2 is correct for this "low" test file
        total += 1

    return correct / total



if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    # put test code here...
    # or run the script from the command line with
    # $ python -i trigram_model.py [corpus_file]
    # >>>
    #
    # you can then call methods on the model instance in the interactive
    # Python prompt.


    # Testing perplexity:
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # Essay scoring experiment:
    acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', 'test_high', 'test_low')
    print(acc)

