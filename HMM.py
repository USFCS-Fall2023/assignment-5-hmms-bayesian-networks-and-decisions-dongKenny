import random
import argparse
import codecs
import os
import numpy


# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq  # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq) + '\n ' + ' '.join(self.outputseq) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


# hmm model
class HMM:
    def __init__(self, transitions=None, emissions=None):
        """creates a model from transition and emission probabilities"""

        if not transitions:
            transitions = {}

        if not emissions:
            emissions = {}

        self.transitions = transitions
        self.emissions = emissions

    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""

        def read_probabilities(file_ending, data):
            with open(f'{basename}.{file_ending}', 'r') as reader:
                for line in reader.readlines():
                    key, value, probability = line.split()
                    if key not in data:
                        data[key] = {}
                    data[key].update({value: float(probability)})

        read_probabilities('trans', self.transitions)
        read_probabilities('emit', self.emissions)

    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        state = '#'
        states = []
        outputs = []
        for _ in range(n):
            transition = random.choices(list(self.transitions[state].keys()), weights=self.transitions[state].values(), k=1)[0]
            emission = random.choices(list(self.emissions[transition].keys()), weights=self.emissions[transition].values(), k=1)[0]

            states.append(state)
            outputs.append(emission)

            state = transition

        return Observation(states, outputs)


    # you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    # determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
