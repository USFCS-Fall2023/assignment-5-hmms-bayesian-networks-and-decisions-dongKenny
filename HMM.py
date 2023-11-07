import random
import argparse
import codecs
import os
import numpy


# observations
import numpy as np


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

            outputs.append(emission)

            state = transition
            states.append(state)

        return Observation(states, outputs)

    def forward(self, observation, show_matrix=False):
        m = [[0. for _ in range(len(observation.outputseq) + 1)] for _ in range(len(self.transitions.keys()))]
        states = {s: i for i, s in enumerate(list(self.transitions.keys()))}
        state_idx = {v: k for k, v in states.items()}
        del states['#']

        m[0][0] = 1.0
        for s in states:
            if observation.outputseq[0] in self.emissions[s]:
                m[states[s]][1] = self.transitions['#'][s] * self.emissions[s][observation.outputseq[0]]

        for i in range(2, len(observation.outputseq) + 1):
            for s in states:
                total = 0
                for s_2 in states:
                    if observation.outputseq[i - 1] in self.emissions[s]:
                        total += m[states[s_2]][i-1] * self.transitions[s_2][s] * self.emissions[s][observation.outputseq[i-1]]
                m[states[s]][i] = total

        if show_matrix:
            for row in m:
                for col in row:
                    print(f'{col: .9f}', end='')
                print()

        values = [row[-1] for row in m]
        idx_max = np.argmax(values)
        print(f'Most likely state is {state_idx[idx_max]}\n')
        return m

    def viterbi(self, observation, show_matrix=False):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        m = [[0. for _ in range(len(observation.outputseq) + 1)] for _ in range(len(self.transitions.keys()))]
        back = [[0 for _ in range(len(observation.outputseq) + 1)] for _ in range(len(self.transitions.keys()))]
        states = {s: i for i, s in enumerate(list(self.transitions.keys()))}
        state_idx = {v: k for k, v in states.items()}
        del states['#']

        # Init forward matrix and backpointers to '#'
        m[0][0] = 1.0
        back[0][0] = -1
        for s in states:
            if observation.outputseq[0] in self.emissions[s]:
                m[states[s]][1] = self.transitions['#'][s] * self.emissions[s][observation.outputseq[0]]
                back[states[s]][1] = 0

        # Forward + Viterbi
        for i in range(2, len(observation.outputseq) + 1):
            for s in states:
                total = 0
                values = []
                for s_2 in states:
                    if observation.outputseq[i - 1] in self.emissions[s]:
                        step = m[states[s_2]][i - 1] * self.transitions[s_2][s] * self.emissions[s][observation.outputseq[i - 1]]
                        total += step
                        values.append(step)
                if values:
                    best_step_idx = np.argmax(values) + 1
                    back[states[s]][i] = int(best_step_idx)
                m[states[s]][i] = total

        values = [row[-1] for row in m]
        idx_max = np.argmax(values)

        if show_matrix:
            for row in back:
                for col in row:
                    print(f'{col: 3d}', end='')
                print()

        curr = idx_max
        path = []
        for i in range(len(back[0]) - 1, 0, -1):
            path.append(curr)
            curr = back[curr][i]

        path = [state_idx[i] for i in reversed(path)]
        return path
