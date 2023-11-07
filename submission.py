import argparse
import sys

from HMM import HMM, Observation
from alarm import alarm_queries
from carnet import car_queries


def parse_args():
    parser = argparse.ArgumentParser(prog='HMM and Bayes')
    parser.add_argument('basename', help='.trans and .emit basename')
    parser.add_argument('--generate', help='n of words to generate')
    parser.add_argument('--forward', help='Name of observations file')
    parser.add_argument('--viterbi', help='Name of observations file')

    return vars(parser.parse_args(sys.argv[1:]))


def hmm_demo(basename, n, forward, viterbi):
    hmm = HMM()
    hmm.load(basename)

    # Generate (prints the states and emissions)
    print(hmm.generate(n))

    # Forward (prints matrix and most likely state)
    with open(forward, 'r') as reader:
        lines = reader.readlines()
        hmm.forward(Observation(stateseq=[], outputseq=lines[0].split()))


def main():
    args = parse_args()

    basename = args['basename']
    generate_n = args['generate'] if args['generate'] else 0
    forward_path = args['forward'] if args['forward'] else None
    viterbi_path = args['viterbi'] if args['viterbi'] else None

    hmm_demo(basename, int(generate_n), forward_path, viterbi_path)
    # alarm_queries()
    # car_queries()


if __name__ == '__main__':
    main()
