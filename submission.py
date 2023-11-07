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
    parser.add_argument('--tagged', help='Name of tagged observations file')

    return vars(parser.parse_args(sys.argv[1:]))


def hmm_demo(basename, n, forward, viterbi, tagged):
    hmm = HMM()
    hmm.load(basename)

    # Generate (prints the states and emissions)
    if n:
        print(hmm.generate(n))

    # Forward (prints matrix and most likely state)
    if forward:
        with open(forward, 'r') as reader:
            for line in reader.readlines():
                if line != '\n':
                    hmm.forward(Observation(stateseq=[], outputseq=line.split()))

    # Viterbi (prints back matrix and state path)
    if viterbi:
        results = []
        with open(viterbi, 'r') as reader:
            for line in reader.readlines():
                if line != '\n':
                    results.append(hmm.viterbi(Observation(stateseq=[], outputseq=line.split())))

        if tagged:
            correct = 0
            with open(tagged, 'r') as reader:
                for idx, line in enumerate(reader.readlines()):
                    if idx % 2 == 0:
                        result = ' '.join(results[idx//2])
                        if result == line.strip():
                            correct += 1
                        else:
                            print(f'Missed: "{result}" vs {line.strip()}')
                print(f'{correct} correct out of {len(results)}\n')


def main():
    args = parse_args()

    basename = args['basename']
    generate_n = args['generate'] if args['generate'] else 0
    forward_path = args['forward'] if args['forward'] else None
    viterbi_path = args['viterbi'] if args['viterbi'] else None
    tagged_path = args['tagged'] if args['tagged'] else None

    hmm_demo(basename, int(generate_n), forward_path, viterbi_path, tagged_path)
    alarm_queries()
    car_queries()


if __name__ == '__main__':
    main()
