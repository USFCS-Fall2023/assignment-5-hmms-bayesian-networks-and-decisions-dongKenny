from HMM import HMM, Observation
from alarm import alarm_queries
from carnet import car_queries


def main():
    hmm = HMM()
    # hmm.load('partofspeech.browntags.trained')
    hmm.load('cat')

    with open('test.obs', 'r') as reader:
        lines = reader.readlines()
        hmm.forward(Observation(stateseq=[], outputseq=lines[0].split()))
    # alarm_queries()
    # car_queries()


if __name__ == '__main__':
    main()
