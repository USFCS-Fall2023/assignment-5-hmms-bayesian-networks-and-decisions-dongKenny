from HMM import HMM
from alarm import alarm_queries
from carnet import car_queries


def main():
    hmm = HMM()
    hmm.load('partofspeech.browntags.trained')
    print(hmm.generate(12))

    # alarm_queries()
    # car_queries()


if __name__ == '__main__':
    main()
