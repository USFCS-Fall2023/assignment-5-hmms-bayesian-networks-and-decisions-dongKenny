from HMM import HMM
from alarm import alarm_queries
from carnet import car_queries


def main():
    # hmm = HMM()
    # hmm.load('partofspeech.browntags.trained')
    # print(hmm.generate(12))
    #
    # alarm_queries()
    car_queries()
    print('Probability unchanged for the radio working given the battery is working if the we discover the car has gas')
    print('Probability affected for the ignition failing given the car does not move and we observe there is no gas')


if __name__ == '__main__':
    main()
