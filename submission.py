from HMM import HMM


def main():
    hmm = HMM()
    hmm.load('partofspeech.browntags.trained')
    print(hmm.generate(12))


if __name__ == '__main__':
    main()
