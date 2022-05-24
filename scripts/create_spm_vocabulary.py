import argparse

from data_processing.vocabulary.spm_vocabulary import SPMVocabularyTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', help='Path to directory with .gz files to train on, e.g. path/to/dataset/train/')
    parser.add_argument('vocab_size', type=int, default=16000, help='Size of generated vocabulary')
    parser.add_argument('model_type', default='bpe', help='Type of generated model, e.g. bpe, unigram, char, word')
    parser.add_argument('model_prefix', help='Output model name prefix. '
                                             'For example, for path/to/vocab/my_model, path/to/vocab/my_model.model and'
                                             ' path/to/vocab/my_model.vocab will be generated')

    args = parser.parse_args()

    spm_trainer = SPMVocabularyTrainer(**vars(args))
    spm_trainer.train()


if __name__ == '__main__':
    main()
