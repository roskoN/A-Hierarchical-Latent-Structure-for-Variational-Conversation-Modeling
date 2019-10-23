from solver import Solver, VariationalSolver
from data_loader import get_loader
from configs import get_config
from utils import Vocab
import pickle
from models import VariationalModels
from datetime import datetime


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    config = get_config(mode='test')

    print('Loading Vocabulary...')
    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f'Vocabulary size: {vocab.vocab_size}')

    config.vocab_size = vocab.vocab_size

    data_loader = get_loader(
        sentences=load_pickle(config.sentences_path),
        conversation_length=load_pickle(config.conversation_length_path),
        sentence_length=load_pickle(config.sentence_length_path),
        vocab=vocab,
        batch_size=1,
        shuffle=False)

    if config.model in VariationalModels:
        solver = VariationalSolver(
            config, None, data_loader, vocab=vocab, is_train=False)
        solver.build()
        recon_loss_history, kl_div_history, bow_loss_history = solver.evaluate_convai()

        losses = {
            'recon': recon_loss_history,
            'kl_div': kl_div_history,
            'bow': bow_loss_history
        }

        dt_string = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")

        for loss_name, loss_data in losses.item():
            with open(config.save_path + '/{}_{}.pkl'.format(loss_name, dt_string), 'rb') as fout:
                pickle.dump(file=fout, obj=loss_data)
    else:
        solver = Solver(config, None, data_loader, vocab=vocab, is_train=False)
        solver.build()
        batch_loss_history = solver.evaluate_convai()

        dt_string = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
        with open(config.save_path + '/eval_{}.pkl'.format(dt_string), 'rb') as fout:
            pickle.dump(file=fout, obj=batch_loss_history)
