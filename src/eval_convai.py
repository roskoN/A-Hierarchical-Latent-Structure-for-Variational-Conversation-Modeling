import pickle
from datetime import datetime

from sqlitedict import SqliteDict

from configs import get_config
from data_loader import get_loader
from models import VariationalModels
from solver import Solver, VariationalSolver
from sqlitedict_compress import my_decode, my_encode
from utils import Vocab


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    config = get_config(mode="test")

    print("Loading Vocabulary...")
    vocab = Vocab()
    vocab.load(config.word2id_path, config.id2word_path)
    print(f"Vocabulary size: {vocab.vocab_size}")

    config.vocab_size = vocab.vocab_size

    data_loader = get_loader(
        sentences=load_pickle(config.sentences_path),
        conversation_length=load_pickle(config.conversation_length_path),
        sentence_length=load_pickle(config.sentence_length_path),
        vocab=vocab,
        batch_size=1,
        shuffle=False,
    )

    dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    probs_dict = SqliteDict(
        config.save_path + "/eval_{}.pkl".format(dt_string),
        encode=my_encode,
        decode=my_decode,
        journal_mode="OFF",
        autocommit=True,
    )

    if config.model in VariationalModels:
        solver = VariationalSolver(
            config, None, data_loader, vocab=vocab, is_train=False
        )
        solver.build()
        batch_probs_history = solver.evaluate_convai()
    else:
        solver = Solver(config, None, data_loader, vocab=vocab, is_train=False)
        solver.build()
        batch_probs_history = solver.evaluate_convai()

    for idx, probs_item in enumerate(batch_probs_history):
        probs_dict[str(idx)] = probs_item
