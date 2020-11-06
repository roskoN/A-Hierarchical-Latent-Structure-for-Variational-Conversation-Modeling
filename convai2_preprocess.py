# Preprocess cornell movie dialogs dataset

import argparse
import json
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm

from src.utils import EOS_TOKEN, PAD_TOKEN, Tokenizer, Vocab

project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath("datasets/")
convai2_dir = datasets_dir.joinpath("convai2/")

# Tokenizer
tokenizer = Tokenizer("spacy")


def prepare_convai2_data():
    """Download and unpack dialogs"""

    url = "http://convai.io/data/summer_wild_evaluation_dialogs.json"
    file_path = convai2_dir.joinpath("convai2.json")

    if not datasets_dir.exists():
        datasets_dir.mkdir()

    # Prepare Dialog data
    if not convai2_dir.exists():
        convai2_dir.mkdir()
    if not file_path.exists():
        print(f"Downloading {url} to {file_path}")
        urlretrieve(url, file_path)
        print(f"Successfully downloaded {file_path}")

    else:
        print("ConvAI2 Data prepared!")

    return file_path


def loadConversations(
    fileName,
):
    """
    Args:
        fileName (str): file to load
        field (set<str>): fields to extract
    Return:
        dict<dict<str>>: the extracted fields for each line
    """
    conversations = []

    with open(fileName, "rt") as f:
        conv_file_data = json.load(f)

        for dialogue in conv_file_data:
            thread = dialogue["dialog"]

            if len(thread) < 3:
                continue

            # Extract fields
            convObj = {}
            convObj["lines"] = []
            for utterance in thread:
                convObj["lines"].append(utterance)

            conversations.append(convObj)

    return conversations


def train_valid_test_split_by_conversation(conversations, split_ratio=[0.8, 0.1, 0.1]):
    return conversations, conversations, conversations


def tokenize_conversation(lines):
    sentence_list = [tokenizer(line["text"]) for line in lines]
    return sentence_list


def pad_sentences(conversations, max_sentence_length=30, max_conversation_length=10):
    def pad_tokens(tokens, max_sentence_length=max_sentence_length):
        n_valid_tokens = len(tokens)
        if n_valid_tokens > max_sentence_length - 1:
            tokens = tokens[: max_sentence_length - 1]
        n_pad = max_sentence_length - n_valid_tokens - 1
        tokens = tokens + [EOS_TOKEN] + [PAD_TOKEN] * n_pad
        return tokens

    def pad_conversation(conversation):
        conversation = [pad_tokens(sentence) for sentence in conversation]
        return conversation

    all_padded_sentences = []
    all_sentence_length = []

    for conversation in conversations:
        if len(conversation) > max_conversation_length:
            conversation = conversation[:max_conversation_length]
        sentence_length = [
            min(len(sentence) + 1, max_sentence_length)  # +1 for EOS token
            for sentence in conversation
        ]
        all_sentence_length.append(sentence_length)

        sentences = pad_conversation(conversation)
        all_padded_sentences.append(sentences)

    sentences = all_padded_sentences
    sentence_length = all_sentence_length
    return sentences, sentence_length


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Maximum valid length of sentence
    # => SOS/EOS will surround sentence (EOS for source / SOS for target)
    # => maximum length of tensor = max_sentence_length + 1
    parser.add_argument("-s", "--max_sentence_length", type=int, default=30)
    parser.add_argument("-c", "--max_conversation_length", type=int, default=10)

    # Split Ratio
    split_ratio = [0.8, 0.1, 0.1]

    # Vocabulary
    parser.add_argument("--max_vocab_size", type=int, default=20000)
    parser.add_argument("--min_vocab_frequency", type=int, default=5)

    # Multiprocess
    parser.add_argument("--n_workers", type=int, default=os.cpu_count())

    args = parser.parse_args()

    max_sent_len = args.max_sentence_length
    max_conv_len = args.max_conversation_length
    max_vocab_size = args.max_vocab_size
    min_freq = args.min_vocab_frequency
    n_workers = args.n_workers

    # Download and extract dialogs if necessary.
    file_path = prepare_convai2_data()

    print("Loading conversations...")
    conversations = loadConversations(
        file_path
    )
    print("Number of conversations:", len(conversations))
    print("Train/Valid/Test Split")
    # train, valid, test = train_valid_test_split_by_movie(conversations, split_ratio)
    train, valid, test = train_valid_test_split_by_conversation(
        conversations, split_ratio
    )

    def to_pickle(obj, path):
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    for split_type, conv_objects in [
        ("test", test),
    ]:
        print(f"Processing {split_type} dataset...")
        split_data_dir = convai2_dir.joinpath(split_type)
        split_data_dir.mkdir(exist_ok=True)

        print(f"Tokenize.. (n_workers={n_workers})")

        def _tokenize_conversation(conv):
            return tokenize_conversation(conv["lines"])

        with Pool(n_workers) as pool:
            conversations = list(
                tqdm(
                    pool.imap(_tokenize_conversation, conv_objects),
                    total=len(conv_objects),
                )
            )

        conversation_length = [
            min(len(conv["lines"]), max_conv_len) for conv in conv_objects
        ]

        sentences, sentence_length = pad_sentences(
            conversations,
            max_sentence_length=max_sent_len,
            max_conversation_length=max_conv_len,
        )

        print("Saving preprocessed data at", split_data_dir)
        to_pickle(
            conversation_length, split_data_dir.joinpath("conversation_length.pkl")
        )
        to_pickle(sentences, split_data_dir.joinpath("sentences.pkl"))
        to_pickle(sentence_length, split_data_dir.joinpath("sentence_length.pkl"))

        if split_type == "train":

            print("Save Vocabulary...")
            vocab = Vocab(tokenizer)
            vocab.add_dataframe(conversations)
            vocab.update(max_size=max_vocab_size, min_freq=min_freq)

            print("Vocabulary size: ", len(vocab))
            vocab.pickle(
                convai2_dir.joinpath("word2id.pkl"), convai2_dir.joinpath("id2word.pkl")
            )

    print("Done!")
