import logging
import os
import numpy as np
from gensim import utils, models
from time import time
from math import log
from scipy.sparse.linalg import svds
from scipy.linalg import sqrtm
from scipy.sparse import coo_matrix
from evaluate import evaluate_models

# Required hyper-parameters
CONTEXT_WINDOWS = (2, 5, 10)
DIMENSIONS = (50, 100, 300)
NEGATIVE_SAMPLES = (1, 5, 15)

# Specifying corpus
PATH_TO_CORPUS = "data/brown.txt"

# Path to directory
PATH_TO_MODELS = "results"

# Parameters for results
PATH_TO_RESULTS = "results"
RESULTS_PREFIX = "results"

# Word2Vec run settings
W2V_PREFIX = "w2v"
W2V_EXT = "bin"
GENERATE_W2V_MODELS = True
EVALUATE_W2V_MODELS = True
EPOCHS = 30  # TODO: Adjust Epochs

# svd run settings
SVD_PREFIX = "svd"
SVD_EXT = "txt"
GENERATE_SVD_MODELS = True
EVALUATE_SVD_MODELS = True


class MemCorpus:
    '''
        Memory-friendly corpus that can be iterated through.
    '''

    # Preprocessing
    def __iter__(self):
        for line in open(PATH_TO_CORPUS):
            # lowercases, tokenizes, and de-accents
            yield utils.simple_preprocess(line)


def create_and_save_w2v_model(
        context_window_size, dimensions, negative_samples,
):
    '''
        Creates, trains, and saves a word2vec model to a file naming
        convention based on context_window_size dimensionality.

    :param context_window_size:
    :param dimensions:
    :param negative_samples:
    :return:
    '''
    model_name = "{}_{}_{}_{}.{}".format(
        W2V_PREFIX, context_window_size, dimensions, negative_samples, W2V_EXT
    )
    print("Building Model: " + model_name)

    sentences = MemCorpus()

    # Create the model
    model = models.Word2Vec(
        # training algorithm
        sg=1,  # skipgram model
        hs=0,  # don't use hierarchical softmax (default 0)
        # hyperparameters we are comparing
        window=context_window_size,
        size=dimensions,
        negative=negative_samples,
        # additional parameters
        min_count=2,  # default 5
        sample=6e-5,  # default 0.001
        alpha=0.03,  # default 0.025
        min_alpha=0.0007,  # default 0.0001
        ns_exponent=0.75,  # default 0.75
        workers=4,  # default 3
    )
    # Build vocab
    t = time()
    model.build_vocab(sentences, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    # Train model
    t = time()
    model.train(sentences, total_examples=model.corpus_count,
                epochs=EPOCHS, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    # Lock in values since we won't be training anymore
    model.init_sims(replace=True)
    # Save model
    model.wv.save_word2vec_format(os.path.join(PATH_TO_MODELS, model_name), binary=True)


def create_ppmi_matrix(context_window):
    '''
    Constructs a ppmi matrix
    :param context_window: an int which represents the size on the context window
    :return: vocab (word-to-index dictionary) and a co-occurrence matrix (coo_matrix)
    '''
    sentences = MemCorpus()
    data = []
    row = []
    col = []
    vocab = {}

    for s in sentences:
        for i, word in enumerate(s):
            r = vocab.setdefault(word, len(vocab))
            start_idx = max(i - context_window, 0)
            end_idx = min(len(s) - 1, i + context_window)
            for j in range(start_idx, end_idx + 1):
                if i == j:
                    continue
                c = vocab.setdefault(s[j], len(vocab))
                row.append(r)
                col.append(c)
                data.append(1)

    cooccurrence_matrix = coo_matrix((data, (row, col)), shape=(len(vocab), len(vocab)))
    cooccurrence_matrix.sum_duplicates()

    # Create PPMI Matrix
    corp_size = cooccurrence_matrix.sum()
    word_counts = cooccurrence_matrix.sum(axis=0).tolist()[0]

    ppmi_data = []
    for r, c, d, in zip(cooccurrence_matrix.row, cooccurrence_matrix.col, cooccurrence_matrix.data):
        ppmi_entry = max(0., log((d * corp_size)/(word_counts[r] * word_counts[c])))
        ppmi_data.append(ppmi_entry)
    ppmi_matrix = coo_matrix((ppmi_data, (cooccurrence_matrix.row, cooccurrence_matrix.col)),
                             shape=(len(vocab), len(vocab)))
    return vocab, ppmi_matrix


def create_and_save_svd_model(context_window_size, dimensions, ppmi_matrix, vocabulary):
    '''
    Creates word embeddings for an svd model and saves them as a txt file.
    Saved file is space delimited txt file with one word vector per line.
    :param context_window_size:
    :param dimensions:
    :param ppmi_matrix:
    :param vocabulary:
    :return:
    '''
    model_name = "{}_{}_{}.{}".format(
        SVD_PREFIX, context_window_size, dimensions, SVD_EXT
    )
    print("Building model: " + model_name)

    # Calculate the word embeddings
    u, s, vt = svds(
        ppmi_matrix,
        k=dimensions,
        maxiter=None,  # default None
    )
    w = np.matmul(u, sqrtm(np.diag(s)))
    word_embeddings = {k: w[v] for (k, v) in vocab.items()}

    # Save model
    with open(os.path.join(PATH_TO_MODELS, model_name), "w") as f:
        for (k, vec) in word_embeddings.items():
            f.write(k + " " + " ".join([str(x) for x in vec]) + "\r\n")
        f.close()


if __name__ == '__main__':

    if not os.path.isdir(PATH_TO_MODELS):
        os.mkdir(PATH_TO_MODELS)

    if not os.path.isdir(PATH_TO_RESULTS):
        os.mkdir(PATH_TO_RESULTS)

    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s",
                        datefmt='%H:%M:%S', level=logging.INFO)

    if GENERATE_W2V_MODELS:
        # Build the 27 word2vec models
        t = time()
        for cw in CONTEXT_WINDOWS:
            for d in DIMENSIONS:
                for ns in NEGATIVE_SAMPLES:
                    create_and_save_w2v_model(cw, d, ns)
        print('Time to build all w2v models: {} mins'.format(round((time() - t) / 60, 2)))

    if GENERATE_SVD_MODELS:
        # Build the 9 svd models
        t = time()
        for cw in CONTEXT_WINDOWS:
            vocab, com = create_ppmi_matrix(cw)
            for d in DIMENSIONS:
                create_and_save_svd_model(cw, d, com, vocab)
        print('Time to build all svd models: {} mins'.format(round((time() - t) / 60, 2)))

    if EVALUATE_SVD_MODELS or EVALUATE_W2V_MODELS:
        t = time()
        files = []
        hyper_params = []

        if EVALUATE_W2V_MODELS:
            for cw in CONTEXT_WINDOWS:
                for d in DIMENSIONS:
                    for ns in NEGATIVE_SAMPLES:
                        model_name = "{}_{}_{}_{}.{}".format(W2V_PREFIX, cw, d, ns, W2V_EXT)
                        model_path = os.path.join(PATH_TO_MODELS, model_name)
                        files.append(model_path)
                        hyper_params.append(("W2V", cw, d, ns))

        if EVALUATE_SVD_MODELS:
            for cw in CONTEXT_WINDOWS:
                for d in DIMENSIONS:
                    model_name = "{}_{}_{}.{}".format(SVD_PREFIX, cw, d, SVD_EXT)
                    model_path = os.path.join(PATH_TO_MODELS, model_name)
                    files.append(model_path)
                    hyper_params.append(("SVD", cw, d, "N/A"))

        # Calculate scores
        scores = evaluate_models(files)

        # Create table
        table = ['ALGORITHM, WINDOW, DIM, NS, WORDSIM, MSR, BATS TOTAL, BATS IM, BATS DM, BATS ES, BATS LS']
        for hp, score in zip(hyper_params, scores):
            table_row = ",".join((
                str(hp[0]),
                str(hp[1]),
                str(hp[2]),
                str(hp[3]),
                str(score["wordsim"][0]),
                str(score["msr"]),
                str(score["bats"]["total"]),
                str(score["bats"]["inflectional_morphology"]),
                str(score["bats"]["derivational_morphology"]),
                str(score["bats"]["encyclopedic_semantics"]),
                str(score["bats"]["lexicographic_semantics"])
            ))
            table.append(table_row)
        # Save table to file
        result_file = os.path.join(PATH_TO_RESULTS, RESULTS_PREFIX + ".csv")
        with open(result_file, "w") as f:
            f.writelines(row + "\n" for row in table)
            f.close()
        print('Time to evaluate all models: {} mins'.format(round((time() - t) / 60, 2)))
