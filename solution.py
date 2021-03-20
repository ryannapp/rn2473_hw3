import gensim.models
import logging
import os
import pickle
from evaluate import evaluate_models
from gensim import utils
from time import time
from numpy import array
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import _sparse_random_matrix
from scipy.sparse import coo_matrix

# Required hyper-parameters
CONTEXT_WINDOWS = (2, 5, 10)
DIMENSIONS = (50, 100, 300)
NEGATIVE_SAMPLES = (1, 5, 15)

# Specifying corpus
PATH_TO_CORPUS = "data/brown.txt"

# Path to directory
PATH_TO_MODELS = "models/"

# Parameters for results
PATH_TO_RESULTS = "results/"
RESULTS_PREFIX = "results"

# Word2Vec run settings
W2V_PREFIX = "w2v"
W2V_EXT = "bin"
GENERATE_W2V_MODELS = False
EVALUATE_W2V_MODELS = False
EPOCHS = 30  # TODO: Adjust Epochs

# svd run settings
SVD_PREFIX = "svd"
SVD_EXT = "bin"
GENERATE_SVD_MODELS = True
EVALUATE_SVD_MODELS = False


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
    model = gensim.models.Word2Vec(
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


def create_co_matrix(context_window):
    '''
    Constructs a co-occurrence matrix
    :param context_window: an int which represents the size on the context window
    :return: vocab (word-to-index dictionary) and a co-occurrence matrix (coo_matrix)
    '''
    sentences = MemCorpus()
    data = []
    rows = []
    cols = []
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
                rows.append(r)
                cols.append(c)
                data.append(1)

    cooccurrence_matrix = coo_matrix((data, (rows, cols)), shape=(len(vocab), len(vocab)))
    return vocab, cooccurrence_matrix


def create_and_save_svd_model(context_window_size, dimensions, co_occurrence_matrix, vocabulary):
    model_name = "{}_{}_{}.{}".format(
        SVD_PREFIX, context_window_size, dimensions, SVD_EXT
    )
    print("Building model: " + model_name)

    # Calculate the co-occurence matrix
    m = _sparse_random_matrix(500, 500)  # ppmi matrix #TODO: UGHHHHHHHHHHHHH

    # Create the model
    svd = TruncatedSVD(
        n_components=dimensions,  # default 2
        algorithm="randomized",  # default "randomized"
        n_iter=5,  # default 5 #TODO: figure out number for best performance
        random_state=42,  # default None
    )
    svd.fit(m)

    # Save model
    with open(os.path.join(PATH_TO_MODELS, model_name), "wb") as f:
        pickle.dump(svd, f)


###### TESTING ######
# vector = model.wv['fruit']
# print(vector)
# similar = model.wv.most_similar(positive=["wallet"])
# print(similar)
# dissimilar = model.wv.doesnt_match(['apple', 'house', 'fire'])
# print(dissimilar)
# analogy_1 = model.wv.most_similar(positive=["actress", "man"], negative=["woman"], topn=3)
# print(analogy_1)
# analogy_2 = model.wv.most_similar(positive=["Paris", "Germany"], negative=["France"], topn=3)
# print(analogy_2)
# analogy_3 = model.wv.most_similar(positive=["puppy", "cat"], negative=["dog"], topn=3)
# print(analogy_3)

def eval_model(path_to_model):
    # model = load_model(path_to_model)
    wordsim_corr, wordsim_pval = evaluate_models()


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
            vocab, com = create_co_matrix(cw)
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
