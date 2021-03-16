import gensim.models
import logging
import os
from evaluate import evaluate_models
from gensim import utils
from time import time

# Required hyper-parameters
CONTEXT_WINDOWS = (2, 5, 10)
DIMENSIONS = (50, 100, 300)
NEGATIVE_SAMPLES = (1, 5, 15)

# For performance tuning
EPOCHS = 2  # TODO: Adjust Epochs

# Specifying corpus
PATH_TO_CORPUS = "data/brown.txt"

# Path to directory
PATH_TO_MODELS = "models/"

# Parameters for results
PATH_TO_RESULTS = "results/"
RESULTS_PREFIX = "results"

# Word2Vec run settings
W2V_PREFIX = "w2v"
W2V_POSTFIX = "model"
GENERATE_W2V_MODELS = True
EVALUATE_W2V_MODELS = True


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
        context_window_size, dimsensions, negative_samples,
        min_count=1,
        sample=6e-5,
        alpha=0.03,
        min_alpha=0.0007,
        workers=4
):
    '''
        Creates, trains, and saves a word2vec model to a file naming
        convention based on context_window_size dimensionality.

    :param context_window_size:
    :param dimsensions:
    :param negative_samples:
    :param min_count:
    :param sample:
    :param alpha:
    :param min_alpha:
    :param workers:
    :return:
    '''
    model_name = "{}_{}_{}_{}.{}".format(
        W2V_PREFIX, context_window_size, dimsensions, negative_samples, W2V_POSTFIX
    )
    print("Building Model: " + model_name)

    # Create the model
    model = gensim.models.Word2Vec(
        window=context_window_size,
        size=dimsensions,
        negative=negative_samples,
        # min_count=min_count,
        # sample=sample,
        # alpha=alpha,
        # min_alpha=min_alpha,
        workers=workers
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
    model.wv.save(os.path.join(PATH_TO_MODELS, model_name))


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

    sentences = MemCorpus()

    if GENERATE_W2V_MODELS:
        # Build the 27 word2vec models
        t = time()
        for cw in CONTEXT_WINDOWS:
            for d in DIMENSIONS:
                for ns in NEGATIVE_SAMPLES:
                    create_and_save_w2v_model(cw, d, ns)
        print('Time to build all w2v models: {} mins'.format(round((time() - t) / 60, 2)))

    if EVALUATE_W2V_MODELS:
        # Evaluate the 27 word2vec models
        t = time()
        files = []
        hyper_params = []
        for cw in CONTEXT_WINDOWS:
            for d in DIMENSIONS:
                for ns in NEGATIVE_SAMPLES:
                    model_path = "{}{}_{}_{}_{}.{}".format(
                        PATH_TO_MODELS, W2V_PREFIX, cw, d, ns, W2V_POSTFIX
                    )
                    files.append(model_path)
                    hyper_params.append((cw, d, ns))
        scores = evaluate_models(files)

        table = ['ALGORITHM, WINDOW, DIM, NS, WORDSIM, MSR, BATS TOTAL, BATS IM, BATS DM, BATS ES, BATS LS']
        for hp, score in zip(hyper_params, scores):
            table_row = ",".join((
                "Word2Vec",
                str(hp[0]),
                str(hp[1]),
                str(hp[2]),
                str(score["wordsim"][0]),
                str(score["msr"]),
                str(score["bats"]["total"]),
                str(score["bats"]["inflectional_morphology"]),
                str(score["bats"]["derivational_morphology"]),
                str(score["bats"]["encyclopedic_semantics"]),
                str(score["bats"]["lexicographic_semantics"])
            ))
            table.append(table_row)
        result_file = os.path.join(PATH_TO_RESULTS, RESULTS_PREFIX + ".csv")
        with open(result_file, "w") as f:
            f.writelines(row + "\n" for row in table)
            f.close()
        print('Time to evaluate all w2v models: {} mins'.format(round((time() - t) / 60, 2)))
