import torch
import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
from process import load_msr


def get_word_vectors(text, model, tokenizer):
    '''
    - Gets word vectors for each token in a text string.
    :param text: Text is a word or sentence
    :param model:
    :param tokenizer:
    :return: a dictionary of tokens to word vectors
    '''
    tokens = tokenizer.tokenize("[CLS] " + text + " [SEP]")
    tokenized_ids = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([tokenized_ids])
    embeddings = []
    with torch.no_grad():
        outputs = model(tokens_tensor)
        last_hidden_state = outputs[0]
        embeddings = last_hidden_state  # TODO: Do summing over hidden states instead
    # Remove dimension 1, the "batches"
    embeddings = torch.squeeze(embeddings, dim=0)
    return {k: v for (k, v) in zip(tokens, embeddings)}


def eval_wordsim(model, tokenizer, f='data/wordsim353/combined.tab'):
    '''

    :param model:
    :param tokenizer:
    :param f:
    :return:
    '''
    sim = []
    pred = []

    for line in open(f, 'r').readlines():
        splits = line.split('\t')
        w1 = splits[0].lower()
        w2 = splits[1].lower()
        if w1 in tokenizer.vocab and w2 in tokenizer.vocab:
            sim.append(float(splits[2]))
            v1 = get_word_vectors(w1, model, tokenizer)[w1]
            v2 = get_word_vectors(w2, model, tokenizer)[w2]
            pred.append(np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)))

    return spearmanr(sim, pred)

def eval_msr(model):
    '''
        Evaluates a trained embedding model on the MSR paraphrase task using
        logistic regression over cosine similarity scores.
    '''
    X_tr, y_tr = load_msr('data/msr/msr_paraphrase_train.txt')
    X_test, y_test = load_msr('data/msr/msr_paraphrase_test.txt')

    train = [[np.sum(get_word_vectors(" ".join(ss[0]), model, tokenizer).values(), axis=0),
              np.sum(get_word_vectors(" ".join(ss[1]), model, tokenizer).values(), axis=0)] for ss in X_tr]
    test = [[np.sum(get_word_vectors(" ".join(ss[0]), model, tokenizer).values(), axis=0),
             np.sum(get_word_vectors(" ".join(ss[1]), model, tokenizer).values(), axis=0)] for ss in X_test]

    tr_cos = np.array([1 - cosine(x[0], x[1]) for x in train]).reshape(-1, 1)
    test_cos = np.array([1 - cosine(x[0], x[1]) for x in test]).reshape(-1, 1)

    lr = LogisticRegression(class_weight='balanced', solver='liblinear')
    lr.fit(tr_cos, y_tr)
    preds = lr.predict(test_cos)

    return accuracy_score(y_test, preds)

if __name__ == "__main__":

    print('[evaluate] Loading model...')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    model.eval()

    # print('[evaluate] WordSim353 correlation:')
    # ws = eval_wordsim(model, tokenizer)
    # print(ws)

    # print('[evaluate] BATS accuracies:')
    # bats = eval_bats(model, matrix, vocab, indices)
    # print(bats)

    print('[evaluate] MSR accuracy:')
    msr = eval_msr(model)
    print(msr)
