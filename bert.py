import torch
import numpy as np
from scipy.stats import spearmanr

from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine
from numpy.linalg import norm

import os
import random

from process import load_msr

def get_one_word_vector(word, model, tokenizer):
    '''

    :param word:
    :param model:
    :param tokenizer:
    :return: A numpy array of a single word vector
    '''
    tokens = tokenizer.tokenize("[CLS] " + word + " [SEP]")
    tokenized_ids = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([tokenized_ids])
    embeddings = []
    with torch.no_grad():
        outputs = model(tokens_tensor)
        last_hidden_state = outputs[0]
        embeddings = last_hidden_state  # TODO: Do summing over hidden states instead
    # Remove dimension 1, the "batches"
    embeddings = torch.squeeze(embeddings, dim=0)
    return embeddings[1].numpy()

def get_word_vector_dict(text, model, tokenizer):
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
        Evaluates a pre-trained bert model on WordSim353 using cosine
        similarity and Spearman's rho. Returns a tuple containing
        (correlation, p-value).
    '''
    sim = []
    pred = []

    for line in open(f, 'r').readlines():
        splits = line.split('\t')
        w1 = splits[0].lower()
        w2 = splits[1].lower()
        if w1 in tokenizer.vocab and w2 in tokenizer.vocab:
            sim.append(float(splits[2]))
            v1 = get_one_word_vector(w1, model, tokenizer)
            v2 = get_one_word_vector(w2, model, tokenizer)
            pred.append(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    return spearmanr(sim, pred)


def eval_msr(model):
    '''
    Evaluates a pre-trained bert model on the MSR paraphrase task using
    logistic regression over cosine similarity scores.
    '''
    X_tr, y_tr = load_msr('data/msr/msr_paraphrase_train.txt')
    X_test, y_test = load_msr('data/msr/msr_paraphrase_test.txt')

    train = [[np.sum([x.numpy() for x in get_word_vector_dict(" ".join(ss[0]), model, tokenizer).values()], axis=0),
              np.sum([x.numpy() for x in get_word_vector_dict(" ".join(ss[1]), model, tokenizer).values()], axis=0)] for ss in X_tr]
    test = [[np.sum([x.numpy() for x in get_word_vector_dict(" ".join(ss[0]), model, tokenizer).values()], axis=0),
             np.sum([x.numpy() for x in get_word_vector_dict(" ".join(ss[1]), model, tokenizer).values()], axis=0)] for ss in X_test]

    tr_cos = np.array([1 - cosine(x[0], x[1]) for x in train]).reshape(-1, 1)
    test_cos = np.array([1 - cosine(x[0], x[1]) for x in test]).reshape(-1, 1)

    lr = LogisticRegression(class_weight='balanced', solver='liblinear')
    lr.fit(tr_cos, y_tr)
    preds = lr.predict(test_cos)

    return accuracy_score(y_test, preds)


def eval_bats_file(model, matrix, vocab, indices, f, repeat=False, multi=0):
    pairs = [line.strip().split() for line in open(f, 'r').readlines()]

    # discard pairs that are not in our vocabulary
    pairs = [[p[0], p[1].split('/')] for p in pairs if p[0] in vocab]
    pairs = [[p[0], [w for w in p[1] if w in vocab]] for p in pairs]
    pairs = [p for p in pairs if len(p[1]) > 0]
    if len(pairs) <= 1:
        return None

    transposed = np.transpose(np.array([x / norm(x) for x in matrix]))

    if not multi:
        qa = []
        qb = []
        qc = []
        targets = []
        exclude = []
        groups = []

        for i in range(len(pairs)):
            j = random.randint(0, len(pairs) - 2)
            if j >= i: j += 1
            a = matrix[indices[pairs[i][0]]]
            c = matrix[indices[pairs[j][0]]]
            for bw in pairs[i][1]:
                qa.append(a)
                qb.append(matrix[indices[bw]])
                qc.append(c)
                groups.append(i)
                targets.append(pairs[j][1])
                exclude.append([pairs[i][0], bw, pairs[j][0]])

        # print("qa ", len(qa))
        # print("qb ", len(qb))
        # print("qc ", len(qc))

        for queries in [qa, qb, qc]:
            queries = np.array([x / norm(x) for x in queries])

        sa = np.matmul(qa, transposed) + .0001
        sb = np.matmul(qb, transposed)
        sc = np.matmul(qc, transposed)
        sims = sb + sc - sa

        # exclude original query words from candidates
        for i in range(len(exclude)):
            for w in exclude[i]:
                sims[i][indices[w]] = 0

    else:
        offsets = []
        exclude = []
        preds = []
        targets = []
        groups = []

        for i in range(len(pairs) // multi):
            qa = [pairs[j][0] for j in range(len(pairs)) if j - i not in range(multi)]
            qb = [[w for w in pairs[j][1] if w in vocab] for j in range(len(pairs)) if j - i not in range(multi)]
            qbs = []
            for ws in qb: qbs += ws
            a = np.mean([matrix[indices[w]] for w in qa], axis=0)
            b = np.mean([np.mean([matrix[indices[w]] for w in ws], axis=0) for ws in qb], axis=0)
            a = a / norm(a)
            b = b / norm(b)

            for k in range(multi):
                c = matrix[indices[pairs[i + k][0]]]
                c = c / norm(c)
                offset = b + c - a
                offsets.append(offset / norm(offset))
                targets.append(pairs[i + k][1])
                exclude.append(qa + qbs + [pairs[i + k][0]])
                groups.append(len(groups))

        # print(np.shape(transposed))

        sims = np.matmul(np.array(offsets), transposed)
        # print(np.shape(sims))
        for i in range(len(exclude)):
            for w in exclude[i]:
                sims[i][indices[w]] = 0

    preds = [vocab[np.argmax(x)] for x in sims]
    accs = [1 if preds[i].lower() in targets[i] else 0 for i in range(len(preds))]
    regrouped = np.zeros(np.max(groups) + 1)
    for a, g in zip(accs, groups):
        regrouped[g] = max(a, regrouped[g])
    return np.mean(regrouped)


def eval_bats(model, matrix, vocab, indices):
    accs = {}
    base = 'data/BATS'
    for dr in os.listdir('data/BATS'):
        if os.path.isdir(os.path.join(base, dr)):
            dk = dr.split('_', 1)[1].lower()
            accs[dk] = []
            for f in os.listdir(os.path.join(base, dr)):
                accs[f.split('.')[0]] = eval_bats_file(model, matrix, vocab, indices, os.path.join(base, dr, f))
                accs[dk].append(accs[f.split('.')[0]])
            accs[dk] = [a for a in accs[dk] if a is not None]
            accs[dk] = np.mean(accs[dk]) if len(accs[dk]) > 0 else None

    accs['total'] = np.mean([accs[k] for k in accs.keys() if accs[k] is not None])

    return accs

def collect(model, tokenizer):
    '''
    Collects matrix and vocabulary list from a pre-trained bert model.
    '''
    vocab = []
    base = 'data/BATS'

    # Build vocab
    for dr in os.listdir('data/BATS'):
        if os.path.isdir(os.path.join(base, dr)):
            for fn in os.listdir(os.path.join(base, dr)):
                f = os.path.join(base, dr, fn)
                # print(f)
                pairs = [line.strip().split() for line in open(f, 'r').readlines()]
                for p in pairs:
                    # print(pairs)
                    vocab.append(p[0])
                    vocab.extend(p[1].split('/'))
    # Build words index dictionary
    indices = {}
    for i in range(len(vocab)):
        indices[vocab[i]] = i

    # Build embedding matrix
    matrix = []
    for w in vocab:
        matrix.append(get_one_word_vector(w, model, tokenizer))
    return np.array(matrix), vocab, indices


if __name__ == "__main__":

    print('[evaluate] Loading model...')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    model.eval()

    print('[evaluate] Collecting matrix...')
    matrix, vocab, indices = collect(model, tokenizer)

    print('[evaluate] WordSim353 correlation:')
    ws = eval_wordsim(model, tokenizer)
    print(ws)

    print('[evaluate] BATS accuracies:')
    bats = eval_bats(model, matrix, vocab, indices)
    print(bats)

    print('[evaluate] MSR accuracy:')
    msr = eval_msr(model)
    print(msr)
