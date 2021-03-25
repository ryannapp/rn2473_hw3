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
            pred.append(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    return spearmanr(sim, pred)


def eval_msr(model):
    X_tr, y_tr = load_msr('data/msr/msr_paraphrase_train.txt')
    X_test, y_test = load_msr('data/msr/msr_paraphrase_test.txt')

    train = [[np.sum([x.numpy() for x in get_word_vectors(" ".join(ss[0]), model, tokenizer).values()], axis=0),
              np.sum([x.numpy() for x in get_word_vectors(" ".join(ss[1]), model, tokenizer).values()], axis=0)] for ss in X_tr]
    test = [[np.sum([x.numpy() for x in get_word_vectors(" ".join(ss[0]), model, tokenizer).values()], axis=0),
             np.sum([x.numpy() for x in get_word_vectors(" ".join(ss[1]), model, tokenizer).values()], axis=0)] for ss in X_test]

    tr_cos = np.array([1 - cosine(x[0], x[1]) for x in train]).reshape(-1, 1)
    test_cos = np.array([1 - cosine(x[0], x[1]) for x in test]).reshape(-1, 1)

    lr = LogisticRegression(class_weight='balanced', solver='liblinear')
    lr.fit(tr_cos, y_tr)
    preds = lr.predict(test_cos)

    return accuracy_score(y_test, preds)


def eval_bats_file(model, matrix, vocab, indices, f, repeat=False,
                   multi=0):
    vocab = tokenizer.get_vocab()
    pairs = [line.strip().split() for line in open(f, 'r').readlines()]

    # discard pairs that are not in our vocabulary
    pairs = [[p[0], p[1].split('/')] for p in pairs if p[0] in vocab]
    pairs = [[p[0], [w for w in p[1] if w in vocab]] for p in pairs]
    pairs = [p for p in pairs if len(p[1]) > 0]
    if len(pairs) <= 1: return None

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
            a = model[pairs[i][0]]
            c = model[pairs[j][0]]
            for bw in pairs[i][1]:
                qa.append(a)
                qb.append(model[bw])
                qc.append(c)
                groups.append(i)
                targets.append(pairs[j][1])
                exclude.append([pairs[i][0], bw, pairs[j][0]])

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
            qb = [[w for w in pairs[j][1] if w in model] for j in range(len(pairs)) if j - i not in range(multi)]
            qbs = []
            for ws in qb: qbs += ws
            a = np.mean([model[w] for w in qa], axis=0)
            b = np.mean([np.mean([model[w] for w in ws], axis=0) for ws in qb], axis=0)
            a = a / norm(a)
            b = b / norm(b)

            for k in range(multi):
                c = model[pairs[i + k][0]]
                c = c / norm(c)
                offset = b + c - a
                offsets.append(offset / norm(offset))
                targets.append(pairs[i + k][1])
                exclude.append(qa + qbs + [pairs[i + k][0]])
                groups.append(len(groups))

        print(np.shape(transposed))

        sims = np.matmul(np.array(offsets), transposed)
        print(np.shape(sims))
        for i in range(len(exclude)):
            for w in exclude[i]:
                sims[i][indices[w]] = 0

    preds = [vocab[np.argmax(x)] for x in sims]
    accs = [1 if preds[i].lower() in targets[i] else 0 for i in range(len(preds))]
    regrouped = np.zeros(np.max(groups) + 1)
    for a, g in zip(accs, groups):
        regrouped[g] = max(a, regrouped[g])
    return np.mean(regrouped)


def eval_bats(model, tokenizer):
    accs = {}
    base = 'data/BATS'
    for dr in os.listdir('data/BATS'):
        if os.path.isdir(os.path.join(base, dr)):
            dk = dr.split('_', 1)[1].lower()
            accs[dk] = []
            for f in os.listdir(os.path.join(base, dr)):
                accs[f.split('.')[0]] = eval_bats_file(model, tokenizer, os.path.join(base, dr, f))
                accs[dk].append(accs[f.split('.')[0]])
            accs[dk] = [a for a in accs[dk] if a is not None]
            accs[dk] = np.mean(accs[dk]) if len(accs[dk]) > 0 else None

    accs['total'] = np.mean([accs[k] for k in accs.keys() if accs[k] is not None])

    return accs


if __name__ == "__main__":

    print('[evaluate] Loading model...')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    model.eval()

    # print('[evaluate] WordSim353 correlation:')
    # ws = eval_wordsim(model, tokenizer)
    # print(ws)

    # print('[evaluate] BATS accuracies:')
    # bats = eval_bats(model, tokenizer)
    # print(bats)

    print('[evaluate] MSR accuracy:')
    msr = eval_msr(model)
    print(msr)
