>Ryan Napolitano (rn2473) - 
>Professor McKeown - 
>Natural Language Processing, Spring 2021 - 
>Homework 3

## Homework 3

## Group Members:
- Ryan Napolitano - rn2473
- Jason Herrera - jjh2210

## solution.py
- Run solution.py at the command line with no arguments.

- solution.py assumes the Brown corpus is available at data/brown.txt

- It will create a directory called "results" if it does not already exist and will 
  save the trained models to this directory.

- It will also create a directory called "results" if it does not already exist, and it 
  will save the results of the evaluation there as a csv file with headers. 
  
## bert.py
- Run bert.py at the command line with no arguments.
  
- It will run all three evaluations and print the results to the console.

## References:
- [Truncated SVD](https://sklearn.org/modules/generated/sklearn.decomposition.TruncatedSVD.html#examples-using-sklearn-decomposition-truncatedsvd)
- [Tutorial for sparse random](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.random.html)
- [SVD tutorial](https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/)
- [Convert embedding dictionary to gensim w2v format](https://www.kaggle.com/matsuik/convert-embedding-dictionary-to-gensim-w2v-format)
- [bert huggingface tutorial](https://huggingface.co/transformers/main_classes/model.html)
- [bert tutorial](https://medium.com/analytics-vidhya/bert-word-embeddings-deep-dive-32f6214f02bf)
- [bert tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/)