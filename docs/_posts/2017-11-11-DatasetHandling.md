---
layout: post
title:  "Dataset Handling"
description: "Handling Dataset with TnsorFlow Dataset APIs"
excerpt: "Handling Dataset with TnsorFlow Dataset APIs"
date:   2017-11-11
mathjax: true
comments: true 
---

**Jupyter notebook avaialble @ [www.github.com/iaja/tf-guru/dataset/2017-11-11-DatasetHandling.ipynb](www.github.com/iaja/tf-guru/dataset/2017-11-11-DatasetHandling.ipynb)**

# Handling TextDataset with TensorFlow APIs

### Preparing vocab list with TF APIs


```python
import tensorflow as tf
from tensorflow.contrib import lookup
from tensorflow.python.platform import gfile

print ('TensorFlow Version: ', tf.__version__)

# Normally this takes the mean length of the words in the dataset documents
MAX_DOCUMENT_LENGTH = 5  
# Padding word that is used when a document has less words than the calculated mean length of the words
PADWORD = 'ZYXW'

# Assume each line to be an document
lines = ['Simple',
         'Some title', 
         'A longer title', 
         'An even longer title', 
         'This is longer than doc length']

# Create vocabulary
# min_frequency -> consider a word if and only it repeats for fiven count
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, 
                                                                     min_frequency=0)
vocab_processor.fit(lines)

#Create a file and store the words
with gfile.Open('vocab_test.tsv', 'wb') as f:
    f.write("{}\n".format(PADWORD))
    for word, index in vocab_processor.vocabulary_._mapping.items():
      f.write("{}\n".format(word))
    
VOCAB_SIZE = len(vocab_processor.vocabulary_)
print ('{} words into vocab.tsv'.format(VOCAB_SIZE+1))

EMBEDDING_SIZE = 50

```

    TensorFlow Version:  1.3.0
    14 words into vocab.tsv



```python
! cat vocab_test.tsv
```

    ZYXW
    <UNK>
    Simple
    Some
    title
    A
    longer
    An
    even
    This
    is
    than
    doc
    length



```python
# can use the vocabulary to convert words to numbers
table = lookup.index_table_from_file(
  vocabulary_file='vocab_test.tsv', num_oov_buckets=1, vocab_size=None, default_value=-1)

numbers = table.lookup(tf.constant(lines[1].split()))

with tf.Session() as sess:
    #Tables needs to be initialized before useing it
    tf.tables_initializer().run()
    print ("{} --> {}".format(lines[1], numbers.eval()))
```

    Some title --> [3 4]



```python
# string operations
# Array of Docs -> Split it into Tokens/words 
#               -> Convert it into Dense Tensor apending PADWORD
#               -> Table lookup 
#               -> Slice it to MAX_DOCUMENT_LENGTH
titles = tf.constant(lines)
words = tf.string_split(titles)
densewords = tf.sparse_tensor_to_dense(words, default_value=PADWORD)
numbers = table.lookup(densewords)

##Following extrasteps are taken care by above 'table.lookup'

# now pad out with zeros and then slice to constant length
# padding = tf.constant([[0,0],[0,MAX_DOCUMENT_LENGTH]])
# this takes care of documents with zero length also
# padded = tf.pad(numbers, padding)

sliced = tf.slice(numbers, [0,0], [-1, MAX_DOCUMENT_LENGTH])

with tf.Session() as sess:
    tf.tables_initializer().run()
    print ("titles=", titles.eval(), titles.shape)
    print('--------------------------------------------------------')
    print ("words=", words.eval())
    print('--------------------------------------------------------')
    print ("dense=", densewords.eval(), densewords.shape)
    print('--------------------------------------------------------')
    print ("numbers=", numbers.eval(), numbers.shape)
    print('--------------------------------------------------------')
#     print ("padding=", padding.eval(), padding.shape)
    print('--------------------------------------------------------')
#     print ("padded=", padded.eval(), padded.shape)
    print('--------------------------------------------------------')
    print ("sliced=", sliced.eval(), sliced.shape)
    print('--------------------------------------------------------')
    
    with tf.device('/cpu:0'), tf.name_scope("embed-layer"):
        # layer to take the words and convert them into vectors (embeddings)
        # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
        # maps word indexes of the sequence into
        # [batch_size, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE].
        word_vectors = tf.contrib.layers.embed_sequence(sliced,
                                                  vocab_size=VOCAB_SIZE,
                                                  embed_dim=EMBEDDING_SIZE)

        # [?, self.MAX_DOCUMENT_LENGTH, self.EMBEDDING_SIZE]
        tf.logging.debug('words_embed={}'.format(word_vectors))
```

    titles= [b'Simple' b'Some title' b'A longer title' b'An even longer title'
     b'This is longer than doc length'] (5,)
    --------------------------------------------------------
    words= SparseTensorValue(indices=array([[0, 0],
           [1, 0],
           [1, 1],
           [2, 0],
           [2, 1],
           [2, 2],
           [3, 0],
           [3, 1],
           [3, 2],
           [3, 3],
           [4, 0],
           [4, 1],
           [4, 2],
           [4, 3],
           [4, 4],
           [4, 5]]), values=array([b'Simple', b'Some', b'title', b'A', b'longer', b'title', b'An',
           b'even', b'longer', b'title', b'This', b'is', b'longer', b'than',
           b'doc', b'length'], dtype=object), dense_shape=array([5, 6]))
    --------------------------------------------------------
    dense= [[b'Simple' b'ZYXW' b'ZYXW' b'ZYXW' b'ZYXW' b'ZYXW']
     [b'Some' b'title' b'ZYXW' b'ZYXW' b'ZYXW' b'ZYXW']
     [b'A' b'longer' b'title' b'ZYXW' b'ZYXW' b'ZYXW']
     [b'An' b'even' b'longer' b'title' b'ZYXW' b'ZYXW']
     [b'This' b'is' b'longer' b'than' b'doc' b'length']] (?, ?)
    --------------------------------------------------------
    numbers= [[ 2  0  0  0  0  0]
     [ 3  4  0  0  0  0]
     [ 5  6  4  0  0  0]
     [ 7  8  6  4  0  0]
     [ 9 10  6 11 12 13]] (?, ?)
    --------------------------------------------------------
    --------------------------------------------------------
    --------------------------------------------------------
    sliced= [[ 2  0  0  0  0]
     [ 3  4  0  0  0]
     [ 5  6  4  0  0]
     [ 7  8  6  4  0]
     [ 9 10  6 11 12]] (?, 5)
    --------------------------------------------------------


# Estimators Inputs
- https://www.tensorflow.org/api_docs/python/tf/estimator/inputs


```python
!ls ../../../data/
```

    ls: cannot access '../../../data/': No such file or directory



```python
import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"


def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle)


# def main(unused_argv):
    # Load datasets
training_set = pd.read_csv("../data/boston_train.csv", skipinitialspace=True,
                         skiprows=1, names=COLUMNS)
test_set = pd.read_csv("../data/boston_test.csv", skipinitialspace=True,
                     skiprows=1, names=COLUMNS)

# Set of 6 examples for which to predict median house values
prediction_set = pd.read_csv("../data/boston_predict.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)

# Feature cols
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

# Build 2 layer fully connected DNN with 10, 10 units respectively.
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                    hidden_units=[10, 10],
                                    model_dir="/tmp/boston_model")

# Train
regressor.train(input_fn=get_input_fn(training_set), steps=5000)

# Evaluate loss over one epoch of test_set.
ev = regressor.evaluate(
  input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score))

# Print out predictions over a slice of prediction_set.
y = regressor.predict(
  input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
# .predict() returns an iterator of dicts; convert to a list and print
# predictions
predictions = list(p["predictions"] for p in itertools.islice(y, 6))
print("Predictions: {}".format(str(predictions)))

```




```python

```


```python

```


```python

```

# References: 
- https://medium.com/towards-data-science/how-to-do-text-classification-using-tensorflow-word-embeddings-and-cnn-edae13b3e575
- https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/blogs/textclassification


```python
# Convert this notebook for Docs
! jupyter nbconvert --to markdown --output-dir ../docs/_posts 2017-11-11-DatasetHandling.ipynb
```

    [NbConvertApp] Converting notebook 2017-11-11-DatasetHandling.ipynb to markdown
    [NbConvertApp] Writing 9279 bytes to ../docs/_posts/2017-11-11-DatasetHandling.md



```python

```
