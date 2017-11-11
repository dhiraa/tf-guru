
# Handling TextDataset with TensorFlow APIs


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


# References: 
- https://medium.com/towards-data-science/how-to-do-text-classification-using-tensorflow-word-embeddings-and-cnn-edae13b3e575
- https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/blogs/textclassification

# Estimators Inputs
- https://www.tensorflow.org/api_docs/python/tf/estimator/inputs


```python
!ls ../../../data/


```

    boston_predict.csv  boston_test.csv  boston_train.csv



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
training_set = pd.read_csv("../../../data/boston_train.csv", skipinitialspace=True,
                         skiprows=1, names=COLUMNS)
test_set = pd.read_csv("../../../data/boston_test.csv", skipinitialspace=True,
                     skiprows=1, names=COLUMNS)

# Set of 6 examples for which to predict median house values
prediction_set = pd.read_csv("../../../data/boston_predict.csv", skipinitialspace=True,
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

    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using config: {'_model_dir': '/tmp/boston_model', '_tf_random_seed': 1, '_save_summary_steps': 100, '_save_checkpoints_secs': 600, '_save_checkpoints_steps': None, '_session_config': None, '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100}
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Restoring parameters from /tmp/boston_model/model.ckpt-5000
    INFO:tensorflow:Saving checkpoints for 5001 into /tmp/boston_model/model.ckpt.
    INFO:tensorflow:loss = 1713.56, step = 5001
    INFO:tensorflow:global_step/sec: 287.837
    INFO:tensorflow:loss = 4552.81, step = 5101 (0.349 sec)
    INFO:tensorflow:global_step/sec: 243.484
    INFO:tensorflow:loss = 4347.34, step = 5201 (0.411 sec)
    INFO:tensorflow:global_step/sec: 244.512
    INFO:tensorflow:loss = 4498.29, step = 5301 (0.409 sec)
    INFO:tensorflow:global_step/sec: 293.096
    INFO:tensorflow:loss = 4358.11, step = 5401 (0.341 sec)
    INFO:tensorflow:global_step/sec: 293.749
    INFO:tensorflow:loss = 6619.55, step = 5501 (0.340 sec)
    INFO:tensorflow:global_step/sec: 290.548
    INFO:tensorflow:loss = 4214.2, step = 5601 (0.344 sec)
    INFO:tensorflow:global_step/sec: 289.119
    INFO:tensorflow:loss = 4452.61, step = 5701 (0.346 sec)
    INFO:tensorflow:global_step/sec: 286.667
    INFO:tensorflow:loss = 4679.92, step = 5801 (0.348 sec)
    INFO:tensorflow:global_step/sec: 199.69
    INFO:tensorflow:loss = 6352.93, step = 5901 (0.503 sec)
    INFO:tensorflow:global_step/sec: 278.689
    INFO:tensorflow:loss = 3737.88, step = 6001 (0.357 sec)
    INFO:tensorflow:global_step/sec: 283.34
    INFO:tensorflow:loss = 5336.66, step = 6101 (0.353 sec)
    INFO:tensorflow:global_step/sec: 288.005
    INFO:tensorflow:loss = 4217.88, step = 6201 (0.347 sec)
    INFO:tensorflow:global_step/sec: 287.639
    INFO:tensorflow:loss = 3340.91, step = 6301 (0.348 sec)
    INFO:tensorflow:global_step/sec: 292.756
    INFO:tensorflow:loss = 4270.69, step = 6401 (0.342 sec)
    INFO:tensorflow:global_step/sec: 281.204
    INFO:tensorflow:loss = 5181.65, step = 6501 (0.356 sec)
    INFO:tensorflow:global_step/sec: 287.603
    INFO:tensorflow:loss = 2999.63, step = 6601 (0.348 sec)
    INFO:tensorflow:global_step/sec: 287.471
    INFO:tensorflow:loss = 1824.25, step = 6701 (0.348 sec)
    INFO:tensorflow:global_step/sec: 294.262
    INFO:tensorflow:loss = 4613.53, step = 6801 (0.340 sec)
    INFO:tensorflow:global_step/sec: 283.177
    INFO:tensorflow:loss = 1867.62, step = 6901 (0.353 sec)
    INFO:tensorflow:global_step/sec: 287.117
    INFO:tensorflow:loss = 5450.2, step = 7001 (0.348 sec)
    INFO:tensorflow:global_step/sec: 277.777
    INFO:tensorflow:loss = 3181.08, step = 7101 (0.360 sec)
    INFO:tensorflow:global_step/sec: 292.91
    INFO:tensorflow:loss = 3770.0, step = 7201 (0.341 sec)
    INFO:tensorflow:global_step/sec: 290.059
    INFO:tensorflow:loss = 3989.91, step = 7301 (0.345 sec)
    INFO:tensorflow:global_step/sec: 289.021
    INFO:tensorflow:loss = 3401.06, step = 7401 (0.346 sec)
    INFO:tensorflow:global_step/sec: 291.806
    INFO:tensorflow:loss = 3408.22, step = 7501 (0.343 sec)
    INFO:tensorflow:global_step/sec: 281.397
    INFO:tensorflow:loss = 6015.98, step = 7601 (0.355 sec)
    INFO:tensorflow:global_step/sec: 196.762
    INFO:tensorflow:loss = 2068.23, step = 7701 (0.509 sec)
    INFO:tensorflow:global_step/sec: 181.937
    INFO:tensorflow:loss = 4346.31, step = 7801 (0.550 sec)
    INFO:tensorflow:global_step/sec: 255.045
    INFO:tensorflow:loss = 4137.44, step = 7901 (0.392 sec)
    INFO:tensorflow:global_step/sec: 262.411
    INFO:tensorflow:loss = 3505.59, step = 8001 (0.381 sec)
    INFO:tensorflow:global_step/sec: 291.505
    INFO:tensorflow:loss = 5116.16, step = 8101 (0.343 sec)
    INFO:tensorflow:global_step/sec: 195.477
    INFO:tensorflow:loss = 4770.48, step = 8201 (0.512 sec)
    INFO:tensorflow:global_step/sec: 258.836
    INFO:tensorflow:loss = 3833.2, step = 8301 (0.387 sec)
    INFO:tensorflow:global_step/sec: 212.659
    INFO:tensorflow:loss = 2380.55, step = 8401 (0.470 sec)
    INFO:tensorflow:global_step/sec: 278.039
    INFO:tensorflow:loss = 4895.08, step = 8501 (0.359 sec)
    INFO:tensorflow:global_step/sec: 192.438
    INFO:tensorflow:loss = 4350.42, step = 8601 (0.520 sec)
    INFO:tensorflow:global_step/sec: 265.761
    INFO:tensorflow:loss = 4644.19, step = 8701 (0.376 sec)
    INFO:tensorflow:global_step/sec: 197.543
    INFO:tensorflow:loss = 2239.86, step = 8801 (0.506 sec)
    INFO:tensorflow:global_step/sec: 181.521
    INFO:tensorflow:loss = 5448.2, step = 8901 (0.551 sec)
    INFO:tensorflow:global_step/sec: 254.588
    INFO:tensorflow:loss = 3133.03, step = 9001 (0.393 sec)
    INFO:tensorflow:global_step/sec: 214.597
    INFO:tensorflow:loss = 3496.03, step = 9101 (0.466 sec)
    INFO:tensorflow:global_step/sec: 228.003
    INFO:tensorflow:loss = 1565.17, step = 9201 (0.438 sec)
    INFO:tensorflow:global_step/sec: 230.899
    INFO:tensorflow:loss = 2542.55, step = 9301 (0.433 sec)
    INFO:tensorflow:global_step/sec: 275.067
    INFO:tensorflow:loss = 2475.57, step = 9401 (0.364 sec)
    INFO:tensorflow:global_step/sec: 258.019
    INFO:tensorflow:loss = 2830.1, step = 9501 (0.387 sec)
    INFO:tensorflow:global_step/sec: 231.913
    INFO:tensorflow:loss = 1851.5, step = 9601 (0.437 sec)
    INFO:tensorflow:global_step/sec: 197.132
    INFO:tensorflow:loss = 3332.6, step = 9701 (0.502 sec)
    INFO:tensorflow:global_step/sec: 267.116
    INFO:tensorflow:loss = 4781.61, step = 9801 (0.374 sec)
    INFO:tensorflow:global_step/sec: 277.116
    INFO:tensorflow:loss = 3258.09, step = 9901 (0.361 sec)
    INFO:tensorflow:Saving checkpoints for 10000 into /tmp/boston_model/model.ckpt.
    INFO:tensorflow:Loss for final step: 4723.48.
    INFO:tensorflow:Starting evaluation at 2017-10-29-13:05:42
    INFO:tensorflow:Restoring parameters from /tmp/boston_model/model.ckpt-10000
    INFO:tensorflow:Finished evaluation at 2017-10-29-13:05:42
    INFO:tensorflow:Saving dict for global step 10000: average_loss = 12.591, global_step = 10000, loss = 1259.1
    Loss: 1259.098389
    INFO:tensorflow:Restoring parameters from /tmp/boston_model/model.ckpt-10000
    Predictions: [array([ 33.89645767], dtype=float32), array([ 18.13626671], dtype=float32), array([ 23.39229202], dtype=float32), array([ 34.82794571], dtype=float32), array([ 15.41587353], dtype=float32), array([ 19.01655006], dtype=float32)]


# Convert this notebook for Docs


```python
! jupyter nbconvert --to markdown 2011-11-11-DatasetHandling.ipynb
```

    [NbConvertApp] Converting notebook 2011-11-11-DatasetHandling.ipynb to markdown
    [NbConvertApp] Writing 15277 bytes to 2011-11-11-DatasetHandling.md



```python

```
