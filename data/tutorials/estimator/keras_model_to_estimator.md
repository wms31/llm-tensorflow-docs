##### Copyright 2019 The TensorFlow Authors.


```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

# Create an Estimator from a Keras model

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/tutorials/estimator/keras_model_to_estimator"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/estimator/keras_model_to_estimator.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/estimator/keras_model_to_estimator.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/estimator/keras_model_to_estimator.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

> Warning: Estimators are not recommended for new code.  Estimators run `v1.Session`-style code which is more difficult to write correctly, and can behave unexpectedly, especially when combined with TF 2 code. Estimators do fall under our [compatibility guarantees](https://tensorflow.org/guide/versions), but will receive no fixes other than security vulnerabilities. See the [migration guide](https://tensorflow.org/guide/migrate) for details.

## Overview

TensorFlow Estimators are supported in TensorFlow, and can be created from new and existing `tf.keras` models. This tutorial contains a complete, minimal example of that process.

Note: If you have a Keras model, you can use it directly with [`tf.distribute` strategies](https://tensorflow.org/guide/migrate/guide/distributed_training) without converting it to an estimator.  As such, `model_to_estimator` is no longer recommended.

## Setup


```
import tensorflow as tf

import numpy as np
import tensorflow_datasets as tfds
```

### Create a simple Keras model.

In Keras, you assemble *layers* to build *models*. A model is (usually) a graph
of layers. The most common type of model is a stack of layers: the
`tf.keras.Sequential` model.

To build a simple, fully-connected network (i.e. multi-layer perceptron):


```
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3)
])
```

Compile the model and get a summary.


```
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam')
model.summary()
```

### Create an input function

Use the [Datasets API](../../guide/data.md) to scale to large datasets
or multi-device training.

Estimators need control of when and how their input pipeline is built. To allow this, they require an "Input function" or `input_fn`. The `Estimator` will call this function with no arguments. The `input_fn` must return a `tf.data.Dataset`.


```
def input_fn():
  split = tfds.Split.TRAIN
  dataset = tfds.load('iris', split=split, as_supervised=True)
  dataset = dataset.map(lambda features, labels: ({'dense_input':features}, labels))
  dataset = dataset.batch(32).repeat()
  return dataset
```

Test out your `input_fn`


```
for features_batch, labels_batch in input_fn().take(1):
  print(features_batch)
  print(labels_batch)
```

### Create an Estimator from the tf.keras model.

A `tf.keras.Model` can be trained with the `tf.estimator` API by converting the
model to an `tf.estimator.Estimator` object with
`tf.keras.estimator.model_to_estimator`.


```
import tempfile
model_dir = tempfile.mkdtemp()
keras_estimator = tf.keras.estimator.model_to_estimator(
    keras_model=model, model_dir=model_dir)
```

Train and evaluate the estimator.


```
keras_estimator.train(input_fn=input_fn, steps=500)
eval_result = keras_estimator.evaluate(input_fn=input_fn, steps=10)
print('Eval result: {}'.format(eval_result))
```
