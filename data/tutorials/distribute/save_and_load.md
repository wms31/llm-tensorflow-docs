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

# Save and load a model using a distribution strategy

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/tutorials/distribute/save_and_load"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/save_and_load.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/save_and_load.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/distribute/save_and_load.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>

</table>

## Overview

This tutorial demonstrates how you can save and load models in a SavedModel format with `tf.distribute.Strategy` during or after training. There are two kinds of APIs for saving and loading a Keras model: high-level (`tf.keras.Model.save` and `tf.keras.models.load_model`) and low-level (`tf.saved_model.save` and `tf.saved_model.load`).

To learn about SavedModel and serialization in general, please read the [saved model guide](../../guide/saved_model.ipynb), and the [Keras model serialization guide](https://www.tensorflow.org/guide/keras/save_and_serialize). Let's start with a simple example.

Caution: TensorFlow models are code and it is important to be careful with untrusted code. Learn more in [Using TensorFlow securely](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md).



Import dependencies:


```
import tensorflow_datasets as tfds

import tensorflow as tf

```

Load and prepare the data with TensorFlow Datasets and `tf.data`, and create the model using `tf.distribute.MirroredStrategy`:


```
mirrored_strategy = tf.distribute.MirroredStrategy()

def get_data():
  datasets = tfds.load(name='mnist', as_supervised=True)
  mnist_train, mnist_test = datasets['train'], datasets['test']

  BUFFER_SIZE = 10000

  BATCH_SIZE_PER_REPLICA = 64
  BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label

  train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

  return train_dataset, eval_dataset

def get_model():
  with mirrored_strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])
    return model
```

Train the model with `tf.keras.Model.fit`: 


```
model = get_model()
train_dataset, eval_dataset = get_data()
model.fit(train_dataset, epochs=2)
```

## Save and load the model

Now that you have a simple model to work with, let's explore the saving/loading APIs. 
There are two kinds of APIs available:

*   High-level (Keras): `Model.save` and `tf.keras.models.load_model` (`.keras` zip archive format)
*   Low-level: `tf.saved_model.save` and `tf.saved_model.load` (TF SavedModel format)


### The Keras API

Here is an example of saving and loading a model with the Keras API:


```
keras_model_path = '/tmp/keras_save.keras'
model.save(keras_model_path)
```

Restore the model without `tf.distribute.Strategy`:


```
restored_keras_model = tf.keras.models.load_model(keras_model_path)
restored_keras_model.fit(train_dataset, epochs=2)
```

After restoring the model, you can continue training on it, even without needing to call `Model.compile` again, since it was already compiled before saving. The model is saved a Keras zip archive format, marked by the `.keras` extension. For more information, please refer to [the guide on Keras saving](https://www.tensorflow.org/guide/keras/save_and_serialize).

Now, restore the model and train it using a `tf.distribute.Strategy`:


```
another_strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
with another_strategy.scope():
  restored_keras_model_ds = tf.keras.models.load_model(keras_model_path)
  restored_keras_model_ds.fit(train_dataset, epochs=2)
```

As the `Model.fit` output shows, loading works as expected with `tf.distribute.Strategy`. The strategy used here does not have to be the same strategy used before saving. 

### The `tf.saved_model` API

Saving the model with lower-level API is similar to the Keras API:


```
model = get_model()  # get a fresh model
saved_model_path = '/tmp/tf_save'
tf.saved_model.save(model, saved_model_path)
```

Loading can be done with `tf.saved_model.load`. However, since it is a lower-level API (and hence has a wider range of use cases), it does not return a Keras model. Instead, it returns an object that contain functions that can be used to do inference. For example:


```
DEFAULT_FUNCTION_KEY = 'serving_default'
loaded = tf.saved_model.load(saved_model_path)
inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]
```

The loaded object may contain multiple functions, each associated with a key. The `"serving_default"` key is the default key for the inference function with a saved Keras model. To do inference with this function: 


```
predict_dataset = eval_dataset.map(lambda image, label: image)
for batch in predict_dataset.take(1):
  print(inference_func(batch))
```

You can also load and do inference in a distributed manner:


```
another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
  loaded = tf.saved_model.load(saved_model_path)
  inference_func = loaded.signatures[DEFAULT_FUNCTION_KEY]

  dist_predict_dataset = another_strategy.experimental_distribute_dataset(
      predict_dataset)

  # Calling the function in a distributed manner
  for batch in dist_predict_dataset:
    result = another_strategy.run(inference_func, args=(batch,))
    print(result)
    break
```

Calling the restored function is just a forward pass on the saved model (`tf.keras.Model.predict`). What if you want to continue training the loaded function? Or what if you need to embed the loaded function into a bigger model? A common practice is to wrap this loaded object into a Keras layer to achieve this. Luckily, [TF Hub](https://www.tensorflow.org/hub) has [`hub.KerasLayer`](https://github.com/tensorflow/hub/blob/master/tensorflow_hub/keras_layer.py) for this purpose, shown here:


```
import tensorflow_hub as hub

def build_model(loaded):
  x = tf.keras.layers.Input(shape=(28, 28, 1), name='input_x')
  # Wrap what's loaded to a KerasLayer
  keras_layer = hub.KerasLayer(loaded, trainable=True)(x)
  model = tf.keras.Model(x, keras_layer)
  return model

another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
  loaded = tf.saved_model.load(saved_model_path)
  model = build_model(loaded)

  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])
  model.fit(train_dataset, epochs=2)
```

In the above example, Tensorflow Hub's `hub.KerasLayer` wraps the result loaded back from `tf.saved_model.load` into a Keras layer that is used to build another model. This is very useful for transfer learning. 

### Which API should I use?

For saving, if you are working with a Keras model, use the Keras `Model.save` API unless you need the additional control allowed by the low-level API. If what you are saving is not a Keras model, then the lower-level API, `tf.saved_model.save`, is your only choice. 

For loading, your API choice depends on what you want to get from the model loading API. If you cannot (or do not want to) get a Keras model, then use `tf.saved_model.load`. Otherwise, use `tf.keras.models.load_model`. Note that you can get a Keras model back only if you saved a Keras model. 

It is possible to mix and match the APIs. You can save a Keras model with `Model.save`, and load a non-Keras model with the low-level API, `tf.saved_model.load`. 


```
model = get_model()

# Saving the model using Keras `Model.save`
model.save(saved_model_path)

another_strategy = tf.distribute.MirroredStrategy()
# Loading the model using the lower-level API
with another_strategy.scope():
  loaded = tf.saved_model.load(saved_model_path)
```

### Saving/Loading from a local device

When saving and loading from a local I/O device while training on remote devices—for example, when using a Cloud TPU—you must use the option `experimental_io_device` in `tf.saved_model.SaveOptions` and `tf.saved_model.LoadOptions` to set the I/O device to `localhost`. For example:


```
model = get_model()

# Saving the model to a path on localhost.
saved_model_path = '/tmp/tf_save'
save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
model.save(saved_model_path, options=save_options)

# Loading the model from a path on localhost.
another_strategy = tf.distribute.MirroredStrategy()
with another_strategy.scope():
  load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
  loaded = tf.keras.models.load_model(saved_model_path, options=load_options)
```

### Caveats

One special case is when you create Keras models in certain ways, and then save them before training. For example:


```
class SubclassedModel(tf.keras.Model):
  """Example model defined by subclassing `tf.keras.Model`."""

  output_name = 'output_layer'

  def __init__(self):
    super(SubclassedModel, self).__init__()
    self._dense_layer = tf.keras.layers.Dense(
        5, dtype=tf.dtypes.float32, name=self.output_name)

  def call(self, inputs):
    return self._dense_layer(inputs)

my_model = SubclassedModel()
try:
  my_model.save(saved_model_path)
except ValueError as e:
  print(f'{type(e).__name__}: ', *e.args)
```

A SavedModel saves the `tf.types.experimental.ConcreteFunction` objects generated when you trace a `tf.function` (check _When is a Function tracing?_ in the [Introduction to graphs and tf.function](../../guide/intro_to_graphs.ipynb) guide to learn more). If you get a `ValueError` like this it's because `Model.save` was not able to find or create a traced `ConcreteFunction`.

**Caution:** You shouldn't save a model without at least one `ConcreteFunction`, since the low-level API will otherwise generate a SavedModel with no `ConcreteFunction` signatures ([learn more](../../guide/saved_model.ipynb) about the SavedModel format). For example:


```
tf.saved_model.save(my_model, saved_model_path)
x = tf.saved_model.load(saved_model_path)
x.signatures
```


Usually the model's forward pass—the `call` method—will be traced automatically when the model is called for the first time, often via the Keras `Model.fit` method. A `ConcreteFunction` can also be generated by the Keras [Sequential](https://www.tensorflow.org/guide/keras/sequential_model) and [Functional](https://www.tensorflow.org/guide/keras/functional) APIs, if you set the input shape, for example, by making the first layer either a `tf.keras.layers.InputLayer` or another layer type, and passing it the `input_shape` keyword argument. 

To verify if your model has any traced `ConcreteFunction`s, check if `Model.save_spec` is `None`:


```
print(my_model.save_spec() is None)
```

Let's use `tf.keras.Model.fit` to train the model, and notice that the `save_spec` gets defined and model saving will work:


```
BATCH_SIZE_PER_REPLICA = 4
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

dataset_size = 100
dataset = tf.data.Dataset.from_tensors(
    (tf.range(5, dtype=tf.float32), tf.range(5, dtype=tf.float32))
    ).repeat(dataset_size).batch(BATCH_SIZE)

my_model.compile(optimizer='adam', loss='mean_squared_error')
my_model.fit(dataset, epochs=2)

print(my_model.save_spec() is None)
my_model.save(saved_model_path)
```
