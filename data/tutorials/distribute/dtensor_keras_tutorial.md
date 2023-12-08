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

# Using DTensors with Keras

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/tutorials/distribute/dtensor_keras_tutorial"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/dtensor_keras_tutorial.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/dtensor_keras_tutorial.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/distribute/dtensor_keras_tutorial.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

## Overview

In this tutorial, you will learn how to use DTensors with Keras.

Through DTensor integration with Keras, you can reuse your existing Keras layers and models to build and train distributed machine learning models.

You will train a multi-layer classification model with the MNIST data. Setting the layout for subclassing model, Sequential model, and functional model will be demonstrated.

This tutorial assumes that you have already read the [DTensor programing guide](/guide/dtensor_overview), and are familiar with basic DTensor concepts like `Mesh` and `Layout`.

This tutorial is based on [Training a neural network on MNIST with Keras](https://www.tensorflow.org/datasets/keras_example).

## Setup

DTensor (`tf.experimental.dtensor`) has been part of TensorFlow since the 2.9.0 release.

First, install or upgrade TensorFlow Datasets:


```
!pip install --quiet --upgrade tensorflow-datasets
```

Next, import TensorFlow and `dtensor`, and configure TensorFlow to use 8 virtual CPUs.

Even though this example uses virtual CPUs, DTensor works the same way on CPU, GPU or TPU devices.


```
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.experimental import dtensor
```


```
def configure_virtual_cpus(ncpu):
  phy_devices = tf.config.list_physical_devices('CPU')
  tf.config.set_logical_device_configuration(
        phy_devices[0], 
        [tf.config.LogicalDeviceConfiguration()] * ncpu)
  
configure_virtual_cpus(8)
tf.config.list_logical_devices('CPU')

devices = [f'CPU:{i}' for i in range(8)]
```

## Deterministic pseudo-random number generators
One thing you should note is that DTensor API requires each of the running client to have the same random seeds, so that it could have deterministic behavior for initializing the weights. You can achieve this by setting the global seeds in keras via `tf.keras.utils.set_random_seed()`.


```
tf.keras.backend.experimental.enable_tf_random_generator()
tf.keras.utils.set_random_seed(1337)
```

## Creating a Data Parallel Mesh

This tutorial demonstrates Data Parallel training. Adapting to Model Parallel training and Spatial Parallel training can be as simple as switching to a different set of `Layout` objects. Refer to the [Distributed training with DTensors](dtensor_ml_tutorial.ipynb) tutorial for more information on distributed training beyond Data Parallel.

Data Parallel training is a commonly used parallel training scheme, also used by, for example, `tf.distribute.MirroredStrategy`.

With DTensor, a Data Parallel training loop uses a `Mesh` that consists of a single 'batch' dimension, where each device runs a replica of the model that receives a shard from the global batch.


```
mesh = dtensor.create_mesh([("batch", 8)], devices=devices)
```

As each device runs a full replica of the model, the model variables shall be fully replicated across the mesh (unsharded). As an example, a fully replicated Layout for a rank-2 weight on this `Mesh` would be as follows:


```
example_weight_layout = dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh)  # or
example_weight_layout = dtensor.Layout.replicated(mesh, rank=2)
```

A layout for a rank-2 data tensor on this `Mesh` would be sharded along the first dimension (sometimes known as `batch_sharded`),


```
example_data_layout = dtensor.Layout(['batch', dtensor.UNSHARDED], mesh)  # or
example_data_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=2)
```

## Create Keras layers with layout

In the data parallel scheme, you usually create your model weights with a fully replicated layout, so that each replica of the model can do calculations with the sharded input data. 

In order to configure the layout information for your layers' weights, Keras has exposed an extra parameter in the layer constructor for most of the built-in layers.

The following example builds a small image classification model with fully replicated weight layout. You can specify layout information `kernel` and `bias` in `tf.keras.layers.Dense` via arguments `kernel_layout` and `bias_layout`. Most of the built-in keras layers are ready for explicitly specifying the `Layout` for the layer weights.


```
unsharded_layout_2d = dtensor.Layout.replicated(mesh, 2)
unsharded_layout_1d = dtensor.Layout.replicated(mesh, 1)
```


```
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, 
                        activation='relu',
                        name='d1',
                        kernel_layout=unsharded_layout_2d, 
                        bias_layout=unsharded_layout_1d),
  tf.keras.layers.Dense(10,
                        name='d2',
                        kernel_layout=unsharded_layout_2d, 
                        bias_layout=unsharded_layout_1d)
])
```

You can check the layout information by examining the `layout` property on the weights.


```
for weight in model.weights:
  print(f'Weight name: {weight.name} with layout: {weight.layout}')
  break
```

## Load a dataset and build input pipeline

Load a MNIST dataset and configure some pre-processing input pipeline for it. The dataset itself is not associated with any DTensor layout information.


```
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
```


```
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label
```


```
batch_size = 128

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
```


```
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
```

## Define the training logic for the model

Next, define the training and evaluation logic for the model. 

As of TensorFlow 2.9, you have to write a custom-training-loop for a DTensor-enabled Keras model. This is to pack the input data with proper layout information, which is not integrated with the standard `tf.keras.Model.fit()` or `tf.keras.Model.eval()` functions from Keras. you will get more `tf.data` support in the upcoming release. 


```
@tf.function
def train_step(model, x, y, optimizer, metrics):
  with tf.GradientTape() as tape:
    logits = model(x, training=True)
    # tf.reduce_sum sums the batch sharded per-example loss to a replicated
    # global loss (scalar).
    loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(
        y, logits, from_logits=True))
    
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  for metric in metrics.values():
    metric.update_state(y_true=y, y_pred=logits)

  loss_per_sample = loss / len(x)
  results = {'loss': loss_per_sample}
  return results
```


```
@tf.function
def eval_step(model, x, y, metrics):
  logits = model(x, training=False)
  loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(
        y, logits, from_logits=True))

  for metric in metrics.values():
    metric.update_state(y_true=y, y_pred=logits)

  loss_per_sample = loss / len(x)
  results = {'eval_loss': loss_per_sample}
  return results
```


```
def pack_dtensor_inputs(images, labels, image_layout, label_layout):
  num_local_devices = image_layout.mesh.num_local_devices()
  images = tf.split(images, num_local_devices)
  labels = tf.split(labels, num_local_devices)
  images = dtensor.pack(images, image_layout)
  labels = dtensor.pack(labels, label_layout)
  return  images, labels
```

## Metrics and optimizers

When using DTensor API with Keras `Metric` and `Optimizer`, you will need to provide the extra mesh information, so that any internal state variables and tensors can work with variables in the model.

- For an optimizer, DTensor introduces a new experimental namespace `keras.dtensor.experimental.optimizers`, where many existing Keras Optimizers are extended to receive an additional `mesh` argument. In future releases, it may be merged with Keras core optimizers.

- For metrics, you can directly specify the `mesh` to the constructor as an argument to make it a DTensor compatible `Metric`.


```
optimizer = tf.keras.dtensor.experimental.optimizers.Adam(0.01, mesh=mesh)
metrics = {'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(mesh=mesh)}
eval_metrics = {'eval_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(mesh=mesh)}
```

## Train the model

The following example demonstrates how to shard the data from input pipeline on the batch dimension, and train with the model, which has fully replicated weights. 

After 3 epochs, the model should achieve about 97% of accuracy:


```
num_epochs = 3

image_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=4)
label_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=1)

for epoch in range(num_epochs):
  print("============================") 
  print("Epoch: ", epoch)
  for metric in metrics.values():
    metric.reset_state()
  step = 0
  results = {}
  pbar = tf.keras.utils.Progbar(target=None, stateful_metrics=[])
  for input in ds_train:
    images, labels = input[0], input[1]
    images, labels = pack_dtensor_inputs(
        images, labels, image_layout, label_layout)

    results.update(train_step(model, images, labels, optimizer, metrics))
    for metric_name, metric in metrics.items():
      results[metric_name] = metric.result()

    pbar.update(step, values=results.items(), finalize=False)
    step += 1
  pbar.update(step, values=results.items(), finalize=True)

  for metric in eval_metrics.values():
    metric.reset_state()
  for input in ds_test:
    images, labels = input[0], input[1]
    images, labels = pack_dtensor_inputs(
        images, labels, image_layout, label_layout)
    results.update(eval_step(model, images, labels, eval_metrics))

  for metric_name, metric in eval_metrics.items():
    results[metric_name] = metric.result()
  
  for metric_name, metric in results.items():
    print(f"{metric_name}: {metric.numpy()}")

```

## Specify Layout for existing model code

Often you have models that work well for your use case. Specifying `Layout` information to each individual layer within the model will be a large amount of work requiring a lot of edits.

To help you easily convert your existing Keras model to work with DTensor API you can use the new `tf.keras.dtensor.experimental.LayoutMap` API that allow you to specify the `Layout` from a global point of view.

First, you need to create a `LayoutMap` instance, which is a dictionary-like object that contains all the `Layout` you would like to specify for your model weights.

`LayoutMap` needs a `Mesh` instance at init, which can be used to provide default replicated `Layout` for any weights that doesn't have Layout configured. In case you would like all your model weights to be just fully replicated, you can provide empty `LayoutMap`, and the default mesh will be used to create replicated `Layout`.

`LayoutMap` uses a string as key and a `Layout` as value. There is a behavior difference between a normal Python dict and this class. The string key will be treated as a regex when retrieving the value.

### Subclassed Model

Consider the following model defined using the Keras subclassing Model syntax.


```
class SubclassedModel(tf.keras.Model):

  def __init__(self, name=None):
    super().__init__(name=name)
    self.feature = tf.keras.layers.Dense(16)
    self.feature_2 = tf.keras.layers.Dense(24)
    self.dropout = tf.keras.layers.Dropout(0.1)

  def call(self, inputs, training=None):
    x = self.feature(inputs)
    x = self.dropout(x, training=training)
    return self.feature_2(x)
```

There are 4 weights in this model, which are `kernel` and `bias` for two `Dense` layers. Each of them are mapped based on the object path:

*   `model.feature.kernel`
*   `model.feature.bias`
*   `model.feature_2.kernel`
*   `model.feature_2.bias`

Note: For subclassed Models, the attribute name, rather than the `.name` attribute of the layer, is used as the key to retrieve the Layout from the mapping. This is consistent with the convention followed by `tf.Module` checkpointing. For complex models with more than a few layers, you can [manually inspect checkpoints](https://www.tensorflow.org/guide/checkpoint#manually_inspecting_checkpoints) to view the attribute mappings. 

Now define the following `LayoutMap` and apply it to the model:


```
layout_map = tf.keras.dtensor.experimental.LayoutMap(mesh=mesh)

layout_map['feature.*kernel'] = dtensor.Layout.batch_sharded(mesh, 'batch', rank=2)
layout_map['feature.*bias'] = dtensor.Layout.batch_sharded(mesh, 'batch', rank=1)

with layout_map.scope():
  subclassed_model = SubclassedModel()
```

The model weights are created on the first call, so call the model with a DTensor input and confirm the weights have the expected layouts:


```
dtensor_input = dtensor.copy_to_mesh(tf.zeros((16, 16)), layout=unsharded_layout_2d)
# Trigger the weights creation for subclass model
subclassed_model(dtensor_input)

print(subclassed_model.feature.kernel.layout)
```

With this, you can quickly map the `Layout` to your models without updating any of your existing code. 

### Sequential and Functional Models

For Keras Functional and Sequential models, you can use `tf.keras.dtensor.experimental.LayoutMap` as well.

Note: For Functional and Sequential models, the mappings are slightly different. The layers in the model don't have a public attribute attached to the model (though you can access them via `Model.layers` as a list). Use the string name as the key in this case. The string name is guaranteed to be unique within a model.


```
layout_map = tf.keras.dtensor.experimental.LayoutMap(mesh=mesh)

layout_map['feature.*kernel'] = dtensor.Layout.batch_sharded(mesh, 'batch', rank=2)
layout_map['feature.*bias'] = dtensor.Layout.batch_sharded(mesh, 'batch', rank=1)
```


```
with layout_map.scope():
  inputs = tf.keras.Input((16,), batch_size=16)
  x = tf.keras.layers.Dense(16, name='feature')(inputs)
  x = tf.keras.layers.Dropout(0.1)(x)
  output = tf.keras.layers.Dense(32, name='feature_2')(x)
  model = tf.keras.Model(inputs, output)

print(model.layers[1].kernel.layout)
```


```
with layout_map.scope():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(16, name='feature', input_shape=(16,)),
      tf.keras.layers.Dropout(0.1),
      tf.keras.layers.Dense(32, name='feature_2')
  ])

print(model.layers[2].kernel.layout)
```
