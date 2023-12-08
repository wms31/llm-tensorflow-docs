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

# Custom training with tf.distribute.Strategy

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/tutorials/distribute/custom_training"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/custom_training.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/custom_training.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/distribute/custom_training.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

This tutorial demonstrates how to use `tf.distribute.Strategy`—a TensorFlow API that provides an abstraction for [distributing your training](../../guide/distributed_training.ipynb) across multiple processing units (GPUs, multiple machines, or TPUs)—with custom training loops. In this example, you will train a simple convolutional neural network on the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) containing 70,000 images of size 28 x 28.

[Custom training loops](../customization/custom_training_walkthrough.ipynb) provide flexibility and a greater control on training. They also make it easier to debug the model and the training loop.


```
# Import TensorFlow
import tensorflow as tf

# Helper libraries
import numpy as np
import os

print(tf.__version__)
```

## Download the Fashion MNIST dataset


```
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Add a dimension to the array -> new shape == (28, 28, 1)
# This is done because the first layer in our model is a convolutional
# layer and it requires a 4D input (batch_size, height, width, channels).
# batch_size dimension will be added later on.
train_images = train_images[..., None]
test_images = test_images[..., None]

# Scale the images to the [0, 1] range.
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)
```

## Create a strategy to distribute the variables and the graph

How does `tf.distribute.MirroredStrategy` strategy work?

*   All the variables and the model graph are replicated across the replicas.
*   Input is evenly distributed across the replicas.
*   Each replica calculates the loss and gradients for the input it received.
*   The gradients are synced across all the replicas by **summing** them.
*   After the sync, the same update is made to the copies of the variables on each replica.

Note: You can put all the code below inside a single scope. This example divides it into several code cells for illustration purposes.



```
# If the list of devices is not specified in
# `tf.distribute.MirroredStrategy` constructor, they will be auto-detected.
strategy = tf.distribute.MirroredStrategy()
```


```
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
```

## Setup input pipeline


```
BUFFER_SIZE = len(train_images)

BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10
```

Create the datasets and distribute them:


```
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)

train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)
```

## Create the model

Create a model using `tf.keras.Sequential`. You can also use the [Model Subclassing API](https://www.tensorflow.org/guide/keras/custom_layers_and_models) or the [functional API](https://www.tensorflow.org/guide/keras/functional) to do this.


```
def create_model():
  regularizer = tf.keras.regularizers.L2(1e-5)
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3,
                             activation='relu',
                             kernel_regularizer=regularizer),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3,
                             activation='relu',
                             kernel_regularizer=regularizer),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64,
                            activation='relu',
                            kernel_regularizer=regularizer),
      tf.keras.layers.Dense(10, kernel_regularizer=regularizer)
    ])

  return model
```


```
# Create a checkpoint directory to store the checkpoints.
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
```

## Define the loss function

Recall that the loss function consists of one or two parts:

  * The **prediction loss** measures how far off the model's predictions are from the training labels for a batch of training examples. It is computed for each labeled example and then reduced across the batch by computing the average value.
  * Optionally, **regularization loss** terms can be added to the prediction loss, to steer the model away from overfitting the training data. A common choice is L2 regularization, which adds a small fixed multiple of the sum of squares of all model weights, independent of the number of examples. The model above uses L2 regularization to demonstrate its handling in the training loop below.

For training on a single machine with a single GPU/CPU, this works as follows:

  * The prediction loss is computed for each example in the batch, summed across the batch, and then divided by the batch size.
  * The regularization loss is added to the prediction loss.
  * The gradient of the total loss is computed w.r.t. each model weight, and the optimizer updates each model weight from the corresponding gradient.

With `tf.distribute.Strategy`, the input batch is split between replicas.
For example, let's say you have 4 GPUs, each with one replica of the model. One batch of 256 input examples is distributed evenly across the 4 replicas, so each replica gets a batch of size 64: We have `256 = 4*64`, or generally `GLOBAL_BATCH_SIZE = num_replicas_in_sync * BATCH_SIZE_PER_REPLICA`.

Each replica computes the loss from the training examples it gets and computes the gradients of the loss w.r.t. each model weight. The optimizer takes care that these **gradients are summed up across replicas** before using them to update the copies of the model weights on each replica.

*So, how should the loss be calculated when using a `tf.distribute.Strategy`?*

  * Each replica computes the prediction loss for all examples distributed to it, sums up the results and divides them by `num_replicas_in_sync * BATCH_SIZE_PER_REPLICA`, or equivently, `GLOBAL_BATCH_SIZE`.
  * Each replica compues the regularization loss(es) and divides them by
  `num_replicas_in_sync`.

Compared to non-distributed training, all per-replica loss terms are scaled down by a factor of `1/num_replicas_in_sync`. On the other hand, all loss terms -- or rather, their gradients -- are summed across that number of replicas before the optimizer applies them. In effect, the optimizer on each replica uses the same gradients as if a non-distributed computation with `GLOBAL_BATCH_SIZE` had happened. This is consistent with the distributed and undistributed behavior of Keras `Model.fit`. See the [Distributed training with Keras](./keras.ipynb) tutorial on how a larger gloabl batch size enables to scale up the learning rate.

*How to do this in TensorFlow?*

  * Loss reduction and scaling is done automatically in Keras `Model.compile` and `Model.fit`

  * If you're writing a custom training loop, as in this tutorial, you should sum the per-example losses and divide the sum by the global batch size using `tf.nn.compute_average_loss`, which takes the per-example losses and
optional sample weights as arguments and returns the scaled loss.

  * If using `tf.keras.losses` classes (as in the example below), the loss reduction needs to be explicitly specified to be one of `NONE` or `SUM`. The default `AUTO` and `SUM_OVER_BATCH_SIZE` are disallowed outside `Model.fit`.
    * `AUTO` is disallowed because the user should explicitly think about what reduction they want to make sure it is correct in the distributed case.
    * `SUM_OVER_BATCH_SIZE` is disallowed because currently it would only divide by per replica batch size, and leave the dividing by number of replicas to the user, which might be easy to miss. So, instead, you need to do the reduction yourself explicitly.

  * If you're writing a custom training loop for a model with a non-empty list of `Model.losses` (e.g., weight regularizers), you should sum them up and divide the sum by the number of replicas. You can do this by using the `tf.nn.scale_regularization_loss` function. The model code itself remains unaware of the number of replicas.

  However, models can define input-dependent regularization losses with Keras APIs such as `Layer.add_loss(...)` and `Layer(activity_regularizer=...)`. For `Layer.add_loss(...)`, it falls on the modeling code to perform the division of the summed per-example terms by the per-replica(!) batch size, e.g., by using `tf.math.reduce_mean()`.


```
with strategy.scope():
  # Set reduction to `NONE` so you can do the reduction yourself.
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True,
      reduction=tf.keras.losses.Reduction.NONE)
  def compute_loss(labels, predictions, model_losses):
    per_example_loss = loss_object(labels, predictions)
    loss = tf.nn.compute_average_loss(per_example_loss)
    if model_losses:
      loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
    return loss
```

### Special cases

Advanced users should also consider the following special cases.

  * Input batches shorter than `GLOBAL_BATCH_SIZE` create unpleasant corner cases in several places. In practice, it often works best to avoid them by allowing batches to span epoch boundaries using `Dataset.repeat().batch()` and defining approximate epochs by step counts, not dataset ends. Alternatively, `Dataset.batch(drop_remainder=True)` maintains the notion of epoch but drops the last few examples.

  For illustration, this example goes the harder route and allows short batches, so that each training epoch contains each training example exactly once.
  
  Which denominator should be used by `tf.nn.compute_average_loss()`?

    * By default, in the example code above and equivalently in `Keras.fit()`, the sum of prediction losses is divided by `num_replicas_in_sync` times the actual batch size seen on the replica (with empty batches silently ignored). This preserves the balance between the prediction loss on the one hand and the regularization losses on the other hand. It is particularly appropriate for models that use input-dependent regularization losses. Plain L2 regularization just superimposes weight decay onto the gradients of the prediction loss and is less in need of such a balance.
    * In practice, many custom training loops pass as a constant Python value into `tf.nn.compute_average_loss(..., global_batch_size=GLOBAL_BATCH_SIZE)` to use it as the denominator. This preserves the relative weighting of training examples between batches. Without it, the smaller denominator in short batches effectively upweights the examples in those. (Before TensorFlow 2.13, this was also needed to avoid NaNs in case some replica received an actual batch size of zero.)
  
  Both options are equivalent if short batches are avoided, as suggested above.

  * Multi-dimensional `labels` require you to average the `per_example_loss` across the number of predictions in each example. Consider a classification task for all pixels of an input image, with `predictions` of shape `(batch_size, H, W, n_classes)` and `labels` of shape `(batch_size, H, W)`. You will need to update `per_example_loss` like: `per_example_loss /= tf.cast(tf.reduce_prod(tf.shape(labels)[1:]), tf.float32)`

  Caution: **Verify the shape of your loss**.
  Loss functions in `tf.losses`/`tf.keras.losses` typically
  return the average over the last dimension of the input. The loss
  classes wrap these functions. Passing `reduction=Reduction.NONE` when
  creating an instance of a loss class means "no **additional** reduction".
  For categorical losses with an example input shape of `[batch, W, H, n_classes]` the `n_classes`
  dimension is reduced. For pointwise losses like
  `losses.mean_squared_error` or `losses.binary_crossentropy` include a
  dummy axis so that `[batch, W, H, 1]` is reduced to `[batch, W, H]`. Without
  the dummy axis  `[batch, W, H]` will be incorrectly reduced to `[batch, W]`.

## Define the metrics to track loss and accuracy

These metrics track the test loss and training and test accuracy. You can use `.result()` to get the accumulated statistics at any time.


```
with strategy.scope():
  test_loss = tf.keras.metrics.Mean(name='test_loss')

  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='test_accuracy')
```

## Training loop


```
# A model, an optimizer, and a checkpoint must be created under `strategy.scope`.
with strategy.scope():
  model = create_model()

  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

  checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
```


```
def train_step(inputs):
  images, labels = inputs

  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = compute_loss(labels, predictions, model.losses)

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_accuracy.update_state(labels, predictions)
  return loss

def test_step(inputs):
  images, labels = inputs

  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss.update_state(t_loss)
  test_accuracy.update_state(labels, predictions)
```


```
# `run` replicates the provided computation and runs it
# with the distributed input.
@tf.function
def distributed_train_step(dataset_inputs):
  per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

@tf.function
def distributed_test_step(dataset_inputs):
  return strategy.run(test_step, args=(dataset_inputs,))

for epoch in range(EPOCHS):
  # TRAIN LOOP
  total_loss = 0.0
  num_batches = 0
  for x in train_dist_dataset:
    total_loss += distributed_train_step(x)
    num_batches += 1
  train_loss = total_loss / num_batches

  # TEST LOOP
  for x in test_dist_dataset:
    distributed_test_step(x)

  if epoch % 2 == 0:
    checkpoint.save(checkpoint_prefix)

  template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
              "Test Accuracy: {}")
  print(template.format(epoch + 1, train_loss,
                         train_accuracy.result() * 100, test_loss.result(),
                         test_accuracy.result() * 100))

  test_loss.reset_states()
  train_accuracy.reset_states()
  test_accuracy.reset_states()
```

### Things to note in the example above

* Iterate over the `train_dist_dataset` and `test_dist_dataset` using  a `for x in ...` construct.
* The scaled loss is the return value of the `distributed_train_step`. This value is aggregated across replicas using the `tf.distribute.Strategy.reduce` call and then across batches by summing the return value of the `tf.distribute.Strategy.reduce` calls.
* `tf.keras.Metrics` should be updated inside `train_step` and `test_step` that gets executed by `tf.distribute.Strategy.run`.
* `tf.distribute.Strategy.run` returns results from each local replica in the strategy, and there are multiple ways to consume this result. You can do `tf.distribute.Strategy.reduce` to get an aggregated value. You can also do `tf.distribute.Strategy.experimental_local_results` to get the list of values contained in the result, one per local replica.


## Restore the latest checkpoint and test

A model checkpointed with a `tf.distribute.Strategy` can be restored with or without a strategy.


```
eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='eval_accuracy')

new_model = create_model()
new_optimizer = tf.keras.optimizers.Adam()

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)
```


```
@tf.function
def eval_step(images, labels):
  predictions = new_model(images, training=False)
  eval_accuracy(labels, predictions)
```


```
checkpoint = tf.train.Checkpoint(optimizer=new_optimizer, model=new_model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

for images, labels in test_dataset:
  eval_step(images, labels)

print('Accuracy after restoring the saved model without strategy: {}'.format(
    eval_accuracy.result() * 100))
```

## Alternate ways of iterating over a dataset

### Using iterators

If you want to iterate over a given number of steps and not through the entire dataset, you can create an iterator using the `iter` call and explicitly call `next` on the iterator. You can choose to iterate over the dataset both inside and outside the `tf.function`. Here is a small snippet demonstrating iteration of the dataset outside the `tf.function` using an iterator.



```
for _ in range(EPOCHS):
  total_loss = 0.0
  num_batches = 0
  train_iter = iter(train_dist_dataset)

  for _ in range(10):
    total_loss += distributed_train_step(next(train_iter))
    num_batches += 1
  average_train_loss = total_loss / num_batches

  template = ("Epoch {}, Loss: {}, Accuracy: {}")
  print(template.format(epoch + 1, average_train_loss, train_accuracy.result() * 100))
  train_accuracy.reset_states()
```

### Iterating inside a `tf.function`

You can also iterate over the entire input `train_dist_dataset` inside a `tf.function` using the `for x in ...` construct or by creating iterators like you did above. The example below demonstrates wrapping one epoch of training with a `@tf.function` decorator and iterating over `train_dist_dataset` inside the function.


```
@tf.function
def distributed_train_epoch(dataset):
  total_loss = 0.0
  num_batches = 0
  for x in dataset:
    per_replica_losses = strategy.run(train_step, args=(x,))
    total_loss += strategy.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    num_batches += 1
  return total_loss / tf.cast(num_batches, dtype=tf.float32)

for epoch in range(EPOCHS):
  train_loss = distributed_train_epoch(train_dist_dataset)

  template = ("Epoch {}, Loss: {}, Accuracy: {}")
  print(template.format(epoch + 1, train_loss, train_accuracy.result() * 100))

  train_accuracy.reset_states()
```

### Tracking training loss across replicas

Note: As a general rule, you should use `tf.keras.Metrics` to track per-sample values and avoid values that have been aggregated within a replica.

Because of the loss scaling computation that is carried out, it's not recommended to use `tf.keras.metrics.Mean` to track the training loss across different replicas.

For example, if you run a training job with the following characteristics:

* Two replicas
* Two samples are processed on each replica
* Resulting loss values: [2,  3] and [4,  5] on each replica
* Global batch size = 4

With loss scaling, you calculate the per-sample value of loss on each replica by adding the loss values, and then dividing by the global batch size. In this case: `(2 + 3) / 4 = 1.25` and `(4 + 5) / 4 = 2.25`.

If you use `tf.keras.metrics.Mean` to track loss across the two replicas, the result is different. In this example, you end up with a `total` of 3.50 and `count` of 2, which results in `total`/`count` = 1.75  when `result()` is called on the metric. Loss calculated with `tf.keras.Metrics` is scaled by an additional factor that is equal to the number of replicas in sync.

### Guide and examples

Here are some examples for using distribution strategy with custom training loops:

1. [Distributed training guide](../../guide/distributed_training)
2. [DenseNet](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/densenet/distributed_train.py) example using `MirroredStrategy`.
1. [BERT](https://github.com/tensorflow/models/blob/master/official/legacy/bert/run_classifier.py) example trained using `MirroredStrategy` and `TPUStrategy`.
This example is particularly helpful for understanding how to load from a checkpoint and generate periodic checkpoints during distributed training etc.
2. [NCF](https://github.com/tensorflow/models/blob/master/official/recommendation/ncf_keras_main.py) example trained using `MirroredStrategy` that can be enabled using the `keras_use_ctl` flag.
3. [NMT](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/nmt_with_attention/distributed_train.py) example trained using `MirroredStrategy`.

You can find more examples listed under _Examples and tutorials_ in the [Distribution strategy guide](../../guide/distributed_training.ipynb).

## Next steps

*   Try out the new `tf.distribute.Strategy` API on your models.
*   Visit the [Better performance with `tf.function`](../../guide/function.ipynb) and [TensorFlow Profiler](../../guide/profiler.md) guides to learn more about tools to optimize the performance of your TensorFlow models.
*   Check out the [Distributed training in TensorFlow](../../guide/distributed_training.ipynb) guide, which provides an overview of the available distribution strategies.
