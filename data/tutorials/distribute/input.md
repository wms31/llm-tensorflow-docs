##### Copyright 2020 The TensorFlow Authors.


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

# Distributed Input

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/tutorials/distribute/input"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/input.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/input.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/distribute/input.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

The [tf.distribute](https://www.tensorflow.org/guide/distributed_training) APIs provide an easy way for users to scale their training from a single machine to multiple machines. When scaling their model, users also have to distribute their input across multiple devices. `tf.distribute` provides APIs using which you can automatically distribute your input across devices.

This guide will show you the different ways in which you can create distributed dataset and iterators using `tf.distribute` APIs. Additionally, the following topics will be covered:
- Usage, sharding and batching options when using  `tf.distribute.Strategy.experimental_distribute_dataset` and `tf.distribute.Strategy.distribute_datasets_from_function`.
- Different ways in which you can iterate over the distributed dataset.
- Differences between `tf.distribute.Strategy.experimental_distribute_dataset`/`tf.distribute.Strategy.distribute_datasets_from_function` APIs and `tf.data` APIs as well as any limitations that users may come across in their usage.

This guide does not cover usage of distributed input with Keras APIs.

## Distributed datasets

To use `tf.distribute` APIs to scale, use `tf.data.Dataset` to represent their input. `tf.distribute` works efficiently with `tf.data.Dataset`—for example, via automatic prefetching onto each accelerator device and regular performance updates. If you have a use case for using something other than `tf.data.Dataset`, please refer to the [Tensor inputs section](#tensorinputs) in this guide.
In a non-distributed training loop, first create a `tf.data.Dataset` instance and then iterate over the elements. For example:



```
import tensorflow as tf

# Helper libraries
import numpy as np
import os

print(tf.__version__)
```


```
# Simulate multiple CPUs with virtual devices
N_VIRTUAL_DEVICES = 2
physical_devices = tf.config.list_physical_devices("CPU")
tf.config.set_logical_device_configuration(
    physical_devices[0], [tf.config.LogicalDeviceConfiguration() for _ in range(N_VIRTUAL_DEVICES)])
```


```
print("Available devices:")
for i, device in enumerate(tf.config.list_logical_devices()):
  print("%d) %s" % (i, device))
```


```
global_batch_size = 16
# Create a tf.data.Dataset object.
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(global_batch_size)

@tf.function
def train_step(inputs):
  features, labels = inputs
  return labels - 0.3 * features

# Iterate over the dataset using the for..in construct.
for inputs in dataset:
  print(train_step(inputs))

```

To allow users to use `tf.distribute` strategy with minimal changes to a user’s existing code, two APIs were introduced which would distribute a `tf.data.Dataset` instance and return a distributed dataset object. A user could then iterate over this distributed dataset instance and train their model as before. Let us now look at the two APIs - `tf.distribute.Strategy.experimental_distribute_dataset` and `tf.distribute.Strategy.distribute_datasets_from_function` in more detail:

### `tf.distribute.Strategy.experimental_distribute_dataset`

#### Usage

This API takes a `tf.data.Dataset` instance as input and returns a `tf.distribute.DistributedDataset` instance. You should batch the input dataset with a value that is equal to the global batch size. This global batch size is the number of samples that you want to process across all devices in 1 step. You can iterate over this distributed dataset in a Pythonic fashion or create an iterator using `iter`. The returned object is not a `tf.data.Dataset` instance and does not support any other APIs that transform or inspect the dataset in any way.
This is the recommended API if you don’t have specific ways in which you want to shard your input over different replicas.



```
global_batch_size = 16
mirrored_strategy = tf.distribute.MirroredStrategy()

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(global_batch_size)
# Distribute input using the `experimental_distribute_dataset`.
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
# 1 global batch of data fed to the model in 1 step.
print(next(iter(dist_dataset)))
```

#### Properties

#####  Batching

`tf.distribute` rebatches the input `tf.data.Dataset` instance with a new batch size that is equal to the global batch size divided by the number of replicas in sync. The number of replicas in sync is equal to the number of devices that are taking part in the gradient allreduce during training. When a user calls `next` on the distributed iterator, a per replica batch size of data is returned on each replica. The rebatched dataset cardinality will always be a multiple of the number of replicas. Here are a couple of
examples:
* `tf.data.Dataset.range(6).batch(4, drop_remainder=False)`
  * Without distribution:
    * Batch 1: [0, 1, 2, 3]
    * Batch 2: [4, 5]
  * With distribution over 2 replicas.
    The last batch ([4, 5]) is split between 2 replicas.

    * Batch 1:
      * Replica 1:[0, 1]
      * Replica 2:[2, 3]
    * Batch 2:
      * Replica 1: [4]
      * Replica 2: [5]



* `tf.data.Dataset.range(4).batch(4)`
  * Without distribution:
    * Batch 1: [0, 1, 2, 3]
  * With distribution over 5 replicas:
    * Batch 1:
      * Replica 1: [0]
      * Replica 2: [1]
      * Replica 3: [2]
      * Replica 4: [3]
      * Replica 5: []

* `tf.data.Dataset.range(8).batch(4)`
  * Without distribution:
    * Batch 1: [0, 1, 2, 3]
    * Batch 2: [4, 5, 6, 7]
  * With distribution over 3 replicas:
    * Batch 1:
      * Replica 1: [0, 1]
      * Replica 2: [2, 3]
      * Replica 3: []
    * Batch 2:
      * Replica 1: [4, 5]
      * Replica 2: [6, 7]
      * Replica 3: []

Note: The above examples only illustrate how a global batch is split on different replicas. It is not advisable to depend on the actual values that might end up on each replica as it can change depending on the implementation.

Rebatching the dataset has a space complexity that increases linearly with the number of replicas. This means that for the multi-worker training use case the input pipeline can run into OOM errors. 

#####  Sharding

`tf.distribute` also autoshards the input dataset in multi-worker training with `MultiWorkerMirroredStrategy` and `TPUStrategy`. Each dataset is created on the CPU device of the worker. Autosharding a dataset over a set of workers means that each worker is assigned a subset of the entire dataset (if the right `tf.data.experimental.AutoShardPolicy` is set). This is to ensure that at each step, a global batch size of non-overlapping dataset elements will be processed by each worker. Autosharding has a couple of different options that can be specified using `tf.data.experimental.DistributeOptions`. Note that there is no autosharding in multi-worker training with `ParameterServerStrategy`, and more information on dataset creation with this strategy can be found in the [ParameterServerStrategy tutorial](parameter_server_training.ipynb). 


```
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(64).batch(16)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset = dataset.with_options(options)
```

There are three different options that you can set for the `tf.data.experimental.AutoShardPolicy`:

* AUTO: This is the default option which means an attempt will be made to shard by FILE. The attempt to shard by FILE fails if a file-based dataset is not detected. `tf.distribute` will then fall back to sharding by DATA. Note that if the input dataset is file-based but the number of files is less than the number of workers, an `InvalidArgumentError` will be raised. If this happens, explicitly set the policy to `AutoShardPolicy.DATA`, or split your input source into smaller files such that number of files is greater than number of workers.
* FILE: This is the option if you want to shard the input files over all the workers. You should use this option if the number of input files is much larger than the number of workers and the data in the files is evenly distributed. The downside of this option is having idle workers if the data in the files is not evenly distributed. If the number of files is less than the number of workers, an `InvalidArgumentError` will be raised. If this happens, explicitly set the policy to `AutoShardPolicy.DATA`.
For example, let us distribute 2 files over 2 workers with 1 replica each. File 1 contains [0, 1, 2, 3, 4, 5] and
File 2 contains [6, 7, 8, 9, 10, 11]. Let the total number of replicas in sync be 2 and global batch size be 4.

  * Worker 0:
    * Batch 1 =  Replica 1: [0, 1]
    * Batch 2 =  Replica 1: [2, 3]
    * Batch 3 =  Replica 1: [4]
    * Batch 4 =  Replica 1: [5]
  * Worker 1:
    * Batch 1 =  Replica 2: [6, 7]
    * Batch 2 =  Replica 2: [8, 9]
    * Batch 3 =  Replica 2: [10]
    * Batch 4 =  Replica 2: [11]

* DATA: This will autoshard the elements across all the workers. Each of the workers will read the entire dataset and only process the shard assigned to it. All other shards will be discarded. This is generally used if the number of input files is less than the number of workers and you want better sharding of data across all workers. The downside is that the entire dataset will be read on each worker.
For example, let us distribute 1 files over 2 workers. File 1 contains [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]. Let the total number of replicas in sync be 2.

  * Worker 0:
    * Batch 1 =  Replica 1: [0, 1]
    * Batch 2 =  Replica 1: [4, 5]
    * Batch 3 =  Replica 1: [8, 9]
  * Worker 1:
    * Batch 1 =  Replica 2: [2, 3]
    * Batch 2 =  Replica 2: [6, 7]
    * Batch 3 =  Replica 2: [10, 11]

* OFF: If you turn off autosharding, each worker will process all the data.
For example, let us distribute 1 files over 2 workers. File 1 contains [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]. Let the total number of replicas in sync be 2. Then each worker will see the following distribution:

  * Worker 0:
    * Batch 1 =  Replica 1: [0, 1]
    * Batch 2 =  Replica 1: [2, 3]
    * Batch 3 =  Replica 1: [4, 5]
    * Batch 4 =  Replica 1: [6, 7]
    * Batch 5 =  Replica 1: [8, 9]
    * Batch 6 =  Replica 1: [10, 11]

  * Worker 1:
    * Batch 1 =  Replica 2: [0, 1]
    * Batch 2 =  Replica 2: [2, 3]
    * Batch 3 =  Replica 2: [4, 5]
    * Batch 4 =  Replica 2: [6, 7]
    * Batch 5 =  Replica 2: [8, 9]
    * Batch 6 =  Replica 2: [10, 11] 

##### Prefetching

By default, `tf.distribute` adds a prefetch transformation at the end of the user provided `tf.data.Dataset` instance. The argument to the prefetch transformation which is `buffer_size` is equal to the number of replicas in sync.

### `tf.distribute.Strategy.distribute_datasets_from_function`

#### Usage

This API takes an input function and returns a `tf.distribute.DistributedDataset` instance. The input function that users pass in has a `tf.distribute.InputContext` argument and should return a `tf.data.Dataset` instance. With this API, `tf.distribute` does not make any further changes to the user’s `tf.data.Dataset` instance returned from the input function. It is the responsibility of the user to batch and shard the dataset. `tf.distribute` calls the input function on the CPU device of each of the workers. Apart from allowing users to specify their own batching and sharding logic, this API also demonstrates better scalability and performance compared to `tf.distribute.Strategy.experimental_distribute_dataset` when used for multi-worker training.


```
mirrored_strategy = tf.distribute.MirroredStrategy()

def dataset_fn(input_context):
  batch_size = input_context.get_per_replica_batch_size(global_batch_size)
  dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(64).batch(16)
  dataset = dataset.shard(
      input_context.num_input_pipelines, input_context.input_pipeline_id)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(2)  # This prefetches 2 batches per device.
  return dataset

dist_dataset = mirrored_strategy.distribute_datasets_from_function(dataset_fn)
```

#### Properties

##### Batching

The `tf.data.Dataset` instance that is the return value of the input function should be batched using the per replica batch size. The per replica batch size is the global batch size divided by the number of replicas that are taking part in sync training. This is because `tf.distribute` calls the input function on the CPU device of each of the workers. The dataset that is created on a given worker should be ready to use by all the replicas on that worker. 

##### Sharding

The `tf.distribute.InputContext` object that is implicitly passed as an argument to the user’s input function is created by `tf.distribute` under the hood. It has information about the number of workers, current worker ID etc. This input function can handle sharding as per policies set by the user using these properties that are part of the `tf.distribute.InputContext` object.


##### Prefetching

`tf.distribute` does not add a prefetch transformation at the end of the `tf.data.Dataset` returned by the user-provided input function, so you explicitly call `Dataset.prefetch` in the example above.

Note:
Both `tf.distribute.Strategy.experimental_distribute_dataset` and `tf.distribute.Strategy.distribute_datasets_from_function` return **`tf.distribute.DistributedDataset` instances that are not of type `tf.data.Dataset`**. You can iterate over these instances (as shown in the Distributed Iterators section) and use the `element_spec`
property.  

## Distributed iterators

Similar to non-distributed `tf.data.Dataset` instances, you will need to create an iterator on the `tf.distribute.DistributedDataset` instances to iterate over it and access the elements in the `tf.distribute.DistributedDataset`.
The following are the ways in which you can create a `tf.distribute.DistributedIterator` and use it to train your model:


### Usages

#### Use a Pythonic for loop construct

You can use a user friendly Pythonic loop to iterate over the `tf.distribute.DistributedDataset`. The elements returned from the `tf.distribute.DistributedIterator` can be a single `tf.Tensor` or a `tf.distribute.DistributedValues` which contains a value per replica. Placing the loop inside a `tf.function` will give a performance boost. However, `break` and `return` are currently not supported for a loop over a `tf.distribute.DistributedDataset` that is placed inside of a `tf.function`.


```
global_batch_size = 16
mirrored_strategy = tf.distribute.MirroredStrategy()

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(global_batch_size)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

@tf.function
def train_step(inputs):
  features, labels = inputs
  return labels - 0.3 * features

for x in dist_dataset:
  # train_step trains the model using the dataset elements
  loss = mirrored_strategy.run(train_step, args=(x,))
  print("Loss is ", loss)
```

#### Use `iter` to create an explicit iterator
To iterate over the elements in a `tf.distribute.DistributedDataset` instance, you can create a `tf.distribute.DistributedIterator` using the `iter` API on it. With an explicit iterator, you can iterate for a fixed number of steps. In order to get the next element from an `tf.distribute.DistributedIterator` instance `dist_iterator`, you can call `next(dist_iterator)`, `dist_iterator.get_next()`, or `dist_iterator.get_next_as_optional()`. The former two are essentially the same:


```
num_epochs = 10
steps_per_epoch = 5
for epoch in range(num_epochs):
  dist_iterator = iter(dist_dataset)
  for step in range(steps_per_epoch):
    # train_step trains the model using the dataset elements
    loss = mirrored_strategy.run(train_step, args=(next(dist_iterator),))
    # which is the same as
    # loss = mirrored_strategy.run(train_step, args=(dist_iterator.get_next(),))
    print("Loss is ", loss)
```

With `next` or `tf.distribute.DistributedIterator.get_next`, if the `tf.distribute.DistributedIterator` has reached its end, an OutOfRange error will be thrown. The client can catch the error on python side and continue doing other work such as checkpointing and evaluation. However, this will not work if you are using a host training loop (i.e., run multiple steps per `tf.function`), which looks like:

```
@tf.function
def train_fn(iterator):
  for _ in tf.range(steps_per_loop):
    strategy.run(step_fn, args=(next(iterator),))
```

This example `train_fn` contains multiple steps by wrapping the step body inside a `tf.range`. In this case, different iterations in the loop with no dependency could start in parallel, so an OutOfRange error can be triggered in later iterations before the computation of previous iterations finishes. Once an OutOfRange error is thrown, all the ops in the function will be terminated right away. If this is some case that you would like to avoid, an alternative that does not throw an OutOfRange error is `tf.distribute.DistributedIterator.get_next_as_optional`. `get_next_as_optional` returns a `tf.experimental.Optional` which contains the next element or no value if the `tf.distribute.DistributedIterator` has reached an end.


```
# You can break the loop with `get_next_as_optional` by checking if the `Optional` contains a value
global_batch_size = 4
steps_per_loop = 5
strategy = tf.distribute.MirroredStrategy()

dataset = tf.data.Dataset.range(9).batch(global_batch_size)
distributed_iterator = iter(strategy.experimental_distribute_dataset(dataset))

@tf.function
def train_fn(distributed_iterator):
  for _ in tf.range(steps_per_loop):
    optional_data = distributed_iterator.get_next_as_optional()
    if not optional_data.has_value():
      break
    per_replica_results = strategy.run(lambda x: x, args=(optional_data.get_value(),))
    tf.print(strategy.experimental_local_results(per_replica_results))
train_fn(distributed_iterator)
```

## Using the `element_spec` property

If you pass the elements of a distributed dataset to a `tf.function` and want a `tf.TypeSpec` guarantee, you can specify the `input_signature` argument of the `tf.function`. The output of a distributed dataset is `tf.distribute.DistributedValues` which can represent the input to a single device or multiple devices. To get the `tf.TypeSpec` corresponding to this distributed value, you can use `tf.distribute.DistributedDataset.element_spec` or `tf.distribute.DistributedIterator.element_spec`.


```
global_batch_size = 16
epochs = 5
steps_per_epoch = 5
mirrored_strategy = tf.distribute.MirroredStrategy()

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(global_batch_size)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

@tf.function(input_signature=[dist_dataset.element_spec])
def train_step(per_replica_inputs):
  def step_fn(inputs):
    return 2 * inputs

  return mirrored_strategy.run(step_fn, args=(per_replica_inputs,))

for _ in range(epochs):
  iterator = iter(dist_dataset)
  for _ in range(steps_per_epoch):
    output = train_step(next(iterator))
    tf.print(output)
```

## Data preprocessing

So far, you have learned how to distribute a `tf.data.Dataset`. Yet before the data is ready for the model, it needs to be preprocessed, for example by cleansing, transforming, and augmenting it. Two sets of those handy tools are:

*   [Keras preprocessing layers](https://www.tensorflow.org/guide/keras/preprocessing_layers): a set of Keras layers that allow developers to build Keras-native input processing pipelines. Some Keras preprocessing layers contain non-trainable states, which can be set on initialization or `adapt`ed (refer to the `adapt` section of the [Keras preprocessing layers guide](https://www.tensorflow.org/guide/keras/preprocessing_layers)). When distributing stateful preprocessing layers, the states should be replicated to all workers. To use these layers, you can either make them part of the model or apply them to the datasets.

*   [TensorFlow Transform (tf.Transform)](https://www.tensorflow.org/tfx/transform/get_started): a library for TensorFlow that allows you to define both instance-level and full-pass data transformation through data preprocessing pipelines. Tensorflow Transform has two phases. The first is the Analyze phase, where the raw training data is analyzed in a full-pass process to compute the statistics needed for the transformations, and the transformation logic is generated as instance-level operations. The second is the Transform phase, where the raw training data is transformed in an instance-level process.



### Keras preprocessing layers vs. Tensorflow Transform 

Both Tensorflow Transform and Keras preprocessing layers provide a way to split out preprocessing during training and bundle preprocessing with a model during inference, reducing train/serve skew.

Tensorflow Transform, deeply integrated with [TFX](https://www.tensorflow.org/tfx), provides a scalable map-reduce solution to analyzing and transforming datasets of any size in a job separate from the training pipeline. If you need to run an analysis on a dataset that cannot fit on a single machine, Tensorflow Transform should be your first choice.

Keras preprocessing layers are more geared towards preprocessing applied during training, after reading data from disk. They fit seamlessly with model development in the Keras library. They support analysis of a smaller dataset via [`adapt`](https://www.tensorflow.org/guide/keras/preprocessing_layers#the_adapt_method) and supports use cases like image data augmentation, where each pass over the input dataset will yield different examples for training.

The two libraries can also be mixed, where Tensorflow Transform is used for analysis and static transformations of input data, and Keras preprocessing layers are used for train-time transformations (e.g., one-hot encoding or data augmentation).



### Best Practice with tf.distribute

Working with both tools involves initializing the transformation logic to apply to data, which might create Tensorflow resources. These resources or states should be replicated to all workers to save inter-workers or worker-coordinator communication. To do so, you are recommended to create Keras preprocessing layers, `tft.TFTransformOutput.transform_features_layer`, or `tft.TransformFeaturesLayer` under `tf.distribute.Strategy.scope`, just like you would for any other Keras layers.

The following examples demonstrate usage of the `tf.distribute.Strategy` API with the high-level Keras `Model.fit` API and with a custom training loop separately.

#### Extra notes for Keras preprocessing layers users:

**Preprocessing layers and large vocabularies**

When dealing with large vocabularies (over one gigabyte) in a multi-worker setting (for example, `tf.distribute.MultiWorkerMirroredStrategy`, `tf.distribute.experimental.ParameterServerStrategy`, `tf.distribute.TPUStrategy`), it is recommended to save the vocabulary to a static file accessible from all workers (for example, with Cloud Storage). This will reduce the time spent replicating the vocabulary to all workers during training.

**Preprocessing in the `tf.data` pipeline versus in the model**

While Keras preprocessing layers can be applied either as part of the model or directly to a `tf.data.Dataset`, each of the options come with their edge:

* Applying the preprocessing layers within the model makes your model portable, and it helps reduce the training/serving skew. (For more details, refer to the _Benefits of doing preprocessing inside the model at inference time_ section in the [Working with preprocessing layers guide](https://www.tensorflow.org/guide/keras/preprocessing_layers#benefits_of_doing_preprocessing_inside_the_model_at_inference_time))
* Applying within the `tf.data` pipeline allows prefetching or offloading to the CPU, which generally gives better performance when using accelerators.

When running on one or more TPUs, users should almost always place Keras preprocessing layers in the `tf.data` pipeline, as not all layers support TPUs, and string ops do not execute on TPUs. (The two exceptions are `tf.keras.layers.Normalization` and `tf.keras.layers.Rescaling`, which run fine on TPUs and are commonly used as the first layer in an image model.)

### Preprocessing with `Model.fit`

When using Keras `Model.fit`, you do not need to distribute data with `tf.distribute.Strategy.experimental_distribute_dataset` nor `tf.distribute.Strategy.distribute_datasets_from_function` themselves. Check out the [Working with preprocessing layers](https://www.tensorflow.org/guide/keras/preprocessing_layers) guide and the [Distributed training with Keras](https://www.tensorflow.org/tutorials/distribute/keras) guide for details. A shortened example may look as below:

```
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
  # Create the layer(s) under scope.
  integer_preprocessing_layer = tf.keras.layers.IntegerLookup(vocabulary=FILE_PATH)
  model = ...
  model.compile(...)
dataset = dataset.map(lambda x, y: (integer_preprocessing_layer(x), y))
model.fit(dataset)
```


Users of `tf.distribute.experimental.ParameterServerStrategy` with the `Model.fit` API need to use a `tf.keras.utils.experimental.DatasetCreator` as the input. (See the [Parameter Server Training](https://www.tensorflow.org/tutorials/distribute/parameter_server_training#parameter_server_training_with_modelfit_api) guide for more)

```
strategy = tf.distribute.experimental.ParameterServerStrategy(
    cluster_resolver,
    variable_partitioner=variable_partitioner)

with strategy.scope():
  preprocessing_layer = tf.keras.layers.StringLookup(vocabulary=FILE_PATH)
  model = ...
  model.compile(...)

def dataset_fn(input_context):
  ...
  dataset = dataset.map(preprocessing_layer)
  ...
  return dataset

dataset_creator = tf.keras.utils.experimental.DatasetCreator(dataset_fn)
model.fit(dataset_creator, epochs=5, steps_per_epoch=20, callbacks=callbacks)

```

### Preprocessing with a custom training loop

When writing a [custom training loop](https://www.tensorflow.org/tutorials/distribute/custom_training), you will distribute your data with either the `tf.distribute.Strategy.experimental_distribute_dataset` API or the `tf.distribute.Strategy.distribute_datasets_from_function` API. If you distribute your dataset through `tf.distribute.Strategy.experimental_distribute_dataset`, applying these preprocessing APIs in your data pipeline will lead the resources automatically co-located with the data pipeline to avoid remote resource access. Thus the examples here will all use `tf.distribute.Strategy.distribute_datasets_from_function`, in which case it is crucial to place initialization of these APIs under `strategy.scope()` for efficiency:


```
strategy = tf.distribute.MirroredStrategy()
vocab = ["a", "b", "c", "d", "f"]

with strategy.scope():
  # Create the layer(s) under scope.
  layer = tf.keras.layers.StringLookup(vocabulary=vocab)

def dataset_fn(input_context):
  # a tf.data.Dataset
  dataset = tf.data.Dataset.from_tensor_slices(["a", "c", "e"]).repeat()

  # Custom your batching, sharding, prefetching, etc.
  global_batch_size = 4
  batch_size = input_context.get_per_replica_batch_size(global_batch_size)
  dataset = dataset.batch(batch_size)
  dataset = dataset.shard(
      input_context.num_input_pipelines,
      input_context.input_pipeline_id)

  # Apply the preprocessing layer(s) to the tf.data.Dataset
  def preprocess_with_kpl(input):
    return layer(input)

  processed_ds = dataset.map(preprocess_with_kpl)
  return processed_ds

distributed_dataset = strategy.distribute_datasets_from_function(dataset_fn)

# Print out a few example batches.
distributed_dataset_iterator = iter(distributed_dataset)
for _ in range(3):
  print(next(distributed_dataset_iterator))
```

Note that if you are training with `tf.distribute.experimental.ParameterServerStrategy`, you'll also call `tf.distribute.experimental.coordinator.ClusterCoordinator.create_per_worker_dataset`

```
@tf.function
def per_worker_dataset_fn():
  return strategy.distribute_datasets_from_function(dataset_fn)

per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
per_worker_iterator = iter(per_worker_dataset)
```


For Tensorflow Transform, as mentioned above, the Analyze stage is done separately from training and thus omitted here. See the [tutorial](https://www.tensorflow.org/tfx/tutorials/transform/census) for a detailed how-to. Usually, this stage includes creating a `tf.Transform` preprocessing function and transforming the data in an [Apache Beam](https://beam.apache.org/) pipeline with this preprocessing function. At the end of the Analyze stage, the output can be exported as a TensorFlow graph which you can use for both training and serving. Our example covers only the training pipeline part:

```
with strategy.scope():
  # working_dir contains the tf.Transform output.
  tf_transform_output = tft.TFTransformOutput(working_dir)
  # Loading from working_dir to create a Keras layer for applying the tf.Transform output to data
  tft_layer = tf_transform_output.transform_features_layer()
  ...

def dataset_fn(input_context):
  ...
  dataset.map(tft_layer, num_parallel_calls=tf.data.AUTOTUNE)
  ...
  return dataset

distributed_dataset = strategy.distribute_datasets_from_function(dataset_fn)
```

## Partial batches

Partial batches are encountered when: 1) `tf.data.Dataset` instances that users create may contain batch sizes that are not evenly divisible by the number of replicas; or 2) when the cardinality of the dataset instance is not divisible by the batch size. This means that when the dataset is distributed over multiple replicas, the `next` call on some iterators will result in an `tf.errors.OutOfRangeError`. To handle this use case, `tf.distribute` returns dummy batches of batch size `0` on replicas that do not have any more data to process.


For the single-worker case, if the data is not returned by the `next` call on the iterator, dummy batches of 0 batch size are created and used along with the real data in the dataset. In the case of partial batches, the last global batch of data will contain real data alongside dummy batches of data. The stopping condition for processing data now checks if any of the replicas have data. If there is no data on any of the replicas, you will get a `tf.errors.OutOfRangeError`.

For the multi-worker case, the boolean value representing presence of data on each of the workers is aggregated using cross replica communication and this is used to identify if all the workers have finished processing the distributed dataset. Since this involves cross worker communication there is some performance penalty involved.


## Caveats

* When using `tf.distribute.Strategy.experimental_distribute_dataset` APIs with a multi-worker setup, you pass a `tf.data.Dataset` that reads from files. If the `tf.data.experimental.AutoShardPolicy` is set to `AUTO` or `FILE`, the actual per-step batch size may be smaller than the one you defined for the global batch size. This can happen when the remaining elements in the file are less than the global batch size. You can either exhaust the dataset without depending on the number of steps to run, or set  `tf.data.experimental.AutoShardPolicy` to `DATA` to work around it.

* Stateful dataset transformations are currently not supported with `tf.distribute` and any stateful ops that the dataset may have are currently ignored. For example, if your dataset has a `map_fn` that uses `tf.random.uniform` to rotate an image, then you have a dataset graph that depends on state (i.e the random seed) on the local machine where the python process is being executed.

* Experimental `tf.data.experimental.OptimizationOptions` that are disabled by default can in certain contexts—such as when used together with `tf.distribute`—cause a performance degradation. You should only enable them after you validate that they benefit the performance of your workload in a distribute setting.

* Please refer to [this guide](https://www.tensorflow.org/guide/data_performance) for how to optimize your input pipeline with `tf.data` in general. A few additional tips:
 * If you have multiple workers and are using `tf.data.Dataset.list_files` to create a dataset from all files matching one or more glob patterns, remember to set the `seed` argument or set `shuffle=False` so that each worker shard the file consistently.

 * If your input pipeline includes both shuffling the data on record level and parsing the data, unless the unparsed data is significantly larger than the parsed data (which is usually not the case), shuffle first and then parse, as shown in the following example. This may benefit memory usage and performance.
```
d = tf.data.Dataset.list_files(pattern, shuffle=False)
d = d.shard(num_workers, worker_index)
d = d.repeat(num_epochs)
d = d.shuffle(shuffle_buffer_size)
d = d.interleave(tf.data.TFRecordDataset,
                 cycle_length=num_readers, block_length=1)
d = d.map(parser_fn, num_parallel_calls=num_map_threads)
```
 * `tf.data.Dataset.shuffle(buffer_size, seed=None, reshuffle_each_iteration=None)` maintain an internal buffer of `buffer_size` elements, and thus reducing `buffer_size` could aleviate OOM issue.

* The order in which the data is processed by the workers when using `tf.distribute.experimental_distribute_dataset` or `tf.distribute.distribute_datasets_from_function` is not guaranteed. This is typically required if you are using `tf.distribute` to scale prediction. You can however insert an index for each element in the batch and order outputs accordingly. The following snippet is an example of how to order outputs.

Note: `tf.distribute.MirroredStrategy` is used here for the sake of convenience. You only need to reorder inputs when you are using multiple workers, but `tf.distribute.MirroredStrategy` is used to distribute training on a single worker.


```
mirrored_strategy = tf.distribute.MirroredStrategy()
dataset_size = 24
batch_size = 6
dataset = tf.data.Dataset.range(dataset_size).enumerate().batch(batch_size)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

def predict(index, inputs):
  outputs = 2 * inputs
  return index, outputs

result = {}
for index, inputs in dist_dataset:
  output_index, outputs = mirrored_strategy.run(predict, args=(index, inputs))
  indices = list(mirrored_strategy.experimental_local_results(output_index))
  rindices = []
  for a in indices:
    rindices.extend(a.numpy())
  outputs = list(mirrored_strategy.experimental_local_results(outputs))
  routputs = []
  for a in outputs:
    routputs.extend(a.numpy())
  for i, value in zip(rindices, routputs):
    result[i] = value

print(result)
```

<a name="tensorinputs">
## Tensor inputs instead of tf.data

Sometimes users cannot use a `tf.data.Dataset` to represent their input and subsequently
the above mentioned APIs to distribute the dataset to multiple devices.
 In such cases you can use raw tensors or inputs from a generator.

### Use experimental_distribute_values_from_function for arbitrary tensor inputs
`strategy.run` accepts `tf.distribute.DistributedValues` which is the output of
`next(iterator)`. To pass the tensor values, use
`tf.distribute.Strategy.experimental_distribute_values_from_function` to construct
`tf.distribute.DistributedValues` from raw tensors. The user will have to specify their own batching and sharding logic in the input function with this option, which can be done using the `tf.distribute.experimental.ValueContext` input object.


```
mirrored_strategy = tf.distribute.MirroredStrategy()

def value_fn(ctx):
  return tf.constant(ctx.replica_id_in_sync_group)

distributed_values = mirrored_strategy.experimental_distribute_values_from_function(value_fn)
for _ in range(4):
  result = mirrored_strategy.run(lambda x: x, args=(distributed_values,))
  print(result)
```

### Use tf.data.Dataset.from_generator if your input is from a generator

If you have a generator function that you want to use, you can create a `tf.data.Dataset` instance using the `from_generator` API.

Note: This is currently not supported for `tf.distribute.TPUStrategy`.


```
mirrored_strategy = tf.distribute.MirroredStrategy()
def input_gen():
  while True:
    yield np.random.rand(4)

# use Dataset.from_generator
dataset = tf.data.Dataset.from_generator(
    input_gen, output_types=(tf.float32), output_shapes=tf.TensorShape([4]))
dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
iterator = iter(dist_dataset)
for _ in range(4):
  result = mirrored_strategy.run(lambda x: x, args=(next(iterator),))
  print(result)
```
