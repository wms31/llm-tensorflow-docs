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

# Parameter server training with ParameterServerStrategy

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/tutorials/distribute/parameter_server_training"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/parameter_server_training.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/parameter_server_training.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/distribute/parameter_server_training.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

## Overview

[Parameter server training](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf) is a common data-parallel method to scale up model training on multiple machines.

A parameter server training cluster consists of _workers_ and _parameter servers_. Variables are created on parameter servers and they are read and updated by workers in each step. By default, workers read and update these variables independently without synchronizing with each other. This is why sometimes parameter server-style training is called _asynchronous training_.

In TensorFlow 2, parameter server training is powered by the `tf.distribute.ParameterServerStrategy` class, which distributes the training steps to a cluster that scales up to thousands of workers (accompanied by parameter servers).

### Supported training methods

There are two main supported training methods:

- The Keras `Model.fit` API: if you prefer a high-level abstraction and handling of training. This is generally recommended if you are training a `tf.keras.Model`.
- A custom training loop: if you prefer to define the details of your training loop (you can refer to guides on [Custom training](../customization/custom_training_walkthrough.ipynb), [Writing a training loop from scratch
](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch) and [Custom training loop with Keras and MultiWorkerMirroredStrategy](multi_worker_with_ctl.ipynb) for more details).

### A cluster with jobs and tasks

Regardless of the API of choice (`Model.fit` or a custom training loop), distributed training in TensorFlow 2 involves a `'cluster'` with several `'jobs'`, and each of the jobs may have one or more `'tasks'`.

When using parameter server training, it is recommended to have:

- One _coordinator_ job (which has the job name `chief`)
- Multiple _worker_ jobs (job name `worker`)
- Multiple _parameter server_ jobs (job name `ps`)

The _coordinator_ creates resources, dispatches training tasks, writes checkpoints, and deals with task failures. The _workers_ and _parameter servers_ run `tf.distribute.Server` instances that listen for requests from the coordinator.

### Parameter server training with the `Model.fit` API

Parameter server training with the `Model.fit` API requires the coordinator to use a `tf.distribute.ParameterServerStrategy` object. Similar to `Model.fit` usage with no strategy, or with other strategies, the workflow involves creating and compiling the model, preparing the callbacks, and calling `Model.fit`.

### Parameter server training with a custom training loop

With custom training loops, the `tf.distribute.coordinator.ClusterCoordinator` class is the key component used for the coordinator.

- The `ClusterCoordinator` class needs to work in conjunction with a `tf.distribute.ParameterServerStrategy` object.
- This `tf.distribute.Strategy` object is needed to provide the information of the cluster and is used to define a training step, as demonstrated in [Custom training with tf.distribute.Strategy](custom_training.ipynb).
- The `ClusterCoordinator` object then dispatches the execution of these training steps to remote workers.

The most important API provided by the `ClusterCoordinator` object is `schedule`:

- The `schedule` API enqueues a `tf.function` and returns a future-like `RemoteValue` immediately.
- The queued functions will be dispatched to remote workers in background threads and their `RemoteValue`s will be filled asynchronously.
- Since `schedule` doesn’t require worker assignment, the `tf.function` passed in can be executed on any available worker.
- If the worker it is executed on becomes unavailable before its completion, the function will be retried on another available worker.
- Because of this fact and the fact that function execution is not atomic, a single function call may be executed more than once.

In addition to dispatching remote functions, the `ClusterCoordinator` also helps
to create datasets on all the workers and rebuild these datasets when a worker recovers from failure.

## Tutorial setup

The tutorial will branch into `Model.fit` and custom training loop paths, and you can choose the one that fits your needs. Sections other than "Training with X" are applicable to both paths.


```
!pip install portpicker
```


```
#@title
import multiprocessing
import os
import random
import portpicker
import tensorflow as tf
```

## Cluster setup

As mentioned above, a parameter server training cluster requires a coordinator task that runs your training program, one or several workers and parameter server tasks that run TensorFlow servers—`tf.distribute.Server`—and possibly an additional evaluation task that runs sidecar evaluation (refer to the [sidecar evaluation section](#sidecar_evaluation) below). The requirements to set them up are:

- The coordinator task needs to know the addresses and ports of all other TensorFlow servers, except the evaluator.
- The workers and parameter servers need to know which port they need to listen to. For the sake of simplicity, you can usually pass in the complete cluster information when creating TensorFlow servers on these tasks.
- The evaluator task doesn’t have to know the setup of the training cluster. If it does, it should not attempt to connect to the training cluster.
- Workers and parameter servers should have task types as `"worker"` and `"ps"`, respectively. The coordinator should use `"chief"` as the task type for legacy reasons.

In this tutorial, you will create an in-process cluster so that the whole parameter server training can be run in Colab. You will learn how to set up [real clusters](#real_clusters) in a later section.

### In-process cluster

You will start by creating several TensorFlow servers in advance and you will connect to them later. Note that this is only for the purpose of this tutorial's demonstration, and in real training the servers will be started on `"worker"` and `"ps"` machines.


```
def create_in_process_cluster(num_workers, num_ps):
  """Creates and starts local servers and returns the cluster_resolver."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {}
  cluster_dict["worker"] = ["localhost:%s" % port for port in worker_ports]
  if num_ps > 0:
    cluster_dict["ps"] = ["localhost:%s" % port for port in ps_ports]

  cluster_spec = tf.train.ClusterSpec(cluster_dict)

  # Workers need some inter_ops threads to work properly.
  worker_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1

  for i in range(num_workers):
    tf.distribute.Server(
        cluster_spec,
        job_name="worker",
        task_index=i,
        config=worker_config,
        protocol="grpc")

  for i in range(num_ps):
    tf.distribute.Server(
        cluster_spec,
        job_name="ps",
        task_index=i,
        protocol="grpc")

  cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, rpc_layer="grpc")
  return cluster_resolver

# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

NUM_WORKERS = 3
NUM_PS = 2
cluster_resolver = create_in_process_cluster(NUM_WORKERS, NUM_PS)
```

The in-process cluster setup is frequently used in unit testing, such as [here](https://github.com/tensorflow/tensorflow/blob/eb4c40fc91da260199fa2aed6fe67d36ad49fafd/tensorflow/python/distribute/coordinator/cluster_coordinator_test.py#L447).

Another option for local testing is to launch processes on the local machine—check out [Multi-worker training with Keras](multi_worker_with_keras.ipynb) for an example of this approach.

## Instantiate a ParameterServerStrategy

Before you dive into the training code, let's instantiate a `tf.distribute.ParameterServerStrategy` object. Note that this is needed regardless of whether you are proceeding with `Model.fit` or a custom training loop. The `variable_partitioner` argument will be explained in the [Variable sharding section](#variable_sharding).


```
variable_partitioner = (
    tf.distribute.experimental.partitioners.MinSizePartitioner(
        min_shard_bytes=(256 << 10),
        max_shards=NUM_PS))

strategy = tf.distribute.ParameterServerStrategy(
    cluster_resolver,
    variable_partitioner=variable_partitioner)
```

In order to use GPUs for training, allocate GPUs visible to each worker. `ParameterServerStrategy` will use all the available GPUs on each worker, with the restriction that all workers should have the same number of GPUs available.

### Variable sharding

Variable sharding refers to splitting a variable into multiple smaller
variables, which are called _shards_. Variable sharding may be useful to distribute the network load when accessing these shards. It is also useful to distribute computation and storage of a normal variable across multiple parameter servers, for example, when using very large embeddings
that may not fit in a single machine's memory.

To enable variable sharding, you can pass in a `variable_partitioner` when
constructing a `ParameterServerStrategy` object. The `variable_partitioner` will
be invoked every time when a variable is created and it is expected to return
the number of shards along each dimension of the variable. Some out-of-box
`variable_partitioner`s are provided such as
`tf.distribute.experimental.partitioners.MinSizePartitioner`. It is recommended to use size-based partitioners like
`tf.distribute.experimental.partitioners.MinSizePartitioner` to avoid
partitioning small variables, which could have a negative impact on model training
speed.

When a `variable_partitioner` is passed in, and you create a variable directly
under `Strategy.scope`, the variable will become a container type with a `variables`
property, which provides access to the list of shards. In most cases, this
container will be automatically converted to a Tensor by concatenating all the
shards. As a result, it can be used as a normal variable. On the other hand,
some TensorFlow methods such as `tf.nn.embedding_lookup` provide efficient
implementation for this container type and in these methods automatic
concatenation will be avoided.

Refer to the API docs of `tf.distribute.ParameterServerStrategy` for more details.

## Training with `Model.fit`
<a id="training_with_modelfit"></a>

Keras provides an easy-to-use training API via `Model.fit` that handles the training loop under the hood, with the flexibility of an overridable `train_step`, and callbacks which provide functionalities such as checkpoint saving or summary saving for TensorBoard. With `Model.fit`, the same training code can be used with other strategies with a simple swap of the strategy object.

### Input data

Keras `Model.fit` with `tf.distribute.ParameterServerStrategy` can take input data in the form of a `tf.data.Dataset`, `tf.distribute.DistributedDataset`, or a `tf.keras.utils.experimental.DatasetCreator`, with `Dataset` being the recommended option for ease of use. If you encounter memory issues using `Dataset`, however, you may need to use `DatasetCreator` with a callable `dataset_fn` argument (refer to the `tf.keras.utils.experimental.DatasetCreator` API documentation for details).

If you transform your dataset into a `tf.data.Dataset`, you should use `Dataset.shuffle` and `Dataset.repeat`, as demonstrated in the code example below.

- Keras `Model.fit` with parameter server training assumes that each worker receives the same dataset, except when it is shuffled differently. Therefore, by calling `Dataset.shuffle`, you ensure more even iterations over the data.
- Because workers do not synchronize, they may finish processing their datasets at different times. Therefore, the easiest way to define epochs with parameter server training is to use `Dataset.repeat`—which repeats a dataset indefinitely when called without an argument—and specify the `steps_per_epoch` argument in the `Model.fit` call.

Refer to the "Training workflows" section of the [tf.data guide](../../guide/data.ipynb) for more details on `shuffle` and `repeat`.


```
global_batch_size = 64

x = tf.random.uniform((10, 10))
y = tf.random.uniform((10,))

dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10).repeat()
dataset = dataset.batch(global_batch_size)
dataset = dataset.prefetch(2)
```

If you instead create your dataset with `tf.keras.utils.experimental.DatasetCreator`, the code in `dataset_fn` will be invoked on the input device, which is usually the CPU, on each of the worker machines.


### Model construction and compiling

Now, you will create a `tf.keras.Model`—a trivial `tf.keras.models.Sequential` model for demonstration purposes—followed by a `Model.compile` call to incorporate components, such as an optimizer, metrics, and other parameters such as `steps_per_execution`:


```
with strategy.scope():
  model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])

  model.compile(tf.keras.optimizers.legacy.SGD(), loss="mse", steps_per_execution=10)
```

### Callbacks and training

<a id="callbacks-and-training"> </a>

Before you call Keras `Model.fit` for the actual training, prepare any needed [callbacks](https://www.tensorflow.org/guide/keras/train_and_evaluate) for common tasks, such as:

- `tf.keras.callbacks.ModelCheckpoint`: saves the model at a certain frequency, such as after every epoch.
- `tf.keras.callbacks.BackupAndRestore`: provides fault tolerance by backing up the model and current epoch number, if the cluster experiences unavailability (such as abort or preemption). You can then restore the training state upon a restart from a job failure, and continue training from the beginning of the interrupted epoch.
- `tf.keras.callbacks.TensorBoard`: periodically writes model logs in summary files that can be visualized in the TensorBoard tool.

Note: Due to performance considerations, custom callbacks cannot have batch level callbacks overridden when used with `ParameterServerStrategy`. Please modify your custom callbacks to make them epoch level calls, and adjust `steps_per_epoch` to a suitable value. In addition, `steps_per_epoch` is a required argument for `Model.fit` when used with `ParameterServerStrategy`.


```
working_dir = "/tmp/my_working_dir"
log_dir = os.path.join(working_dir, "log")
ckpt_filepath = os.path.join(working_dir, "ckpt")
backup_dir = os.path.join(working_dir, "backup")

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_filepath),
    tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir),
]

model.fit(dataset, epochs=5, steps_per_epoch=20, callbacks=callbacks)
```

### Direct usage with `ClusterCoordinator` (optional)

Even if you choose the `Model.fit` training path, you can optionally instantiate a `tf.distribute.coordinator.ClusterCoordinator` object to schedule other functions you would like to be executed on the workers. Refer to the [Training with a custom training loop](#training_with_custom_training_loop) section for more details and examples.

## Training with a custom training loop

<a id="training_with_custom_training_loop"> </a>

Using custom training loops with `tf.distribute.Strategy` provides great flexibility to define training loops. With the `ParameterServerStrategy` defined above (as `strategy`), you will use a `tf.distribute.coordinator.ClusterCoordinator` to dispatch the execution of training steps to remote workers.

Then, you will create a model, define a dataset, and define a step function, as you have done in the training loop with other `tf.distribute.Strategy`s. You can find more details in the [Custom training with tf.distribute.Strategy](custom_training.ipynb) tutorial.

To ensure efficient dataset prefetching, use the recommended distributed dataset creation APIs mentioned in the [Dispatch training steps to remote workers](#dispatch_training_steps_to_remote_workers) section below. Also, make sure to call `Strategy.run` inside `worker_fn` to take full advantage of GPUs allocated to workers. The rest of the steps are the same for training with or without GPUs.

Let’s create these components in the following steps:


### Set up the data

First, write a function that creates a dataset.

If you would like to preprocess the data with [Keras preprocessing layers](https://www.tensorflow.org/guide/keras/preprocessing_layers) or [Tensorflow Transform layers](https://www.tensorflow.org/tfx/tutorials/transform/simple), create these layers **outside the `dataset_fn`** and **under `Strategy.scope`**, like you would do for any other Keras layers. This is because the `dataset_fn` will be wrapped into a `tf.function` and then executed on each worker to generate the data pipeline.

If you don't follow the above procedure, creating the layers might create Tensorflow states which will be lifted out of the `tf.function` to the coordinator. Thus, accessing them on workers would incur repetitive RPC calls between coordinator and workers, and cause significant slowdown.

Placing the layers under `Strategy.scope` will instead create them on all workers. Then, you will apply the transformation inside the `dataset_fn` via `tf.data.Dataset.map`. Refer to _Data preprocessing_ in the [Distributed input](input.ipynb) tutorial for more information on data preprocessing with distributed input.


```
feature_vocab = [
    "avenger", "ironman", "batman", "hulk", "spiderman", "kingkong", "wonder_woman"
]
label_vocab = ["yes", "no"]

with strategy.scope():
  feature_lookup_layer = tf.keras.layers.StringLookup(
      vocabulary=feature_vocab,
      mask_token=None)
  label_lookup_layer = tf.keras.layers.StringLookup(
      vocabulary=label_vocab,
      num_oov_indices=0,
      mask_token=None)

  raw_feature_input = tf.keras.layers.Input(
      shape=(3,),
      dtype=tf.string,
      name="feature")
  feature_id_input = feature_lookup_layer(raw_feature_input)
  feature_preprocess_stage = tf.keras.Model(
      {"features": raw_feature_input},
      feature_id_input)

  raw_label_input = tf.keras.layers.Input(
      shape=(1,),
      dtype=tf.string,
      name="label")
  label_id_input = label_lookup_layer(raw_label_input)

  label_preprocess_stage = tf.keras.Model(
      {"label": raw_label_input},
      label_id_input)
```

Generate toy examples in a dataset:


```
def feature_and_label_gen(num_examples=200):
  examples = {"features": [], "label": []}
  for _ in range(num_examples):
    features = random.sample(feature_vocab, 3)
    label = ["yes"] if "avenger" in features else ["no"]
    examples["features"].append(features)
    examples["label"].append(label)
  return examples

examples = feature_and_label_gen()
```

Then, create the training dataset wrapped in a `dataset_fn`:


```
def dataset_fn(_):
  raw_dataset = tf.data.Dataset.from_tensor_slices(examples)

  train_dataset = raw_dataset.map(
      lambda x: (
          {"features": feature_preprocess_stage(x["features"])},
          label_preprocess_stage(x["label"])
      )).shuffle(200).batch(32).repeat()
  return train_dataset
```

### Build the model

Next, create the model and other objects. Make sure to create all variables under `Strategy.scope`.


```
# These variables created under the `Strategy.scope` will be placed on parameter
# servers in a round-robin fashion.
with strategy.scope():
  # Create the model. The input needs to be compatible with Keras processing layers.
  model_input = tf.keras.layers.Input(
      shape=(3,), dtype=tf.int64, name="model_input")

  emb_layer = tf.keras.layers.Embedding(
      input_dim=len(feature_lookup_layer.get_vocabulary()), output_dim=16384)
  emb_output = tf.reduce_mean(emb_layer(model_input), axis=1)
  dense_output = tf.keras.layers.Dense(
      units=1, activation="sigmoid",
      kernel_regularizer=tf.keras.regularizers.L2(1e-4),
  )(emb_output)
  model = tf.keras.Model({"features": model_input}, dense_output)

  optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=0.1)
  accuracy = tf.keras.metrics.Accuracy()
```

Let's confirm that the use of `FixedShardsPartitioner` split all variables into two shards and that each shard was assigned to a different parameter server:


```
assert len(emb_layer.weights) == 2
assert emb_layer.weights[0].shape == (4, 16384)
assert emb_layer.weights[1].shape == (4, 16384)

print(emb_layer.weights[0].device)
print(emb_layer.weights[1].device)

```

### Define the training step
Third, create the training step wrapped into a `tf.function`:


```
@tf.function
def step_fn(iterator):

  def replica_fn(batch_data, labels):
    with tf.GradientTape() as tape:
      pred = model(batch_data, training=True)
      per_example_loss = tf.keras.losses.BinaryCrossentropy(
          reduction=tf.keras.losses.Reduction.NONE)(labels, pred)
      loss = tf.nn.compute_average_loss(per_example_loss)
      model_losses = model.losses
      if model_losses:
        loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
    accuracy.update_state(labels, actual_pred)
    return loss

  batch_data, labels = next(iterator)
  losses = strategy.run(replica_fn, args=(batch_data, labels))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)
```

In the above training step function, calling `Strategy.run` and `Strategy.reduce` in the `step_fn` can support multiple GPUs per worker. If the workers have GPUs allocated, `Strategy.run` will distribute the datasets on multiple replicas (GPUs). Their parallel calls to `tf.nn.compute_average_loss()` compute the average of the loss across the replicas (GPUs) of one worker, independent of the total number of workers.

### Dispatch training steps to remote workers
<a id="dispatch_training_steps_to_remote_workers"> </a>

After all the computations are defined by `ParameterServerStrategy`, you will use the `tf.distribute.coordinator.ClusterCoordinator` class to create resources and distribute the training steps to remote workers.

Let’s first create a `ClusterCoordinator` object and pass in the strategy object:


```
coordinator = tf.distribute.coordinator.ClusterCoordinator(strategy)
```

Then, create a per-worker dataset and an iterator using the `ClusterCoordinator.create_per_worker_dataset` API, which replicates the dataset to all workers. In the `per_worker_dataset_fn` below, wrapping the `dataset_fn` into `strategy.distribute_datasets_from_function` is recommended to allow efficient prefetching to GPUs seamlessly.


```
@tf.function
def per_worker_dataset_fn():
  return strategy.distribute_datasets_from_function(dataset_fn)

per_worker_dataset = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
per_worker_iterator = iter(per_worker_dataset)
```

The final step is to distribute the computation to remote workers using `ClusterCoordinator.schedule`:

- The `schedule` method enqueues a `tf.function` and returns a future-like `RemoteValue` immediately. The queued functions will be dispatched to remote workers in background threads and the `RemoteValue` will be filled asynchronously.
- The `join` method (`ClusterCoordinator.join`) can be used to wait until all scheduled functions are executed.


```
num_epochs = 4
steps_per_epoch = 5
for i in range(num_epochs):
  accuracy.reset_states()
  for _ in range(steps_per_epoch):
    coordinator.schedule(step_fn, args=(per_worker_iterator,))
  # Wait at epoch boundaries.
  coordinator.join()
  print("Finished epoch %d, accuracy is %f." % (i, accuracy.result().numpy()))
```

Here is how you can fetch the result of a `RemoteValue`:


```
loss = coordinator.schedule(step_fn, args=(per_worker_iterator,))
print("Final loss is %f" % loss.fetch())
```

Alternatively, you can launch all steps and do something while waiting for
completion:

```python
for _ in range(total_steps):
  coordinator.schedule(step_fn, args=(per_worker_iterator,))
while not coordinator.done():
  time.sleep(10)
  # Do something like logging metrics or writing checkpoints.
```

For the complete training and serving workflow for this particular example, please check out this [test](https://github.com/keras-team/keras/blob/master/keras/integration_test/parameter_server_keras_preprocessing_test.py).


### More about dataset creation

The dataset in the above code is created using the `ClusterCoordinator.create_per_worker_dataset` API. It creates one dataset per worker and returns a container object. You can call the `iter` method on it to create a per-worker iterator. The per-worker iterator contains one iterator per worker and the corresponding slice of a worker will be substituted in the input argument of the function passed to the `ClusterCoordinator.schedule` method before the function is executed on a particular worker.

The `ClusterCoordinator.schedule` method assumes workers are equivalent and thus assumes the datasets on different workers are the same (except that they may be shuffled differently). Because of this, it is also recommended to repeat datasets, and schedule a finite number of steps instead of relying on receiving an `OutOfRangeError` from a dataset.

Another important note is that `tf.data` datasets don’t support implicit serialization and deserialization across task boundaries. So it is important to create the whole dataset inside the function passed to `ClusterCoordinator.create_per_worker_dataset`. The `create_per_worker_dataset` API can also directly take a `tf.data.Dataset` or `tf.distribute.DistributedDataset` as input.

## Evaluation

The two main approaches to performing evaluation with `tf.distribute.ParameterServerStrategy` training are inline evaluation and sidecar evaluation. Each has its own pros and cons as described below. The inline evaluation method is recommended if you don't have a preference. For users using `Model.fit`, `Model.evaluate` uses inline (distributed) evaluation under the hood.

### Inline evaluation

In this method, the coordinator alternates between training and evaluation, and thus it is called _inline evaluation_.

There are several benefits of inline evaluation. For example:

- It can support large evaluation models and evaluation datasets that a single task cannot hold.
- The evaluation results can be used to make decisions for training the next epoch, for example, whether to stop training early.

There are two ways to implement inline evaluation: direct evaluation and distributed evaluation.

- **Direct evaluation**: For small models and evaluation datasets, the coordinator can run evaluation directly on the distributed model with the evaluation dataset on the coordinator:


```
eval_dataset = tf.data.Dataset.from_tensor_slices(
    feature_and_label_gen(num_examples=16)).map(
          lambda x: (
              {"features": feature_preprocess_stage(x["features"])},
              label_preprocess_stage(x["label"])
          )).batch(8)

eval_accuracy = tf.keras.metrics.Accuracy()

for batch_data, labels in eval_dataset:
  pred = model(batch_data, training=False)
  actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
  eval_accuracy.update_state(labels, actual_pred)

print("Evaluation accuracy: %f" % eval_accuracy.result())
```

- **Distributed evaluation**: For large models or datasets that are infeasible to run directly on the coordinator, the coordinator task can distribute evaluation tasks to the workers via the `ClusterCoordinator.schedule`/`ClusterCoordinator.join` methods:


```
with strategy.scope():
  # Define the eval metric on parameter servers.
  eval_accuracy = tf.keras.metrics.Accuracy()

@tf.function
def eval_step(iterator):
  def replica_fn(batch_data, labels):
    pred = model(batch_data, training=False)
    actual_pred = tf.cast(tf.greater(pred, 0.5), tf.int64)
    eval_accuracy.update_state(labels, actual_pred)
  batch_data, labels = next(iterator)
  strategy.run(replica_fn, args=(batch_data, labels))

def eval_dataset_fn():
  return tf.data.Dataset.from_tensor_slices(
      feature_and_label_gen(num_examples=16)).map(
          lambda x: (
              {"features": feature_preprocess_stage(x["features"])},
              label_preprocess_stage(x["label"])
          )).shuffle(16).repeat().batch(8)

per_worker_eval_dataset = coordinator.create_per_worker_dataset(eval_dataset_fn)
per_worker_eval_iterator = iter(per_worker_eval_dataset)

eval_steps_per_epoch = 2
for _ in range(eval_steps_per_epoch):
  coordinator.schedule(eval_step, args=(per_worker_eval_iterator,))
coordinator.join()
print("Evaluation accuracy: %f" % eval_accuracy.result())
```

#### Enabling exactly-once evaluation
<a id="exactly_once_evaluation"></a>

The `schedule` and `join` methods of `tf.distribute.coordinator.ClusterCoordinator` don’t support visitation guarantees or exactly-once semantics by default. In other words, in the above example there is no guarantee that all evaluation examples in a dataset will be evaluated exactly once; some may not be visited and some may be evaluated multiple times.

Exactly-once evaluation may be preferred to reduce the variance of evaluation across epochs, and improve model selection done via early stopping,  hyperparameter tuning, or other methods. There are different ways to enable exactly-once evaluation:

- With a `Model.fit/.evaluate` workflow, it can be enabled by adding an argument to `Model.compile`. Refer to docs for the `pss_evaluation_shards` argument.
- The `tf.data` service API can be used to provide exactly-once visitation for evaluation when using `ParameterServerStrategy` (refer to the _Dynamic Sharding_ section of the `tf.data.experimental.service` API documentation).
- [Sidecar evaluation](#sidecar_evaluation) provides exactly-once evaluation by default, since the evaluation happens on a single machine. However this can be much slower than performing evaluation distributed across many workers.

The first option, using `Model.compile`, is the suggested solution for most users.

Exactly-once evaluation has some limitations:

- It is not supported to write a custom distributed evaluation loop with an exactly-once visitation guarantee. File a GitHub issue if you need support for this.
- It cannot automatically handle computation of metrics that use the `Layer.add_metric` API. These should be excluded from evaluation, or reworked into `Metric` objects.

### Sidecar evaluation
<a id="sidecar_evaluation"></a>

Another method for defining and running an evaluation loop in `tf.distribute.ParameterServerStrategy` training is called _sidecar evaluation_, in which you create a dedicated evaluator task that repeatedly reads checkpoints and runs evaluation on the latest checkpoint (refer to [this guide](../../guide/checkpoint.ipynb) for more details on checkpointing). The coordinator and worker tasks do not spend any time on evaluation, so for a fixed number of iterations the overall training time should be shorter than using other evaluation methods. However, it requires an additional evaluator task and periodic checkpointing to trigger evaluation.

To write an evaluation loop for sidecar evaluation, you have two
options:

1. Use the `tf.keras.utils.SidecarEvaluator` API.
2. Create a custom evaluation loop.

Refer to the `tf.keras.utils.SidecarEvaluator` API documentation for more details on option 1.

Sidecar evaluation is supported only with a single task. This means:

*   It is guaranteed that each example is evaluated once. In the event the
    evaluator is preempted or restarted, it simply restarts the
    evaluation loop from the latest checkpoint, and the partial evaluation
    progress made before the restart is discarded.

*   However, running evaluation on a single task implies that a full evaluation
    can possibly take a long time.

*   If the size of the model is too large to fit into an evaluator's memory,
    single sidecar evaluation is not applicable.

Another caveat is that the `tf.keras.utils.SidecarEvaluator` implementation, and the custom
evaluation loop below, may skip some checkpoints because it always picks up the
latest checkpoint available, and during an evaluation epoch, multiple
checkpoints can be produced from the training cluster. You can write a custom
evaluation loop that evaluates every checkpoint, but it is not covered in this
tutorial. On the other hand, it may sit idle if checkpoints are produced less
frequently than how long it takes to run evaluation.

A custom evaluation loop provides more control over the details, such as choosing which checkpoint to evaluate, or providing any additional logic to run along with evaluation. The following is a possible custom sidecar evaluation loop:

```python
checkpoint_dir = ...
eval_model = ...
eval_data = ...
checkpoint = tf.train.Checkpoint(model=eval_model)

for latest_checkpoint in tf.train.checkpoints_iterator(
    checkpoint_dir):
  try:
    checkpoint.restore(latest_checkpoint).expect_partial()
  except (tf.errors.OpError,) as e:
    # checkpoint may be deleted by training when it is about to read it.
    continue

  # Optionally add callbacks to write summaries.
  eval_model.evaluate(eval_data)

  # Evaluation finishes when it has evaluated the last epoch.
  if latest_checkpoint.endswith('-{}'.format(train_epochs)):
    break
```

## Clusters in the real world
<a id="real_clusters"></a>

Note: this section is not necessary for running the tutorial code in this page.

In a real production environment, you will run all tasks in different processes on different machines. The simplest way to configure cluster information on each task is to set `"TF_CONFIG"` environment variables and use a `tf.distribute.cluster_resolver.TFConfigClusterResolver` to parse `"TF_CONFIG"`.

For a general description of `"TF_CONFIG"` environment variables, refer to "Setting up the `TF_CONFIG` environment variable" in the [Distributed training](../../guide/distributed_training.ipynb) guide.

If you start your training tasks using Kubernetes or other configuration templates, likely, these templates have already set `“TF_CONFIG"` for you.

### Set the `"TF_CONFIG"` environment variable

Suppose you have 3 workers and 2 parameter servers. Then the `"TF_CONFIG"` of worker 1 can be:

```python
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["host1:port", "host2:port", "host3:port"],
        "ps": ["host4:port", "host5:port"],
        "chief": ["host6:port"]
    },
    "task": {"type": "worker", "index": 1}
})
```

The `"TF_CONFIG"` of the evaluator can be:

```python
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "evaluator": ["host7:port"]
    },
    "task": {"type": "evaluator", "index": 0}
})
```

The `"cluster"` part in the above `"TF_CONFIG"` string for the evaluator is optional.

### If you use the same binary for all tasks

If you prefer to run all these tasks using a single binary, you will need to let your program branch into different roles at the very beginning:

```python
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
if cluster_resolver.task_type in ("worker", "ps"):
  # Start a TensorFlow server and wait.
elif cluster_resolver.task_type == "evaluator":
  # Run sidecar evaluation
else:
  # Run the coordinator.
```

The following code starts a TensorFlow server and waits, useful for the `"worker"` and `"ps"` roles:

```python
# Set the environment variable to allow reporting worker and ps failure to the
# coordinator. This is a workaround and won't be necessary in the future.
os.environ["GRPC_FAIL_FAST"] = "use_caller"

server = tf.distribute.Server(
    cluster_resolver.cluster_spec(),
    job_name=cluster_resolver.task_type,
    task_index=cluster_resolver.task_id,
    protocol=cluster_resolver.rpc_layer or "grpc",
    start=True)
server.join()
```

## Handling task failure

### Worker failure

Both the `tf.distribute.coordinator.ClusterCoordinator` custom training loop and `Model.fit` approaches provide built-in fault tolerance for worker failure. Upon worker recovery, the `ClusterCoordinator` invokes dataset re-creation on the workers.

### Parameter server or coordinator failure

However, when the coordinator sees a parameter server error, it will raise an `UnavailableError` or `AbortedError` immediately. You can restart the coordinator in this case. The coordinator itself can also become unavailable. Therefore, certain tooling is recommended in order to not lose the training progress:

- For `Model.fit`, you should use a `BackupAndRestore` callback, which handles the progress saving and restoration automatically. See [Callbacks and training](#callbacks-and-training) section above for an example.

- For a custom training loop, you should checkpoint the model variables periodically and load model variables from a checkpoint, if any, before training starts. The training progress can be inferred approximately from `optimizer.iterations` if an optimizer is checkpointed:

```python
checkpoint_manager = tf.train.CheckpointManager(
    tf.train.Checkpoint(model=model, optimizer=optimizer),
    checkpoint_dir,
    max_to_keep=3)
if checkpoint_manager.latest_checkpoint:
  checkpoint = checkpoint_manager.checkpoint
  checkpoint.restore(
      checkpoint_manager.latest_checkpoint).assert_existing_objects_matched()

global_steps = int(optimizer.iterations.numpy())
starting_epoch = global_steps // steps_per_epoch

for _ in range(starting_epoch, num_epochs):
  for _ in range(steps_per_epoch):
    coordinator.schedule(step_fn, args=(per_worker_iterator,))
  coordinator.join()
  checkpoint_manager.save()
```

### Fetching a `RemoteValue`

Fetching a `RemoteValue` is guaranteed to succeed if a function is executed successfully. This is because currently the return value is immediately copied to the coordinator after a function is executed. If there is any worker failure during the copy, the function will be retried on another available worker. Therefore, if you want to optimize for performance, you can schedule functions without a return value.

## Error reporting

Once the coordinator sees an error such as `UnavailableError` from parameter servers or other application errors such as an `InvalidArgument` from `tf.debugging.check_numerics`, it will cancel all pending and queued functions before raising the error. Fetching their corresponding `RemoteValue`s will raise a `CancelledError`.

After an error is raised, the coordinator will not raise the same error or any error from cancelled functions.

## Performance improvement

There are several possible reasons you may experience performance issues when you train with `tf.distribute.ParameterServerStrategy` and `tf.distribute.coordinator.ClusterCoordinator`.

One common reason is that the parameter servers have unbalanced load and some heavily-loaded parameter servers have reached capacity. There can also be multiple root causes. Some simple methods to mitigate this issue are to:

1. Shard your large model variables via specifying a `variable_partitioner` when constructing a `ParameterServerStrategy`.
2. Avoid creating a hotspot variable that is required by all parameter servers in a single step, by both:

  1) Using a constant learning rate or subclass `tf.keras.optimizers.schedules.LearningRateSchedule` in optimizers. This is because the default behavior is that the learning rate will become a variable placed on a particular parameter server, and requested by all other parameter servers in each step); and

  2) Using a `tf.keras.optimizers.legacy.Optimizer` (the standard `tf.keras.optimizers.Optimizer`s could still lead to hotspot variables).
3. Shuffle your large vocabularies before passing them to Keras preprocessing layers.

Another possible reason for performance issues is the coordinator. The implementation of `schedule`/`join` is Python-based and thus may have threading overhead. Also, the latency between the coordinator and the workers can be large. If this is the case:

- For `Model.fit`, you can set the `steps_per_execution` argument provided at `Model.compile` to a value larger than 1.

- For a custom training loop, you can pack multiple steps into a single `tf.function`:

```python
steps_per_invocation = 10

@tf.function
def step_fn(iterator):
  for _ in range(steps_per_invocation):
    features, labels = next(iterator)
    def replica_fn(features, labels):
      ...

    strategy.run(replica_fn, args=(features, labels))
```

As the library is optimized further, hopefully most users won't have to manually pack steps in the future.

In addition, a small trick for performance improvement is to schedule functions without a return value as explained in the [handling task failure section](#handling_task_failure) above.

## Known limitations

<a id="known_limitations"> </a>

Most of the known limitations are already covered in the above sections. This section provides a summary.

### `ParameterServerStrategy` general

- `os.environment["grpc_fail_fast"]="use_caller"` is needed on every task including the coordinator, to make fault tolerance work properly.
- Synchronous parameter server training is not supported.
- It is usually necessary to pack multiple steps into a single function to achieve optimal performance.
- It is not supported to load a saved_model via `tf.saved_model.load` containing sharded variables. Note loading such a saved_model using TensorFlow Serving is expected to work (refer to the [serving tutorial](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple) for details).
- It is not supported to recover from parameter server failure without restarting the coordinator task.
- Creation of `tf.lookup.StaticHashTable`, commonly employed by some Keras preprocessing layers, such as `tf.keras.layers.IntegerLookup`, `tf.keras.layers.StringLookup`, and `tf.keras.layers.TextVectorization`, should be placed under `Strategy.scope`. Otherwise, resources will be placed on the coordinator, and lookup RPCs from workers to the coordinator incur performance implications.


### `Model.fit` specifics

- `steps_per_epoch` argument is required in `Model.fit`. You can select a value that provides appropriate intervals in an epoch.
- `ParameterServerStrategy` does not have support for custom callbacks that have batch-level calls for performance reasons. You should convert those calls into epoch-level calls with suitably picked `steps_per_epoch`, so that they are called every `steps_per_epoch` number of steps. Built-in callbacks are not affected: their batch-level calls have been modified to be performant. Supporting batch-level calls for `ParameterServerStrategy` is being planned.
- For the same reason, unlike other strategies, progress bars and metrics are logged only at epoch boundaries.
- `run_eagerly` is not supported.


### Custom training loop specifics

- `ClusterCoordinator.schedule` doesn't support visitation guarantees for a dataset in general, although a visitation guarantee for evaluation is possible through `Model.fit/.evaluate`. See [Enabling exactly-once evaluation](#exactly_once_evaluation).
- When `ClusterCoordinator.create_per_worker_dataset` is used with a callable as input, the whole dataset must be created inside the function passed to it.
- `tf.data.Options` is ignored in a dataset created by `ClusterCoordinator.create_per_worker_dataset`.
