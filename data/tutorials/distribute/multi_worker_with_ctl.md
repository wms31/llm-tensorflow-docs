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

# Custom training loop with Keras and MultiWorkerMirroredStrategy

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/tutorials/distribute/multi_worker_with_ctl"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/multi_worker_with_ctl.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/multi_worker_with_ctl.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/distribute/multi_worker_with_ctl.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

## Overview

This tutorial demonstrates how to perform multi-worker distributed training with a Keras model and with [custom training loops](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch) using the `tf.distribute.Strategy` API. The training loop is distributed via `tf.distribute.MultiWorkerMirroredStrategy`, such that a `tf.keras` model—designed to run on [single-worker](custom_training.ipynb)—can seamlessly work on multiple workers with minimal code changes. Custom training loops provide flexibility and a greater control on training, while also making it easier to debug the model. Learn more about [writing a basic training loop](../../guide/basic_training_loops.ipynb), [writing a training loop from scratch](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch) and [custom training](../customization/custom_training_walkthrough.ipynb).

If you are looking for how to use `MultiWorkerMirroredStrategy` with `tf.keras.Model.fit`, refer to this [tutorial](multi_worker_with_keras.ipynb) instead.

[Distributed Training in TensorFlow](../../guide/distributed_training.ipynb) guide is available for an overview of the distribution strategies TensorFlow supports for those interested in a deeper understanding of `tf.distribute.Strategy` APIs.

## Setup

First, some necessary imports.


```
import json
import os
import sys
```

Before importing TensorFlow, make a few changes to the environment:
* Disable all GPUs. This prevents errors caused by all workers trying to use the same GPU. In a real-world application, each worker would be on a different machine.


```
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

* Reset the `'TF_CONFIG'` environment variable (you'll see more about this later).


```
os.environ.pop('TF_CONFIG', None)
```

* Make sure that the current directory is on Python's path. This allows the notebook to import the files written by `%%writefile` later.



```
if '.' not in sys.path:
  sys.path.insert(0, '.')
```

Now import TensorFlow.


```
import tensorflow as tf
```

### Dataset and model definition

Next, create an `mnist.py` file with a simple model and dataset setup. This Python file will be used by the worker-processes in this tutorial:


```
%%writefile mnist.py

import os
import tensorflow as tf
import numpy as np

def mnist_dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the range [0, 255].
  # You need to convert them to float32 with values in the range [0, 1]
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000)
  return train_dataset

def dataset_fn(global_batch_size, input_context):
  batch_size = input_context.get_per_replica_batch_size(global_batch_size)
  dataset = mnist_dataset(batch_size)
  dataset = dataset.shard(input_context.num_input_pipelines,
                          input_context.input_pipeline_id)
  dataset = dataset.batch(batch_size)
  return dataset

def build_cnn_model():
  regularizer = tf.keras.regularizers.L2(1e-5)
  return tf.keras.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3,
                             activation='relu',
                             kernel_regularizer=regularizer),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128,
                            activation='relu',
                            kernel_regularizer=regularizer),
      tf.keras.layers.Dense(10, kernel_regularizer=regularizer)
  ])
```

## Multi-worker configuration

Now let's enter the world of multi-worker training. In TensorFlow, the `'TF_CONFIG'` environment variable is required for training on multiple machines. Each machine may have a different role. The `'TF_CONFIG'` variable used below is a JSON string that specifies the cluster configuration on each worker that is part of the cluster. This is the default method for specifying a cluster, using `cluster_resolver.TFConfigClusterResolver`,  but there are other options available in the `distribute.cluster_resolver` module. Learn more about setting up the `'TF_CONFIG'` variable in the [Distributed training guide](../../guide/distributed_training.ipynb).

### Describe your cluster
Here is an example configuration:


```
tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}
```

Note that `tf_config` is just a local variable in Python. To use it for training configuration, serialize it as a JSON and place it in a `'TF_CONFIG'` environment variable. Here is the same `'TF_CONFIG'` serialized as a JSON string:


```
json.dumps(tf_config)
```

There are two components of `'TF_CONFIG'`: `'cluster'` and `'task'`.

* `'cluster'` is the same for all workers and provides information about the training cluster, which is a dict consisting of different types of jobs such as `'worker'`. In multi-worker training with `MultiWorkerMirroredStrategy`, there is usually one `'worker'` that takes on a little more responsibility like saving checkpoints and writing summary files for TensorBoard in addition to what a regular `'worker'` does. Such a worker is referred to as the `'chief'` worker, and it is customary that the `'worker'` with `'index'` 0 is appointed as the chief `worker`.

* `'task'` provides information of the current task and is different on each worker. It specifies the `'type'` and `'index'` of that worker.

In this example, you set the task `'type'` to `'worker'` and the task `'index'` to `0`. This machine is the first worker and will be appointed as the chief worker and do more work than the others. Note that other machines will need to have the `'TF_CONFIG'` environment variable set as well, and it should have the same `'cluster'` dict, but different task `'type'` or task `'index'` depending on what the roles of those machines are.


For illustration purposes, this tutorial shows how one may set a `'TF_CONFIG'` with two workers on `'localhost'`.  In practice, users would create multiple workers on external IP addresses/ports, and set `'TF_CONFIG'` on each worker appropriately.

This example uses two workers. The first worker's `'TF_CONFIG'` is shown above. For the second worker, set `tf_config['task']['index']=1`.

### Environment variables and subprocesses in notebooks

Subprocesses inherit environment variables from their parent. So if you set an environment variable in this Jupyter Notebook process:


```
os.environ['GREETINGS'] = 'Hello TensorFlow!'
```

you can then access the environment variable from a subprocess:


```bash
%%bash
echo ${GREETINGS}
```

In the next section, you'll use this to pass the `'TF_CONFIG'` to the worker subprocesses. You would never really launch your jobs this way, but it's sufficient for the purposes of this tutorial: To demonstrate a minimal multi-worker example.

## MultiWorkerMirroredStrategy

Before training the model, first create an instance of `tf.distribute.MultiWorkerMirroredStrategy`:


```
strategy = tf.distribute.MultiWorkerMirroredStrategy()
```

Note: `'TF_CONFIG'` is parsed and TensorFlow's GRPC servers are started at the time you call `tf.distribute.MultiWorkerMirroredStrategy.` Therefore, you must set the `'TF_CONFIG'` environment variable before you instantiate a `tf.distribute.Strategy`. To save time in this illustrative example, this is not demonstrated in this tutorial, so that servers do not need to start. You can find a full example in the last section of this tutorial.

Use `tf.distribute.Strategy.scope` to specify that a strategy should be used when building your model. This allows the strategy to control things like variable placement—it will create copies of all variables in the model's layers on each device across all workers.


```
import mnist
with strategy.scope():
  # Model building needs to be within `strategy.scope()`.
  multi_worker_model = mnist.build_cnn_model()
```

## Auto-shard your data across workers

In multi-worker training, _dataset sharding_ is needed to ensure convergence and reproducibility. Sharding means handing each worker a subset of the entire dataset—it helps create the experience similar to training on a single worker. In the example below, you're relying on the default autosharding policy of `tf.distribute`. You can also customize it by setting the `tf.data.experimental.AutoShardPolicy` of the `tf.data.experimental.DistributeOptions`. To learn more, refer to the _Sharding_ section of the [Distributed input tutorial](input.ipynb).


```
per_worker_batch_size = 64
num_workers = len(tf_config['cluster']['worker'])
global_batch_size = per_worker_batch_size * num_workers

with strategy.scope():
  multi_worker_dataset = strategy.distribute_datasets_from_function(
      lambda input_context: mnist.dataset_fn(global_batch_size, input_context))
```

## Define a custom training loop and train the model
Specify an optimizer:


```
with strategy.scope():
  # The creation of optimizer and train_accuracy needs to be in
  # `strategy.scope()` as well, since they create variables.
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')
```

Define a training step with `tf.function`:



```
@tf.function
def train_step(iterator):
  """Training step function."""

  def step_fn(inputs):
    """Per-Replica step function."""
    x, y = inputs
    with tf.GradientTape() as tape:
      predictions = multi_worker_model(x, training=True)
      per_example_loss = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True,
          reduction=tf.keras.losses.Reduction.NONE)(y, predictions)
      loss = tf.nn.compute_average_loss(per_example_loss)
      model_losses = multi_worker_model.losses
      if model_losses:
        loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))

    grads = tape.gradient(loss, multi_worker_model.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, multi_worker_model.trainable_variables))
    train_accuracy.update_state(y, predictions)
    return loss

  per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
  return strategy.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
```

### Checkpoint saving and restoring

As you write a custom training loop, you need to handle [checkpoint saving](../../guide/checkpoint.ipynb) manually instead of relying on a Keras callback. Note that for `MultiWorkerMirroredStrategy`, saving a checkpoint or a complete model requires the participation of all workers, because attempting to save only on the chief worker could lead to a deadlock. Workers also need to write to different paths to avoid overwriting each other. Here's an example of how to configure the directories:


```
from multiprocessing import util
checkpoint_dir = os.path.join(util.get_temp_dir(), 'ckpt')

def _is_chief(task_type, task_id, cluster_spec):
  return (task_type is None
          or task_type == 'chief'
          or (task_type == 'worker'
              and task_id == 0
              and "chief" not in cluster_spec.as_dict()))

def _get_temp_dir(dirpath, task_id):
  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir

def write_filepath(filepath, task_type, task_id, cluster_spec):
  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id, cluster_spec):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)
```

Create one `tf.train.Checkpoint` that tracks the model, which is managed by a `tf.train.CheckpointManager`, so that only the latest checkpoints are preserved:


```
epoch = tf.Variable(
    initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')
step_in_epoch = tf.Variable(
    initial_value=tf.constant(0, dtype=tf.dtypes.int64),
    name='step_in_epoch')
task_type, task_id = (strategy.cluster_resolver.task_type,
                      strategy.cluster_resolver.task_id)
# Normally, you don't need to manually instantiate a `ClusterSpec`, but in this
# illustrative example you did not set `'TF_CONFIG'` before initializing the
# strategy. Check out the next section for "real-world" usage.
cluster_spec = tf.train.ClusterSpec(tf_config['cluster'])

checkpoint = tf.train.Checkpoint(
    model=multi_worker_model, epoch=epoch, step_in_epoch=step_in_epoch)

write_checkpoint_dir = write_filepath(checkpoint_dir, task_type, task_id,
                                      cluster_spec)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, directory=write_checkpoint_dir, max_to_keep=1)
```

Now, when you need to restore a checkpoint, you can find the latest checkpoint saved using the convenient `tf.train.latest_checkpoint` function (or by calling `tf.train.CheckpointManager.restore_or_initialize`).


```
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
  checkpoint.restore(latest_checkpoint)
```

After restoring the checkpoint, you can continue with training your custom training loop.


```
num_epochs = 3
num_steps_per_epoch = 70

while epoch.numpy() < num_epochs:
  iterator = iter(multi_worker_dataset)
  total_loss = 0.0
  num_batches = 0

  while step_in_epoch.numpy() < num_steps_per_epoch:
    total_loss += train_step(iterator)
    num_batches += 1
    step_in_epoch.assign_add(1)

  train_loss = total_loss / num_batches
  print('Epoch: %d, accuracy: %f, train_loss: %f.'
                %(epoch.numpy(), train_accuracy.result(), train_loss))

  train_accuracy.reset_states()

  # Once the `CheckpointManager` is set up, you're now ready to save, and remove
  # the checkpoints non-chief workers saved.
  checkpoint_manager.save()
  if not _is_chief(task_type, task_id, cluster_spec):
    tf.io.gfile.rmtree(write_checkpoint_dir)

  epoch.assign_add(1)
  step_in_epoch.assign(0)
```

## Complete code at a glance

To sum up all the procedures discussed so far:

1. You create worker processes.
2. Pass `'TF_CONFIG'`s to the worker processes.
3. Let each work process run the script below that contains the training code.


```
%%writefile main.py
#@title File: `main.py`
import os
import json
import tensorflow as tf
import mnist
from multiprocessing import util

per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])
global_batch_size = per_worker_batch_size * num_workers

num_epochs = 3
num_steps_per_epoch=70

# Checkpoint saving and restoring
def _is_chief(task_type, task_id, cluster_spec):
  return (task_type is None
          or task_type == 'chief'
          or (task_type == 'worker'
              and task_id == 0
              and 'chief' not in cluster_spec.as_dict()))

def _get_temp_dir(dirpath, task_id):
  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir

def write_filepath(filepath, task_type, task_id, cluster_spec):
  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id, cluster_spec):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)

checkpoint_dir = os.path.join(util.get_temp_dir(), 'ckpt')

# Define Strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
  # Model building/compiling need to be within `tf.distribute.Strategy.scope`.
  multi_worker_model = mnist.build_cnn_model()

  multi_worker_dataset = strategy.distribute_datasets_from_function(
      lambda input_context: mnist.dataset_fn(global_batch_size, input_context))
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='train_accuracy')

@tf.function
def train_step(iterator):
  """Training step function."""

  def step_fn(inputs):
    """Per-Replica step function."""
    x, y = inputs
    with tf.GradientTape() as tape:
      predictions = multi_worker_model(x, training=True)
      per_example_loss = tf.keras.losses.SparseCategoricalCrossentropy(
          from_logits=True,
          reduction=tf.keras.losses.Reduction.NONE)(y, predictions)
      loss = tf.nn.compute_average_loss(per_example_loss)
      model_losses = multi_worker_model.losses
      if model_losses:
        loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))

    grads = tape.gradient(loss, multi_worker_model.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, multi_worker_model.trainable_variables))
    train_accuracy.update_state(y, predictions)

    return loss

  per_replica_losses = strategy.run(step_fn, args=(next(iterator),))
  return strategy.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

epoch = tf.Variable(
    initial_value=tf.constant(0, dtype=tf.dtypes.int64), name='epoch')
step_in_epoch = tf.Variable(
    initial_value=tf.constant(0, dtype=tf.dtypes.int64),
    name='step_in_epoch')

task_type, task_id, cluster_spec = (strategy.cluster_resolver.task_type,
                                    strategy.cluster_resolver.task_id,
                                    strategy.cluster_resolver.cluster_spec())

checkpoint = tf.train.Checkpoint(
    model=multi_worker_model, epoch=epoch, step_in_epoch=step_in_epoch)

write_checkpoint_dir = write_filepath(checkpoint_dir, task_type, task_id,
                                      cluster_spec)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, directory=write_checkpoint_dir, max_to_keep=1)

# Restoring the checkpoint
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
  checkpoint.restore(latest_checkpoint)

# Resume our CTL training
while epoch.numpy() < num_epochs:
  iterator = iter(multi_worker_dataset)
  total_loss = 0.0
  num_batches = 0

  while step_in_epoch.numpy() < num_steps_per_epoch:
    total_loss += train_step(iterator)
    num_batches += 1
    step_in_epoch.assign_add(1)

  train_loss = total_loss / num_batches
  print('Epoch: %d, accuracy: %f, train_loss: %f.'
                %(epoch.numpy(), train_accuracy.result(), train_loss))

  train_accuracy.reset_states()

  checkpoint_manager.save()
  if not _is_chief(task_type, task_id, cluster_spec):
    tf.io.gfile.rmtree(write_checkpoint_dir)

  epoch.assign_add(1)
  step_in_epoch.assign(0)
```

The current directory now contains both Python files:


```bash
%%bash
ls *.py
```

So JSON-serialize the `'TF_CONFIG'` and add it to the environment variables:


```
os.environ['TF_CONFIG'] = json.dumps(tf_config)
```

Now, you can launch a worker process that will run the `main.py` and use the `'TF_CONFIG'`:


```
# first kill any previous runs
%killbgscripts
```


```bash
%%bash --bg
python main.py &> job_0.log
```

There are a few things to note about the above command:

1. It uses the `%%bash` which is a [notebook "magic"](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to run some bash commands.
2. It uses the `--bg` flag to run the `bash` process in the background, because this worker will not terminate. It waits for all the workers before it starts.

The backgrounded worker process won't print the output to this notebook. The `&>` redirects its output to a file, so that you can inspect what happened.

Wait a few seconds for the process to start up:


```
import time
time.sleep(20)
```

Now, check the output to the worker's log file so far:


```bash
%%bash
cat job_0.log
```

The last line of the log file should say: `Started server with target: grpc://localhost:12345`. The first worker is now ready, and is waiting for all the other worker(s) to be ready to proceed.

Update the `tf_config` for the second worker's process to pick up:


```
tf_config['task']['index'] = 1
os.environ['TF_CONFIG'] = json.dumps(tf_config)
```

Now launch the second worker. This will start the training since all the workers are active (so there's no need to background this process):


```bash
%%bash
python main.py > /dev/null 2>&1
```

If you recheck the logs written by the first worker, notice that it participated in training that model:


```bash
%%bash
cat job_0.log
```


```
# Delete the `'TF_CONFIG'`, and kill any background tasks so they don't affect the next section.
os.environ.pop('TF_CONFIG', None)
%killbgscripts
```

## Multi-worker training in depth

This tutorial has demonstrated a custom training loop workflow of the multi-worker setup. Detailed descriptions of other topics is available in the [Multi-worker training with Keras (`tf.keras.Model.fit`)](multi_worker_with_keras.ipynb) tutorial applicable to custom training loops.

## Learn more

1. The [Distributed training in TensorFlow](../../guide/distributed_training.ipynb) guide provides an overview of the available distribution strategies.
2. [Official models](https://github.com/tensorflow/models/tree/master/official), many of which can be configured to run multiple distribution strategies.
3. The [Performance section](../../guide/function.ipynb) in the `tf.function` guide provides information about other strategies and [tools](../../guide/profiler.md) you can use to optimize the performance of your TensorFlow models.

