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

# Multi-worker training with Keras

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/multi_worker_with_keras.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/multi_worker_with_keras.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/distribute/multi_worker_with_keras.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

## Overview

This tutorial demonstrates how to perform multi-worker distributed training with a Keras model and the `Model.fit` API using the `tf.distribute.MultiWorkerMirroredStrategy` API. With the help of this strategy, a Keras model that was designed to run on a single-worker can seamlessly work on multiple workers with minimal code changes.

To learn how to use the `MultiWorkerMirroredStrategy` with Keras and a custom training loop, refer to [Custom training loop with Keras and MultiWorkerMirroredStrategy](multi_worker_with_ctl.ipynb).

This tutorial contains a minimal multi-worker example with two workers for demonstration purposes.

### Choose the right strategy

Before you dive in, make sure that `tf.distribute.MultiWorkerMirroredStrategy` is the right choice for your accelerator(s) and training. These are two common ways of distributing training with data parallelism:

* _Synchronous training_, where the steps of training are synced across the workers and replicas, such as `tf.distribute.MirroredStrategy`, `tf.distribute.TPUStrategy`, and `tf.distribute.MultiWorkerMirroredStrategy`. All workers train over different slices of input data in sync, and aggregating gradients at each step.
* _Asynchronous training_, where the training steps are not strictly synced, such as `tf.distribute.experimental.ParameterServerStrategy`. All workers are independently training over the input data and updating variables asynchronously.

If you are looking for multi-worker synchronous training without TPU, then `tf.distribute.MultiWorkerMirroredStrategy` is your choice. It creates copies of all variables in the model's layers on each device across all workers. It uses `CollectiveOps`, a TensorFlow op for collective communication, to aggregate gradients and keeps the variables in sync. For those interested, check out the `tf.distribute.experimental.CommunicationOptions` parameter for the collective implementation options.

For an overview of `tf.distribute.Strategy` APIs, refer to [Distributed training in TensorFlow](../../guide/distributed_training.ipynb).

## Setup

Start with some necessary imports:


```
import json
import os
import sys
```

Before importing TensorFlow, make a few changes to the environment:

* In a real-world application, each worker would be on a different machine. For the purposes of this tutorial, all the workers will run on the **this** machine. Therefore, disable all GPUs to prevent errors caused by all workers trying to use the same GPU.


```
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

* Reset the `TF_CONFIG` environment variable (you'll learn more about this later):


```
os.environ.pop('TF_CONFIG', None)
```

* Make sure that the current directory is on Python's path—this allows the notebook to import the files written by `%%writefile` later:



```
if '.' not in sys.path:
  sys.path.insert(0, '.')
```

Install `tf-nightly`, as the frequency of checkpoint saving at a particular step with the `save_freq` argument in `tf.keras.callbacks.BackupAndRestore` is introduced from TensorFlow 2.10:


```
!pip install tf-nightly
```

Finally, import TensorFlow:


```
import tensorflow as tf
```

### Dataset and model definition

Next, create an `mnist_setup.py` file with a simple model and dataset setup. This Python file will be used by the worker processes in this tutorial:


```
%%writefile mnist_setup.py

import os
import tensorflow as tf
import numpy as np

def mnist_dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the [0, 255] range.
  # You need to convert them to float32 with values in the [0, 1] range.
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
  return train_dataset

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer(input_shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model
```

### Model training on a single worker

Try training the model for a small number of epochs and observe the results of _a single worker_ to make sure everything works correctly. As training progresses, the loss should drop and the accuracy should increase.


```
import mnist_setup

batch_size = 64
single_worker_dataset = mnist_setup.mnist_dataset(batch_size)
single_worker_model = mnist_setup.build_and_compile_cnn_model()
single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)
```

## Multi-worker configuration

Now let's enter the world of multi-worker training.

### A cluster with jobs and tasks

In TensorFlow, distributed training involves a `'cluster'`
with several jobs, and each of the jobs may have one or more `'task'`s.

You will need the `TF_CONFIG` configuration environment variable for training on multiple machines, each of which possibly has a different role. `TF_CONFIG` is a JSON string used to specify the cluster configuration for each worker that is part of the cluster.

There are two components of a `TF_CONFIG` variable: `'cluster'` and `'task'`.

* A `'cluster'` is the same for all workers and provides information about the training cluster, which is a dict consisting of different types of jobs, such as `'worker'` or `'chief'`.
    - In multi-worker training with `tf.distribute.MultiWorkerMirroredStrategy`, there is usually one `'worker'` that takes on more responsibilities, such as saving a checkpoint and writing a summary file for TensorBoard, in addition to what a regular `'worker'` does. Such `'worker'` is referred to as the chief worker (with a job name `'chief'`).
    - It is customary for the worker with `'index'` `0` to be the `'chief'`.

* A `'task'` provides information on the current task and is different for each worker. It specifies the `'type'` and `'index'` of that worker.

Below is an example configuration:


```
tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}
```

Note that `tf_config` is just a local variable in Python. To use it for training configuration, serialize it as a JSON and place it in a `TF_CONFIG` environment variable.


```
json.dumps(tf_config)
```

In the example configuration above, you set the task `'type'` to `'worker'` and the task `'index'` to `0`. Therefore, this machine is the _first_ worker. It will be appointed as the `'chief'` worker.

Note: Other machines will need to have the `TF_CONFIG` environment variable set as well, and it should have the same `'cluster'` dict, but different task `'type'`s or task `'index'`es, depending on the roles of those machines.

In practice, you would create multiple workers on external IP addresses/ports and set a `TF_CONFIG` variable on each worker accordingly. For illustration purposes, this tutorial shows how you may set up a `TF_CONFIG` variable with two workers on a `localhost`:
- The first (`'chief'`) worker's `TF_CONFIG` as shown above.
- For the second worker, you will set `tf_config['task']['index']=1`

### Environment variables and subprocesses in notebooks

Subprocesses inherit environment variables from their parent. So if you set an environment variable in this Jupyter Notebook process:


```
os.environ['GREETINGS'] = 'Hello TensorFlow!'
```

... then you can access the environment variable from the subprocesses:


```bash
%%bash
echo ${GREETINGS}
```

In the next section, you'll use this method to pass the `TF_CONFIG` to the worker subprocesses. You would never really launch your jobs this way in a real-world scenario—this tutorial is just showing how to do it with a minimal multi-worker example.

## Train the model

To train the model, firstly create an instance of the `tf.distribute.MultiWorkerMirroredStrategy`:


```
strategy = tf.distribute.MultiWorkerMirroredStrategy()
```

Note: `TF_CONFIG` is parsed and TensorFlow's GRPC servers are started at the time `MultiWorkerMirroredStrategy` is called, so the `TF_CONFIG` environment variable must be set before a `tf.distribute.Strategy` instance is created. Since `TF_CONFIG` is not set yet, the above strategy is effectively single-worker training.

With the integration of `tf.distribute.Strategy` API into `tf.keras`, the only change you will make to distribute the training to multiple-workers is enclosing the model building and `model.compile()` call inside `strategy.scope()`. The distribution strategy's scope dictates how and where the variables are created, and in the case of `MultiWorkerMirroredStrategy`, the variables created are `MirroredVariable`s, and they are replicated on each of the workers.



```
with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = mnist_setup.build_and_compile_cnn_model()
```

Note: Currently there is a limitation in `MultiWorkerMirroredStrategy` where TensorFlow ops need to be created after the instance of strategy is created. If you encounter `RuntimeError: Collective ops must be configured at program startup`, try creating the instance of `MultiWorkerMirroredStrategy` at the beginning of the program and put the code that may create ops after the strategy is instantiated.

To actually run with `MultiWorkerMirroredStrategy` you'll need to run worker processes and pass a `TF_CONFIG` to them.

Like the `mnist_setup.py` file written earlier, here is the `main.py` that each of the workers will run:


```
%%writefile main.py

import os
import json

import tensorflow as tf
import mnist_setup

per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_setup.mnist_dataset(global_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = mnist_setup.build_and_compile_cnn_model()


multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)
```

In the code snippet above note that the `global_batch_size`, which gets passed to `Dataset.batch`, is set to `per_worker_batch_size * num_workers`. This ensures that each worker processes batches of `per_worker_batch_size` examples regardless of the number of workers.

The current directory now contains both Python files:


```bash
%%bash
ls *.py
```

Serialize the `TF_CONFIG` to JSON and add it to the environment variables:


```
os.environ['TF_CONFIG'] = json.dumps(tf_config)
```

Now, you can launch a worker process that will run the `main.py` and use the `TF_CONFIG`:


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

The backgrounded worker process won't print output to this notebook, so the `&>` redirects its output to a file so that you can inspect what happened in a log file later.

So, wait a few seconds for the process to start up:


```
import time
time.sleep(10)
```

Now, inspect what's been output to the worker's log file so far:


```bash
%%bash
cat job_0.log
```

The last line of the log file should say: `Started server with target: grpc://localhost:12345`. The first worker is now ready and is waiting for all the other worker(s) to be ready to proceed.

So update the `tf_config` for the second worker's process to pick up:


```
tf_config['task']['index'] = 1
os.environ['TF_CONFIG'] = json.dumps(tf_config)
```

Launch the second worker. This will start the training since all the workers are active (so there's no need to background this process):


```bash
%%bash
python main.py
```

If you recheck the logs written by the first worker, you'll learn that it participated in training that model:


```bash
%%bash
cat job_0.log
```

Note: This may run slower than the test run at the beginning of this tutorial because running multiple workers on a single machine only adds overhead. The goal here is not to improve the training time but to give an example of multi-worker training.



```
# Delete the `TF_CONFIG`, and kill any background tasks so they don't affect the next section.
os.environ.pop('TF_CONFIG', None)
%killbgscripts
```

## Multi-worker training in depth


So far, you have learned how to perform a basic multi-worker setup. The rest of the tutorial goes over other factors, which may be useful or important for real use cases, in detail.

### Dataset sharding

In multi-worker training, _dataset sharding_ is needed to ensure convergence and performance.

The example in the previous section relies on the default autosharding provided by the `tf.distribute.Strategy` API. You can control the sharding by setting the `tf.data.experimental.AutoShardPolicy` of the `tf.data.experimental.DistributeOptions`.

To learn more about _auto-sharding_, refer to the [Distributed input guide](https://www.tensorflow.org/tutorials/distribute/input#sharding).

Here is a quick example of how to turn the auto sharding off, so that each replica processes every example (_not recommended_):



```
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

global_batch_size = 64
multi_worker_dataset = mnist_setup.mnist_dataset(batch_size=64)
dataset_no_auto_shard = multi_worker_dataset.with_options(options)
```

### Evaluation

If you pass the `validation_data` into `Model.fit` as well, it will alternate between training and evaluation for each epoch. The evaluation work is distributed across the same set of workers, and its results are aggregated and available to all workers.

Similar to training, the validation dataset is automatically sharded at the file level. You need to set a global batch size in the validation dataset and set the `validation_steps`.

A repeated dataset (by calling `tf.data.Dataset.repeat`) is recommended for evaluation.

Alternatively, you can also create another task that periodically reads checkpoints and runs the evaluation. This is what an Estimator does. But this is not a recommended way to perform evaluation and thus its details are omitted.

### Performance

To tweak the performance of multi-worker training, you can try the following:

- `tf.distribute.MultiWorkerMirroredStrategy` provides multiple [collective communication implementations](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/CommunicationImplementation):
    - `RING` implements ring-based collectives using gRPC as the cross-host communication layer.
    - `NCCL` uses the [NVIDIA Collective Communication Library](https://developer.nvidia.com/nccl) to implement collectives.
    -  `AUTO` defers the choice to the runtime.
    
    The best choice of collective implementation depends upon the number of GPUs, the type of GPUs, and the network interconnects in the cluster. To override the automatic choice, specify the `communication_options` parameter of `MultiWorkerMirroredStrategy`'s constructor. For example:
    
    ```python
    communication_options=tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
    ```

- Cast the variables to `tf.float` if possible:
    - The official ResNet model includes [an example](https://github.com/tensorflow/models/blob/8367cf6dabe11adf7628541706b660821f397dce/official/resnet/resnet_model.py#L466) of how to do this.

### Fault tolerance

In synchronous training, the cluster would fail if one of the workers fails and no failure-recovery mechanism exists.

Using Keras with `tf.distribute.Strategy` comes with the advantage of fault tolerance in cases where workers die or are otherwise unstable. You can do this by preserving the training state in the distributed file system of your choice, such that upon a restart of the instance that previously failed or preempted, the training state is recovered.

When a worker becomes unavailable, other workers will fail (possibly after a timeout). In such cases, the unavailable worker needs to be restarted, as well as other workers that have failed.

Note: Previously, the `ModelCheckpoint` callback provided a mechanism to restore the training state upon a restart from a job failure for multi-worker training. The TensorFlow team is introducing a new [`BackupAndRestore`](#scrollTo=kmH8uCUhfn4w) callback, which also adds the support to single-worker training for a consistent experience, and removed the fault tolerance functionality from existing `ModelCheckpoint` callback. From now on, applications that rely on this behavior should migrate to the new `BackupAndRestore` callback.

#### The `ModelCheckpoint` callback

`ModelCheckpoint` callback no longer provides fault tolerance functionality, please use [`BackupAndRestore`](#scrollTo=kmH8uCUhfn4w) callback instead.

The `ModelCheckpoint` callback can still be used to save checkpoints. But with this, if training was interrupted or successfully finished, in order to continue training from the checkpoint, the user is responsible to load the model manually.

Optionally, users can choose to save and restore model/weights outside `ModelCheckpoint` callback.

### Model saving and loading

To save your model using `model.save` or `tf.saved_model.save`, the saving destination needs to be different for each worker.

- For non-chief workers, you will need to save the model to a temporary directory.
- For the chief, you will need to save to the provided model directory.

The temporary directories on the worker need to be unique to prevent errors resulting from multiple workers trying to write to the same location.

The model saved in all the directories is identical, and typically only the model saved by the chief should be referenced for restoring or serving.

You should have some cleanup logic that deletes the temporary directories created by the workers once your training has completed.

The reason for saving on the chief and workers at the same time is because you might be aggregating variables during checkpointing, which requires both the chief and workers to participate in the allreduce communication protocol. On the other hand, letting chief and workers save to the same model directory will result in errors due to contention.

Using the `MultiWorkerMirroredStrategy`, the program is run on every worker, and in order to know whether the current worker is the chief, it takes advantage of the cluster resolver object that has attributes `task_type` and `task_id`:
- `task_type` tells you what the current job is (for example, `'worker'`).
- `task_id` tells you the identifier of the worker.
- The worker with `task_id == 0` is designated as the chief worker.

In the code snippet below, the `write_filepath` function provides the file path to write, which depends on the worker's `task_id`:

- For the chief worker (with `task_id == 0`), it writes to the original file path. 
- For other workers, it creates a temporary directory—`temp_dir`—with the `task_id` in the directory path to write in:


```
model_path = '/tmp/keras-model'

def _is_chief(task_type, task_id):
  # Note: there are two possible `TF_CONFIG` configurations.
  #   1) In addition to `worker` tasks, a `chief` task type is use;
  #      in this case, this function should be modified to
  #      `return task_type == 'chief'`.
  #   2) Only `worker` task type is used; in this case, worker 0 is
  #      regarded as the chief. The implementation demonstrated here
  #      is for this case.
  # For the purpose of this Colab section, the `task_type` is `None` case
  # is added because it is effectively run with only a single worker.
  return (task_type == 'worker' and task_id == 0) or task_type is None

def _get_temp_dir(dirpath, task_id):
  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir

def write_filepath(filepath, task_type, task_id):
  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)

task_type, task_id = (strategy.cluster_resolver.task_type,
                      strategy.cluster_resolver.task_id)
write_model_path = write_filepath(model_path, task_type, task_id)
```

With that, you're now ready to save:

Deprecated: For Keras objects, it's recommended to use the new high-level `.keras` format and `tf.keras.Model.export`, as demonstrated in the guide [here](https://www.tensorflow.org/guide/keras/save_and_serialize). The low-level SavedModel format continues to be supported for existing code.


```
multi_worker_model.save(write_model_path)
```

As described above, later on the model should only be loaded from the file path the chief worker saved to. Therefore, remove the temporary ones the non-chief workers have saved:


```
if not _is_chief(task_type, task_id):
  tf.io.gfile.rmtree(os.path.dirname(write_model_path))
```

Now, when it's time to load, use the convenient `tf.keras.models.load_model` API, and continue with further work.

Here, assume only using single worker to load and continue training, in which case you do not call `tf.keras.models.load_model` within another `strategy.scope()` (note that `strategy = tf.distribute.MultiWorkerMirroredStrategy()`, as defined earlier):


```
loaded_model = tf.keras.models.load_model(model_path)

# Now that the model is restored, and can continue with the training.
loaded_model.fit(single_worker_dataset, epochs=2, steps_per_epoch=20)
```

### Checkpoint saving and restoring

On the other hand, checkpointing allows you to save your model's weights and restore them without having to save the whole model.

Here, you'll create one `tf.train.Checkpoint` that tracks the model, which is managed by the `tf.train.CheckpointManager`, so that only the latest checkpoint is preserved:


```
checkpoint_dir = '/tmp/ckpt'

checkpoint = tf.train.Checkpoint(model=multi_worker_model)
write_checkpoint_dir = write_filepath(checkpoint_dir, task_type, task_id)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, directory=write_checkpoint_dir, max_to_keep=1)
```

Once the `CheckpointManager` is set up, you're now ready to save and remove the checkpoints the non-chief workers had saved:


```
checkpoint_manager.save()
if not _is_chief(task_type, task_id):
  tf.io.gfile.rmtree(write_checkpoint_dir)
```

Now, when you need to restore the model, you can find the latest checkpoint saved using the convenient `tf.train.latest_checkpoint` function. After restoring the checkpoint, you can continue with training.


```
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint.restore(latest_checkpoint)
multi_worker_model.fit(multi_worker_dataset, epochs=2, steps_per_epoch=20)
```

#### The `BackupAndRestore` callback

The `tf.keras.callbacks.BackupAndRestore` callback provides the fault tolerance functionality by backing up the model and current training state in a temporary checkpoint file under `backup_dir` argument to `BackupAndRestore`. 

Note: In Tensorflow 2.9, the current model and the training state is backed up at epoch boundaries. In the `tf-nightly` version and from TensorFlow 2.10, the `BackupAndRestore` callback can back up the model and the training state at epoch or step boundaries. `BackupAndRestore` accepts an optional `save_freq` argument. `save_freq` accepts either `'epoch'` or an `int` value. If `save_freq` is set to `'epoch'` the model is backed up after every epoch. If `save_freq` is set to an integer value greater than `0`, the model is backed up after every `save_freq` number of batches.

Once the jobs get interrupted and restarted, the `BackupAndRestore` callback restores the last checkpoint, and you can continue training from the beginning of the epoch and step at which the training state was last saved.

To use it, provide an instance of `tf.keras.callbacks.BackupAndRestore` at the `Model.fit` call.

With `MultiWorkerMirroredStrategy`, if a worker gets interrupted, the whole cluster will pause until the interrupted worker is restarted. Other workers will also restart, and the interrupted worker will rejoin the cluster. Then, every worker will read the checkpoint file that was previously saved and pick up its former state, thereby allowing the cluster to get back in sync. Then, the training will continue. The distributed dataset iterator state will be re-initialized and not restored.

The `BackupAndRestore` callback uses the `CheckpointManager` to save and restore the training state, which generates a file called checkpoint that tracks existing checkpoints together with the latest one. For this reason, `backup_dir` should not be re-used to store other checkpoints in order to avoid name collision.

Currently, the `BackupAndRestore` callback supports single-worker training with no strategy—`MirroredStrategy`—and multi-worker training with `MultiWorkerMirroredStrategy`.

Below are two examples for both multi-worker training and single-worker training:


```
# Multi-worker training with `MultiWorkerMirroredStrategy`
# and the `BackupAndRestore` callback. The training state 
# is backed up at epoch boundaries by default.

callbacks = [tf.keras.callbacks.BackupAndRestore(backup_dir='/tmp/backup')]
with strategy.scope():
  multi_worker_model = mnist_setup.build_and_compile_cnn_model()
multi_worker_model.fit(multi_worker_dataset,
                       epochs=3,
                       steps_per_epoch=70,
                       callbacks=callbacks)
```

If the `save_freq` argument in the `BackupAndRestore` callback is set to `'epoch'`, the model is backed up after every epoch.


```
# The training state is backed up at epoch boundaries because `save_freq` is
# set to `epoch`.

callbacks = [tf.keras.callbacks.BackupAndRestore(backup_dir='/tmp/backup')]
with strategy.scope():
  multi_worker_model = mnist_setup.build_and_compile_cnn_model()
multi_worker_model.fit(multi_worker_dataset,
                       epochs=3,
                       steps_per_epoch=70,
                       callbacks=callbacks)

```

Note: The next code block uses features that are only available in `tf-nightly` until Tensorflow 2.10 is released.

If the `save_freq` argument in the `BackupAndRestore` callback is set to an integer value greater than `0`, the model is backed up after every `save_freq` number of batches.


```
# The training state is backed up at every 30 steps because `save_freq` is set
# to an integer value of `30`.

callbacks = [tf.keras.callbacks.BackupAndRestore(backup_dir='/tmp/backup', save_freq=30)]
with strategy.scope():
  multi_worker_model = mnist_setup.build_and_compile_cnn_model()
multi_worker_model.fit(multi_worker_dataset,
                       epochs=3,
                       steps_per_epoch=70,
                       callbacks=callbacks)
```

If you inspect the directory of `backup_dir` you specified in `BackupAndRestore`, you may notice some temporarily generated checkpoint files. Those files are needed for recovering the previously lost instances, and they will be removed by the library at the end of `Model.fit` upon successful exiting of your training.

Note: Currently the `BackupAndRestore` callback only supports eager mode. In graph mode, consider using `Model.save`/`tf.saved_model.save` and `tf.keras.models.load_model` for saving and restoring models, respectively, as described in the _Model saving and loading_ section above, and by providing `initial_epoch` in `Model.fit` during training.

## Additional resources

1. The [Distributed training in TensorFlow](../../guide/distributed_training.ipynb) guide provides an overview of the available distribution strategies.
1. The [Custom training loop with Keras and MultiWorkerMirroredStrategy](multi_worker_with_ctl.ipynb) tutorial shows how to use the `MultiWorkerMirroredStrategy` with Keras and a custom training loop.
1. Check out the [official models](https://github.com/tensorflow/models/tree/master/official), many of which can be configured to run multiple distribution strategies.
1. The [Better performance with tf.function](../../guide/function.ipynb) guide provides information about other strategies and tools, such as the [TensorFlow Profiler](../../guide/profiler.md) you can use to optimize the performance of your TensorFlow models.
