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

# Multi-worker training with Estimator

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/distribute/multi_worker_with_estimator.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/multi_worker_with_estimator.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/distribute/multi_worker_with_estimator.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
</table>

> Warning: Estimators are not recommended for new code.  Estimators run `v1.Session`-style code which is more difficult to write correctly, and can behave unexpectedly, especially when combined with TF 2 code. Estimators do fall under [compatibility guarantees](https://tensorflow.org/guide/versions), but will receive no fixes other than security vulnerabilities. See the [migration guide](https://tensorflow.org/guide/migrate) for details.

## Overview

Note: While you can use Estimators with `tf.distribute` API, it's recommended to use Keras with `tf.distribute`, see [multi-worker training with Keras](multi_worker_with_keras.ipynb). Estimator training with `tf.distribute.Strategy` has limited support.


This tutorial demonstrates how `tf.distribute.Strategy` can be used for distributed multi-worker training with `tf.estimator`.  If you write your code using `tf.estimator`, and you're interested in scaling beyond a single machine with high performance, this tutorial is for you.

Before getting started, please read the [distribution strategy](../../guide/distributed_training.ipynb) guide.  The [multi-GPU training tutorial](./keras.ipynb) is also relevant, because this tutorial uses the same model.


## Setup

First, setup TensorFlow and the necessary imports.


```
import tensorflow_datasets as tfds
import tensorflow as tf

import os, json
```

Note: Starting from TF2.4 multi worker mirrored strategy fails with estimators if run with eager enabled (the default). The error in TF2.4 is `TypeError: cannot pickle '_thread.lock' object`, See [issue #46556](https://github.com/tensorflow/tensorflow/issues/46556) for details. The workaround is to disable eager execution.


```
tf.compat.v1.disable_eager_execution()
```

## Input function

This tutorial uses the MNIST dataset from [TensorFlow Datasets](https://www.tensorflow.org/datasets).  The code here is similar to the [multi-GPU training tutorial](./keras.ipynb) with one key difference: when using Estimator for multi-worker training, it is necessary to shard the dataset by the number of workers to ensure model convergence.  The input data is sharded by worker index, so that each worker processes `1/num_workers` distinct portions of the dataset.


```
BUFFER_SIZE = 10000
BATCH_SIZE = 64

def input_fn(mode, input_context=None):
  datasets, info = tfds.load(name='mnist',
                                with_info=True,
                                as_supervised=True)
  mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else
                   datasets['test'])

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  if input_context:
    mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines,
                                        input_context.input_pipeline_id)
  return mnist_dataset.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```

Another reasonable approach to achieve convergence would be to shuffle the dataset with distinct seeds at each worker.

## Multi-worker configuration

One of the key differences in this tutorial (compared to the [multi-GPU training tutorial](./keras.ipynb)) is the multi-worker setup.  The `TF_CONFIG` environment variable is the standard way to specify the cluster configuration to each worker that is part of the cluster.

There are two components of `TF_CONFIG`: `cluster` and `task`.  `cluster` provides information about the entire cluster, namely the workers and parameter servers in the cluster.  `task` provides information about the current task. The first component `cluster` is the same for all workers and parameter servers in the cluster, and the second component `task` is different on each worker and parameter server and specifies its own `type` and `index`. In this example, the task `type` is `worker` and the task `index` is `0`.

For illustration purposes, this tutorial shows how to set a `TF_CONFIG` with 2 workers on `localhost`.  In practice, you would create multiple workers on an external IP address and port, and set `TF_CONFIG` on each worker appropriately, i.e. modify the task `index`.

Warning: *Do not execute the following code in Colab.*  TensorFlow's runtime will attempt to create a gRPC server at the specified IP address and port, which will likely fail. See the [keras version](multi_worker_with_keras.ipynb) of this tutorial for an example of how you can test run multiple workers on a single machine.

```
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})
```


## Define the model

Write the layers, the optimizer, and the loss function for training.  This tutorial defines the model with Keras layers, similar to the [multi-GPU training tutorial](./keras.ipynb).


```
LEARNING_RATE = 1e-4
def model_fn(features, labels, mode):
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  logits = model(features, training=False)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {'logits': logits}
    return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

  optimizer = tf.compat.v1.train.GradientDescentOptimizer(
      learning_rate=LEARNING_RATE)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels, logits)
  loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=optimizer.minimize(
          loss, tf.compat.v1.train.get_or_create_global_step()))
```

Note: Although the learning rate is fixed in this example, in general it may be necessary to adjust the learning rate based on the global batch size.

## MultiWorkerMirroredStrategy

To train the model, use an instance of `tf.distribute.experimental.MultiWorkerMirroredStrategy`.  `MultiWorkerMirroredStrategy` creates copies of all variables in the model's layers on each device across all workers.  It uses `CollectiveOps`, a TensorFlow op for collective communication, to aggregate gradients and keep the variables in sync.  The [`tf.distribute.Strategy` guide](../../guide/distributed_training.ipynb) has more details about this strategy.


```
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
```

## Train and evaluate the model

Next, specify the distribution strategy in the `RunConfig` for the estimator, and train and evaluate by invoking `tf.estimator.train_and_evaluate`.  This tutorial distributes only the training by specifying the strategy via `train_distribute`.  It is also possible to distribute the evaluation via `eval_distribute`.


```
config = tf.estimator.RunConfig(train_distribute=strategy)

classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir='/tmp/multiworker', config=config)
tf.estimator.train_and_evaluate(
    classifier,
    train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
    eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)
)
```

## Optimize training performance

You now have a model and a multi-worker capable Estimator powered by `tf.distribute.Strategy`.  You can try the following techniques to optimize performance of multi-worker training:

*   *Increase the batch size:* The batch size specified here is per-GPU.  In general, the largest batch size that fits the GPU memory is advisable.
*   *Cast variables:* Cast the variables to `tf.float` if possible.  The official ResNet model includes [an example](https://github.com/tensorflow/models/blob/8367cf6dabe11adf7628541706b660821f397dce/official/resnet/resnet_model.py#L466) of how this can be done.
*   *Use collective communication:* `MultiWorkerMirroredStrategy` provides multiple [collective communication implementations](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/distribute/cross_device_ops.py).  
    * `RING` implements ring-based collectives using gRPC as the cross-host communication layer.  
    * `NCCL` uses [Nvidia's NCCL](https://developer.nvidia.com/nccl) to implement collectives.  
    * `AUTO` defers the choice to the runtime.
    
    The best choice of collective implementation depends upon the number and kind of GPUs, and the network interconnect in the cluster.  To override the automatic choice, specify a valid value to the `communication` parameter of `MultiWorkerMirroredStrategy`'s constructor, e.g. `communication=tf.distribute.experimental.CollectiveCommunication.NCCL`.

Visit the [Performance section](../../guide/function.ipynb) in the guide to learn more about other strategies and [tools](../../guide/profiler.md) you can use to optimize the performance of your TensorFlow models.


## Other code examples

1.   [End to end example](https://github.com/tensorflow/ecosystem/tree/master/distribution_strategy) for multi worker training in tensorflow/ecosystem using Kubernetes templates. This example starts with a Keras model and converts it to an Estimator using the `tf.keras.estimator.model_to_estimator` API.
2.   [Official models](https://github.com/tensorflow/models/tree/master/official), many of which can be configured to run multiple distribution strategies.

