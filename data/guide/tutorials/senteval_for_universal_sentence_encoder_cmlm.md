**Copyright 2021 The TensorFlow Hub Authors.**

Licensed under the Apache License, Version 2.0 (the "License");


```
# Copyright 2021 The TensorFlow Hub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
```

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/hub/tutorials/senteval_for_universal_sentence_encoder_cmlm"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/hub/tutorials/senteval_for_universal_sentence_encoder_cmlm.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/hub/tutorials/senteval_for_universal_sentence_encoder_cmlm.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/hub/tutorials/senteval_for_universal_sentence_encoder_cmlm.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
  <td>
    <a href="https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1"><img src="https://www.tensorflow.org/images/hub_logo_32px.png" />See TF Hub model</a>
  </td>
</table>

# Universal Sentence Encoder SentEval demo
This colab demostrates the [Universal Sentence Encoder CMLM model](https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1) using the [SentEval](https://github.com/facebookresearch/SentEval) toolkit, which is a library for measuring the quality of sentence embeddings. The SentEval toolkit includes a diverse set of downstream tasks that are able to evaluate the generalization power of an embedding model and to evaluate the linguistic properties encoded.

Run the first two code blocks to setup the environment, in the third code block you can pick a SentEval task to evaluate the model. A GPU runtime is recommended to run this Colab.

To learn more about the Universal Sentence Encoder CMLM model, see https://openreview.net/forum?id=WDVD4lUCTzU.


```
#@title Install dependencies
!pip install --quiet "tensorflow-text==2.11.*"
!pip install --quiet torch==1.8.1
```

## Download SentEval and task data
This step download SentEval from github and execute the data script to download the task data. It may take up to 5 minutes to complete.


```
#@title Install SentEval and download task data
!rm -rf ./SentEval
!git clone https://github.com/facebookresearch/SentEval.git
!cd $PWD/SentEval/data/downstream && bash get_transfer_data.bash > /dev/null 2>&1
```

#Execute a SentEval evaulation task
The following code block executes a SentEval task and output the results, choose one of the following tasks to evaluate the USE CMLM model:

```
MR	CR	SUBJ	MPQA	SST	TREC	MRPC	SICK-E
```

Select a model, params and task to run. The rapid prototyping params can be used for reducing computation time for faster result.

It typically takes 5-15 mins to complete a task with the **'rapid prototyping'** params and up to an hour with the **'slower, best performance'** params.

```
params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
```

For better result, use the slower **'slower, best performance'** params, computation may take up to 1 hour:

```
params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 16,
                                 'tenacity': 5, 'epoch_size': 6}
```



```
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append(f'{os.getcwd()}/SentEval')

import tensorflow as tf

# Prevent TF from claiming all GPU memory so there is some left for pytorch.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Memory growth needs to be the same across GPUs.
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import tensorflow_hub as hub
import tensorflow_text
import senteval
import time

PATH_TO_DATA = f'{os.getcwd()}/SentEval/data'
MODEL = 'https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1' #@param ['https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1', 'https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-large/1']
PARAMS = 'rapid prototyping' #@param ['slower, best performance', 'rapid prototyping']
TASK = 'CR' #@param ['CR','MR', 'MPQA', 'MRPC', 'SICKEntailment', 'SNLI', 'SST2', 'SUBJ', 'TREC']

params_prototyping = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_prototyping['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

params_best = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_best['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 16,
                                 'tenacity': 5, 'epoch_size': 6}

params = params_best if PARAMS == 'slower, best performance' else params_prototyping

preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1")

inputs = tf.keras.Input(shape=tf.shape(''), dtype=tf.string)
outputs = encoder(preprocessor(inputs))

model = tf.keras.Model(inputs=inputs, outputs=outputs)

def prepare(params, samples):
    return

def batcher(_, batch):
    batch = [' '.join(sent) if sent else '.' for sent in batch]
    return model.predict(tf.constant(batch))["default"]


se = senteval.engine.SE(params, batcher, prepare)
print("Evaluating task %s with %s parameters" % (TASK, PARAMS))
start = time.time()
results = se.eval(TASK)
end = time.time()
print('Time took on task %s : %.1f. seconds' % (TASK, end - start))
print(results)

```

#Learn More

*   Find more text embedding models on [TensorFlow Hub](https://tfhub.dev)
*   See also the [Multilingual Universal Sentence Encoder CMLM model](https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1)
*   Check out other [Universal Sentence Encoder models](https://tfhub.dev/google/collections/universal-sentence-encoder/1)

## Reference

*   Ziyi Yang, Yinfei Yang, Daniel Cer, Jax Law, Eric Darve. [Universal Sentence Representations Learning with Conditional Masked Language Model. November 2020](https://openreview.net/forum?id=WDVD4lUCTzU)

