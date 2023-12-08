##### Copyright 2021 The TensorFlow Hub Authors.

Licensed under the Apache License, Version 2.0 (the "License");


```
#@title Copyright 2021 The TensorFlow Hub Authors. All Rights Reserved.
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
    <a target="_blank" href="https://www.tensorflow.org/hub/tutorials/wav2vec2_saved_model_finetuning"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/hub/tutorials/wav2vec2_saved_model_finetuning.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/hub/tutorials/wav2vec2_saved_model_finetuning.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/hub/tutorials/wav2vec2_saved_model_finetuning.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
  <td>
    <a href="https://tfhub.dev/vasudevgupta7/wav2vec2/1"><img src="https://www.tensorflow.org/images/hub_logo_32px.png" />See TF Hub model</a>
  </td>
</table>

# Fine-tuning Wav2Vec2 with an LM head

In this notebook, we will load the pre-trained wav2vec2 model from [TFHub](https://tfhub.dev) and will fine-tune it on [LibriSpeech dataset](https://huggingface.co/datasets/librispeech_asr) by appending Language Modeling head (LM) over the top of our pre-trained model. The underlying task is to build a model for **Automatic Speech Recognition** i.e. given some speech, the model should be able to transcribe it into text.

## Setting Up

Before running this notebook, please ensure that you are on GPU runtime (`Runtime` > `Change runtime type` > `GPU`). The following cell will install [`gsoc-wav2vec2`](https://github.com/vasudevgupta7/gsoc-wav2vec2) package & its dependencies.


```
!pip3 install -q git+https://github.com/vasudevgupta7/gsoc-wav2vec2@main
!sudo apt-get install -y libsndfile1-dev
!pip3 install -q SoundFile
```

## Model setup using `TFHub`

We will start by importing some libraries/modules.


```
import os

import tensorflow as tf
import tensorflow_hub as hub
from wav2vec2 import Wav2Vec2Config

config = Wav2Vec2Config()

print("TF version:", tf.__version__)
```

First, we will download our model from TFHub & will wrap our model signature with [`hub.KerasLayer`](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer) to be able to use this model like any other Keras layer. Fortunately, `hub.KerasLayer` can do both in just 1 line.

**Note:** When loading model with `hub.KerasLayer`, model becomes a bit opaque but sometimes we need finer controls over the model, then we can load the model with `tf.keras.models.load_model(...)`.


```
pretrained_layer = hub.KerasLayer("https://tfhub.dev/vasudevgupta7/wav2vec2/1", trainable=True)
```

You can refer to this [script](https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/src/export2hub.py) in case you are interested in the model exporting script. Object `pretrained_layer` is the freezed version of [`Wav2Vec2Model`](https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/src/wav2vec2/modeling.py). These pre-trained weights were converted from HuggingFace PyTorch [pre-trained weights](https://huggingface.co/facebook/wav2vec2-base) using [this script](https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/src/convert_torch_to_tf.py).

Originally, wav2vec2 was pre-trained with a masked language modelling approach with the objective to identify the true quantized latent speech representation for a masked time step. You can read more about the training objective in the paper- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477).

Now, we will be defining a few constants and hyper-parameters which will be useful in the next few cells. `AUDIO_MAXLEN` is intentionally set to `246000` as the model signature only accepts static sequence length of `246000`.


```
AUDIO_MAXLEN = 246000
LABEL_MAXLEN = 256
BATCH_SIZE = 2
```

In the following cell, we will wrap `pretrained_layer` & a dense layer (LM head) with the [Keras's Functional API](https://www.tensorflow.org/guide/keras/functional).


```
inputs = tf.keras.Input(shape=(AUDIO_MAXLEN,))
hidden_states = pretrained_layer(inputs)
outputs = tf.keras.layers.Dense(config.vocab_size)(hidden_states)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

The dense layer (defined above) is having an output dimension of `vocab_size` as we want to predict probabilities of each token in the vocabulary at each time step.

## Setting up training state

In TensorFlow, model weights are built only when `model.call` or `model.build` is called for the first time, so the following cell will build the model weights for us. Further, we will be running `model.summary()` to check the total number of trainable parameters.


```
model(tf.random.uniform(shape=(BATCH_SIZE, AUDIO_MAXLEN)))
model.summary()
```

Now, we need to define the `loss_fn` and optimizer to be able to train the model. The following cell will do that for us. We will be using the `Adam` optimizer for simplicity. `CTCLoss` is a common loss type that is used for tasks (like `ASR`) where input sub-parts can't be easily aligned with output sub-parts. You can read more about CTC-loss from this amazing [blog post](https://distill.pub/2017/ctc/).


`CTCLoss` (from [`gsoc-wav2vec2`](https://github.com/vasudevgupta7/gsoc-wav2vec2) package) accepts 3 arguments: `config`, `model_input_shape` & `division_factor`. If `division_factor=1`, then loss will simply get summed, so pass `division_factor` accordingly to get mean over batch.


```
from wav2vec2 import CTCLoss

LEARNING_RATE = 5e-5

loss_fn = CTCLoss(config, (BATCH_SIZE, AUDIO_MAXLEN), division_factor=BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
```

## Loading & Pre-processing data

Let's now download the LibriSpeech dataset from the [official website](http://www.openslr.org/12) and set it up.


```
!wget https://www.openslr.org/resources/12/dev-clean.tar.gz -P ./data/train/
!tar -xf ./data/train/dev-clean.tar.gz -C ./data/train/
```

**Note:** We are using `dev-clean` configuration as this notebook is just for demonstration purposes, so we need a small amount of data. Complete training data can be easily downloaded from [LibriSpeech website](http://www.openslr.org/12).


```
ls ./data/train/
```

Our dataset lies in the LibriSpeech directory. Let's explore these files.


```
data_dir = "./data/train/LibriSpeech/dev-clean/2428/83705/"
all_files = os.listdir(data_dir)

flac_files = [f for f in all_files if f.endswith(".flac")]
txt_files = [f for f in all_files if f.endswith(".txt")]

print("Transcription files:", txt_files, "\nSound files:", flac_files)
```

Alright, so each sub-directory has many `.flac` files and a `.txt` file. The `.txt` file contains text transcriptions for all the speech samples (i.e. `.flac` files) present in that sub-directory.

We can load this text data as follows:


```
def read_txt_file(f):
  with open(f, "r") as f:
    samples = f.read().split("\n")
    samples = {s.split()[0]: " ".join(s.split()[1:]) for s in samples if len(s.split()) > 2}
  return samples
```

Similarly, we will define a function for loading a speech sample from a `.flac` file.

`REQUIRED_SAMPLE_RATE` is set to `16000` as wav2vec2 was pre-trained with `16K` frequency and it's recommended to fine-tune it without any major change in data distribution due to frequency.


```
import soundfile as sf

REQUIRED_SAMPLE_RATE = 16000

def read_flac_file(file_path):
  with open(file_path, "rb") as f:
      audio, sample_rate = sf.read(f)
  if sample_rate != REQUIRED_SAMPLE_RATE:
      raise ValueError(
          f"sample rate (={sample_rate}) of your files must be {REQUIRED_SAMPLE_RATE}"
      )
  file_id = os.path.split(file_path)[-1][:-len(".flac")]
  return {file_id: audio}
```

Now, we will pick some random samples & will try to visualize them.


```
from IPython.display import Audio
import random

file_id = random.choice([f[:-len(".flac")] for f in flac_files])
flac_file_path, txt_file_path = os.path.join(data_dir, f"{file_id}.flac"), os.path.join(data_dir, "2428-83705.trans.txt")

print("Text Transcription:", read_txt_file(txt_file_path)[file_id], "\nAudio:")
Audio(filename=flac_file_path)
```

Now, we will combine all the speech & text samples and will define the function (in next cell) for that purpose.


```
def fetch_sound_text_mapping(data_dir):
  all_files = os.listdir(data_dir)

  flac_files = [os.path.join(data_dir, f) for f in all_files if f.endswith(".flac")]
  txt_files = [os.path.join(data_dir, f) for f in all_files if f.endswith(".txt")]

  txt_samples = {}
  for f in txt_files:
    txt_samples.update(read_txt_file(f))

  speech_samples = {}
  for f in flac_files:
    speech_samples.update(read_flac_file(f))

  assert len(txt_samples) == len(speech_samples)

  samples = [(speech_samples[file_id], txt_samples[file_id]) for file_id in speech_samples.keys() if len(speech_samples[file_id]) < AUDIO_MAXLEN]
  return samples
```

It's time to have a look at a few samples ...


```
samples = fetch_sound_text_mapping(data_dir)
samples[:5]
```

Note: We are loading this data into memory as we working with a small amount of dataset in this notebook. But for training on the complete dataset (~300 GBs), you will have to load data lazily. You can refer to [this script](https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/src/data_utils.py) to know more on that.

Let's pre-process the data now !!!

We will first define the tokenizer & processor using `gsoc-wav2vec2` package. Then, we will do very simple pre-processing. `processor` will normalize raw speech w.r.to frames axis and `tokenizer` will convert our model outputs into the string (using the defined vocabulary) & will take care of the removal of special tokens (depending on your tokenizer configuration).


```
from wav2vec2 import Wav2Vec2Processor
tokenizer = Wav2Vec2Processor(is_tokenizer=True)
processor = Wav2Vec2Processor(is_tokenizer=False)

def preprocess_text(text):
  label = tokenizer(text)
  return tf.constant(label, dtype=tf.int32)

def preprocess_speech(audio):
  audio = tf.constant(audio, dtype=tf.float32)
  return processor(tf.transpose(audio))
```

Now, we will define the python generator to call the preprocessing functions we defined in above cells.


```
def inputs_generator():
  for speech, text in samples:
    yield preprocess_speech(speech), preprocess_text(text)
```

## Setting up `tf.data.Dataset`

Following cell will setup `tf.data.Dataset` object using its `.from_generator(...)` method. We will be using the `generator` object, we defined in the above cell.

**Note:** For distributed training (especially on TPUs), `.from_generator(...)` doesn't work currently and it is recommended to train on data stored in `.tfrecord` format (Note: The TFRecords should ideally be stored inside a GCS Bucket in order for the TPUs to work to the fullest extent).

You can refer to [this script](https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/src/make_tfrecords.py) for more details on how to convert LibriSpeech data into tfrecords.


```
output_signature = (
    tf.TensorSpec(shape=(None),  dtype=tf.float32),
    tf.TensorSpec(shape=(None), dtype=tf.int32),
)

dataset = tf.data.Dataset.from_generator(inputs_generator, output_signature=output_signature)
```


```
BUFFER_SIZE = len(flac_files)
SEED = 42

dataset = dataset.shuffle(BUFFER_SIZE, seed=SEED)
```

We will pass the dataset into multiple batches, so let's prepare batches in the following cell. Now, all the sequences in a batch should be padded to a constant length. We will use the`.padded_batch(...)` method for that purpose.


```
dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=(AUDIO_MAXLEN, LABEL_MAXLEN), padding_values=(0.0, 0))
```

Accelerators (like GPUs/TPUs) are very fast and often data-loading (& pre-processing) becomes the bottleneck during training as the data-loading part happens on CPUs. This can increase the training time significantly especially when there is a lot of online pre-processing involved or data is streamed online from GCS buckets. To handle those issues, `tf.data.Dataset` offers the `.prefetch(...)` method. This method helps in preparing the next few batches in parallel (on CPUs) while the model is making predictions (on GPUs/TPUs) on the current batch.


```
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

Since this notebook is made for demonstration purposes, we will be taking first `num_train_batches` and will perform training over only that. You are encouraged to train on the whole dataset though. Similarly, we will evaluate only `num_val_batches`.


```
num_train_batches = 10
num_val_batches = 4

train_dataset = dataset.take(num_train_batches)
val_dataset = dataset.skip(num_train_batches).take(num_val_batches)
```

## Model training

For training our model, we will be directly calling `.fit(...)` method after compiling our model with `.compile(...)`.


```
model.compile(optimizer, loss=loss_fn)
```

The above cell will set up our training state. Now we can initiate training with the `.fit(...)` method.


```
history = model.fit(train_dataset, validation_data=val_dataset, epochs=3)
history.history
```

Let's save our model with `.save(...)` method to be able to perform inference later. You can also export this SavedModel to TFHub by following [TFHub documentation](https://www.tensorflow.org/hub/publish).


```
save_dir = "finetuned-wav2vec2"
model.save(save_dir, include_optimizer=False)
```

Note: We are setting `include_optimizer=False` as we want to use this model for inference only.

## Evaluation

Now we will be computing Word Error Rate over the validation dataset

**Word error rate** (WER) is a common metric for measuring the performance of an automatic speech recognition system. The WER is derived from the Levenshtein distance, working at the word level. Word error rate can then be computed as: WER = (S + D + I) / N = (S + D + I) / (S + D + C) where S is the number of substitutions, D is the number of deletions, I is the number of insertions, C is the number of correct words, N is the number of words in the reference (N=S+D+C). This value indicates the percentage of words that were incorrectly predicted. 

You can refer to [this paper](https://www.isca-speech.org/archive_v0/interspeech_2004/i04_2765.html) to learn more about WER.

We will use `load_metric(...)` function from [HuggingFace datasets](https://huggingface.co/docs/datasets/) library. Let's first install the `datasets` library using `pip` and then define the `metric` object.


```
!pip3 install -q datasets

from datasets import load_metric
metric = load_metric("wer")
```


```
@tf.function(jit_compile=True)
def eval_fwd(batch):
  logits = model(batch, training=False)
  return tf.argmax(logits, axis=-1)
```

It's time to run the evaluation on validation data now.


```
from tqdm.auto import tqdm

for speech, labels in tqdm(val_dataset, total=num_val_batches):
    predictions  = eval_fwd(speech)
    predictions = [tokenizer.decode(pred) for pred in predictions.numpy().tolist()]
    references = [tokenizer.decode(label, group_tokens=False) for label in labels.numpy().tolist()]
    metric.add_batch(references=references, predictions=predictions)
```

We are using the `tokenizer.decode(...)` method for decoding our predictions and labels back into the text and will add them to the metric for `WER` computation later.

Now, let's calculate the metric value in following cell:


```
metric.compute()
```

**Note:** Here metric value doesn't make any sense as the model is trained on very small data and ASR-like tasks often require a large amount of data to learn a mapping from speech to text. You should probably train on large data to get some good results. This notebook gives you a template to fine-tune a pre-trained speech model.

## Inference

Now that we are satisfied with the training process & have saved the model in `save_dir`, we will see how this model can be used for inference.

First, we will load our model using `tf.keras.models.load_model(...)`.


```
finetuned_model = tf.keras.models.load_model(save_dir)
```

Let's download some speech samples for performing inference. You can replace the following sample with your speech sample also.


```
!wget https://github.com/vasudevgupta7/gsoc-wav2vec2/raw/main/data/SA2.wav
```

Now, we will read the speech sample using `soundfile.read(...)` and pad it to `AUDIO_MAXLEN` to satisfy the model signature. Then we will normalize that speech sample using the `Wav2Vec2Processor` instance & will feed it into the model.


```
import numpy as np

speech, _ = sf.read("SA2.wav")
speech = np.pad(speech, (0, AUDIO_MAXLEN - len(speech)))
speech = tf.expand_dims(processor(tf.constant(speech)), 0)

outputs = finetuned_model(speech)
outputs
```

Let's decode numbers back into text sequence using the `Wav2Vec2tokenizer` instance, we defined above.


```
predictions = tf.argmax(outputs, axis=-1)
predictions = [tokenizer.decode(pred) for pred in predictions.numpy().tolist()]
predictions
```

This prediction is quite random as the model was never trained on large data in this notebook (as this notebook is not meant for doing complete training). You will get good predictions if you train this model on complete LibriSpeech dataset.

Finally, we have reached an end to this notebook. But it's not an end of learning TensorFlow for speech-related tasks, this [repository](https://github.com/tulasiram58827/TTS_TFLite) contains some more amazing tutorials. In case you encountered any bug in this notebook, please create an issue [here](https://github.com/vasudevgupta7/gsoc-wav2vec2/issues).
