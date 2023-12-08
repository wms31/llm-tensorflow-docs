##### Copyright 2023 The TensorFlow Hub Authors.

Licensed under the Apache License, Version 2.0 (the "License");


```
#@title Copyright 2023 The TensorFlow Hub Authors. All Rights Reserved.
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
  <!-- <td>
    <a target="_blank" href="https://www.tensorflow.org/hub/tutorials/bird_vocalization_classifier"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td> -->
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/hub/tutorials/bird_vocalization_classifier.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/hub/tutorials/bird_vocalization_classifier.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/hub/tutorials/bird_vocalization_classifier.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
  <td>
    <a href="https://tfhub.dev/google/bird-vocalization-classifier/1"><img src="https://www.tensorflow.org/images/hub_logo_32px.png" />See TF Hub model</a>
  </td>
</table>

# Using Google Bird Vocalization model

The Google Bird Vocalization is a global bird embedding and classification model.

This model expects as input a 5-second audio segment sampled at 32kHz

The model outputs both the logits and the embeddigs for each input window of audio.

On this notebook you'll learn how to feed the audio properly to the model and how to use the logits for inference.



```
!pip install -q "tensorflow_io==0.28.*"
!pip install -q librosa
```


```
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

import numpy as np
import librosa

import csv
import io

from IPython.display import Audio
```

Loading the Model from TFHub


```
model_handle = "https://tfhub.dev/google/bird-vocalization-classifier/1"
model = hub.load(model_handle)
```

Lets load the labels that the model was trained on.

The labels file is in the assets forlder under label.csv. Each line is an ebird id.


```
# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  with open(labels_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    class_names = [mid for mid, desc in csv_reader]
    return class_names[1:]

labels_path = hub.resolve(model_handle) + "/assets/label.csv"
classes = class_names_from_csv(labels_path)
print(classes)
```

The ```frame_audio``` function is based on the [Chirp lib](https://github.com/google-research/chirp/blob/10c5faa325a3c3468fa6f18a736fc1aeb9bf8129/chirp/inference/interface.py#L128) version but using tf.signal instead of librosa.

The `ensure_sample_rate` is a function to make sure that any audio used with the model has the expected sample rate of 32kHz


```
def frame_audio(
      audio_array: np.ndarray,
      window_size_s: float = 5.0,
      hop_size_s: float = 5.0,
      sample_rate = 32000,
  ) -> np.ndarray:
    """Helper function for framing audio for inference."""
    if window_size_s is None or window_size_s < 0:
      return audio_array[np.newaxis, :]
    frame_length = int(window_size_s * sample_rate)
    hop_length = int(hop_size_s * sample_rate)
    framed_audio = tf.signal.frame(audio_array, frame_length, hop_length, pad_end=True)
    return framed_audio

def ensure_sample_rate(waveform, original_sample_rate,
                       desired_sample_rate=32000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    waveform = tfio.audio.resample(waveform, original_sample_rate, desired_sample_rate)
  return desired_sample_rate, waveform
```

Lets load a file from Wikipedia.

To be more precise, the audio of a [Common Blackbird](https://es.wikipedia.org/wiki/Turdus_merula)

|<p><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Common_Blackbird.jpg/1200px-Common_Blackbird.jpg" alt="Common Blackbird.jpg">|
|:--:|
| *By <a rel="nofollow" class="external text" href="http://photo-natur.de">Andreas Trepte</a> - <span class="int-own-work" lang="en">Own work</span>, <a href="https://creativecommons.org/licenses/by-sa/2.5" title="Creative Commons Attribution-Share Alike 2.5">CC BY-SA 2.5</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=16110223">Link*</a></p> |


The audio was contributed by Oona Räisänen (Mysid) under the public domain license.


```
!curl -O  "https://upload.wikimedia.org/wikipedia/commons/7/7c/Turdus_merula_2.ogg"
```


```
turdus_merula = "Turdus_merula_2.ogg"

audio, sample_rate = librosa.load(turdus_merula)

sample_rate, wav_data_turdus = ensure_sample_rate(audio, sample_rate)
Audio(wav_data_turdus, rate=sample_rate)
```

The audio has 24 seconds and the model expects chunks of 5 seconds.

The `frame_audio` function can fix that and split the audio in proper frames


```
fixed_tm = frame_audio(wav_data_turdus)
fixed_tm.shape
```

Let's apply the model only on the first frame:


```
logits, embeddings = model.infer_tf(fixed_tm[:1])
```

The label.csv file contains ebirds ids.
The ebird id for Turdus Merula is eurbla


```
probabilities = tf.nn.softmax(logits)
argmax = np.argmax(probabilities)
print(f"The audio is from the class {classes[argmax]} (element:{argmax} in the label.csv file), with probability of {probabilities[0][argmax]}")
```

Lets apply the model on all the frames now:

*note*: this code is also based on the [Chirp library](https://github.com/google-research/chirp/blob/d6ff5e7cee3865940f31697bf4b70176c1072572/chirp/inference/models.py#L174)


```
all_logits, all_embeddings = model.infer_tf(fixed_tm[:1])
for window in fixed_tm[1:]:
  logits, embeddings = model.infer_tf(window[np.newaxis, :])
  all_logits = np.concatenate([all_logits, logits], axis=0)

all_logits.shape
```


```
frame = 0
for frame_logits in all_logits:
  probabilities = tf.nn.softmax(frame_logits)
  argmax = np.argmax(probabilities)
  print(f"For frame {frame}, the audio is from the class {classes[argmax]} (element:{argmax} in the label.csv file), with probability of {probabilities[argmax]}")
  frame += 1
```
