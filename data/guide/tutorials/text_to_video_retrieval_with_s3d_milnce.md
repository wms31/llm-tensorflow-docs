```
# Copyright 2019 The TensorFlow Hub Authors. All Rights Reserved.
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

# Text-to-Video retrieval with S3D MIL-NCE

<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/hub/tutorials/text_to_video_retrieval_with_s3d_milnce"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/hub/tutorials/text_to_video_retrieval_with_s3d_milnce.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/hub/tutorials/text_to_video_retrieval_with_s3d_milnce.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/hub/tutorials/text_to_video_retrieval_with_s3d_milnce.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
  <td>
    <a href="https://tfhub.dev/deepmind/mil-nce/s3d/1"><img src="https://www.tensorflow.org/images/hub_logo_32px.png" />See TF Hub model</a>
  </td>
</table>


```
!pip install -q opencv-python

import os

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

import numpy as np
import cv2
from IPython import display
import math
```

## Import TF-Hub model

This tutorial demonstrates how to use the [S3D MIL-NCE model](https://tfhub.dev/deepmind/mil-nce/s3d/1) from TensorFlow Hub to do **text-to-video retrieval** to find the most similar videos for a given text query.

The model has 2 signatures, one for generating *video embeddings* and one for generating *text embeddings*. We will use these embedding to find the nearest neighbors in the embedding space.


```
# Load the model once from TF-Hub.
hub_handle = 'https://tfhub.dev/deepmind/mil-nce/s3d/1'
hub_model = hub.load(hub_handle)

def generate_embeddings(model, input_frames, input_words):
  """Generate embeddings from the model from video frames and input words."""
  # Input_frames must be normalized in [0, 1] and of the shape Batch x T x H x W x 3
  vision_output = model.signatures['video'](tf.constant(tf.cast(input_frames, dtype=tf.float32)))
  text_output = model.signatures['text'](tf.constant(input_words))
  return vision_output['video_embedding'], text_output['text_embedding']
```


```
# @title Define video loading and visualization functions  { display-mode: "form" }

# Utilities to open video files using CV2
def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]


def load_video(video_url, max_frames=32, resize=(224, 224)):
  path = tf.keras.utils.get_file(os.path.basename(video_url)[-128:], video_url)
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)

      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  frames = np.array(frames)
  if len(frames) < max_frames:
    n_repeat = int(math.ceil(max_frames / float(len(frames))))
    frames = frames.repeat(n_repeat, axis=0)
  frames = frames[:max_frames]
  return frames / 255.0

def display_video(urls):
    html = '<table>'
    html += '<tr><th>Video 1</th><th>Video 2</th><th>Video 3</th></tr><tr>'
    for url in urls:
        html += '<td>'
        html += '<img src="{}" height="224">'.format(url)
        html += '</td>'
    html += '</tr></table>'
    return display.HTML(html)

def display_query_and_results_video(query, urls, scores):
  """Display a text query and the top result videos and scores."""
  sorted_ix = np.argsort(-scores)
  html = ''
  html += '<h2>Input query: <i>{}</i> </h2><div>'.format(query)
  html += 'Results: <div>'
  html += '<table>'
  html += '<tr><th>Rank #1, Score:{:.2f}</th>'.format(scores[sorted_ix[0]])
  html += '<th>Rank #2, Score:{:.2f}</th>'.format(scores[sorted_ix[1]])
  html += '<th>Rank #3, Score:{:.2f}</th></tr><tr>'.format(scores[sorted_ix[2]])
  for i, idx in enumerate(sorted_ix):
    url = urls[sorted_ix[i]];
    html += '<td>'
    html += '<img src="{}" height="224">'.format(url)
    html += '</td>'
  html += '</tr></table>'
  return html

```


```
# @title Load example videos and define text queries  { display-mode: "form" }

video_1_url = 'https://upload.wikimedia.org/wikipedia/commons/b/b0/YosriAirTerjun.gif' # @param {type:"string"}
video_2_url = 'https://upload.wikimedia.org/wikipedia/commons/e/e6/Guitar_solo_gif.gif' # @param {type:"string"}
video_3_url = 'https://upload.wikimedia.org/wikipedia/commons/3/30/2009-08-16-autodrift-by-RalfR-gif-by-wau.gif' # @param {type:"string"}

video_1 = load_video(video_1_url)
video_2 = load_video(video_2_url)
video_3 = load_video(video_3_url)
all_videos = [video_1, video_2, video_3]

query_1_video = 'waterfall' # @param {type:"string"}
query_2_video = 'playing guitar' # @param {type:"string"}
query_3_video = 'car drifting' # @param {type:"string"}
all_queries_video = [query_1_video, query_2_video, query_3_video]
all_videos_urls = [video_1_url, video_2_url, video_3_url]
display_video(all_videos_urls)
```

## Demonstrate text to video retrieval



```
# Prepare video inputs.
videos_np = np.stack(all_videos, axis=0)

# Prepare text input.
words_np = np.array(all_queries_video)

# Generate the video and text embeddings.
video_embd, text_embd = generate_embeddings(hub_model, videos_np, words_np)

# Scores between video and text is computed by dot products.
all_scores = np.dot(text_embd, tf.transpose(video_embd))
```


```
# Display results.
html = ''
for i, words in enumerate(words_np):
  html += display_query_and_results_video(words, all_videos_urls, all_scores[i, :])
  html += '<br>'
display.HTML(html)
```
