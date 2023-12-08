##### Copyright 2019 The TensorFlow Hub Authors.

Licensed under the Apache License, Version 2.0 (the "License");


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

# Multilingual Universal Sentence Encoder Q&A Retrieval


<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/hub/tutorials/retrieval_with_tf_hub_universal_encoder_qa"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/hub/tutorials/retrieval_with_tf_hub_universal_encoder_qa.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/hub/tutorials/retrieval_with_tf_hub_universal_encoder_qa.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/hub/tutorials/retrieval_with_tf_hub_universal_encoder_qa.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
  <td>
    <a href="https://tfhub.dev/s?q=google%2Funiversal-sentence-encoder-multilingual-qa%2F3%20OR%20google%2Funiversal-sentence-encoder-qa%2F3"><img src="https://www.tensorflow.org/images/hub_logo_32px.png" />See TF Hub models</a>
  </td>
</table>

This is a demo for using [Universal Encoder Multilingual Q&A model](https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3) for question-answer retrieval of text, illustrating the use of **question_encoder** and **response_encoder** of the model. We use sentences from [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) paragraphs as the demo dataset, each sentence and its context (the text surrounding the sentence) is encoded into high dimension embeddings with the **response_encoder**. These embeddings are stored in an index built using the [simpleneighbors](https://pypi.org/project/simpleneighbors/) library for question-answer retrieval.

On retrieval a random question is selected from the [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) dataset and encoded into high dimension embedding with the **question_encoder** and query the  simpleneighbors index returning a list of approximate nearest neighbors in semantic space.

### More models
You can find all currently hosted text embedding models [here](https://tfhub.dev/s?module-type=text-embedding) and all models that have been trained on SQuAD as well [here](https://tfhub.dev/s?dataset=squad).

## Setup



```
%%capture
#@title Setup Environment
# Install the latest Tensorflow version.
!pip install -q "tensorflow-text==2.11.*"
!pip install -q simpleneighbors[annoy]
!pip install -q nltk
!pip install -q tqdm
```


```
#@title Setup common imports and functions
import json
import nltk
import os
import pprint
import random
import simpleneighbors
import urllib
from IPython.display import HTML, display
from tqdm.notebook import tqdm

import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer

nltk.download('punkt')


def download_squad(url):
  return json.load(urllib.request.urlopen(url))

def extract_sentences_from_squad_json(squad):
  all_sentences = []
  for data in squad['data']:
    for paragraph in data['paragraphs']:
      sentences = nltk.tokenize.sent_tokenize(paragraph['context'])
      all_sentences.extend(zip(sentences, [paragraph['context']] * len(sentences)))
  return list(set(all_sentences)) # remove duplicates

def extract_questions_from_squad_json(squad):
  questions = []
  for data in squad['data']:
    for paragraph in data['paragraphs']:
      for qas in paragraph['qas']:
        if qas['answers']:
          questions.append((qas['question'], qas['answers'][0]['text']))
  return list(set(questions))

def output_with_highlight(text, highlight):
  output = "<li> "
  i = text.find(highlight)
  while True:
    if i == -1:
      output += text
      break
    output += text[0:i]
    output += '<b>'+text[i:i+len(highlight)]+'</b>'
    text = text[i+len(highlight):]
    i = text.find(highlight)
  return output + "</li>\n"

def display_nearest_neighbors(query_text, answer_text=None):
  query_embedding = model.signatures['question_encoder'](tf.constant([query_text]))['outputs'][0]
  search_results = index.nearest(query_embedding, n=num_results)

  if answer_text:
    result_md = '''
    <p>Random Question from SQuAD:</p>
    <p>&nbsp;&nbsp;<b>%s</b></p>
    <p>Answer:</p>
    <p>&nbsp;&nbsp;<b>%s</b></p>
    ''' % (query_text , answer_text)
  else:
    result_md = '''
    <p>Question:</p>
    <p>&nbsp;&nbsp;<b>%s</b></p>
    ''' % query_text

  result_md += '''
    <p>Retrieved sentences :
    <ol>
  '''

  if answer_text:
    for s in search_results:
      result_md += output_with_highlight(s, answer_text)
  else:
    for s in search_results:
      result_md += '<li>' + s + '</li>\n'

  result_md += "</ol>"
  display(HTML(result_md))
```

Run the following code block to download and extract the SQuAD dataset into:

* **sentences** is a list of (text, context) tuples - each paragraph from the SQuAD dataset are split into sentences using nltk library and the sentence and paragraph text forms the (text, context) tuple.
* **questions** is a list of (question, answer) tuples.

Note: You can use this demo to index the SQuAD train dataset or the smaller dev dataset (1.1 or 2.0) by selecting the **squad_url** below.



```
#@title Download and extract SQuAD data
squad_url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json' #@param ["https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json", "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"]

squad_json = download_squad(squad_url)
sentences = extract_sentences_from_squad_json(squad_json)
questions = extract_questions_from_squad_json(squad_json)
print("%s sentences, %s questions extracted from SQuAD %s" % (len(sentences), len(questions), squad_url))

print("\nExample sentence and context:\n")
sentence = random.choice(sentences)
print("sentence:\n")
pprint.pprint(sentence[0])
print("\ncontext:\n")
pprint.pprint(sentence[1])
print()
```

The following code block setup the tensorflow graph **g** and **session** with the [Universal Encoder Multilingual Q&A model](https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3)'s **question_encoder** and **response_encoder** signatures.


```
#@title Load model from tensorflow hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3", "https://tfhub.dev/google/universal-sentence-encoder-qa/3"]
model = hub.load(module_url)

```

The following code block compute the embeddings for all the text, context tuples and store them in a [simpleneighbors](https://pypi.org/project/simpleneighbors/) index using the **response_encoder**.



```
#@title Compute embeddings and build simpleneighbors index
batch_size = 100

encodings = model.signatures['response_encoder'](
  input=tf.constant([sentences[0][0]]),
  context=tf.constant([sentences[0][1]]))
index = simpleneighbors.SimpleNeighbors(
    len(encodings['outputs'][0]), metric='angular')

print('Computing embeddings for %s sentences' % len(sentences))
slices = zip(*(iter(sentences),) * batch_size)
num_batches = int(len(sentences) / batch_size)
for s in tqdm(slices, total=num_batches):
  response_batch = list([r for r, c in s])
  context_batch = list([c for r, c in s])
  encodings = model.signatures['response_encoder'](
    input=tf.constant(response_batch),
    context=tf.constant(context_batch)
  )
  for batch_index, batch in enumerate(response_batch):
    index.add_one(batch, encodings['outputs'][batch_index])

index.build()
print('simpleneighbors index for %s sentences built.' % len(sentences))

```

On retrieval, the question is encoded using the **question_encoder** and the question embedding is used to query the simpleneighbors index.


```
#@title Retrieve nearest neighbors for a random question from SQuAD
num_results = 25 #@param {type:"slider", min:5, max:40, step:1}

query = random.choice(questions)
display_nearest_neighbors(query[0], query[1])
```
