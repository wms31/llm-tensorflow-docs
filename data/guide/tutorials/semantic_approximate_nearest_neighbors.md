##### Copyright 2019 The TensorFlow Hub Authors.

Licensed under the Apache License, Version 2.0 (the "License");


```
# Copyright 2018 The TensorFlow Hub Authors. All Rights Reserved.
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

# Semantic Search with Approximate Nearest Neighbors and Text Embeddings


<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/hub/tutorials/semantic_approximate_nearest_neighbors"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/hub/tutorials/semantic_approximate_nearest_neighbors.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/hub/tutorials/semantic_approximate_nearest_neighbors.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/hub/tutorials/semantic_approximate_nearest_neighbors.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
  <td>
    <a href="https://tfhub.dev/google/universal-sentence-encoder/2"><img src="https://www.tensorflow.org/images/hub_logo_32px.png" />See TF Hub models</a>
  </td>
</table>

This tutorial illustrates how to generate embeddings from a [TensorFlow Hub](https://tfhub.dev) (TF-Hub) module given input data, and build an approximate nearest neighbours (ANN) index using the extracted embeddings. The index can then be used for real-time similarity matching and retrieval. 

When dealing with a large corpus of data, it's not efficient to perform exact matching by scanning the whole repository to find the most similar items to a given query in real-time. Thus, we use an approximate similarity matching algorithm which allows us to trade off a little bit of accuracy in finding exact nearest neighbor matches for a significant boost in speed. 

In this tutorial, we show an example of real-time text search over a corpus of news headlines to find the headlines that are most similar to a query. Unlike keyword search, this captures the semantic similarity encoded in the text embedding.

The steps of this tutorial are:
1. Download sample data.
2. Generate embeddings for the data using a TF-Hub module
3. Build an ANN index for the embeddings
4. Use the index for similarity matching

We use [Apache Beam](https://beam.apache.org/documentation/programming-guide/) with [TensorFlow Transform](https://www.tensorflow.org/tfx/tutorials/transform/simple) (TF-Transform) to generate the embeddings from the TF-Hub module. We also use Spotify's [ANNOY](https://github.com/spotify/annoy) library to build the approximate nearest neighbours index. You can find benchmarking of ANN framework in this [Github repository](https://github.com/erikbern/ann-benchmarks).

This tutorial uses TensorFlow 1.0 and works only with TF1 [Hub modules](https://www.tensorflow.org/hub/tf1_hub_module) from TF-Hub. See the updated [TF2 version of this tutorial](https://github.com/tensorflow/docs/blob/master/site/en/hub/tutorials/tf2_semantic_approximate_nearest_neighbors.ipynb).

## Setup

Install the required libraries.


```
!pip install -q apache_beam
!pip install -q 'scikit_learn~=0.23.0'  # For gaussian_random_matrix.
!pip install -q annoy
```

Import the required libraries


```
import os
import sys
import pathlib
import pickle
from collections import namedtuple
from datetime import datetime

import numpy as np
import apache_beam as beam
import annoy
from sklearn.random_projection import gaussian_random_matrix

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
```


```
# TFT needs to be installed afterwards
!pip install -q tensorflow_transform==0.24
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
```


```
print('TF version: {}'.format(tf.__version__))
print('TF-Hub version: {}'.format(hub.__version__))
print('TF-Transform version: {}'.format(tft.__version__))
print('Apache Beam version: {}'.format(beam.__version__))
```

## 1. Download Sample Data

[A Million News Headlines](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SYBGZL#) dataset contains news headlines published over a period of 15 years sourced from the reputable Australian Broadcasting Corp. (ABC). This news dataset has a summarised historical record of noteworthy events in the globe from early-2003 to end-2017 with a more granular focus on Australia. 

**Format**: Tab-separated two-column data: 1) publication date and 2) headline text. We are only interested in the headline text.



```
!wget 'https://dataverse.harvard.edu/api/access/datafile/3450625?format=tab&gbrecs=true' -O raw.tsv
!wc -l raw.tsv
!head raw.tsv
```

For simplicity, we only keep the headline text and remove the publication date


```
!rm -r corpus
!mkdir corpus

with open('corpus/text.txt', 'w') as out_file:
  with open('raw.tsv', 'r') as in_file:
    for line in in_file:
      headline = line.split('\t')[1].strip().strip('"')
      out_file.write(headline+"\n")
```


```
!tail corpus/text.txt
```

## Helper function to load a TF-Hub module


```
def load_module(module_url):
  embed_module = hub.Module(module_url)
  placeholder = tf.placeholder(dtype=tf.string)
  embed = embed_module(placeholder)
  session = tf.Session()
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  print('TF-Hub module is loaded.')

  def _embeddings_fn(sentences):
    computed_embeddings = session.run(
        embed, feed_dict={placeholder: sentences})
    return computed_embeddings

  return _embeddings_fn
```

## 2. Generate Embeddings for the Data.

In this tutorial, we use the [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/2) to generate embeddings for the headline data. The sentence embeddings can then be easily used to compute sentence level meaning similarity. We run the embedding generation process using Apache Beam and TF-Transform.

### Embedding extraction method


```
encoder = None

def embed_text(text, module_url, random_projection_matrix):
  # Beam will run this function in different processes that need to
  # import hub and load embed_fn (if not previously loaded)
  global encoder
  if not encoder:
    encoder = hub.Module(module_url)
  embedding = encoder(text)
  if random_projection_matrix is not None:
    # Perform random projection for the embedding
    embedding = tf.matmul(
        embedding, tf.cast(random_projection_matrix, embedding.dtype))
  return embedding

```

### Make TFT preprocess_fn method


```
def make_preprocess_fn(module_url, random_projection_matrix=None):
  '''Makes a tft preprocess_fn'''

  def _preprocess_fn(input_features):
    '''tft preprocess_fn'''
    text = input_features['text']
    # Generate the embedding for the input text
    embedding = embed_text(text, module_url, random_projection_matrix)
    
    output_features = {
        'text': text, 
        'embedding': embedding
        }
        
    return output_features
  
  return _preprocess_fn
```

### Create dataset metadata


```
def create_metadata():
  '''Creates metadata for the raw data'''
  from tensorflow_transform.tf_metadata import dataset_metadata
  from tensorflow_transform.tf_metadata import schema_utils
  feature_spec = {'text': tf.FixedLenFeature([], dtype=tf.string)}
  schema = schema_utils.schema_from_feature_spec(feature_spec)
  metadata = dataset_metadata.DatasetMetadata(schema)
  return metadata
```

### Beam pipeline


```
def run_hub2emb(args):
  '''Runs the embedding generation pipeline'''

  options = beam.options.pipeline_options.PipelineOptions(**args)
  args = namedtuple("options", args.keys())(*args.values())

  raw_metadata = create_metadata()
  converter = tft.coders.CsvCoder(
      column_names=['text'], schema=raw_metadata.schema)

  with beam.Pipeline(args.runner, options=options) as pipeline:
    with tft_beam.Context(args.temporary_dir):
      # Read the sentences from the input file
      sentences = ( 
          pipeline
          | 'Read sentences from files' >> beam.io.ReadFromText(
              file_pattern=args.data_dir)
          | 'Convert to dictionary' >> beam.Map(converter.decode)
      )

      sentences_dataset = (sentences, raw_metadata)
      preprocess_fn = make_preprocess_fn(args.module_url, args.random_projection_matrix)
      # Generate the embeddings for the sentence using the TF-Hub module
      embeddings_dataset, _ = (
          sentences_dataset
          | 'Extract embeddings' >> tft_beam.AnalyzeAndTransformDataset(preprocess_fn)
      )

      embeddings, transformed_metadata = embeddings_dataset
      # Write the embeddings to TFRecords files
      embeddings | 'Write embeddings to TFRecords' >> beam.io.tfrecordio.WriteToTFRecord(
          file_path_prefix='{}/emb'.format(args.output_dir),
          file_name_suffix='.tfrecords',
          coder=tft.coders.ExampleProtoCoder(transformed_metadata.schema))
```

### Generaring Random Projection Weight Matrix

[Random projection](https://en.wikipedia.org/wiki/Random_projection) is a simple, yet powerfull technique used to reduce the dimensionality of a set of points which lie in Euclidean space. For a theoretical background, see the [Johnson-Lindenstrauss lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma).

Reducing the dimensionality of the embeddings with random projection means less time needed to build and query the ANN index.

In this tutorial we use [Gaussian Random Projection](https://en.wikipedia.org/wiki/Random_projection#Gaussian_random_projection) from the [Scikit-learn](https://scikit-learn.org/stable/modules/random_projection.html#gaussian-random-projection) library.


```
def generate_random_projection_weights(original_dim, projected_dim):
  random_projection_matrix = None
  if projected_dim and original_dim > projected_dim:
    random_projection_matrix = gaussian_random_matrix(
        n_components=projected_dim, n_features=original_dim).T
    print("A Gaussian random weight matrix was creates with shape of {}".format(random_projection_matrix.shape))
    print('Storing random projection matrix to disk...')
    with open('random_projection_matrix', 'wb') as handle:
      pickle.dump(random_projection_matrix, 
                  handle, protocol=pickle.HIGHEST_PROTOCOL)
        
  return random_projection_matrix
```

### Set parameters
If you want to build an index using the original embedding space without random projection, set the `projected_dim` parameter to `None`. Note that this will slow down the indexing step for high-dimensional embeddings.


```
module_url = 'https://tfhub.dev/google/universal-sentence-encoder/2' #@param {type:"string"}
projected_dim = 64  #@param {type:"number"}
```

### Run pipeline


```
import tempfile

output_dir = pathlib.Path(tempfile.mkdtemp())
temporary_dir = pathlib.Path(tempfile.mkdtemp())

g = tf.Graph()
with g.as_default():
  original_dim = load_module(module_url)(['']).shape[1]
  random_projection_matrix = None

  if projected_dim:
    random_projection_matrix = generate_random_projection_weights(
        original_dim, projected_dim)

args = {
    'job_name': 'hub2emb-{}'.format(datetime.utcnow().strftime('%y%m%d-%H%M%S')),
    'runner': 'DirectRunner',
    'batch_size': 1024,
    'data_dir': 'corpus/*.txt',
    'output_dir': output_dir,
    'temporary_dir': temporary_dir,
    'module_url': module_url,
    'random_projection_matrix': random_projection_matrix,
}

print("Pipeline args are set.")
args
```


```
!rm -r {output_dir}
!rm -r {temporary_dir}

print("Running pipeline...")
%time run_hub2emb(args)
print("Pipeline is done.")
```


```
!ls {output_dir}
```

Read some of the generated embeddings...


```
import itertools

embed_file = os.path.join(output_dir, 'emb-00000-of-00001.tfrecords')
sample = 5
record_iterator =  tf.io.tf_record_iterator(path=embed_file)
for string_record in itertools.islice(record_iterator, sample):
  example = tf.train.Example()
  example.ParseFromString(string_record)
  text = example.features.feature['text'].bytes_list.value
  embedding = np.array(example.features.feature['embedding'].float_list.value)
  print("Embedding dimensions: {}".format(embedding.shape[0]))
  print("{}: {}".format(text, embedding[:10]))

```

## 3. Build the ANN Index for the Embeddings

[ANNOY](https://github.com/spotify/annoy) (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings to search for points in space that are close to a given query point. It also creates large read-only file-based data structures that are mmapped into memory. It is built and used by [Spotify](https://www.spotify.com) for music recommendations.


```
def build_index(embedding_files_pattern, index_filename, vector_length, 
    metric='angular', num_trees=100):
  '''Builds an ANNOY index'''

  annoy_index = annoy.AnnoyIndex(vector_length, metric=metric)
  # Mapping between the item and its identifier in the index
  mapping = {}

  embed_files = tf.gfile.Glob(embedding_files_pattern)
  print('Found {} embedding file(s).'.format(len(embed_files)))

  item_counter = 0
  for f, embed_file in enumerate(embed_files):
    print('Loading embeddings in file {} of {}...'.format(
      f+1, len(embed_files)))
    record_iterator = tf.io.tf_record_iterator(
      path=embed_file)

    for string_record in record_iterator:
      example = tf.train.Example()
      example.ParseFromString(string_record)
      text = example.features.feature['text'].bytes_list.value[0].decode("utf-8")
      mapping[item_counter] = text
      embedding = np.array(
        example.features.feature['embedding'].float_list.value)
      annoy_index.add_item(item_counter, embedding)
      item_counter += 1
      if item_counter % 100000 == 0:
        print('{} items loaded to the index'.format(item_counter))

  print('A total of {} items added to the index'.format(item_counter))

  print('Building the index with {} trees...'.format(num_trees))
  annoy_index.build(n_trees=num_trees)
  print('Index is successfully built.')
  
  print('Saving index to disk...')
  annoy_index.save(index_filename)
  print('Index is saved to disk.')
  print("Index file size: {} GB".format(
    round(os.path.getsize(index_filename) / float(1024 ** 3), 2)))
  annoy_index.unload()

  print('Saving mapping to disk...')
  with open(index_filename + '.mapping', 'wb') as handle:
    pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
  print('Mapping is saved to disk.')
  print("Mapping file size: {} MB".format(
    round(os.path.getsize(index_filename + '.mapping') / float(1024 ** 2), 2)))
```


```
embedding_files = "{}/emb-*.tfrecords".format(output_dir)
embedding_dimension = projected_dim
index_filename = "index"

!rm {index_filename}
!rm {index_filename}.mapping

%time build_index(embedding_files, index_filename, embedding_dimension)
```


```
!ls
```

## 4. Use the Index for Similarity Matching
Now we can use the ANN index to find news headlines that are semantically close to an input query.

### Load the index and the mapping files


```
index = annoy.AnnoyIndex(embedding_dimension)
index.load(index_filename, prefault=True)
print('Annoy index is loaded.')
with open(index_filename + '.mapping', 'rb') as handle:
  mapping = pickle.load(handle)
print('Mapping file is loaded.')

```

### Similarity matching method


```
def find_similar_items(embedding, num_matches=5):
  '''Finds similar items to a given embedding in the ANN index'''
  ids = index.get_nns_by_vector(
  embedding, num_matches, search_k=-1, include_distances=False)
  items = [mapping[i] for i in ids]
  return items
```

### Extract embedding from a given query


```
# Load the TF-Hub module
print("Loading the TF-Hub module...")
g = tf.Graph()
with g.as_default():
  embed_fn = load_module(module_url)
print("TF-Hub module is loaded.")

random_projection_matrix = None
if os.path.exists('random_projection_matrix'):
  print("Loading random projection matrix...")
  with open('random_projection_matrix', 'rb') as handle:
    random_projection_matrix = pickle.load(handle)
  print('random projection matrix is loaded.')

def extract_embeddings(query):
  '''Generates the embedding for the query'''
  query_embedding =  embed_fn([query])[0]
  if random_projection_matrix is not None:
    query_embedding = query_embedding.dot(random_projection_matrix)
  return query_embedding
```


```
extract_embeddings("Hello Machine Learning!")[:10]
```

### Enter a query to find the most similar items


```
#@title { run: "auto" }
query = "confronting global challenges" #@param {type:"string"}
print("Generating embedding for the query...")
%time query_embedding = extract_embeddings(query)

print("")
print("Finding relevant items in the index...")
%time items = find_similar_items(query_embedding, 10)

print("")
print("Results:")
print("=========")
for item in items:
  print(item)
```

## Want to learn more?

You can learn more about TensorFlow at [tensorflow.org](https://www.tensorflow.org/) and see the TF-Hub API documentation at [tensorflow.org/hub](https://www.tensorflow.org/hub/). Find available TensorFlow Hub modules at [tfhub.dev](https://tfhub.dev/) including more text embedding modules and image feature vector modules.

Also check out the [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/) which is Google's fast-paced, practical introduction to machine learning.
