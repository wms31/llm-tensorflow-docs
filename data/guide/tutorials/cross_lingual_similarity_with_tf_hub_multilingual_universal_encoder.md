**Copyright 2019 The TensorFlow Hub Authors.**

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

# Cross-Lingual Similarity and Semantic Search Engine with Multilingual Universal Sentence Encoder


<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://www.tensorflow.org/hub/tutorials/cross_lingual_similarity_with_tf_hub_multilingual_universal_encoder"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/hub/tutorials/cross_lingual_similarity_with_tf_hub_multilingual_universal_encoder.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/hub/tutorials/cross_lingual_similarity_with_tf_hub_multilingual_universal_encoder.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View on GitHub</a>
  </td>
  <td>
    <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/hub/tutorials/cross_lingual_similarity_with_tf_hub_multilingual_universal_encoder.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
  </td>
  <td>
    <a href="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"><img src="https://www.tensorflow.org/images/hub_logo_32px.png" />See TF Hub model</a>
  </td>
</table>

This notebook illustrates how to access the Multilingual Universal Sentence Encoder module and use it for sentence similarity across multiple languages. This module is an extension of the [original Universal Encoder module](https://tfhub.dev/google/universal-sentence-encoder/2).

The notebook is divided as follows:

*   The first section shows a visualization of sentences between pair of languages. This is a more academic exercise. 
*   In the second section, we show how to build a semantic search engine from a sample of a Wikipedia corpus in multiple languages.

## Citation

*Research papers that make use of the models explored in this colab should cite:*

### [Multilingual universal sentence encoder for semantic retrieval](https://arxiv.org/abs/1907.04307)
Yinfei Yang, Daniel Cer, Amin Ahmad, Mandy Guo, Jax Law, Noah Constant, Gustavo Hernandez Abrego, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, and Ray Kurzweil. 2019.
 arXiv preprint arXiv:1907.04307

## Setup

This section sets up the environment for access to the Multilingual Universal Sentence Encoder Module and also prepares a set of English sentences and their translations. In the following sections, the multilingual module will be used to compute similarity *across languages*.


```
%%capture
#@title Setup Environment
# Install the latest Tensorflow version.
!pip install "tensorflow-text==2.11.*"
!pip install bokeh
!pip install simpleneighbors[annoy]
!pip install tqdm
```


```
#@title Setup common imports and functions
import bokeh
import bokeh.models
import bokeh.plotting
import numpy as np
import os
import pandas as pd
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
import sklearn.metrics.pairwise

from simpleneighbors import SimpleNeighbors
from tqdm import tqdm
from tqdm import trange

def visualize_similarity(embeddings_1, embeddings_2, labels_1, labels_2,
                         plot_title,
                         plot_width=1200, plot_height=600,
                         xaxis_font_size='12pt', yaxis_font_size='12pt'):

  assert len(embeddings_1) == len(labels_1)
  assert len(embeddings_2) == len(labels_2)

  # arccos based text similarity (Yang et al. 2019; Cer et al. 2019)
  sim = 1 - np.arccos(
      sklearn.metrics.pairwise.cosine_similarity(embeddings_1,
                                                 embeddings_2))/np.pi

  embeddings_1_col, embeddings_2_col, sim_col = [], [], []
  for i in range(len(embeddings_1)):
    for j in range(len(embeddings_2)):
      embeddings_1_col.append(labels_1[i])
      embeddings_2_col.append(labels_2[j])
      sim_col.append(sim[i][j])
  df = pd.DataFrame(zip(embeddings_1_col, embeddings_2_col, sim_col),
                    columns=['embeddings_1', 'embeddings_2', 'sim'])

  mapper = bokeh.models.LinearColorMapper(
      palette=[*reversed(bokeh.palettes.YlOrRd[9])], low=df.sim.min(),
      high=df.sim.max())

  p = bokeh.plotting.figure(title=plot_title, x_range=labels_1,
                            x_axis_location="above",
                            y_range=[*reversed(labels_2)],
                            plot_width=plot_width, plot_height=plot_height,
                            tools="save",toolbar_location='below', tooltips=[
                                ('pair', '@embeddings_1 ||| @embeddings_2'),
                                ('sim', '@sim')])
  p.rect(x="embeddings_1", y="embeddings_2", width=1, height=1, source=df,
         fill_color={'field': 'sim', 'transform': mapper}, line_color=None)

  p.title.text_font_size = '12pt'
  p.axis.axis_line_color = None
  p.axis.major_tick_line_color = None
  p.axis.major_label_standoff = 16
  p.xaxis.major_label_text_font_size = xaxis_font_size
  p.xaxis.major_label_orientation = 0.25 * np.pi
  p.yaxis.major_label_text_font_size = yaxis_font_size
  p.min_border_right = 300

  bokeh.io.output_notebook()
  bokeh.io.show(p)

```

This is additional boilerplate code where we import the pre-trained ML model we will use to encode text throughout this notebook.


```
# The 16-language multilingual module is the default but feel free
# to pick others from the list and compare the results.
module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3' #@param ['https://tfhub.dev/google/universal-sentence-encoder-multilingual/3', 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3']

model = hub.load(module_url)

def embed_text(input):
  return model(input)
```

# Visualize Text Similarity Between Languages
With the sentence embeddings now in hand, we can visualize semantic similarity across different languages.

## Computing Text Embeddings

We first define a set of sentences translated to various languages in parallel. Then, we precompute the embeddings for all of our sentences.


```
# Some texts of different lengths in different languages.
arabic_sentences = ['كلب', 'الجراء لطيفة.', 'أستمتع بالمشي لمسافات طويلة على طول الشاطئ مع كلبي.']
chinese_sentences = ['狗', '小狗很好。', '我喜欢和我的狗一起沿着海滩散步。']
english_sentences = ['dog', 'Puppies are nice.', 'I enjoy taking long walks along the beach with my dog.']
french_sentences = ['chien', 'Les chiots sont gentils.', 'J\'aime faire de longues promenades sur la plage avec mon chien.']
german_sentences = ['Hund', 'Welpen sind nett.', 'Ich genieße lange Spaziergänge am Strand entlang mit meinem Hund.']
italian_sentences = ['cane', 'I cuccioli sono carini.', 'Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane.']
japanese_sentences = ['犬', '子犬はいいです', '私は犬と一緒にビーチを散歩するのが好きです']
korean_sentences = ['개', '강아지가 좋다.', '나는 나의 개와 해변을 따라 길게 산책하는 것을 즐긴다.']
russian_sentences = ['собака', 'Милые щенки.', 'Мне нравится подолгу гулять по пляжу со своей собакой.']
spanish_sentences = ['perro', 'Los cachorros son agradables.', 'Disfruto de dar largos paseos por la playa con mi perro.']

# Multilingual example
multilingual_example = ["Willkommen zu einfachen, aber", "verrassend krachtige", "multilingüe", "compréhension du langage naturel", "модели.", "大家是什么意思" , "보다 중요한", ".اللغة التي يتحدثونها"]
multilingual_example_in_en =  ["Welcome to simple yet", "surprisingly powerful", "multilingual", "natural language understanding", "models.", "What people mean", "matters more than", "the language they speak."]

```


```
# Compute embeddings.
ar_result = embed_text(arabic_sentences)
en_result = embed_text(english_sentences)
es_result = embed_text(spanish_sentences)
de_result = embed_text(german_sentences)
fr_result = embed_text(french_sentences)
it_result = embed_text(italian_sentences)
ja_result = embed_text(japanese_sentences)
ko_result = embed_text(korean_sentences)
ru_result = embed_text(russian_sentences)
zh_result = embed_text(chinese_sentences)

multilingual_result = embed_text(multilingual_example)
multilingual_in_en_result = embed_text(multilingual_example_in_en)
```

## Visualizing Similarity

With text embeddings in hand, we can take their dot-product to visualize how similar sentences are between languages. A darker color indicates the embeddings are semantically similar.

### Multilingual Similarity


```
visualize_similarity(multilingual_in_en_result, multilingual_result,
                     multilingual_example_in_en, multilingual_example,  "Multilingual Universal Sentence Encoder for Semantic Retrieval (Yang et al., 2019)")

```










<div class="bk-root" id="c23bc7cf-8f04-4fa0-a38c-20c29e5098b9" data-root-id="1002"></div>





### English-Arabic Similarity


```
visualize_similarity(en_result, ar_result, english_sentences, arabic_sentences, 'English-Arabic Similarity')
```










<div class="bk-root" id="d20b027b-6bf0-444d-8763-df8d299e1642" data-root-id="1082"></div>





### Engish-Russian Similarity


```
visualize_similarity(en_result, ru_result, english_sentences, russian_sentences, 'English-Russian Similarity')
```










<div class="bk-root" id="02b6f17f-c980-4d2c-b9d7-58e2d001b1bf" data-root-id="1169"></div>





### English-Spanish Similarity


```
visualize_similarity(en_result, es_result, english_sentences, spanish_sentences, 'English-Spanish Similarity')
```










<div class="bk-root" id="81e993c9-fc6b-4169-8c6b-a0101097b959" data-root-id="1263"></div>





### English-Italian Similarity


```
visualize_similarity(en_result, it_result, english_sentences, italian_sentences, 'English-Italian Similarity')
```










<div class="bk-root" id="5e20475c-62a7-4a19-87ed-a605dc444c96" data-root-id="1364"></div>





### Italian-Spanish Similarity


```
visualize_similarity(it_result, es_result, italian_sentences, spanish_sentences, 'Italian-Spanish Similarity')
```










<div class="bk-root" id="6f559e42-2dec-4a29-a3d9-62c969a8c08a" data-root-id="1472"></div>





### English-Chinese Similarity


```
visualize_similarity(en_result, zh_result, english_sentences, chinese_sentences, 'English-Chinese Similarity')
```










<div class="bk-root" id="5b8e7d08-b7e7-4a05-a22d-c27aa1873e6d" data-root-id="1587"></div>





### English-Korean Similarity


```
visualize_similarity(en_result, ko_result, english_sentences, korean_sentences, 'English-Korean Similarity')
```










<div class="bk-root" id="7b449243-0dbd-46b6-8b02-a89fdf92645e" data-root-id="1709"></div>





### Chinese-Korean Similarity


```
visualize_similarity(zh_result, ko_result, chinese_sentences, korean_sentences, 'Chinese-Korean Similarity')
```










<div class="bk-root" id="63952aa4-d54a-4445-ad10-ef5bef98f1ef" data-root-id="1838"></div>





### And more...

The above examples can be extended to any language pair from **English, Arabic, Chinese, Dutch, French, German, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Thai and Turkish**. Happy coding!

# Creating a Multilingual Semantic-Similarity Search Engine

Whereas in the previous example we visualized a handful of sentences, in this section we will build a semantic-search index of about 200,000 sentences from a Wikipedia Corpus. About half will be in English and the other half in Spanish to demonstrate the multilingual capabilities of the Universal Sentence Encoder.

## Download Data to Index
First, we will download news sentences in multiples languages from the [News Commentary Corpus](http://opus.nlpl.eu/News-Commentary-v11.php) [1].  Without loss of generality, this approach should also work for indexing the rest of the supported languages.

To speed up the demo, we limit to 1000 sentences per language.


```
corpus_metadata = [
    ('ar', 'ar-en.txt.zip', 'News-Commentary.ar-en.ar', 'Arabic'),
    ('zh', 'en-zh.txt.zip', 'News-Commentary.en-zh.zh', 'Chinese'),
    ('en', 'en-es.txt.zip', 'News-Commentary.en-es.en', 'English'),
    ('ru', 'en-ru.txt.zip', 'News-Commentary.en-ru.ru', 'Russian'),
    ('es', 'en-es.txt.zip', 'News-Commentary.en-es.es', 'Spanish'),
]

language_to_sentences = {}
language_to_news_path = {}
for language_code, zip_file, news_file, language_name in corpus_metadata:
  zip_path = tf.keras.utils.get_file(
      fname=zip_file,
      origin='http://opus.nlpl.eu/download.php?f=News-Commentary/v11/moses/' + zip_file,
      extract=True)
  news_path = os.path.join(os.path.dirname(zip_path), news_file)
  language_to_sentences[language_code] = pd.read_csv(news_path, sep='\t', header=None)[0][:1000]
  language_to_news_path[language_code] = news_path

  print('{:,} {} sentences'.format(len(language_to_sentences[language_code]), language_name))
```

    Downloading data from http://opus.nlpl.eu/download.php?f=News-Commentary/v11/moses/ar-en.txt.zip
    24715264/24714354 [==============================] - 2s 0us/step
    1,000 Arabic sentences
    Downloading data from http://opus.nlpl.eu/download.php?f=News-Commentary/v11/moses/en-zh.txt.zip
    18104320/18101984 [==============================] - 2s 0us/step
    1,000 Chinese sentences
    Downloading data from http://opus.nlpl.eu/download.php?f=News-Commentary/v11/moses/en-es.txt.zip
    28106752/28106064 [==============================] - 2s 0us/step
    1,000 English sentences
    Downloading data from http://opus.nlpl.eu/download.php?f=News-Commentary/v11/moses/en-ru.txt.zip
    24854528/24849511 [==============================] - 2s 0us/step
    1,000 Russian sentences
    1,000 Spanish sentences
    

## Using a pre-trained model to transform sentences into vectors

We compute embeddings in _batches_ so that they fit in the GPU's RAM.


```
# Takes about 3 minutes

batch_size = 2048
language_to_embeddings = {}
for language_code, zip_file, news_file, language_name in corpus_metadata:
  print('\nComputing {} embeddings'.format(language_name))
  with tqdm(total=len(language_to_sentences[language_code])) as pbar:
    for batch in pd.read_csv(language_to_news_path[language_code], sep='\t',header=None, chunksize=batch_size):
      language_to_embeddings.setdefault(language_code, []).extend(embed_text(batch[0]))
      pbar.update(len(batch))
```

      0%|          | 0/1000 [00:00<?, ?it/s]

    
    Computing Arabic embeddings
    

    83178it [00:30, 2768.60it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]

    
    Computing Chinese embeddings
    

    69206it [00:18, 3664.60it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]

    
    Computing English embeddings
    

    238853it [00:37, 6319.00it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]

    
    Computing Russian embeddings
    

    190092it [00:34, 5589.16it/s]
      0%|          | 0/1000 [00:00<?, ?it/s]

    
    Computing Spanish embeddings
    

    238819it [00:41, 5754.02it/s]
    

## Building an index of semantic vectors

We use the [SimpleNeighbors](https://pypi.org/project/simpleneighbors/) library---which is a wrapper for the [Annoy](https://github.com/spotify/annoy) library---to efficiently look up results from the corpus.


```
%%time

# Takes about 8 minutes

num_index_trees = 40
language_name_to_index = {}
embedding_dimensions = len(list(language_to_embeddings.values())[0][0])
for language_code, zip_file, news_file, language_name in corpus_metadata:
  print('\nAdding {} embeddings to index'.format(language_name))
  index = SimpleNeighbors(embedding_dimensions, metric='dot')

  for i in trange(len(language_to_sentences[language_code])):
    index.add_one(language_to_sentences[language_code][i], language_to_embeddings[language_code][i])

  print('Building {} index with {} trees...'.format(language_name, num_index_trees))
  index.build(n=num_index_trees)
  language_name_to_index[language_name] = index
```

      0%|          | 1/1000 [00:00<02:21,  7.04it/s]

    
    Adding Arabic embeddings to index
    

    100%|██████████| 1000/1000 [02:06<00:00,  7.90it/s]
      0%|          | 1/1000 [00:00<01:53,  8.84it/s]

    Building Arabic index with 40 trees...
    
    Adding Chinese embeddings to index
    

    100%|██████████| 1000/1000 [02:05<00:00,  7.99it/s]
      0%|          | 1/1000 [00:00<01:59,  8.39it/s]

    Building Chinese index with 40 trees...
    
    Adding English embeddings to index
    

    100%|██████████| 1000/1000 [02:07<00:00,  7.86it/s]
      0%|          | 1/1000 [00:00<02:17,  7.26it/s]

    Building English index with 40 trees...
    
    Adding Russian embeddings to index
    

    100%|██████████| 1000/1000 [02:06<00:00,  7.91it/s]
      0%|          | 1/1000 [00:00<02:03,  8.06it/s]

    Building Russian index with 40 trees...
    
    Adding Spanish embeddings to index
    

    100%|██████████| 1000/1000 [02:07<00:00,  7.84it/s]

    Building Spanish index with 40 trees...
    CPU times: user 11min 21s, sys: 2min 14s, total: 13min 35s
    Wall time: 10min 33s
    

    
    


```
%%time

# Takes about 13 minutes

num_index_trees = 60
print('Computing mixed-language index')
combined_index = SimpleNeighbors(embedding_dimensions, metric='dot')
for language_code, zip_file, news_file, language_name in corpus_metadata:
  print('Adding {} embeddings to mixed-language index'.format(language_name))
  for i in trange(len(language_to_sentences[language_code])):
    annotated_sentence = '({}) {}'.format(language_name, language_to_sentences[language_code][i])
    combined_index.add_one(annotated_sentence, language_to_embeddings[language_code][i])

print('Building mixed-language index with {} trees...'.format(num_index_trees))
combined_index.build(n=num_index_trees)
```

      0%|          | 1/1000 [00:00<02:00,  8.29it/s]

    Computing mixed-language index
    Adding Arabic embeddings to mixed-language index
    

    100%|██████████| 1000/1000 [02:06<00:00,  7.92it/s]
      0%|          | 1/1000 [00:00<02:24,  6.89it/s]

    Adding Chinese embeddings to mixed-language index
    

    100%|██████████| 1000/1000 [02:05<00:00,  7.95it/s]
      0%|          | 1/1000 [00:00<02:05,  7.98it/s]

    Adding English embeddings to mixed-language index
    

    100%|██████████| 1000/1000 [02:06<00:00,  7.88it/s]
      0%|          | 1/1000 [00:00<02:18,  7.20it/s]

    Adding Russian embeddings to mixed-language index
    

    100%|██████████| 1000/1000 [02:04<00:00,  8.03it/s]
      0%|          | 1/1000 [00:00<02:17,  7.28it/s]

    Adding Spanish embeddings to mixed-language index
    

    100%|██████████| 1000/1000 [02:06<00:00,  7.90it/s]
    

    Building mixed-language index with 60 trees...
    CPU times: user 11min 18s, sys: 2min 13s, total: 13min 32s
    Wall time: 10min 30s
    

## Verify that the semantic-similarity search engine works

In this section we will demonstrate:

1.   Semantic-search capabilities: retrieving sentences from the corpus that are semantically similar to the given query.
2.   Multilingual capabilities: doing so in multiple languages when they query language and index language match
3.   Cross-lingual capabilities: issuing queries in a distinct language than the indexed corpus
4.   Mixed-language corpus: all of the above on a single index containing entries from all languages


### Semantic-search crosss-lingual capabilities

In this section we show how to retrieve sentences related to a set of sample English sentences. Things to try:

*   Try a few different sample sentences
*   Try changing the number of returned results (they are returned in order of similarity)
*   Try cross-lingual capabilities by returning results in different languages (might want to use [Google Translate](http://translate.google.com) on some results to your native language for sanity check)



```
sample_query = 'The stock market fell four points.'  #@param ["Global warming", "Researchers made a surprising new discovery last week.", "The stock market fell four points.", "Lawmakers will vote on the proposal tomorrow."] {allow-input: true}
index_language = 'English'  #@param ["Arabic", "Chinese", "English", "French", "German", "Russian", "Spanish"]
num_results = 10  #@param {type:"slider", min:0, max:100, step:10}

query_embedding = embed_text(sample_query)[0]
search_results = language_name_to_index[index_language].nearest(query_embedding, n=num_results)

print('{} sentences similar to: "{}"\n'.format(index_language, sample_query))
search_results
```

    English sentences similar to: "The stock market fell four points."
    
    




    ['Nobel laureate Amartya Sen attributed the European crisis to four failures – political, economic, social, and intellectual.',
     'Just last December, fellow economists Martin Feldstein and Nouriel Roubini each penned op-eds bravely questioning bullish market sentiment, sensibly pointing out gold’s risks.',
     'His ratings have dipped below 50% for the first time.',
     'As a result, markets were deregulated, making it easier to trade assets that were perceived to be safe, but were in fact not.',
     'Consider the advanced economies.',
     'But the agreement has three major flaws.',
     'This “predetermined equilibrium” thinking – reflected in the view that markets always self-correct – led to policy paralysis until the Great Depression, when John Maynard Keynes’s argument for government intervention to address unemployment and output gaps gained traction.',
     'Officials underestimated tail risks.',
     'Consider a couple of notorious examples.',
     'Stalin was content to settle for an empire in Eastern Europe.']



### Mixed-corpus capabilities

We will now issue a query in English, but the results will come from the any of the indexed languages.


```
sample_query = 'The stock market fell four points.'  #@param ["Global warming", "Researchers made a surprising new discovery last week.", "The stock market fell four points.", "Lawmakers will vote on the proposal tomorrow."] {allow-input: true}
num_results = 40  #@param {type:"slider", min:0, max:100, step:10}

query_embedding = embed_text(sample_query)[0]
search_results = language_name_to_index[index_language].nearest(query_embedding, n=num_results)

print('{} sentences similar to: "{}"\n'.format(index_language, sample_query))
search_results
```

    English sentences similar to: "The stock market fell four points."
    
    




    ['Nobel laureate Amartya Sen attributed the European crisis to four failures – political, economic, social, and intellectual.',
     'It was part of the 1945 consensus.',
     'The end of the East-West ideological divide and the end of absolute faith in markets are historical turning points.',
     'Just last December, fellow economists Martin Feldstein and Nouriel Roubini each penned op-eds bravely questioning bullish market sentiment, sensibly pointing out gold’s risks.',
     'His ratings have dipped below 50% for the first time.',
     'As a result, markets were deregulated, making it easier to trade assets that were perceived to be safe, but were in fact not.',
     'Consider the advanced economies.',
     'Since their articles appeared, the price of gold has moved up still further.',
     'But the agreement has three major flaws.',
     'Gold prices even hit a record-high $1,300 recently.',
     'This “predetermined equilibrium” thinking – reflected in the view that markets always self-correct – led to policy paralysis until the Great Depression, when John Maynard Keynes’s argument for government intervention to address unemployment and output gaps gained traction.',
     'What Failed in 2008?',
     'Officials underestimated tail risks.',
     'Consider a couple of notorious examples.',
     'One of these species, orange roughy, has been caught commercially for only around a quarter-century, but already is being fished to the point of collapse.',
     'Meanwhile, policymakers were lulled into complacency by the widespread acceptance of economic theories such as the “efficient-market hypothesis,” which assumes that investors act rationally and use all available information when making their decisions.',
     'Stalin was content to settle for an empire in Eastern Europe.',
     'Intelligence assets have been redirected.',
     'A new wave of what the economist Joseph Schumpeter famously called “creative destruction” is under way: even as central banks struggle to maintain stability by flooding markets with liquidity, credit to business and households is shrinking.',
     'It all came about in a number of ways.',
     'The UN, like the dream of European unity, was also part of the 1945 consensus.',
     'The End of 1945',
     'The Global Economy’s New Path',
     'But this scenario failed to materialize.',
     'Gold prices are extremely sensitive to global interest-rate movements.',
     'Fukushima has presented the world with a far-reaching, fundamental choice.',
     'It was Japan, the high-tech country par excellence (not the latter-day Soviet Union) that proved unable to take adequate precautions to avert disaster in four reactor blocks.',
     'Some European academics tried to argue that there was no need for US-like fiscal transfers, because any desired degree of risk sharing can, in theory, be achieved through financial markets.',
     '$10,000 Gold?',
     'One answer, of course, is a complete collapse of the US dollar.',
     '1929 or 1989?',
     'The goods we made were what economists call “rival" and “excludible" commodities.',
     'This dream quickly faded when the Cold War divided the world into two hostile blocs. But in some ways the 1945 consensus, in the West, was strengthened by Cold War politics.',
     'The first flaw is that the spending reductions are badly timed: coming as they do when the US economy is weak, they risk triggering another recession.',
     'One successful gold investor recently explained to me that stock prices languished for a more than a decade before the Dow Jones index crossed the 1,000 mark in the early 1980’s.',
     'Eichengreen traces our tepid response to the crisis to the triumph of monetarist economists, the disciples of Milton Friedman, over their Keynesian and Minskyite peers – at least when it comes to interpretations of the causes and consequences of the Great Depression.',
     "However, America's unilateral options are limited.",
     'Once it was dark, a screen was set up and Mark showed home videos from space.',
     'These aspirations were often voiced in the United Nations, founded in 1945.',
     'Then I got distracted for about 40 years.']



Try your own queries:


```
query = 'The stock market fell four points.'  #@param {type:"string"}
num_results = 30  #@param {type:"slider", min:0, max:100, step:10}

query_embedding = embed_text(sample_query)[0]
search_results = combined_index.nearest(query_embedding, n=num_results)

print('{} sentences similar to: "{}"\n'.format(index_language, query))
search_results
```

    English sentences similar to: "The stock market fell four points."
    
    




    ['(Chinese) 新兴市场的号角',
     '(English) It was part of the 1945 consensus.',
     '(Russian) Брюссель. Цунами, пронёсшееся по финансовым рынкам, является глобальной катастрофой.',
     '(Arabic) هناك أربعة شروط مسبقة لتحقيق النجاح الأوروبي في أفغانستان:',
     '(Spanish) Su índice de popularidad ha caído por primera vez por debajo del 50 por ciento.',
     '(English) His ratings have dipped below 50% for the first time.',
     '(Russian) Впервые его рейтинг опустился ниже 50%.',
     '(English) As a result, markets were deregulated, making it easier to trade assets that were perceived to be safe, but were in fact not.',
     '(Arabic) وكانت التطورات التي شهدتها سوق العمل أكثر تشجيعا، فهي على النقيض من أسواق الأصول تعكس النتائج وليس التوقعات. وهنا أيضاً كانت الأخبار طيبة. فقد أصبحت سوق العمل أكثر إحكاما، حيث ظلت البطالة عند مستوى 3.5% وكانت نسبة الوظائف إلى الطلبات المقدمة فوق مستوى التعادل.',
     '(Russian) Это было частью консенсуса 1945 года.',
     '(English) Consider the advanced economies.',
     '(English) Since their articles appeared, the price of gold has moved up still further.',
     '(Russian) Тогда они не только смогут накормить свои семьи, но и начать получать рыночную прибыль и откладывать деньги на будущее.',
     '(English) Gold prices even hit a record-high $1,300 recently.',
     '(Chinese) 另一种金融危机',
     '(Russian) Европейская мечта находится в кризисе.',
     '(English) What Failed in 2008?',
     '(Spanish) Pero el acuerdo alcanzado tiene tres grandes defectos.',
     '(English) Officials underestimated tail risks.',
     '(English) Consider a couple of notorious examples.',
     '(Spanish) Los mercados financieros pueden ser frágiles y ofrecen muy poca capacidad de compartir los riesgos relacionados con el ingreso de los trabajadores, que constituye la mayor parte de la renta de cualquier economía avanzada.',
     '(Chinese) 2008年败在何处？',
     '(Spanish) Consideremos las economías avanzadas.',
     '(Spanish) Los bienes producidos se caracterizaron por ser, como señalaron algunos economistas, mercancías “rivales” y “excluyentes”.',
     '(Arabic) إغلاق الفجوة الاستراتيجية في أوروبا',
     '(English) Stalin was content to settle for an empire in Eastern Europe.',
     '(English) Intelligence assets have been redirected.',
     '(Spanish) Hoy, envalentonados por la apreciación continua, algunos están sugiriendo que el oro podría llegar incluso a superar esa cifra.',
     '(Russian) Цены на золото чрезвычайно чувствительны к мировым движениям процентных ставок.',
     '(Russian) Однако у достигнутой договоренности есть три основных недостатка.']



# Further topics

## Multilingual

Finally, we encourage you to try queries in any of the supported languages: **English, Arabic, Chinese, Dutch, French, German, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Thai and Turkish**.

Also, even though we only indexed in a subset of the languages, you can also index content in any of the supported languages.


## Model variations

We offer variations of the Universal Encoder models optimized for various things like memory, latency and/or quality. Please feel free to experiment with them to find a suitable one.

## Nearest neighbor libraries

We used Annoy to efficiently look up nearest neighbors. See the [tradeoffs section](https://github.com/spotify/annoy/blob/master/README.rst#tradeoffs) to read about the number of trees (memory-dependent) and number of items to search (latency-dependent)---SimpleNeighbors only allows to control the number of trees, but refactoring the code to use Annoy directly should be simple, we just wanted to keep this code as simple as possible for the general user.

If Annoy does not scale for your application, please also check out [FAISS](https://github.com/facebookresearch/faiss).

*All the best building your multilingual semantic applications!*

[1] J. Tiedemann, 2012, [Parallel Data, Tools and Interfaces in OPUS](http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf). In Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC 2012)
