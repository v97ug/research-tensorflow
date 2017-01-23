# coding=utf-8
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import types

# Step 1: Read the data.
words = open('input-file/ALL-HADOOP-API-ascii.txt').read().split()

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 20000

all_method_words = open('all-methods-really.txt', 'r').read().split("\n")  # 改行でsplitするよ


def build_dataset(param_words):
    pair_word_and_word_freq = [['UNK', -1]]
    pair_word_and_word_freq.extend(collections.Counter(param_words).most_common(vocabulary_size - 1))
    word_freq_rank_dict = dict()  # 単語をキー、その頻出順を値とする。（第一要素は、['UNK', -1]）
    for word, _ in pair_word_and_word_freq:
        word_freq_rank_dict[word] = len(word_freq_rank_dict)
    rank_list = list()  # それぞれの単語が何番目に多く出現しているかのリスト
    unk_count = 0
    for word in param_words:
        if word in word_freq_rank_dict:
            rank = word_freq_rank_dict[word]
        else:
            rank = 0  # dictionary['UNK']
            unk_count += 1
        rank_list.append(rank)
    pair_word_and_word_freq[0][1] = unk_count
    # print('Most common words (+UNK)', pair_word_and_word_freq[:5])

    reverse_dictionary = dict(zip(word_freq_rank_dict.values(), word_freq_rank_dict.keys()))
    return rank_list, reverse_dictionary


# 以下の２つの変数の必要性は？
word_ranks, freq_rank_word_dict = build_dataset(words)
del words  # Hint to reduce memory.

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        target_rank = word_ranks[data_index]
        buffer.append(target_rank)
        data_index = (data_index + 1) % len(word_ranks)
    for i in range(batch_size // num_skips):
        target_index = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target_index in targets_to_avoid:
                target_index = random.randint(0, span - 1)
            targets_to_avoid.append(target_index)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target_index]
        buffer.append(word_ranks[data_index])
        data_index = (data_index + 1) % len(word_ranks)
    return batch, labels


# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 20  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
# 0からvalid_window-1の範囲で、valid_sizeの、値がそれぞれ違う一次元配列を生成
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()

# Step 5: Begin training.
# num_steps = 100001
num_steps = 10001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

    final_embeddings = normalized_embeddings.eval()


def cal_similarity(mat_1m, mat_mn):
    # 正規化する
    mat_1m_norm = np.sqrt(np.sum(np.square(mat_1m)))
    mat_1m_normalized = resource_method_mat / mat_1m_norm

    mat_mn_norm = np.sqrt(np.sum(np.square(mat_mn), axis=1)).reshape(vocabulary_size, 1)
    mat_nm_normalized = final_embeddings / mat_mn_norm

    return np.matmul(mat_1m_normalized,
                     np.transpose(mat_nm_normalized))[0, :]  # その単語と、その他全ての単語とのsimilarityを求めた一次元配列


count = 0
for i in xrange(vocabulary_size):
    if freq_rank_word_dict[i] in all_method_words:
        count += 1

        resource_method_mat = final_embeddings[i, :].reshape(1, embedding_size)  # 縦ベクトルになってるので、reshapeで行列にする
        sim_array = cal_similarity(resource_method_mat, final_embeddings)

        # sim_arrayを逆順ソートし、上位３つのindexを取ってくる。
        # なお、indexとは、ソートする前にどこのindexにあったかである.
        large_sim_indices = np.argsort(sim_array)[::-1]
        most_sim_values = np.sort(sim_array)[::-1]

        resource_name = freq_rank_word_dict[i]
        log_str = "Nearest to %s:" % resource_name
        for k in xrange(100):
            close_word_index = large_sim_indices[k]
            close_word = freq_rank_word_dict[close_word_index]
            if close_word in all_method_words and close_word != resource_name:
                log_str = "%s %s %s," % (log_str, close_word, most_sim_values[k])

        print(log_str)

print(count)


# Step 6: Visualize the embeddings.


# def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
#     assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
#     plt.figure(figsize=(18, 18))  # in inches
#     for i, label in enumerate(labels):
#         x, y = low_dim_embs[i, :]
#         plt.scatter(x, y)  # plotする
#         plt.annotate(label,  # 文字列をつけたり、位置を調整したり
#                      xy=(x, y),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
#
#     plt.savefig(filename)
#
#
# try:
#     from sklearn.manifold import TSNE
#     import matplotlib.pyplot as plt
#
#     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#     plot_only = 500
#     low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])  # [[x,y]]
#     labels = [freq_rank_word_dict[i] for i in xrange(plot_only)]
#     plot_with_labels(low_dim_embs, labels)
#
# except ImportError:
#     print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
