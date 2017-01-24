# coding=utf-8

###
# cal_method_sim.py is a module.
##

import numpy as np


def cal_similarity(mat_1m, mat_mn):
    # 正規化する
    mat_1m_norm = np.sqrt(np.sum(np.square(mat_1m)))
    mat_1m_normalized = mat_1m / mat_1m_norm

    mat_mn_norm = np.sqrt(np.sum(np.square(mat_mn), axis=1)).reshape(mat_mn.shape[0], 1)
    mat_nm_normalized = mat_mn / mat_mn_norm

    return np.matmul(mat_1m_normalized,
                     np.transpose(mat_nm_normalized))[0, :]  # その単語と、その他全ての単語とのsimilarityを求めた一次元配列


def find_near_words(resource_name, large_sim_indices, freq_rank_word_dict, num_slice_words):
    near_word_list = list()
    close_word_indices = large_sim_indices[1:num_slice_words + 1]  # 0番目は自分自身が入っている
    for close_word_index in close_word_indices:
        close_w = freq_rank_word_dict[close_word_index]
        if close_w != resource_name:
            near_word_list.append(close_w)
    return near_word_list


# 返り値は、(word, sim_value)のリスト
def find_near_words_sim(resource_name, large_sim_indices, most_sim_values, freq_rank_word_dict, num_slice_words):
    near_word_list = list()
    close_word_indices = large_sim_indices[1:num_slice_words + 1]  # 0番目は自分自身が入っている
    slice_most_sim = most_sim_values[1:num_slice_words + 1]

    for close_word_index, sim_value in zip(close_word_indices, slice_most_sim):
        close_w = freq_rank_word_dict[close_word_index]
        if close_w != resource_name:
            near_word_list.append( (close_w, sim_value) )
    return near_word_list