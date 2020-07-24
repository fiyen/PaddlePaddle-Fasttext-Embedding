"""
对论文Piotr Bojanowski and Edouard Grave and Armand Joulin and Tomas Mikolov， Enriching Word Vectors with Subword Information
Piotr, 2017的实现
"""

import numpy as np
from collections import Counter
from itertools import chain
import random


class FastText_py:
    """
    min_count: only the words with frequency higher than min_count will be considered to be trained.
    embedding_size: dimensionality of word embedding
    window_size: the words number that will be consider around a given word. For the start and end of the text, the words
    number is not equal enough to windows, then the lacked words will be neglected.
    init_scale: initialization scope for word embedding
    neg_sampling_num: A negative sampling set of all words will be generated. neg_sampling_num
    define the number of negative samples of every word.
    skip_ratio: for some words with high frequency, the redundant training of them will not get better embeddings while
    cost extra time. These words that exceed skip probability will not be trained every meeting. Instead, they will be skipped
    in training with probability 1 - (skip_ratio / f(w)) ^ 1/2, where f(w) = Count(w) / Count(all words)
    """
    def __init__(self, embedding_size,
                 min_count=5,
                 window_size=5,
                 learning_rate=0.001,
                 init_scale=2.5,
                 ngrams=(3,4,5,6),
                 skip_ratio=1.0,
                 neg_sampling_num=4,
                 ):
        self.min_count = min_count
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.init_scale = init_scale
        self.ngrams = ngrams
        self.skip_ratio = skip_ratio
        self.neg_sampling_num = neg_sampling_num
        self.act = {x: 1.0 / (1 + np.exp(- x / 1000)) for x in range(-8000, 8001, 1)}

    def activation(self, x):
        try:
            if x <= -8:
                return 0
            elif x < 8:
                index = int(np.ceil(x / 0.001))
                return self.act[index]
            else:
                return 1
        except:
            print(x)
            raise ValueError('error')

    def fit(self, data):
        """
        fit the data, get the vocabulary and vocabulary size, the code the data
        :param data: text string or splitted words. Note that the data should be a (batch, length)
        form, where the length dim corresponds the text string or splitted words.
        :return: the fitted data
        """
        try:
            data = [line.split(' ') for line in data]
            print("The form of input text is [text1, text2, ...].")
        except AttributeError:
            print("The form of input text is [[word11, word12, ...], [word21, word22, ...]].")
        data = [[word.lower() for word in line] for line in data]
        words_counter = [(word, count) for word, count in Counter(chain(*data)).most_common() if count >= self.min_count]
        self.vocab_size = len(words_counter)
        # generate 3 different dicts to store:
        # word to id
        # word to id's frequency
        # id to word
        self.word2id_dict = {word: id for id, (word, _) in enumerate(words_counter)}
        self.id2word_dict = {id: word for id, (word, _) in enumerate(words_counter)}
        self.word2id_freq = {id: freq for id, (_, freq) in enumerate(words_counter)}
        # n-grams internal structural of words
        self.word2sub_dict = {id: self.gen_ngrams(self.id2word_dict[id]) for id in self.id2word_dict.keys()}
        subwords_counter = [(sub, count) for sub, count in Counter(chain(*self.word2sub_dict.values())).most_common()]
        self.subvocab_size = len(subwords_counter)
        # generate 2 different dicts to store for sub words
        self.subword2id_dict = {subword: id for id, (subword, _) in enumerate(subwords_counter)}
        self.id2subword_dict = {id: subword for id, (subword, _) in enumerate(subwords_counter)}
        self.word2sub_dict = {id: [self.subword2id_dict[sub] for sub in self.word2sub_dict[id]] for id in self.word2sub_dict.keys()}

        # transfer data to ids
        data = [[self.word2id_dict[word] for word in line if word in self.word2id_dict.keys()] for line in data]
        # subsampling, reduce the size of the data to boost the training process
        data = self.subsampling(data)
        self._init_embedding()
        return data

    def subsampling(self, data):
        """
        discard some words according to their frequency. The higher the more likely to be discarded.
        :param data:
        :return:
        """
        total_num = np.sum([freq for freq in self.word2id_freq.values()])
        skip_prob = {id: 1 - (self.skip_ratio / freq * total_num) ** 0.5 for (id, freq) in self.word2id_freq.items()}
        return [[word for word in line if np.random.rand() > skip_prob[word]] for line in data]

    def gen_ngrams(self, word):
        word = '<' + word + '>'
        len_word = len(word)
        sub_words = []
        for n in self.ngrams:
            if len_word >= n:
                end_index = len_word - n + 1
                for i in range(end_index):
                    sub_words.append(word[i:i+n])
        return sub_words

    def build_data(self, data, batch_size=1000, test=False):
        """
        to define a generator to reduce the memory cost
        the sample form is [[center_word, [target_word1, neg_word11, neg_word12, ...], ..., [target_wordn, neg_wordn1,
        neg_wordn2, ...], ...]
        :param data:
        :param batch_size:
        :param test: whether to test the length of the data
        :return:
        """
        batch_pin = 0
        batch = []
        # from left to right, enumerate every centre word
        for line in data:
            for idx, center_word in enumerate(line):
                if not test:
                    sample = [center_word]
                    # let window_size be the up bound, random sample window
                    window = np.random.randint(1, self.window_size)
                    # positive_word_candidates
                    pos_word_range = (max(0, idx - window), idx + window)
                    pos_word = line[pos_word_range[0]:idx] + line[idx + 1:pos_word_range[1]]
                    for pos_w in pos_word:
                        sample_ = [pos_w]
                        # negative sampling
                        neg_num = 0
                        negs = []
                        while neg_num < self.neg_sampling_num:
                            neg_word = np.random.randint(0, self.vocab_size - 1)
                            if neg_word not in pos_word:
                                if not test:
                                    negs.append(neg_word)
                                neg_num += 1
                        # the idea in skip gram to boost training
                        for neg in negs:
                            if neg != pos_w and neg != center_word:
                                sample_.append(neg)
                                break
                        sample.append(sample_)
                    batch.append(sample)
                    batch_pin += 1
                    if batch_pin == batch_size:
                        yield batch
                        batch = []
                        batch_pin = 0
                else:
                    batch_pin += 1
                    if batch_pin == batch_size:
                        yield batch_pin
                        batch_pin = 0

        if len(batch) > 0:
            if not test:
                yield batch
            else:
                yield batch_pin

    def _init_embedding(self):
        # the embedding of all subwords
        # padding embedding, this embedding is used to match the embedding of padding label, without updating gradient
        self.subword_emb = np.random.uniform(-self.init_scale, self.init_scale, (self.subvocab_size, self.embedding_size))
        # the embedding of all the words, different from word_emb, this is a hidden parameter that will not be the final
        # embedding
        self.word_emb_hidden = np.random.uniform(-self.init_scale, self.init_scale, (self.vocab_size, self.embedding_size))

    def get_embedding(self, word):
        """
        get the embedding of word
        :param word:
        :return:
        """
        emb = self.subword_emb
        if word in self.word2id_dict.keys():
            subwords = self.word2sub_dict[self.word2id_dict[word]]
        else:
            subwords = self.gen_ngrams(word)
            subwords = [self.subword2id_dict[sub] for sub in subwords if sub in self.subword2id_dict.keys()]
        index = np.array(subwords)
        if index.shape[0] == 0:
            raise KeyError("Can't find the word or any subword in dictionary.")
        else:
            embedding = emb[index]
            return np.sum(embedding, axis=0)

    def get_similarity(self, word1, word2):
        """
        get two the cos similarity of two words
        :param word1:
        :param word2:
        :return:
        """
        emb_1 = self.get_embedding(word1)
        emb_2 = self.get_embedding(word2)
        return np.dot(emb_1, emb_2) / (np.sum(emb_1 * emb_1) ** 0.5 * np.sum(emb_2 * emb_2) ** 0.5 + 1e-9)

    def word_analogy(self, word1, word2, word3, words_list=None, verbose=1):
        """
        word1 is to word2 as word3 is to ? (? in the words_list)
        emb_target = emb_word1 + emb_word2 - emb_word3
        :param verbose: whether or not to show the target(?).
        :param words_list: provide words_list to choose target(?). if is None, the words_list is the vocabulary
        :param word1:
        :param word2:
        :param word3:
        :return:
        """
        target = self.get_embedding(word1) + self.get_embedding(word2) - self.get_embedding(word3)
        target = self.get_similar_word(target, 1, words_list=words_list, verbose=0)
        if verbose:
            print("{} is to {} as {} is to {}".format(word1, word2, word3, target))
        return target

    def get_similar_word(self, word, k, words_list=None, verbose=1):
        """
        get the top_k most similar words of word in the words_list.
        :param words_list: provide words_list to choose the k most similar words. if is None, the words_list is the vocabulary
        :param verbose: whether (1) or not (0) to print the k words
        :param word: string or embedding
        :param k:
        :return:
        """
        if words_list is None:
            vocab_emb = self.get_vocab_emb()
        else:
            if k > len(words_list):
                raise ValueError('Not enough words to choose {} most similar words. {} > the length of words_list'.format(k, k))
            vocab_emb = {word: self.get_embedding(word) for word in words_list}
        if isinstance(word, str):
            emb = self.get_embedding(word)
        else:
            emb = word
        word2emb_list = [w for w in vocab_emb.items()]
        vocab_emb = np.array([x[1] for x in word2emb_list])
        cos = np.dot(vocab_emb, emb) / np.sqrt(np.sum(vocab_emb * vocab_emb, axis=1) * np.sum(emb * emb) + 1e-9)
        flat = cos.flatten()
        indices = np.argpartition(flat, -k)[-k:]
        indices = indices[np.argsort(-flat[indices])]
        k_words = [word2emb_list[i][0] for i in indices]
        if verbose:
            print('The {} most similar words to {} are(is) {}.'.format(k, word, str(k_words)))
        return k_words

    def get_vocab_emb(self):
        """
        get the embedding of the vocabulary.
        :return: a dict with words as the keys and embeddings as the values.
        """
        return {w: self.get_embedding(w) for w in self.word2id_dict.keys()}

    def train(self, inputs):
        """
        :param center_words:
        :param target_words:
        :param label:
        :return:
        """
        for sample in inputs:
            center = sample[0]
            target_neg = sample[1:]
            for line in target_neg:
                epsilon = 0
                target = line[0]
                target_subs = self.word2sub_dict[target]
                target_emb = np.sum(np.array([self.subword_emb[sub] for sub in target_subs]), axis=0)
                neg = [center] + line[1:]
                l_w = [1] + [0] * len(line[1:])
                for index, w in enumerate(neg):
                    q = np.matmul(self.word_emb_hidden[w], target_emb)
                    q = self.activation(q)
                    g = self.learning_rate * (l_w[index] - q)
                    epsilon += g * self.word_emb_hidden[w]
                    self.word_emb_hidden[w] += g * target_emb
                for id in target_subs:
                    self.subword_emb[id] += epsilon


if __name__ == '__main__':
    import chardet
    import time

    folder_prefix = 'D:/OneDrive/work/datasets/'
    x_train = list(open(folder_prefix + "amazon-reviews-train-no-stop.txt", 'rb').readlines())
    x_test = list(open(folder_prefix + "amazon-reviews-test-no-stop.txt", 'rb').readlines())
    x_all = []
    x_all = x_all + x_train + x_test

    le = len(x_all)
    for i in range(le):
        encode_type = chardet.detect(x_all[i])
        x_all[i] = x_all[i].decode(encode_type['encoding'])  # 进行相应解码，赋给原标识符（变量
    x_all = [s.split()[1:] for s in x_all]

    embedding_size = 100
    learning_rate = 0.0001
    batch_size = 256
    epoch_num = 3
    multiple = 20
    ft = FastText_py(embedding_size=embedding_size, min_count=5, window_size=5, learning_rate=learning_rate,
                     ngrams=(3, 4, 5, 6), neg_sampling_num=4, skip_ratio=0.001)
    data = ft.fit(x_all)
    total_x_num = 0
    total_steps = 0
    x = ft.build_data(data, batch_size=batch_size, test=True)
    for num in x:
        total_x_num += num
        total_steps += 1
    print('Total number of samples: ', total_x_num)
    for epoch in range(epoch_num):
        step = 0
        ave_loss = 0
        start = time.time()
        # get a larger batch_size to shuffle and a relatively small batch_size to train
        x = ft.build_data(data, batch_size=batch_size*multiple, test=False)
        for x_ in x:
            # shuffle
            data_length = len(x_)
            random.shuffle(x_)
            # split batches
            for i in range(multiple):
                s = i*batch_size
                e = np.min((data_length, (i+1)*batch_size))
                if s >= data_length:
                    break
                inputs = x_[s:e]
                len_inputs = len(inputs)
                # train
                ft.train(inputs)
                # show progress
                step += 1
                if step % 100 == 0:
                    print("epoch {} - step {} / {} - ETA {:.0f}s.".format(epoch + 1,
                        str(step).rjust(len(str(total_steps))), total_steps, (time.time() - start) /
                        step * (total_steps - step)))
        print('epoch {} done. cost time: {:.0f}s'.format(epoch + 1, time.time() - start))
