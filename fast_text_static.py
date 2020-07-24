"""
对论文Piotr Bojanowski and Edouard Grave and Armand Joulin and Tomas Mikolov， Enriching Word Vectors with Subword Information
Piotr, 2017的实现
"""

import numpy as np
from paddle import fluid
from paddle.fluid.dygraph.nn import Embedding
from collections import Counter
from itertools import chain


class FastText_pp:
    """
    使用paddlepaddle框架编写
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
    learning_rate: learning rate for training
    use_gpu: whether or not to use gpu for training
    """
    def __init__(self, embedding_size,
                 min_count=5,
                 window_size=5,
                 init_scale=2.5,
                 ngrams=(3,4,5,6),
                 skip_ratio=1.0,
                 neg_sampling_num=4,
                 learning_rate=0.01,
                 use_gpu=False
                 ):
        self.min_count = min_count
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.init_scale = init_scale
        self.ngrams = ngrams
        self.skip_ratio = skip_ratio
        self.neg_sampling_num = neg_sampling_num
        self.sub_emb_to_array = None
        self.learning_rate = learning_rate
        if use_gpu:
            self.place = fluid.CUDAPlace(0)
        else:
            self.place = fluid.CPUPlace()
        self.deployed = False

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
        max_length_word2sub = np.max([len(sub) for sub in self.word2sub_dict.values()])
        self.max_length_word2sub = max_length_word2sub
        self.word2sub_length = {id: len(sub) for id, sub in self.word2sub_dict.items()}
        self.word2sub_dict = {id: sub + [self.subvocab_size] * (max_length_word2sub - len(sub)) for id, sub in self.word2sub_dict.items()}

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
                # let window_size be the up bound, random sample window
                window = np.random.randint(1, self.window_size)
                # positive_word_candidates
                pos_word_range = (max(0, idx - window), idx + window)
                pos_word = line[pos_word_range[0]:idx] + line[idx+1:pos_word_range[1]]
                for pos_w in pos_word:
                    if not test:
                        batch.append((center_word, pos_w, 1))
                    batch_pin += 1
                    if batch_pin == batch_size:
                        if not test:
                            batch = np.array(batch)
                            center_word_batch, target_word_batch, label_batch = np.hsplit(batch, 3)
                            center_word_batch = np.squeeze(center_word_batch)
                            center_word2sub_length_batch = (1.0 / np.array([self.word2sub_length[id] for id in center_word_batch])).astype('float32')
                            center_word_batch = np.array([self.word2sub_dict[id] for id in center_word_batch]).astype('int64')
                            target_word_batch = np.squeeze(target_word_batch).astype('int64')
                            label_batch = np.squeeze(label_batch).astype('float32')
                            yield center_word_batch, target_word_batch, label_batch, center_word2sub_length_batch
                            batch = []
                            batch_pin = 0
                        else:
                            yield batch_pin
                            batch = []
                            batch_pin = 0
                    # negative sampling
                    neg_num = 0
                    while neg_num < self.neg_sampling_num:
                        neg_word = np.random.randint(0, self.vocab_size-1)
                        if neg_word not in pos_word:
                            if not test:
                                batch.append((center_word, neg_word, 0))
                            neg_num += 1
                            batch_pin += 1
                        if batch_pin == batch_size:
                            if not test:
                                batch = np.array(batch)
                                center_word_batch, target_word_batch, label_batch = np.hsplit(batch, 3)
                                center_word_batch = np.squeeze(center_word_batch)
                                center_word2sub_length_batch = (1.0 / np.array([self.word2sub_length[id] for id in center_word_batch])).astype('float32')
                                center_word_batch = np.array([self.word2sub_dict[id] for id in center_word_batch]).astype('int64')
                                target_word_batch = np.squeeze(target_word_batch).astype('int64')
                                label_batch = np.squeeze(label_batch).astype('float32')
                                yield center_word_batch, target_word_batch, label_batch, center_word2sub_length_batch
                                batch = []
                                batch_pin = 0
                            else:
                                yield batch_pin
                                batch = []
                                batch_pin = 0

        if len(batch) > 0:
            if len(batch) == 1:
                if not test:
                    yield np.array([self.word2sub_dict[batch[0][0]]]).astype('int64'), \
                          np.array([batch[0][1]]).astype('int64'), \
                          np.array([batch[0][2]]).astype('float32'), \
                          (1.0 / np.array(self.word2sub_length[batch[0][0]])).astype('float32')
                else:
                    yield 1
            else:
                if not test:
                    batch = np.array(batch)
                    center_word_batch, target_word_batch, label_batch = np.hsplit(batch, 3)
                    center_word_batch = np.squeeze(center_word_batch)
                    center_word2sub_length_batch = (1.0 / np.array([self.word2sub_length[id] for id in center_word_batch])).astype('float32')
                    center_word_batch = np.array([self.word2sub_dict[id] for id in center_word_batch]).astype('int64')
                    target_word_batch = np.squeeze(target_word_batch).astype('int64')
                    label_batch = np.squeeze(label_batch).astype('float32')
                    yield center_word_batch, target_word_batch, label_batch, center_word2sub_length_batch
                else:
                    yield batch_pin

    def _init_embedding(self):
        # the embedding of all words
        '''self.word_emb = Embedding(size=(self.vocab_size, self.embedding_size),
                                  dtype='float32',
                                  param_attr=fluid.ParamAttr(name='word_emb_para',
                                                             initializer=fluid.initializer.UniformInitializer(
                                                                 low=-self.init_scale/self.embedding_size,
                                                                 high=self.init_scale/self.embedding_size)))'''
        # the embedding of all subwords
        # padding embedding, this embedding is used to match the embedding of padding label, without updating gradient
        self.subword_emb = Embedding(size=(self.subvocab_size + 1, self.embedding_size),
                                     dtype='float32',
                                     padding_idx=self.subvocab_size,
                                     param_attr=fluid.ParamAttr(name='subword_emb_para',
                                                                initializer=fluid.initializer.UniformInitializer(
                                                                    low=-self.init_scale / self.embedding_size,
                                                                    high=self.init_scale / self.embedding_size),
                                                                trainable=True)
           )
        # the embedding of all the words, different from word_emb, this is a hidden parameter that will not be the final
        # embedding
        self.word_emb_hidden = Embedding(size=(self.vocab_size, self.embedding_size),
                                         dtype='float32',
                                         param_attr=fluid.ParamAttr(name='hidden_emb_para',
                                                                    initializer=fluid.initializer.UniformInitializer(
                                                                        low=-self.init_scale/self.embedding_size,
                                                                        high=self.init_scale/self.embedding_size),
                                                                    trainable=True))

    def get_embedding(self, word):
        """
        get the embedding of word
        :param word:
        :return:
        """
        emb = self.sub_emb_to_array
        if word in self.word2id_dict.keys():
            subwords = self.word2sub_dict[self.word2id_dict[word]]
            len_sub = self.word2sub_length[self.word2id_dict[word]]
        else:
            subwords = self.gen_ngrams(word)
            subwords = [self.subword2id_dict[sub] for sub in subwords if sub in self.subword2id_dict.keys()]
            len_sub = len(subwords)
        index = np.array(subwords)
        if index.shape[0] == 0:
            raise KeyError("Can't find the word or any subword in dictionary.")
        else:
            embedding = emb[index]
            return np.sum(embedding, axis=0) / len_sub

    def get_similarity(self, word1, word2):
        """
        get two the cos similarity of two words
        :param word1:
        :param word2:
        :return:
        """
        emb_1 = self.get_embedding(word1)
        emb_2 = self.get_embedding(word2)
        return np.dot(emb_1, emb_2) / np.sqrt(np.dot(emb_1, emb_1) * np.dot(emb_2, emb_2) + 1e-9)

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
        target = self.get_similar_word(target, 2, words_list=words_list, verbose=0)[1]
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

    def forward(self, center_words, len_sub, target_words, label):
        """
        :param center_words:
        :param target_words:
        :param label:
        :return:
        """
        target_words_emb = self.word_emb_hidden(target_words)
        center_words_emb = self.subword_emb(center_words)
        center_words_emb = fluid.layers.reduce_sum(center_words_emb, dim=1)
        len_sub = fluid.layers.reshape(len_sub, [-1, 1])
        center_words_emb = fluid.layers.elementwise_mul(center_words_emb, len_sub)
        # mul
        word_sim = fluid.layers.elementwise_mul(center_words_emb, target_words_emb)
        word_sim = fluid.layers.reduce_sum(word_sim, dim=-1)
        pred = fluid.layers.sigmoid(word_sim)
        # calculate the loss
        loss = fluid.layers.sigmoid_cross_entropy_with_logits(word_sim, label)
        loss = fluid.layers.reduce_mean(loss)
        return pred, loss

    def train(self, data, epochs=2, batch_size=32, multiple=500):
        """
        To train the model
        multiple is the size of data that pre released time the batch_size, multiple is used to get
        the shuffled data to get stable training
        """
        if not self.deployed:
            center_words_var = fluid.data(name='center_word', shape=[None, self.max_length_word2sub], dtype='int64')
            target_words_var = fluid.data(name='target_word', shape=[None], dtype='int64')
            label_var = fluid.data(name='label', shape=[None], dtype='float32')
            len_sub_var = fluid.data(name='len_sub', shape=[None], dtype='float32')

            self.main = fluid.default_main_program()
            self.startup = fluid.default_startup_program()

            pred, self.loss = self.forward(center_words_var, len_sub_var, target_words_var, label_var)

            optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
            optimizer.minimize(self.loss)

            #self.emb_ = self.main.all_parameters()[0]

            self.exe = fluid.Executor(self.place)
            self.exe.run(self.startup)

            self.deployed = True

        feed_order = ['center_word', 'len_sub', 'target_word', 'label']
        feed_list = [self.main.global_block().var(var_name) for var_name in feed_order]
        self.feeder = fluid.DataFeeder(feed_list=feed_list, place=self.place)

        total_x_num = 0
        total_steps = 0
        x = self.build_data(data, batch_size=batch_size, test=True)
        for num in x:
            total_x_num += num
            total_steps += 1
        print('Total number of samples: ', total_x_num)

        for epoch in range(epochs):
            start = time.time()
            step = 0
            ave_loss = 0
            # get a larger batch_size to shuffle and a relatively small batch_size to train
            x = self.build_data(data, batch_size=batch_size * multiple, test=False)
            for center, target, label, len_sub in x:
                # shuffle
                data_length = center.shape[0]
                shuffle_ix = np.random.permutation(np.arange(data_length))
                center_batch = center[shuffle_ix]
                target_batch = target[shuffle_ix]
                label_batch = label[shuffle_ix]
                len_sub_batch = len_sub[shuffle_ix]
                # split batches
                for i in range(multiple):
                    s = i * batch_size
                    e = np.min((data_length, (i + 1) * batch_size))
                    if s >= data_length:
                        break
                    ix = np.arange(s, e)
                    center = center_batch[ix]
                    target = target_batch[ix]
                    label = label_batch[ix]
                    len_sub = len_sub_batch[ix]

                    step += 1

                    loss, = self.exe.run(self.main, feed=self.feeder.feed(zip(center, len_sub, target, label)), fetch_list=[self.loss])
                    ave_loss = (ave_loss * (float(step) - 1.0) + loss[0]) / float(step)

                    if step % 10 == 0:
                        print("epoch {} - step {} / {} - ETA {:.0f}s - loss {:.6f}".format(epoch + 1,
                                                                                           str(step).rjust(
                                                                                               len(str(total_steps))),
                                                                                           total_steps,
                                                                                           (time.time() - start) /
                                                                                           step * (total_steps - step),
                                                                                           loss[0]))
            self.sub_emb_to_array, = self.exe.run(self.main, feed=self.feeder.feed(zip(center, len_sub, target, label)), fetch_list=[self.main.all_parameters()[0]])
            print('epoch {} done - cost time {:.0f}s - ave loss {:.6f}'.format(epoch + 1, time.time() - start, ave_loss))



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

    embedding_size = 256
    learning_rate = 0.01
    epoch_num = 1
    multiple = 20
    batch_size = 512

    ft = FastText_pp(embedding_size=embedding_size, min_count=5, window_size=5, ngrams=(3, 4, 5), neg_sampling_num=4,
                     skip_ratio=0.001, learning_rate=learning_rate, use_gpu=False)
    data = ft.fit(x_all[:1000])
    ft.train(data, epochs=epoch_num, batch_size=512, multiple=1000)

    '''center_words_var = fluid.data(name='center_word', shape=[None, ft.max_length_word2sub], dtype='int64')
    target_words_var = fluid.data(name='target_word', shape=[None], dtype='int64')
    label_var = fluid.data(name='label', shape=[None], dtype='float32')
    len_sub_var = fluid.data(name='len_sub', shape=[None], dtype='float32')

    main = fluid.default_main_program()
    startup = fluid.default_startup_program()

    pred, loss_ = ft.forward(center_words_var, len_sub_var, target_words_var, label_var)

    optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
    optimizer.minimize(loss_)

    # self.emb_ = self.main.all_parameters()[0]

    exe = fluid.Executor(ft.place)
    exe.run(startup)

    feed_order = ['center_word', 'len_sub', 'target_word', 'label']
    feed_list = [main.global_block().var(var_name) for var_name in feed_order]
    feeder = fluid.DataFeeder(feed_list=feed_list, place=ft.place)

    total_x_num = 0
    total_steps = 0
    x = ft.build_data(data, batch_size=batch_size, test=True)
    for num in x:
        total_x_num += num
        total_steps += 1
    print('Total number of samples: ', total_x_num)

    for epoch in range(epoch_num):
        start = time.time()
        step = 0
        ave_loss = 0
        # get a larger batch_size to shuffle and a relatively small batch_size to train
        x = ft.build_data(data, batch_size=batch_size * multiple, test=False)
        for center, target, label, len_sub in x:
            # shuffle
            data_length = center.shape[0]
            shuffle_ix = np.random.permutation(np.arange(data_length))
            center_batch = center[shuffle_ix]
            target_batch = target[shuffle_ix]
            label_batch = label[shuffle_ix]
            len_sub_batch = len_sub[shuffle_ix]
            # split batches
            for i in range(multiple):
                s = i * batch_size
                e = np.min((data_length, (i + 1) * batch_size))
                if s >= data_length:
                    break
                ix = np.arange(s, e)
                center = center_batch[ix]
                target = target_batch[ix]
                label = label_batch[ix]
                len_sub = len_sub_batch[ix]

                step += 1

                loss, = exe.run(main, feed=feeder.feed(zip(center, len_sub, target, label)),
                                     fetch_list=[loss_])
                ave_loss = (ave_loss * (float(step) - 1.0) + loss[0]) / float(step)

                if step % 10 == 0:
                    print("epoch {} - step {} / {} - ETA {:.0f}s - loss {:.6f}".format(epoch + 1,
                                                                                       str(step).rjust(
                                                                                           len(str(total_steps))),
                                                                                       total_steps,
                                                                                       (time.time() - start) /
                                                                                       step * (total_steps - step),
                                                                                       loss[0]))
        ft.sub_emb_to_array, = exe.run(main, feed=feeder.feed(zip(center, len_sub, target, label)),
                                              fetch_list=[main.all_parameters()[0]])
        print('epoch {} done - cost time {:.0f}s - ave loss {:.6f}'.format(epoch + 1, time.time() - start, ave_loss))'''
