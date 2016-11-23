import os
os.environ["THEANO_FLAGS"] = "device=gpu1,floatX=float32,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic"

import numpy as np
np.random.seed(1000)
from scipy.spatial.distance import cdist
from collections import namedtuple
from keras.models import Model
from keras.layers import Dense, Activation, Input, merge, Lambda
from keras.layers.convolutional import Convolution1D
from keras.optimizers import Adam
from keras.layers.pooling import AveragePooling1D, MaxPooling1D
import sys
import re
import Score
from collections import defaultdict
from sklearn import linear_model
from keras.models import load_model

vec_dim=300
sent_vec_dim=300
ans_len_cut_off=40
word_vecs = {}
stop_words=[]
idf = defaultdict(float)
data_folder=""
qtype_map, qtype_invmap = {}, {}

def load_word2vec(fname):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
    word_vecs["<unk>"] = np.random.uniform(-0.25, 0.25, vec_dim)
    return word_vecs

def load_vocab(vocab_file):
    file_reader = open(vocab_file)
    lines = file_reader.readlines()
    file_reader.close()
    vocab = {}
    for line in lines:
        parts = line.split('\t')
        qs = parts[0]
        ans = parts[1]
        qwords = qs.split()
        for word in qwords:
            if vocab.has_key(word):
                vocab[word] += 1
            else:
                vocab[word] = 1
        answords = ans.split()
        for word in answords:
            if vocab.has_key(word):
                vocab[word] += 1
            else:
                vocab[word] = 1
    return vocab

def load_samples(file):
    file_reader = open(file)
    lines = file_reader.readlines()
    file_reader.close()
    samples = []
    for line in lines:
        parts = line.split('\t')
        qs = parts[0]
        qs=qs.replace('\n','')
        ans = parts[1]
        ans=ans.replace('\n','')
        qwords = qs.split()
        answords = ans.split()
        label = int(parts[2].replace('\n', ''))
        sample = QASample(Qs=qs, Ans=ans, QsWords=qwords, AnsWords=answords, Label=label)
        samples.append(sample)
    return samples

def load_stop_words(stop_file):
    file_reader=open(stop_file)
    lines=file_reader.readlines()
    for line in lines:
        line=line.replace('\n','')
        stop_words.append(line)
    return stop_words

def load_bag_of_words_based_neural_net_data(samples):
    qsdata = []
    ansdata = []
    labels = []
    for sample in samples:
        qsvec = get_bag_of_words_based_sentence_vec(sample.QsWords)
        ansvec=get_bag_of_words_based_sentence_vec(sample.AnsWords)
        qsdata.append(qsvec)
        ansdata.append(ansvec)
        labels.append(sample.Label)

    qsdata_nn = np.array(qsdata)
    ansdata_nn = np.array(ansdata)
    label_nn = np.array(labels)
    return qsdata_nn,ansdata_nn,label_nn

def get_bag_of_words_based_sentence_vec(words):
    vec = np.zeros(vec_dim, dtype='float32')
    word_count = 0
    for word in words:
        if stop_words.count(word) > 0:
            continue
        if word_vecs.has_key(word):
            vec += word_vecs[word]
        else:
            vec += word_vecs["<unk>"]
        word_count += 1
    #vec *= 100
    #vec /= word_count
    return vec

def run_neural_model(train_qsdata, train_ansdata, train_label, dev_qsdata, dev_ansdata, dev_ref_lines, test_qsdata, test_ansdata, test_ref_lines):

    qs_input = Input(shape=(vec_dim,), dtype='float32', name='qs_input')
    ans_input = Input(shape=(vec_dim,), dtype='float32', name='ans_input')
    qtm = Dense(output_dim=vec_dim, input_dim=vec_dim, activation='linear')(qs_input)
    merged = merge([qtm, ans_input], mode='dot', dot_axes=(1, 1))
    labels = Activation('sigmoid', name='labels')(merged)
    bow_model = Model(input=[qs_input, ans_input], output=[labels])

    bow_model.compile(loss={'labels': 'binary_crossentropy'}, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  metrics=['accuracy'])
    #SGD(lr=0.001, momentum=0.9, nesterov=True)

    batch_size = 100
    epoch = 50
    best_MAP = -10.0
    best_bow_model_file = os.path.join(data_folder, "best_bow_model.h5")

    for epoch_count in range(0, epoch):

        bow_model.fit({'qs_input': train_qsdata, 'ans_input': train_ansdata}, {'labels': train_label}, nb_epoch=1,
                  batch_size=batch_size, verbose=2)
        dev_probs = bow_model.predict([dev_qsdata, dev_ansdata], batch_size=batch_size)
        MAP, MRR = cal_score(dev_ref_lines, dev_probs)
        if MAP > best_MAP:
            best_MAP = MAP
            bow_model.save(best_bow_model_file)

    best_bow_model = load_model(best_bow_model_file)

    test_probs = best_bow_model.predict([test_qsdata, test_ansdata], batch_size=batch_size)

    MAP, MRR = cal_score(test_ref_lines, test_probs)
    return MAP, MRR

def run_bag_of_words_model(train_samples, dev_samples, test_samples, dev_ref_lines, test_ref_lines):

    train_qsdata, train_ansdata, train_label = load_bag_of_words_based_neural_net_data(train_samples)
    dev_qsdata, dev_ansdata, dev_label = load_bag_of_words_based_neural_net_data(dev_samples)
    test_qsdata, test_ansdata, test_label = load_bag_of_words_based_neural_net_data(test_samples)
    MAP, MRR = run_neural_model(train_qsdata, train_ansdata, train_label, dev_qsdata, dev_ansdata, dev_ref_lines, test_qsdata, test_ansdata, test_ref_lines)
    return MAP, MRR

def get_cnn_data(samples, max_qs_l, max_ans_l):
    qsdata = np.zeros(shape=(len(samples), max_qs_l, vec_dim), dtype="float32")
    ansdata = np.zeros(shape=(len(samples), max_ans_l, vec_dim), dtype="float32")
    labeldata = np.zeros(len(samples), dtype="int32")
    sent_count = 0
    for sample in samples:
        word_count = 0
        for word in sample.QsWords:
            if (word_vecs.has_key(word)):
                qsdata[sent_count][word_count] = word_vecs[word]
            else:
                qsdata[sent_count][word_count] = word_vecs["<unk>"]
            word_count += 1
        word_count = 0
        for word in sample.AnsWords:
            if (word_vecs.has_key(word)):
                ansdata[sent_count][word_count] = word_vecs[word]
            else:
                ansdata[sent_count][word_count] = word_vecs["<unk>"]
            word_count += 1
            if word_count==40:
                break
        labeldata[sent_count] = sample.Label
        sent_count += 1
    return qsdata,ansdata,labeldata

def get_max_len(train_samples,dev_samples,test_samples):
    max_qs_l = len(train_samples[0].QsWords)
    for i in range(1, len(train_samples)):
        if len(train_samples[i].QsWords) > max_qs_l:
            max_qs_l = len(train_samples[i].QsWords)

    for i in range(0, len(dev_samples)):
        if len(dev_samples[i].QsWords) > max_qs_l:
            max_qs_l = len(dev_samples[i].QsWords)

    for i in range(0, len(test_samples)):
        if len(test_samples[i].QsWords) > max_qs_l:
            max_qs_l = len(test_samples[i].QsWords)

    max_ans_l = len(train_samples[0].AnsWords)
    for i in range(1, len(train_samples)):
        if len(train_samples[i].AnsWords) > max_ans_l:
            max_ans_l = len(train_samples[i].AnsWords)

    for i in range(0, len(dev_samples)):
        if len(dev_samples[i].AnsWords) > max_ans_l:
            max_ans_l = len(dev_samples[i].AnsWords)

    for i in range(0, len(test_samples)):
        if len(test_samples[i].AnsWords) > max_ans_l:
            max_ans_l = len(test_samples[i].AnsWords)
    return max_qs_l, max_ans_l

def train_cnn(ngram, data_folder, max_qs_l, max_ans_l,
              train_qsdata, train_ansdata, train_labeldata,
              dev_qsdata, dev_ansdata,
              test_qsdata, test_ansdata,
              dev_ref_lines, test_ref_lines):
    Reduce = Lambda(lambda x: x[:, 0, :], output_shape=lambda shape: (shape[0], shape[-1]))
    qs_input = Input(shape=(max_qs_l, vec_dim,), dtype='float32', name='qs_input')
    qsconvmodel = Convolution1D(nb_filter=sent_vec_dim, filter_length=ngram, activation="tanh", border_mode='valid')(
        qs_input)
    qsconvmodel = AveragePooling1D(pool_length=max_qs_l - ngram + 1)(qsconvmodel)
    qsconvmodel = Reduce(qsconvmodel)

    qtm = Dense(output_dim=sent_vec_dim, activation='linear')(qsconvmodel)

    ans_input = Input(shape=(max_ans_l, vec_dim,), dtype='float32', name='ans_input')
    ansconvmodel = Convolution1D(nb_filter=sent_vec_dim, filter_length=ngram, activation="tanh", border_mode='valid')(
        ans_input)
    ansconvmodel = AveragePooling1D(pool_length=max_ans_l - ngram + 1)(ansconvmodel)
    ansconvmodel = Reduce(ansconvmodel)

    merged = merge([qtm, ansconvmodel], mode='dot', dot_axes=(1, 1))
    labels = Activation('sigmoid', name='labels')(merged)
    cnn_model = Model(input=[qs_input, ans_input], output=[labels])

    cnn_model.compile(loss={'labels': 'binary_crossentropy'},
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), metrics=['accuracy'])
    # Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    # SGD(lr=0.01, momentum=0.9, nesterov=True)


    batch_size = 10
    epoch = 20
    best_MAP=-10.0
    best_cnn_model_file=os.path.join(data_folder,"best_cnn_model.h5")
    train_probs_epochs=[]
    dev_probs_epochs=[]
    test_probs_epochs=[]
    for epoch_count in range(0,epoch):

        cnn_model.fit({'qs_input': train_qsdata, 'ans_input': train_ansdata}, {'labels': train_labeldata}, nb_epoch=1,
              batch_size=batch_size,verbose=2)
        train_probs=cnn_model.predict([train_qsdata, train_ansdata], batch_size=batch_size)
        train_probs_epochs.append(train_probs)
        dev_probs=cnn_model.predict([dev_qsdata,dev_ansdata],batch_size=batch_size)
        dev_probs_epochs.append(dev_probs)
        test_probs = cnn_model.predict([test_qsdata, test_ansdata], batch_size=batch_size)
        test_probs_epochs.append(test_probs)
        MAP, MRR=cal_score(dev_ref_lines,dev_probs)
        if MAP > best_MAP :
            best_MAP=MAP
            cnn_model.save(best_cnn_model_file)

    best_cnn_model=load_model(best_cnn_model_file)

    train_probs = best_cnn_model.predict([train_qsdata, train_ansdata], batch_size=batch_size)
    dev_probs=best_cnn_model.predict([dev_qsdata, dev_ansdata], batch_size=batch_size)
    test_probs = best_cnn_model.predict([test_qsdata, test_ansdata], batch_size=batch_size)

    MAP, MRR = cal_score(test_ref_lines, test_probs)

    return MAP, MRR, train_probs_epochs, dev_probs_epochs, test_probs_epochs, train_probs, dev_probs, test_probs

def train_lr_using_dense_layer(reg_train_data_np, reg_dev_data_np, reg_test_data_np, train_labeldata, dev_ref_lines, test_ref_lines):
    reg_feature_dim = len(reg_train_data_np[0])
    print reg_feature_dim
    reg_input = Input(shape=(reg_feature_dim,), dtype='float32', name='reg_input')
    reg_layer = Dense(output_dim=1)(reg_input)
    reg_output = Activation('sigmoid', name='reg_output')(reg_layer)
    reg_model = Model(input=[reg_input], output=[reg_output])
    reg_model.compile(loss={'reg_output': 'binary_crossentropy'},
                      optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                      metrics=['accuracy'])
    batch_size = 1
    epoch=20
    best_MAP = -10.0
    best_reg_model_file = os.path.join(data_folder, "best_lr_dense_model.h5")
    for epoch_count in range(0, epoch):
        reg_model.fit({'reg_input': reg_train_data_np}, {'reg_output': train_labeldata}, nb_epoch=1,
                      batch_size=batch_size,verbose=2)

        dev_probs = reg_model.predict([reg_dev_data_np], batch_size=batch_size)
        MAP, MRR=cal_score(dev_ref_lines,dev_probs)
        if MAP > best_MAP:
            best_MAP=MAP
            reg_model.save(best_reg_model_file)
    best_reg_model=load_model(best_reg_model_file)
    test_probs=best_reg_model.predict([reg_test_data_np], batch_size=batch_size)

    MAP, MRR = cal_score(test_ref_lines, test_probs)
    return MAP, MRR

def train_lr_using_sklearn(train_probs_epochs, dev_probs_epochs, test_probs_epochs,
                           train_samples, dev_samples, test_samples, train_labeldata,
                           dev_ref_lines, test_ref_lines):
    best_dev_MAP=-10.0
    best_test_MAP=0.0
    best_test_MRR=0.0
    for i in range(0,len(train_probs_epochs)):
        train_probs=train_probs_epochs[i]
        dev_probs=dev_probs_epochs[i]
        test_probs=test_probs_epochs[i]
        reg_train_data_np = get_lr_data(train_samples, train_probs)
        reg_dev_data_np = get_lr_data(dev_samples, dev_probs)
        reg_test_data_np = get_lr_data(test_samples, test_probs)
        clf = linear_model.LogisticRegression(C=0.01, solver='lbfgs')
        clf = clf.fit(reg_train_data_np, train_labeldata)
        lr_dev_preds = clf.predict_proba(reg_dev_data_np)
        dev_probs = []
        for lr_dev_pred in lr_dev_preds:
            dev_probs.append(lr_dev_pred[1])
        dev_MAP, dev_MRR = cal_score(dev_ref_lines, dev_probs)

        lr_test_preds = clf.predict_proba(reg_test_data_np)
        test_probs = []
        for lr_test_pred in lr_test_preds:
            test_probs.append(lr_test_pred[1])
        test_MAP, test_MRR = cal_score(test_ref_lines, test_probs)
        if dev_MAP > best_dev_MAP :
            best_dev_MAP=dev_MAP
            best_test_MAP=test_MAP
            best_test_MRR=test_MRR

    return best_test_MAP, best_test_MRR

def get_lr_data(samples, probs):
    reg_data = []
    data_index = 0
    for sample in samples:
        feat = cali_feature_extractor(sample, probs[data_index])
        reg_data.append(feat)
        data_index += 1

    reg_data_np = np.array(reg_data)
    return reg_data_np

def run_bigram_model(ngram, train_samples, dev_samples, dev_ref_lines, test_samples, test_ref_lines):

    max_qs_l, max_ans_l=get_max_len(train_samples,dev_samples,test_samples)
    if max_ans_l > ans_len_cut_off:
        max_ans_l = ans_len_cut_off
    train_qsdata, train_ansdata, train_labeldata = get_cnn_data(train_samples, max_qs_l, max_ans_l)
    dev_qsdata, dev_ansdata, dev_labeldata=get_cnn_data(dev_samples, max_qs_l, max_ans_l)
    test_qsdata, test_ansdata, test_labeldata = get_cnn_data(test_samples, max_qs_l, max_ans_l)

    CNN_MAP, CNN_MRR, train_probs_epochs, dev_probs_epochs, test_probs_epochs, train_probs, dev_probs, test_probs=train_cnn(ngram, data_folder, max_qs_l, max_ans_l,
                                      train_qsdata, train_ansdata, train_labeldata,
                                      dev_qsdata, dev_ansdata,
                                      test_qsdata, test_ansdata,
                                      dev_ref_lines, test_ref_lines)


    reg_train_data_np=get_lr_data(train_samples,train_probs)
    reg_dev_data_np=get_lr_data(dev_samples,dev_probs)
    reg_test_data_np = get_lr_data(test_samples, test_probs)

    LR_Dense_MAP, LR_Dense_MRR = train_lr_using_dense_layer(reg_train_data_np, reg_dev_data_np, reg_test_data_np, train_labeldata, dev_ref_lines, test_ref_lines)

    #LR_Sklearn_MAP, LR_Sklearn_MRR= train_lr_using_sklearn(train_probs_epochs, dev_probs_epochs, test_probs_epochs, train_samples, dev_samples, test_samples, train_labeldata,
    #                                                   dev_ref_lines, test_ref_lines)

    return CNN_MAP, CNN_MRR, LR_Dense_MAP, LR_Dense_MRR #, LR_Sklearn_MAP, LR_Sklearn_MRR

def cal_score(ref_lines, probs):
    line_count = 0
    pred_lines = defaultdict(list)
    for ref_line in ref_lines:
        ref_line = ref_line.replace('\n', '')
        parts = ref_line.strip().split()
        qid, aid, lbl = int(parts[0]), int(parts[2]), int(parts[3])
        pred_lines[qid].append((aid, lbl, probs[line_count]))
        line_count += 1
    MAP=Score.calc_mean_avg_prec(pred_lines)
    MRR=Score.calc_mean_reciprocal_rank(pred_lines)
    return MAP, MRR

def clean_str(string):
    """
    Tokenization/string cleaning
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def build_idf(files):
    n = 0
    ques_list, qtype_list = [], []
    for fname in files:
        with open(fname, "rb") as f:
            for line in f:
                n += 1
                parts = line.strip().split("\t")
                question=parts[0].replace('\n','')
                ques_list.append(question)
                question = clean_str(question)
                words = set(question.split())
                for word in words:
                    if word in stop_words:
                        continue
                    idf[word] += 1
    for word in idf.keys():
        idf[word] = np.log(n / idf[word])
    for fname in files:
        with open(fname[:fname.rfind(".") + 1] + "qtype", "rb") as f:
            for line in f:
                parts = line.strip().split(":")
                qtype_list.append(parts[0])
    for q, qt in zip(ques_list, qtype_list):
        qtype_map[q] = qt
        if qt not in qtype_invmap:
            qtype_invmap[qt] = len(qtype_invmap)
    return idf


def count_feature_extractor(qtoks, atoks):
    qset, aset = set(qtoks), set(atoks)
    count, weighted_count = 0.0, 0.0
    for word in qset:
        if word not in stop_words and word in aset:
            count += 1.0
            weighted_count += idf[word]
    return [count, weighted_count]


def cali_feature_extractor(sample, sim_probs):
    qtoks=sample.QsWords
    atoks=sample.AnsWords
    #if len(atoks) > ans_len_cut_off:
    #    atoks=atoks[0:ans_len_cut_off-1]
    question=sample.Qs
    question=question.replace('\n','')
    feat = count_feature_extractor(qtoks, atoks)
    feat.append(len(qtoks))
    ans_len=len(atoks)
    if ans_len > ans_len_cut_off:
        ans_len=ans_len_cut_off
    feat.append(ans_len)
    count, idf_sum = 1.0, 0.0
    for word in qtoks:
        if word not in stop_words:
            count += 1.0
            idf_sum += idf[word]
    feat.append(idf_sum / count)
    count, idf_sum = 1.0, 0.0
    for word in atoks:
        if word not in stop_words:
            count += 1.0
            idf_sum += idf[word]
    feat.append(idf_sum / count)
    qtype_vec = np.zeros(len(qtype_invmap))
    qtype_vec[qtype_invmap[qtype_map[question]]] = 1.0
    feat += qtype_vec.tolist()
    #for i in range(0,5):
    feat.append(sim_probs)
    return feat

# Implementation of Wang model

def compose_decompose(qmatrix, amatrix):
    qhatmatrix, ahatmatrix = f_match(qmatrix, amatrix, window_size=3)
    qplus, qminus = f_decompose(qmatrix, qhatmatrix)
    aplus, aminus = f_decompose(amatrix, ahatmatrix)
    return qplus, qminus, aplus, aminus


def f_match(qmatrix, amatrix, window_size=3):
    A = 1 - cdist(qmatrix, amatrix, metric='cosine')  # Similarity matrix
    Atranspose = np.transpose(A)
    qa_max_indices = np.argmax(A,
                               axis=1)  # 1-d array: for each question word, the index of the answer word which is most similar
    # Selecting answer word vectors in a window surrounding the most closest answer word
    qa_window = [range(max(0, max_idx - window_size), min(amatrix.shape[0], max_idx + window_size + 1)) for max_idx in
                 qa_max_indices]
    # Selecting question word vectors in a window surrounding the most closest answer word
    # Finding weights and its sum (for normalization) to find f_match for question for the corresponding window of answer words
    qa_weights = [(np.sum(A[qword_idx][aword_indices]), A[qword_idx][aword_indices]) for qword_idx, aword_indices in
                  enumerate(qa_window)]
    # Then multiply each vector in the window with the weights, sum up the vectors and normalize it with the sum of weights
    # This will give the local-w vecotrs for the Question sentence words and Answer sentence words.
    qhatmatrix = np.array([np.sum(weights.reshape(-1, 1) * amatrix[aword_indices], axis=0) / weight_sum for
                           ((qword_idx, aword_indices), (weight_sum, weights)) in
                           zip(enumerate(qa_window), qa_weights)])

    # Doing similar stuff for answer words
    aq_max_indices = np.argmax(A,
                               axis=0)  # 1-d array: for each   answer word, the index of the question word which is most similar
    aq_window = [range(max(0, max_idx - window_size), min(qmatrix.shape[0], max_idx + window_size + 1)) for max_idx in
                 aq_max_indices]
    aq_weights = [(np.sum(Atranspose[aword_idx][qword_indices]), Atranspose[aword_idx][qword_indices]) for
                  aword_idx, qword_indices in enumerate(aq_window)]
    ahatmatrix = np.array([np.sum(weights.reshape(-1, 1) * qmatrix[qword_indices], axis=0) / weight_sum for
                           ((aword_idx, qword_indices), (weight_sum, weights)) in
                           zip(enumerate(aq_window), aq_weights)])
    return qhatmatrix, ahatmatrix


def f_decompose(matrix, hatmatrix):
    # finding magnitude of parallel vector
    mag = np.sum(hatmatrix * matrix, axis=1) / np.sum(hatmatrix * hatmatrix, axis=1)
    # multiplying magnitude with hatmatrix vector
    plus = mag.reshape(-1, 1) * hatmatrix
    minus = matrix - plus
    return plus, minus

def get_wang_conv_model_input(samples_sent_matrix, max_qs_l, max_ans_l):
    token = np.zeros((2, vec_dim), dtype='float')
    qsamples = samples_sent_matrix[0]
    asamples = samples_sent_matrix[1]
    q_list = []
    a_list = []
    for qmatrix, amatrix in zip(qsamples, asamples):
        qplus, qminus, aplus, aminus = compose_decompose(qmatrix, amatrix)
        # Padding questions
        qpad_width = ((0, max_qs_l - qplus.shape[0]), (0, 0))
        qplus_pad = np.pad(qplus, pad_width=qpad_width, mode='constant', constant_values=0.0)
        qminus_pad = np.pad(qminus, pad_width=qpad_width, mode='constant', constant_values=0.0)
        # Padding answers
        apad_width = ((0, max_ans_l - aplus.shape[0]), (0, 0))
        aplus_pad = np.pad(aplus, pad_width=apad_width, mode='constant', constant_values=0.0)
        aminus_pad = np.pad(aminus, pad_width=apad_width, mode='constant', constant_values=0.0)
        qplusminus=np.concatenate((qplus_pad, token, qminus_pad))
        aplusminus = np.concatenate((aplus_pad, token, aminus_pad))
        # Adding these padded matrices to list
        q_list.append(qplusminus)
        a_list.append(aplusminus)
    q_tensor = np.array(q_list)
    a_tensor = np.array(a_list)
    return q_tensor, a_tensor

def get_wang_model_input(samples, max_qs_l, max_ans_l):
    """
    Returns the training samples and labels as numpy array
    """
    s1samples_list = []
    s2samples_list = []
    labels_list = []

    for sample in samples:
        q_len=len(sample.QsWords)
        if q_len > max_qs_l:
            q_len=max_qs_l
        a_len=len(sample.AnsWords)
        if a_len > max_ans_l:
            a_len=max_ans_l
        s1samples_list.append(get_sent_matrix(sample.QsWords[0:q_len]))
        s2samples_list.append(get_sent_matrix(sample.AnsWords[0:a_len]))
        labels_list.append(sample.Label)

    samples_sent_matrix = [s1samples_list, s2samples_list]
    labels = labels_list
    return samples_sent_matrix, labels


def get_sent_matrix(words):
    """
    Given a sentence, gets the input in the required format.
    """
    vecs = []
    vec = np.zeros(vec_dim, dtype='float32')
    for word in words:
        if word_vecs.has_key(word):
            vec = word_vecs[word]
        else:
            vec = word_vecs["<unk>"]
        vecs.append(np.array(vec))
    return np.array(vecs)


def run_wang_cnn_model(train_samples, dev_samples, dev_ref_lines, test_samples, test_ref_lines):
    max_qs_l, max_ans_l = get_max_len(train_samples, dev_samples, test_samples)

    if max_ans_l > ans_len_cut_off:
        max_ans_l = ans_len_cut_off
    train_samples_sent_matrix, train_labels = get_wang_model_input(train_samples, max_qs_l, max_ans_l)
    train_labels_np=np.array(train_labels)
    dev_samples_sent_matrix, dev_labels = get_wang_model_input(dev_samples, max_qs_l, max_ans_l)
    dev_labels_np = np.array(dev_labels)
    test_samples_sent_matrix, test_labels = get_wang_model_input(test_samples, max_qs_l, max_ans_l)
    test_labels_np = np.array(test_labels)

    train_q_tensor, train_a_tensor = get_wang_conv_model_input(train_samples_sent_matrix, max_qs_l, max_ans_l)
    dev_q_tensor, dev_a_tensor = get_wang_conv_model_input(dev_samples_sent_matrix, max_qs_l, max_ans_l)
    test_q_tensor, test_a_tensor = get_wang_conv_model_input(test_samples_sent_matrix, max_qs_l, max_ans_l)
    max_qs_l = 2 * max_qs_l + 2
    max_ans_l = 2 * max_ans_l + 2
    Reduce = Lambda(lambda x: x[:, 0, :], output_shape=lambda shape: (shape[0], shape[-1]))

    nb_filter = 500

    qs_input = Input(shape=(max_qs_l, vec_dim,), dtype='float32', name='qs_input')
    qs_convmodel_3 = Convolution1D(nb_filter=nb_filter, filter_length=3, activation="tanh", border_mode='valid')(qs_input)
    qs_convmodel_3 = MaxPooling1D(pool_length=max_qs_l - 2)(qs_convmodel_3)
    qs_convmodel_3 = Reduce(qs_convmodel_3)
    qs_convmodel_2 = Convolution1D(nb_filter=nb_filter, filter_length=2, activation="tanh", border_mode='valid')(qs_input)
    qs_convmodel_2 = MaxPooling1D(pool_length=max_qs_l - 1)(qs_convmodel_2)
    qs_convmodel_2 = Reduce(qs_convmodel_2)
    qs_convmodel_1 = Convolution1D(nb_filter=nb_filter, filter_length=1, activation="tanh", border_mode='valid')(qs_input)
    qs_convmodel_1 = MaxPooling1D(pool_length=max_qs_l)(qs_convmodel_1)
    qs_convmodel_1 = Reduce(qs_convmodel_1)
    qs_concat = merge([qs_convmodel_1, qs_convmodel_2, qs_convmodel_3], mode='concat', concat_axis=-1)

    ans_input = Input(shape=(max_ans_l, vec_dim,), dtype='float32', name='ans_input')
    ans_convmodel_3 = Convolution1D(nb_filter=nb_filter, filter_length=3, activation="tanh", border_mode='valid')(ans_input)
    ans_convmodel_3 = MaxPooling1D(pool_length=max_ans_l - 2)(ans_convmodel_3)
    ans_convmodel_3 = Reduce(ans_convmodel_3)
    ans_convmodel_2 = Convolution1D(nb_filter=nb_filter, filter_length=2, activation="tanh", border_mode='valid')(ans_input)
    ans_convmodel_2 = MaxPooling1D(pool_length=max_ans_l - 1)(ans_convmodel_2)
    ans_convmodel_2 = Reduce(ans_convmodel_2)
    ans_convmodel_1 = Convolution1D(nb_filter=nb_filter, filter_length=1, activation="tanh", border_mode='valid')(ans_input)
    ans_convmodel_1 = MaxPooling1D(pool_length=max_ans_l)(ans_convmodel_1)
    ans_convmodel_1 = Reduce(ans_convmodel_1)
    ans_concat = merge([ans_convmodel_1, ans_convmodel_2, ans_convmodel_3], mode='concat', concat_axis=-1)

    q_a_model=merge([qs_concat, ans_concat], mode='concat', concat_axis=-1)
    sim_model = Dense(output_dim=1, activation = 'linear')(q_a_model)
    labels = Activation('sigmoid', name='labels')(sim_model)

    wang_model = Model(input=[qs_input, ans_input], output=[labels])

    #model.summary()

    wang_model.compile(loss={'labels': 'binary_crossentropy'},
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), metrics=['accuracy'])


    batch_size = 10
    epoch = 5
    best_MAP = -10.0
    best_wang_model_file = os.path.join(data_folder, "best_wang_cnn_model.h5")

    for epoch_count in range(0, epoch):

        wang_model.fit({'qs_input': train_q_tensor, 'ans_input': train_a_tensor}, {'labels': train_labels_np}, nb_epoch=1,
                  batch_size=batch_size, verbose=2)
        dev_probs = wang_model.predict([dev_q_tensor, dev_a_tensor], batch_size=batch_size)
        MAP, MRR = cal_score(dev_ref_lines, dev_probs)
        if MAP > best_MAP:
            best_MAP = MAP
            wang_model.save(best_wang_model_file)

    best_wang_model = load_model(best_wang_model_file)
    test_probs = best_wang_model.predict([test_q_tensor, test_a_tensor], batch_size=batch_size)
    MAP, MRR = cal_score(test_ref_lines, test_probs)

    train_probs = best_wang_model.predict([train_q_tensor, train_a_tensor], batch_size=batch_size)
    dev_probs = best_wang_model.predict([dev_q_tensor, dev_a_tensor], batch_size=batch_size)
    reg_train_data_np = get_lr_data(train_samples, train_probs)
    reg_dev_data_np = get_lr_data(dev_samples, dev_probs)
    reg_test_data_np = get_lr_data(test_samples, test_probs)

    LR_Dense_MAP, LR_Dense_MRR = train_lr_using_dense_layer(reg_train_data_np, reg_dev_data_np, reg_test_data_np,
                                                            train_labels_np, dev_ref_lines, test_ref_lines)
    return MAP, MRR, LR_Dense_MAP, LR_Dense_MRR

if __name__=="__main__":

    model_name = sys.argv[1]
    data_folder = os.path.join("../data")
    word_vec_file = os.path.join(data_folder, sys.argv[2])
    stop_words_file = os.path.join(data_folder, sys.argv[3])
    train_file = os.path.join(data_folder, sys.argv[4])
    dev_file = os.path.join(data_folder, sys.argv[5])
    dev_ref_file = os.path.join(data_folder, sys.argv[6])
    test_file = os.path.join(data_folder, sys.argv[7])
    test_ref_file = os.path.join(data_folder, sys.argv[8])

    QASample=namedtuple("QASample","Qs Ans QsWords AnsWords Label")

    word_vecs=load_word2vec(word_vec_file)
    stop_words=load_stop_words(stop_file=stop_words_file)

    files=[]
    files.append(train_file)
    files.append(dev_file)
    files.append(test_file)
    idf=build_idf(files)

    train_samples = load_samples(train_file)
    dev_samples=load_samples(dev_file)
    test_samples = load_samples(test_file)

    file_reader = open(dev_ref_file)
    dev_ref_lines = file_reader.readlines()
    file_reader.close()

    file_reader=open(test_ref_file)
    test_ref_lines=file_reader.readlines()
    file_reader.close()

    if model_name=="BoW":
        # Bar of words model
        print "Bag of words model started......"
        MAP, MRR = run_bag_of_words_model(train_samples, dev_samples, test_samples, dev_ref_lines, test_ref_lines)
        print "MAP:", MAP
        print "MRR:", MRR
    elif model_name=="BigramCNN":
        #Bigram model
        print "Convolutional bigram model started......"
        CNN_MAP, CNN_MRR, LR_Dense_MAP, LR_Dense_MRR = run_bigram_model(2, train_samples, dev_samples, dev_ref_lines, test_samples, test_ref_lines)
        print "Bigram CNN"
        print "MAP:", CNN_MAP
        print "MRR:", CNN_MRR
        print "Bigram CNN with Features"
        print "MAP:", LR_Dense_MAP
        print "MRR:", LR_Dense_MRR
    elif model_name=="TrigramCNN":
        #Bigram model
        print "Convolutional bigram model started......"
        CNN_MAP, CNN_MRR, LR_Dense_MAP, LR_Dense_MRR = run_bigram_model(3, train_samples, dev_samples, dev_ref_lines, test_samples, test_ref_lines)
        print "Trigram CNN"
        print "MAP:", CNN_MAP
        print "MRR:", CNN_MRR
        print "Trigram CNN with Features"
        print "MAP:", LR_Dense_MAP
        print "MRR:", LR_Dense_MRR
    elif model_name=="DecompCompCNN":
        #Decomposition and Composition based CNN model
        print "Decomposition and Composition based CNN model started......"
        MAP, MRR, LR_Dense_MAP, LR_Dense_MRR = run_wang_cnn_model(train_samples, dev_samples, dev_ref_lines, test_samples, test_ref_lines)
        print "Decomp Comp CNN"
        print "MAP:", MAP
        print "MRR:", MRR
        print "Decomp Comp CNN with Features"
        print "MAP:", LR_Dense_MAP
        print "MRR:", LR_Dense_MRR


















































