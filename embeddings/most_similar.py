import pandas as pd
import subprocess
import os
import numpy as np
import nltk
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import time
from string_grouper import match_strings, match_most_similar, compute_pairwise_similarities

from tqdm import tqdm
from collections import Counter
import re
import math
from difflib import SequenceMatcher
import string

from multiprocessing import Process, Manager, Pipe

from numba import jit

# Import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.models import FastText
from numpy import dot
from numpy.linalg import norm

import fasttext as ft
import spacy
import wmd
# import libwmdrelax
# nlp = spacy.load('fr_core_news_lg')
# nlp.add_pipe(wmd.WMD.SpacySimilarityHook(nlp), last=True)

WORD = re.compile(r"\w+")

model = gensim.models.fasttext.load_facebook_model('/Volumes/Elements/medical_word_embeddings/data/big_model.fr.bin')

from nltk.corpus import stopwords
# from nltk import download
# download('stopwords')
stop_words = stopwords.words('french')


def create_excel_file():
    f = open('../data/CEPIDC_embeddings_top10_best_matches_NEW.txt','r',encoding='utf-8')
    lines = f.readlines()
    f.close()

    f = open('../results/CEPIDC_embeddings_top10_best_matches_NEW.csv','w',encoding='utf-8')
    for line in lines:
        l = line.split(';')
        label = l[0].replace('|','')
        ln = line.split('{')
        sim = ln[1].strip('}')
        sall = sim.split(',')
        f.write(label)
        for si in sall:
            sw = si.split(":")
            f.write("|"+sw[0].strip().strip("'"))
        f.write("\n")
    f.close()


def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))


def most_similar(word, topn=1):
    word = nlp.vocab[str(word)]
    queries = [
        w for w in word.vocab
        if w.is_lower == word.is_lower and w.prob >= -15 and np.count_nonzero(w.vector)
    ]

    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    return [(w.lower_,w.similarity(word)) for w in by_similarity[:topn+1] if w.lower_ != word.lower_]


def preprocess(sentence):

    sentence = sentence.lower().split()

    # Retain alphabetic words: alpha_only
    # alpha_only = [t for t in sentence if t.isalpha()]
    no_punc = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentence]

    # Remove all stop words: no_stops
    no_stops = [t for t in no_punc if t not in stop_words]

    # Instantiate the WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()

    # Lemmatize all tokens into a new list: lemmatized
    lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

    return " ".join(lemmatized)


@jit(nopython=True)
def cosine_similarity_numba(u:np.ndarray, v:np.ndarray):
    assert(u.shape[0] == v.shape[0])
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 1
    if uu!=0 and vv!=0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta

def similar(a_s, b_s):
    sims = []
    for ind, a in enumerate(a_s):
        b = b_s[ind]
        sim = SequenceMatcher(None, a, b).ratio()
        sims.append(sim)
    return sims


def get_cosine(texts1, texts2):

    vecs1 = text_to_vectors(texts1)
    vecs2 = text_to_vectors(texts2)

    sims = []
    for ind, vec1 in enumerate(vecs1):
        vec2 = vecs2[ind]
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
        sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            sim = 0.0
        else:
            sim = float(numerator) / denominator

        sims.append(sim)
    return sims


def text_to_vectors(texts):
    vecs = []
    for text in texts:
        words = WORD.findall(text)
        vecs.append(Counter(words))
    return vecs


def avg_sentence_vector(sen):
    # featureVec = nlp(sen).vector
    words = sen.split()
    featureVec = np.zeros((100,), dtype="float32")
    nwords = 0
    #
    for word in words:
        # if word in model.wv.key_to_index:
        featureVec = np.add(featureVec, model.wv[word])
        nwords = nwords+1
        # if word in model.words:
        # if nlp(word).has_vector:
            # featureVec = np.add(featureVec, model[word])
            # featureVec = np.add(featureVec, nlp(word).vector)
            # nwords = nwords+1
        # else:
            # result = model.wv.most_similar(word)
            # most_similar_key, similarity = result[0]
            # featureVec = np.add(featureVec, model.wv[most_similar_key])
            # ms = most_similar(word, topn=1)
            # featureVec = np.add(featureVec, ms[0][0])


    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
        # featureVec = model.get_sentence_vector(sen)
        return featureVec
    else:
        return np.array([])


def avg_sentence_vectors(sens):

    manager = Manager()
    return_dict1 = manager.dict()
    return_dict2 = manager.dict()
    processes = []
    l_split = np.array_split(sens, 6)

    def split_stuff(i, s_split, return_dict1, return_dict2):

        featureVecs = []
        found_s = []

        for sen in tqdm(s_split.tolist()):
            words = sen.split()
            featureVec = np.zeros((100,), dtype="float32")

            nwords = 0

            for word in words:
                # if word in model.wv.key_to_index:
                featureVec = np.add(featureVec, model.wv[word])
                nwords = nwords+1
                # else:
                #     result = model.wv.most_similar(word)
                #     most_similar_key, similarity = result[0]
                #     featureVec = np.add(featureVec, model.wv[most_similar_key])
                # # if word in model.words:
                # #     featureVec = np.add(featureVec, model[word])
                # nwords = nwords+1


                # if nlp(word).has_vector:
                #     featureVec = np.add(featureVec, nlp(word).vector)
                #     nwords = nwords+1
                # else:
                #     ms = most_similar(word, topn=1)
                #     featureVec = np.add(featureVec, ms[0][0])

            # featureVec = np.array([nlp(token).vector for token in sen if nlp(token).has_vector]).mean(axis=0)

            if nwords>0:
                featureVec = np.divide(featureVec, nwords)
                # featureVec = nlp(sen).vector
                # sen = " ".join(preprocess(sen))
                # featureVec = model.get_sentence_vector(sen)
                found_s.append(sen)
                featureVecs.append(featureVec)

        return_dict1[i] = featureVecs
        return_dict2[i] = found_s

    for i, s in enumerate(l_split):
        p = Process(target=split_stuff, args=(i, s, return_dict1, return_dict2))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    return return_dict1, return_dict2


def run_cepidc_new():
    f = open('cepidc_sims.txt','w')
    for label in labels:
        sims = model.get_nearest_neighbors(word, k=100)
        f.write(label+"|"+sims+"\n")
    f.close()


def run_cepidc_map(method, match_most_s=True, match_s=True, word_emb=True):

    df_icd11 = pd.read_excel('../data/ATIH_deliv_15-03-2021.xlsx', na_values='')
    df_icd11 = df_icd11.replace(np.nan, '', regex=True)

    df_icd11 = df_icd11.loc[(df_icd11['Chapter']!=10) & (df_icd11['Chapter']!=5) & (df_icd11['Chapter']!=26)]

    # labels = [label.strip() for label in labels]
    # df_icd11 = pd.DataFrame(labels, columns=['ICD11'])
    labels = pd.Series(df_icd11['Label_FR(ANS, 4th round)'].values)
    chapters = pd.Series(df_icd11['Chapter'].values)
    terms_ids = pd.Series(df_icd11['TermId'].values)
    master_id = pd.Series([i for i in range(len(labels))])
    df_icd11['proc_labels'] = df_icd11['Label_FR(ANS, 4th round)'].apply(preprocess)
    proc_labels = pd.Series(df_icd11['proc_labels'].values)

    ## read CEPIDC
    df = pd.read_excel('../data/CEPIDC_matches_most_similar_PROCESSED.xlsx', na_values='')
    df = df.replace(np.nan, '', regex=True)

    # to_match_new = df['CEPIDC_original'].values
    # df['proc_match'] = df['Terme CepiDC'].apply(preprocess)
    to_match = pd.Series(df['CEPIDC_original'].values)
    print(to_match.head())


    ## use word embeddings
    if word_emb:
        icd11_sens = proc_labels.tolist()
        cepidc_sens = to_match.tolist()

        icd11_avg_vectors, icd11_found = avg_sentence_vectors(icd11_sens)

        avg_1 = []
        for avg_1_v in icd11_avg_vectors.keys():
            avg_1.extend(icd11_avg_vectors[avg_1_v])
        icd11_avg_vectors = avg_1

        avg_f = []
        for avg_1_f in icd11_found.keys():
            avg_f.extend(icd11_found[avg_1_f])
        icd11_found = avg_f

        def split_stuff2(i, s, icd11_sens):
            f = open("../results/CEPIDC_embeddings_top10_best_matches"+str(i)+"_NEW.txt","w")
            for ind2, sen2 in enumerate(tqdm(s)):
                sen2 = str(sen2)
                sen2_orig = sen2
                # sen2 = preprocess(sen2)
                # icd11_label = row1['ICD11']
                cepidc_avg_vector = avg_sentence_vector(sen2)
                if cepidc_avg_vector.size == 0:
                    continue
                min_dist = 9999999999
                max_sim = 0
                best_sen = ""
                best_ind = 0
                sims = {}

                # for index, row2 in data.iterrows():
                for ind, icd11_avg_vector in enumerate(icd11_avg_vectors):
                    label = icd11_found[ind]
                    sim = cosine_similarity_numba(np.array(cepidc_avg_vector), np.array(icd11_avg_vector))
                    sims[label] = sim
                    if sim>max_sim:
                        # min_dist = dist
                        max_sim = sim
                        best_sen = label
                        best_ind = ind

                # f.write(sen2_orig.replace(";","")+";"+best_sen.replace(";","")+";"+str(best_ind)+";"+str(max_sim)+"\n")
                sims = dict(sorted(sims.items(), key=lambda item: item[1], reverse=True)[:10])
                # str_sim = sims[0]
                # for si in sims:
                #     str_sim += ";"
                f.write(sen2_orig.replace(";","")+";"+best_sen.replace(";","")+";"+str(best_ind)+";"+str(max_sim)+";"+str(sims)+"\n")
            f.close()

        jobs = []
        df_split = np.array_split(cepidc_sens, 6)
        for i, s in enumerate(df_split):
            j = Process(target=split_stuff2, args=(i, s, icd11_avg_vectors))
            # j = Process(target=split_stuff2, args=(i, s, icd11_sens))
            j.start()
            jobs.append(j)

        for j in jobs:
            j.join()

        f = open("../results/CEPIDC_embeddings_top10_best_matches_NEW.txt","w")

        for i in range(6):
            fr = open("../results/CEPIDC_embeddings_top10_best_matches"+str(i)+"_NEW.txt","r")
            st = fr.read()
            f.write(st)
            fr.close()
        f.close()


if __name__ == "__main__":
    # run_cepidc_map('top10', match_most_s=False, match_s=False, word_emb=True)
    create_excel_file()
