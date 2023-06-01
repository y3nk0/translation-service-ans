import numpy as np
import string
import math
import pickle
import spacy
from pattern.fr import singularize, pluralize
import nltk
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
import spacy
import re
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sentence_transformers import SentenceTransformer, util
import torch

from nltk.tokenize import word_tokenize
from Levenshtein import distance

from rules import *

cc = nltk.translate.bleu_score.SmoothingFunction()

stopwords = set(nltk.corpus.stopwords.words('french'))

nlp = spacy.load("fr_core_news_lg")
# nlp = spacy.load("fr_dep_news_trf")

def get_dicts():

    bs2_dict = {}
    df_bs = pd.read_excel("../../resources/Avancement_traduction_anatomie-1.xlsx", sheet_name="bs2")
    df_bs = df_bs.fillna('')
    for index, row in df_bs.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation_prefLabel'])
        if corr!="":
            bs2_dict[err] = corr

    bs3_dict = {}
    df_bs = pd.read_excel("../../resources/Avancement_traduction_anatomie-1.xlsx", sheet_name="bs3")
    df_bs = df_bs.fillna('')
    for index, row in df_bs.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation_prefLabel'])
        if corr!="":
            bs3_dict[err] = corr

    hyphen_dict_pref = {}
    hyphen_dict_alt = {}
    df_hyphen = pd.read_excel("../../resources/Avancement_traduction_anatomie-1.xlsx", sheet_name="Hyphen")
    df_hyphen = df_hyphen.fillna('')
    for index, row in df_hyphen.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation_prefLabel'])
        corr_alt = str(row['correct_translation_altLabel'])

        if "[" not in corr:
            if corr!="":
                hyphen_dict_pref[err] = corr
        else:
            if corr=="[TO REMOVE]":
                hyphen_dict_pref[err] = ""

        if err!="" and corr_alt!="":
            hyphen_dict_alt[err] = corr_alt

    ar7_dict_pref = {}
    ar7_dict_alt = {}
    ar7_dict_eng = {}
    df_ar = pd.read_excel("../../resources/Avancement_traduction_anatomie-1.xlsx", sheet_name="ar7")
    df_ar = df_ar.fillna('')
    for index, row in df_ar.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation_prefLabel'])
        corr_alt = str(row['correct_translation_altLabel'])
        eng = str(row['english_term'])

        if err!="" and corr!="":
            ar7_dict_pref[err] = corr
        if err!="" and corr_alt!="":
            ar7_dict_alt[err] = corr_alt
        if err!="" and eng!="":
            ar7_dict_eng[err] = eng

    ll1_dict = {}
    df_ll1 = pd.read_excel("../../resources/Avancement_traduction_anatomie-1.xlsx", sheet_name="ll1")
    df_ll1 = df_ll1.fillna('')
    for index, row in df_ll1.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation'])
        if corr!="":
            ll1_dict[err] = corr

    custom_dict = {}
    df_custom = pd.read_excel("../../resources/Avancement_traduction_anatomie-1.xlsx", sheet_name="custom")
    df_custom = df_custom.fillna('')
    for index, row in df_custom.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation'])
        if corr!="":
            custom_dict[err] = corr

    return bs2_dict, bs3_dict, hyphen_dict_pref, hyphen_dict_alt, ar7_dict_pref, ar7_dict_alt, ar7_dict_eng, ll1_dict, custom_dict



def examine_casing():
    import truecase
    # print(truecase.get_true_case('hey, what is the weather in new york?'))
    # print(truecase.get_true_case('Stanford Sleepiness Scale SSS0101 Standardized Character Result 1'))
    import json
    import stanza
    from stanza.server import CoreNLPClient
    # stanza.install_corenlp()
    english_properties = {'annotators': 'truecase','tokenize_pretokenized': True,'tokenize.whitespace': True, 'quote.asciiQuotes': True}

    client_eng = CoreNLPClient(properties=english_properties,  timeout=70000, memory='6G', be_quiet=False, max_char_length=100000, endpoint='http://localhost:9937')
    client_eng.start()
    # text = 'Barack was born in hawaii. his wife Michelle was born in Milan. He says that she is very smart.'
    # print(f"Input text: {text}")

    # submit the request to the server
    # ann = client_eng.annotate(text)

    # text = 'Bcl-XL Proteolysis Targeting Chimera DT2216'
    # text = 'Stanford Sleepiness Scale SSS0101 Standardized Character Result 1'
    # text = 'DI - Distal-incisal'
    # text = 'Smoking Not Allowed in Any Work Areas'
    # text = 'SNARE'
    # text = "A question about whether an individual's cravings get intolerable if they have not been able to smoke for a few hours."
    text = "Gower's tract"
    ann = client_eng.annotate(text)
    # import ipdb; ipdb.set_trace()

    spl = str(ann.sentence).replace("\n","").split("}token {")
    # import ipdb; ipdb.set_trace()
    label = ""
    for i in range(0,len(spl)-1):
        label += " "+spl[i].split()[-1].replace('"','')

    label += " "+spl[len(spl)-1].split("}")[0].split()[-1].replace('"','')
    label = label.strip()
    print(label)
    # print("Result = ", ann.text)


def check_upper():
    st = "lymphocyte cD4 + HLA-DR + T"
    if st.isupper():
        print("True")
    else:
        print("False")
    if st.islower():
        print("True")

    import spacy

    import spacy
    from spacy.lang.fr.examples import sentences

    # nlp = spacy.load("fr_dep_news_trf")
    nlp = spacy.load("fr_core_news_lg")
    # nlp = spacy.load("en_core_web_trf")
    # doc = nlp("Lymphocyte T positif pour l'antigène CD4 et l'antigène HLA-DR")
    # doc = nlp("PEComa bénin")
    # doc = nlp("Bone structure of right navicular")
    # doc = nlp("ISO designation 48")
    doc = nlp("Gène IL23A")
    print(doc.text)
    for token in doc:
        print(token.text, token.pos_, token.dep_)


def detect_acronym(st_en, st_fr):
    acronyms = []

    spl = st_en.split()
    spl_fr = st_fr.split()
    if len(spl)==1 and len(spl_fr)==1:
        if spl[0].isupper():
            return st_fr.upper()

    found_u = False
    for ind, s in enumerate(spl):
        if s.isupper():
            # if s.upper() in spl_fr or s.lower() in spl_fr:
            for ind_fr, s_fr in enumerate(spl_fr):
                if s.upper()==s_fr.upper():
                    spl_fr[ind_fr] = s_fr.upper()

        if s.islower():
            if s.upper() in spl_fr or s.lower() in spl_fr:
                for ind_fr, s_fr in enumerate(spl_fr):
                    if s.lower()==s_fr.lower():
                        spl_fr[ind_fr] = s_fr.lower()

    for ind_fr, s_fr in enumerate(spl_fr):
        if s_fr.isupper():
            for ind, s in enumerate(spl):
                if s.lower()==s_fr.lower() and s.upper() not in spl:
                    spl_fr[ind_fr] = s_fr.lower()

    for ind_fr, s_fr in enumerate(spl_fr):
        for ind, s in enumerate(spl):
            if s.strip(".").strip(",").lower()==s_fr.strip(".").strip(",").lower():
                s_tok = nltk.word_tokenize(s.strip(".").strip(","))
                s_pos = nltk.pos_tag(s_tok)
                # if s_fr!="structure" and s_fr!="tendon" and s_fr!="muscle" and s_fr!="base":
                if s_pos[0][1]=="NNP":
                    spl_fr[ind_fr] = s.strip(".").strip(",")


    ## check for symbols
    for ind, s in enumerate(spl):
        if "-" in s or "+" in s:
            for ind_fr, s_fr in enumerate(spl_fr):
                if "-" in s or "+" in s_fr:
                    if s.lower()==s_fr.lower() and s.upper()==s_fr.upper():
                        spl_fr[ind_fr] = s

    st_fr = " ".join(spl_fr)

    return st_fr


def get_all_acronyms_english():
    from os import listdir
    from os.path import isfile, join

    mypath = '../medical_abbreviations-master/CSVs'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    acronyms = []
    for f in onlyfiles:
        df = pd.read_csv(f,sep=",")

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            label = str(row['Abbreviation/Shorthand'])
            acronyms.append(label)

    return acronyms


def casePresent(str):

    # ReGex to check if a string
    # contains uppercase, lowercase
    # special character & numeric value
    regex = ("^(?=.*[a-z])(?=." + "*[A-Z])")

    # Compile the ReGex
    p = re.compile(regex)

    # If the string is empty
    # return false
    if (str == None):
        print("No")
        return

    # Print Yes if string
    # matches ReGex
    if(re.search(p, str)):
        print("Yes")
    else:
        print("No")


def singular_or_plural_fr(token):
    # from pattern.fr import singularize, pluralize
    import pluralizefr
    if pluralizefr.pluralize(token) == token:
        return 'plural'
    else:
        return 'singular'


def general_ortho(st_fr):
    # import nltk
    # nltk.download('averaged_perceptron_tagger')
    from nltk import word_tokenize
    from nltk.tag import pos_tag
    #
    text = word_tokenize(st_fr)
    tagged = pos_tag(text)
    # print(tagged)
    # tense = {}
    # tense["future"] = len([word for word in tagged if word[1] == "MD"])
    # tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
    # tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]])
    # print(tense)

    # import spacy
    # nlp = spacy.load("fr_core_news_lg")
    doc = nlp(st_fr)
    for token in doc:
        print(token.text,list(token.morph), token.lemma_)

    for token in doc:
        print(token.text, token.pos_, token.dep_)

    # return(tense)


def check_for_mg(st):
    check = st.split("m")
    print(check)
    if check[len(check)-1] == "g":
        print("already in mg")
    else:
        step = st.split(" ")
        step2 = step[len(step)-1].split("mg")
        newstring = " ".join(step)
        step3 = step2[0].split("g")
        step4 = float(step3[0])
        newstring = newstring.replace(step2[0], str(step4*10000)+"mg")
        print(newstring)
    return newstring


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def singular_or_plural(token):
    return 'plural' if pluralize(token) == token else 'singular'

def check_if_1st_letter_upper_and_lower_and_change(st_en, st_fr):
    if st_en[0].isupper():
        st_fr = st_fr[0].upper()+st_fr[1:]
    if st_en[0].islower():
        st_fr = st_fr[0].lower()+st_fr[1:]
    return st_fr

def replace_1st_letter_with_upper(st):
    st = st[0].upper() + st[1:]
    return st

def replace_nos_with_sai(st):
    if " NOS" in st:
        st = st.replace(" NOS", " SAI")
    return st

def replace_1st_letter_with_lower(st_en, st_fr):
    if not st_fr.split()[0].isupper():
        doc = nlp(st_fr)
        if doc[0].pos_=="NOUN" or doc[0].pos_=="ADJ" or doc[0].pos_=="ADP":
            st_fr = st_fr[0].lower() + "".join(st_fr[1:])

        if doc[0].pos_=="PROPN":
            st_fr = st_fr[0].upper() + "".join(st_fr[1:])

        # regexp = re.compile('[^0-9a-zA-Z]+')
        # if regexp.search(word):
        #     special_char = True
        #
    return st_fr


def remove_space_after_acc():
    st = st.replace("' ","'")
    return st


def remove_space_after_symb(st):
    st = st.replace(" / ","/")
    return st
