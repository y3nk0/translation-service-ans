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

stopwords = set(nltk.corpus.stopwords.words('french'))

nlp = spacy.load("fr_core_news_lg")
# nlp = spacy.load("fr_dep_news_trf")

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
    # changed = False
    # if st_en.split()[0].lower()==st_fr.split()[0].lower():
    #     if not st_en.split()[0].isupper():
    #         st_fr = st_fr[0].lower() + "".join(st_fr[1:])
    #         changed = True
    # else:
    #     if len(st_en.split())>1:
    #         if st_en.split()[1].lower()==st_fr.split()[0].lower():
    #             if not st_en.split()[1].isupper():
    #                 st_fr = st_fr[0].lower() + "".join(st_fr[1:])
    #                 changed = True
    #
    # if not changed:
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


def bs1_replace_with_right_trans(st_en, st_fr):
    if st_en.endswith("(body structure)") and st_fr.endswith(", sai"):
        st_fr = st_fr.replace(", sai"," (structure corporelle)")
    elif st_en.endswith("(disorder)") and st_fr.endswith(", sai"):
        st_fr = st_fr.replace(", sai"," (désordre)")
    elif st_en.endswith("(cell structure)") and st_fr.endswith(", sai"):
        st_fr = st_fr.replace(", sai"," (structure cellulaire)")
    elif st_en.endswith("(cell)") and st_fr.endswith(", sai"):
        st_fr = st_fr.replace(", sai"," (cellule)")
    elif st_fr.endswith(", sai"):
        st_fr = st_fr.replace(", sai","")

    if "body structure" in st_en or "Body structure" in st_en:
        if "structures du corps" in st_fr:
            st_fr = st_fr.replace("structures du corps", "structure corporelle")
        elif "structure du corps" in st_fr:
            st_fr = st_fr.replace("structure du corps", "structure corporelle")

    if "structure" in st_fr and "du corps" in st_fr:
        st_fr = st_fr.replace("structure", "structure corporelle")
        st_fr = st_fr.replace("du corps", "")

    return st_fr


def bs2_replace(st_en, st_fr, bs_dict):

    for key in bs_dict:

        if key in st_fr:
            st_fr = st_fr.replace(key,bs_dict[key])

        if key=="^structure articulaire$":
            if st_fr=="structure articulaire":
                st_fr = "structure d'une articulation"

        if key=="^structure articulaire$":
            if st_fr.startswith("structure articulaire"):
                st_fr = st_fr.lstrip("structure articulaire").strip()
                st_fr = "structure de l'articulation "+ st_fr

    return st_fr


def bs3_replace(st_en, st_fr, bs_dict, prop):

    # if "structure" in st_fr:
    #     if "prefLabel"==prop:
    #         st_fr = st_fr.replace("structure","")

    found = False
    for key in bs_dict:
        if key==st_fr:
            st_fr = st_fr.replace(key,bs_dict[key])
            found=True

    # for key in bs_dict:
    if not found:
        changes = {}
        for key in bs_dict:

            if "X" in key:
                if "/" in bs_dict[key]:
                    bs_dict[key] = bs_dict[key].split("/")[0]

            if "X" in key:
                if key.replace("X","").replace("Y","").strip() in st_fr:

                    if key[-1]=="X" and bs_dict[key][-1]=="X":
                        st_fr = st_fr.replace(key[0:-1], bs_dict[key][0:-1])
                        changes[key] = st_fr

                    if key[-1]=="X" and bs_dict[key][0]=="X":
                        st_fr = st_fr.replace(key[0:-1],"")
                        st_fr = st_fr+" "+bs_dict[key][1:]
                        changes[key] = st_fr

                if "Y" in key:
                    if key[0]=="X" and key[-1]=="Y":
                        st_fr = st_fr.replace(key[1:-1],bs_dict[key][1:-1])
                        changes[key] = st_fr

            if key in st_fr:
                st_fr = st_fr.replace(key,bs_dict[key])
                changes[key] = st_fr


        new_changes = []
        for c in changes:
            found = False
            last_cf = ""
            for cf in changes:
                if c in cf.replace("la","").replace("le","").replace("les","").replace("l'","") and c!=cf and len(cf)>len(c):
                    found=True
                    last_cf = cf

            if found:
                if last_cf not in new_changes:
                    new_changes.append(last_cf)
            else:
                if c not in new_changes:
                    new_changes.append(c)

        for c in new_changes:
            st_fr = st_fr.replace(c,changes[c])

    return st_fr


def bs5_replace(st_fr):
    # "region" must be translated as "région"
    st_fr = st_fr.replace("region", "région")
    return st_fr


def bs6_replace(st_fr):
    # """zone"" must be translated as ""zone""
    # ""area"" must be translated as ""zone"", ""surface"" or ""aire"""
    st_fr = st_fr.replace("area", "zone")
    return st_fr


def bs8_replace(st_fr, prop):
    ## "apex" must be translated as "apex" in prefLabel. "pointe", "bout" or "cime" can be used in altLabel
    if "alt" in prop:
        st_fr = st_fr.replace("apex", "pointe")
    return st_fr


def ss1(st_en, st_fr):
    st_all = st.split()
    if len(st_all)==1:
        if st_en[0].isupper():
            st_fr = st_fr[0].upper()+st_fr[1:]
        if st_en[0].islower():
            st_fr = st_fr[0].lower()+st_fr[1:]
        return st_fr
    # else:
    #     for st in st_all:
    #         st_fr[]

    return st

# def ss2(st):
#     # mod = spacy.load('fr_core_news_md', disable=['parser', 'textcat', 'ner', 'tagger'])
#     # assert singular_or_plural("garçons") == 'plural'
#     # assert singular_or_plural("chienne") == 'singular'
#
#     return st

def ar2_remove_article_from_start(st_fr):
    ## this is to catch ar7

    if st_fr.startswith("l' "):
        st_fr = st_fr.strip("l' ")

    if st_fr.startswith("l'"):
        st_fr = st_fr.strip("l'")

    if st_fr.startswith("le "):
        st_fr = st_fr.strip("le ")

    if st_fr.startswith("la "):
        st_fr = st_fr.strip("la ")

    st = st_fr.split()

    if st[0] in stopwords:
        st = st[1:]

    if st[0]=="deux":
        st[0] = "les deux"

    st = " ".join(st)

    return st


def ar7_plural_if_both_or_all(st_en, st_fr, ar7_dict_pref, ar7_dict_alt, ar7_dict_eng, prop): ## to check
    # import pluralizefr
    # if "both" in st_en or "all" in st_fr:
    #     for ind, word in enumerate(st_fr.split()):
    #         st_fr[ind] = pluralizefr.plularize(word)

    ar7_dict = ar7_dict_pref
    if "alt" in prop:
        ar7_dict = ar7_dict_alt

    found = False
    for key in ar7_dict:
        if key==st_fr:
            if key in ar7_dict_eng:
                if ar7_dict_eng[key] in st_en:
                    st_fr = st_fr.replace(key,ar7_dict[key])
            else:
                st_fr = st_fr.replace(key,ar7_dict[key])
            found=True

    if not found:
        changes = []
        for key in ar7_dict:
            if key in st_fr:

                if key in ar7_dict_eng:
                    if ar7_dict_eng[key] in st_en:
                        # st_fr = st_fr.replace(key,ar7_dict_pref[key])
                        changes.append(key)
                else:
                    # st_fr = st_fr.replace(key,ar7_dict_pref[key])
                    changes.append(key)

        new_changes = []
        for c in changes:
            found = False
            last_cf = ""
            for cf in changes:
                if c in cf and c!=cf and len(cf)>len(c):
                    found=True
                    last_cf = cf

            if found:
                if last_cf not in new_changes:
                    new_changes.append(last_cf)
            else:
                if c not in new_changes:
                    new_changes.append(c)

        for c in new_changes:
            st_fr = st_fr.replace(c,ar7_dict[c])


        # for key in ar7_dict_pref:
        #     if key in st_fr:
        #
        #         if ar7_dict_eng[key]!="":
        #             if ar7_dict_eng[key] in st_en:
        #                 st_fr = st_fr.replace(key,ar7_dict_pref[key])
        #         else:
        #             st_fr = st_fr.replace(key,ar7_dict_pref[key])


    return st_fr


def ll1_replace(st_fr, ll1_dict):
    for key in ll1_dict:
        if key in st_fr:
            st_fr = st_fr.replace(key,ll1_dict[key])

    return st_fr

def custom_replace(st_fr, custom_dict):
    for key in custom_dict:
        if key in st_fr:
            st_fr = st_fr.replace(key,custom_dict[key])

    return st_fr

def ab1_repl_words_with_periods_with_fulls(st):
    if "." in st:
        st = st.replace("","")
    return st


def sc1_roman_uppercase(st_fr):
    """Uppercase roman numerals"""

    ROMAN_CONSTANTS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC" ,
            "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM", "M", "MM", "MMM"]

    spl = st_fr.split()
    for ind, word in enumerate(spl):
        if word.upper() in ROMAN_CONSTANTS:
            spl[ind] = word.upper()
    return " ".join(spl)


def um2_temperature(st_orig):
    st = st_orig.split()
    if "Celsius" in st:
        ind = st.index("Celsius")
        if st[ind-1]!="degrés":
            st_fr = st_orig.replace("Celsius","degrés Celsius")
            return st_fr
    return st_orig


def um8_repl_unit(st):
    """Rule um8
    The letter μ meaning micro in a unit is replaced by the letter u.
    """
    if 'mol / L' in st:
        st = st.replace("μ","u")
    return st


def sc6_replace_comp_symbols(st):
    """Rule sc6
    """
    if ' > ' in st:
        st = st.replace(" > "," supérieure à ")
    if ' >= ' in st:
        st = st.replace(" >= "," supérieure ou égale à ")
    if ' < ' in st:
        st = st.replace(" < "," inférieure à ")
    if ' <= ' in st:
        st = st.replace(" <= "," inférieure ou égale à ")
    return st


def gr1_replace_greek_letter_with_long(st):
    """Rule gr1
    Greek letters are written in long form.
    """
    greek_letters = {}
    greek_letters['α'] = 'alpha'
    greek_letters["β"] = 'beta'
    greek_letters["γ"] = 'gamma'
    greek_letters['δ'] = 'delta'
    greek_letters['ε'] = 'epsilon'

    greek_letters['μ'] = 'mu'
    greek_letters['σ'] = 'sigma'

    for let, initial in greek_letters.items():
        if let in st:
            st = st.replace(let, initial)
    return st


def or1_orthographe_replace(st):
    """
    """
    if "uë" in st:
        st = st.replace("uë", "üe")
    if "uï" in st:
        st = st.replace("uï", "üi")
    if "oe" in st:
        st = st.replace("oe", "œ")
    if "ae" in st:
        st = st.replace("ae", "æ")

    return st


def or2_accent(st_fr):

    doc = nlp(st_fr)
    for token in doc:
        prev = str(token)
        if token.pos_=="VERB":
        # print(token.text,list(token.morph), token.lemma_)
            regexp = re.compile(r'é[bcdfklmnpqrstvz]er')
            # import ipdb; ipdb.set_trace()
            match = regexp.search(str(token))
            if match:
                sp = match.span()
                # import ipdb; ipdb.set_trace()
                st_token = prev[sp[0]:sp[1]]
                for st in list(token.morph):
                    if 'Tense' in st:
                        s = st.strip("'").split("=")[1]
                        if 'Fut' in s:
                            st_token = prev[0:sp[0]] + prev[sp[0]].replace("é","è") + prev[sp[0]+1:]
                            print("st_token:"+st_token)

                            st_fr = st_fr.replace(prev, st_token)

                    if 'Mood' in st:
                        s = st.strip("'").split("=")[1]
                        if 'Cnd' in s:
                            st_token = prev[0:sp[0]] + prev[sp[0]].replace("é","è") + prev[sp[0]+1:]
                            print("st_token:"+st_token)

                            st_fr = st_fr.replace(prev, st_token)

            regexp = re.compile(r'e[bcdfklmnpqrstvz]er')
            match = regexp.search(str(token))
            if match:
                sp = match.span()
                st_token = prev[0:sp[0]] + prev[sp[0]].replace("e","é") + prev[sp[0]+1:]
                print("st_token:"+st_token)

                st_fr = st_fr.replace(prev, st_token)

    ## evenement
    if st_fr[0]=="é":
        st_fr = st_fr[0] + st_fr[1] + st_fr[2].replace("é","è") + "".join(st_fr[3:])

    if "é-je" in st_fr:
        st_fr = st_fr.replace("é-je","è-je")
    return st_fr


def or3_or4_or5_hyphen(st_fr, hyphen_dict_pref, hyphen_dict_alt, prop):
    from nltk import word_tokenize
    from nltk.tag import pos_tag

    hyphen_dict = hyphen_dict_pref
    if "alt" in prop:
        hyphen_dict = hyphen_dict_alt

    found = False
    for key in hyphen_dict:
        if key==st_fr:
            st_fr = st_fr.replace(key,hyphen_dict[key])
            found=True

    if not found:
        changes = []
        for key in hyphen_dict:
            if key in st_fr:
                changes.append(key)

        new_changes = []
        for c in changes:
            found = False
            last_cf = ""
            for cf in changes:
                if c in cf and c!=cf and len(cf)>len(c):
                    last_cf = cf
                    found=True

            if found:
                if last_cf not in new_changes:
                    new_changes.append(last_cf)
            else:
                if c not in new_changes:
                    new_changes.append(c)

        for c in new_changes:
            st_fr = st_fr.replace(c,hyphen_dict[c])

    pws = ['demi', 'mi', 'semi', 'ex', 'sous', 'vice', 'non']

    st_spl = st_fr.split()
    for ind, word in enumerate(st_spl):
        if "-" in word:
            spl = word.split("-")
            if spl[0] not in pws and spl[0]!="" and len(spl)>1 and spl[1]!="" :
                tag0 = pos_tag(word_tokenize(spl[0]))[0]
                tag1 = pos_tag(word_tokenize(spl[1]))[0]

                if len(spl)>2:
                    continue

                if spl[0].isdigit() or spl[1].isdigit() or has_numbers(spl[0]) or has_numbers(spl[1]):
                    continue

                if spl[0][-1]+spl[1][0]=="ou" or spl[0][-1]+spl[1][0]=="ii" or spl[0][-1]+spl[1][0]=="éh" or spl[0][-1]+spl[1][0]=="io":
                    continue

                # import ipdb; ipdb.set_trace()
                nlp_word = nlp(" ".join(spl))
                found = False
                for ent in nlp_word:
                    if ent.pos_=="PROPN":
                        found = True
                        break

                if found:
                    break
                # else:
                #     # import ipdb; ipdb.set_trace()
                #     if tag0[0]!="NNP" and tag0[1]!="NNS" and tag1[1]!="NNP" and tag1[1]!="NNS":
                #         st_spl[ind] = word.replace("-","")

    st_fr = " ".join(st_spl)


    return st_fr


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


def disorder_fix(st_en,st_fr):
    """
    "The word 'disorder' appearing in the preferred English term is translated as appropriate by:
      - trouble (if the object is a function),
      - affection (if the object is a body structure),
      - disorder or anomaly or nothing in other cases"

      Eating disorder -> trouble de l’alimentation
      Sleep disorder -> trouble du sommeil
      Developmental disorder -> trouble du développement Disorder of skin -> affection de la peau
      Lung disorder -> affection du poumon
      Rectal disorder -> affection rectale
      Disorder of electrolytes -> désordre hydroélectrolytique
      Chromosomal disorder -> anomalie chromosomique
      Disorder of acid-base balance -> déséquilibre acidobasique"
    """

    # import scispacy
    import spacy

    # nlp = spacy.load("en_ner_bionlp13cg_md")
    # nlp = spacy.load("en_core_sci_md")

    functions = ['eat', 'sleep', 'develop']
    structures = ['lung', 'rectal', 'skin']
    anomalies = ['chromosomal']
    if "disorder" in st_en:
        trans = "désordre"
        found = False
        for function in functions:
            if function.lower() in st_en.lower():
                trans = "trouble"
                found = True
                break

        if not found:
            for structure in structures:
                if structure.lower() in st_en.lower():
                    trans = 'affection'
                    found = True
                    break

        if not found:
            if "structure corporelle" in st_fr:
                trans = 'affection'
                found = True

        if not found:
            f = st_en.split()[0].lower()
            for anom in anomalies:
                if anom.lower() == f:
                    trans = 'anomalie'
                    found = True

        # if found = False:
        #     trans = 'anomalie'

        if "trouble" in st_fr or "Trouble" in st_fr:
            st_fr = st_fr.replace("trouble",trans)
            st_fr = st_fr.replace("Trouble",trans[0].upper()+""+trans[1:])
        if "affection" in st_fr or "Affection" in st_fr:
            st_fr = st_fr.replace("affection",trans)
            st_fr = st_fr.replace("Affection",trans[0].upper()+""+trans[1:])
        # import ipdb; ipdb.set_trace()
    return st_fr


def pa2_disorder_fix_with_embeddings():
    """
    check if function or structure with topics from word embeddings or with distance to keyword
    """
    return


def ss1_ss2_rule(line_en, line):
    import inflect
    import pluralizefr
    from nltk.tag import pos_tag

    po = inflect.engine()

    # for line in lines:
    # for index, row in df.iterrows():
    # line = lines[index].strip()
    # line = line.replace(" - ","-").strip()
    line = line.replace(" / ","/")
    line = line.replace("~ ","~")
    prev_line = line.split().copy()

    # if row['Type']=="def":
    #     line = line[0].upper() + "".join(line[1:])
    # else:
    # FIX case
    line_s = line.split().copy()

    ### check extreme cases
    line_en_s = line_en.split()

    if not line_s[0].isupper():
        # line = line_s[0].lower() + " " + " ".join(line_s[1:])

        if line_en_s[0]==line_s[0]:
            if line_en_s[0].strip("s").isupper():
                line = prev_line[0] + " " + " ".join(line_s[1:])

        if line_en_s[0].split("-")[0]==line_s[0].split("-")[0]:
            if line_en_s[0].split("-")[0].isupper():
                line = prev_line[0] + " " + " ".join(line_s[1:])

    if line_en_s[0].isupper() and line_en_s[0].upper()==line_s[0].upper():
        line = line_s[0].upper() + " " + " ".join(line_s[1:])


    if " - " in line_en and " - " not in line:
        line_en_s = line_en.split(" - ")
        if line_en_s[0] in line:
            line = line.replace(line_en_s[0]+"-",line_en_s[0]+" - ")

    # FIX singular
    # single_en_all = True
    # for word in line_en.split():
    #     if po.singular_noun(word)!=False:
    #     # if po.plural(word)==word:
    #         single_en_all = False
    #         break
    #
    # lemma_tags = ["NNS", "NNPS"]
    # single_fr = True
    # if single_en_all:
    #
    #     # for token in nlp(line):
    #     for word in line.split():
    #         if singular_or_plural_fr(word)=='plural':
    #             # print(word, singular_or_plural_fr(word))
    #         # lemma = token.text
    #         # print(lemma, token.tag_)
    #         # if token.tag_ in lemma_tags:
    #             # lemma = token.lemma_
    #             single_fr = False
    #             break
    #
    #     ## if english is singular and french not singular
    #
    #     if not single_fr:
    #         sent = ""
    #         # line_n = nlp(line)
    #         for word in line.split():
    #         # for token in nlp(line):
    #
    #             # ptag = pos_tag(word)
    #             # if 'NN' in ptag:
    #             # word = token.text
    #             # print(word, token.tag_)
    #             # if token.tag_!="":
    #                 # word = token.lemma_
    #                 # sent += " " + word
    #             if "eux" not in word:
    #                 sent += " " + pluralizefr.singularize(word)
    #             else:
    #                 sent += " " + word
    #
    #         line = sent.strip()

    return line


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


def test_pluralize():
    import pluralizefr
    import inflect
    p = inflect.engine()

    sent = 'Lymphoma Cell to Total Cell Ratio Measurement'
    for word in sent.split():
        print(word+" "+str(p.singular_noun(word)))

    sent = "Mesure du ratio lymphome-cellules totales"
    for word in sent.split():
        # pluralizefr.pluralize("fromage") # return fromages
        # print(pluralizefr.singularize(word)) # return fromage
        print(word+" "+singular_or_plural_fr(word))
        print(pluralizefr.singularize(word))


def my_pos_tag():
    from nltk.tag import StanfordPOSTagger
    jar = '../../stanford-postagger-full-2020-11-17/stanford-postagger-4.2.0.jar'
    model = '../../stanford-postagger-full-2020-11-17/models/french-ud.tagger'
    import os
    java_path = "/usr/local/opt/openjdk/bin/java"
    os.environ['JAVAHOME'] = java_path

    pos_tagger = StanfordPOSTagger(model, jar, encoding='utf8' )
    res = pos_tagger.tag('je suis libre'.split())
    print(res)


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

# class TestSum(unittest.TestCase):
#
#     def test_sum(self):
#         self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")
#
#     def test_sum_tuple(self):
#         self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")
#
# if __name__ == '__main__':
#     unittest.main()

def rule_CS(st_en, st_fr):
    if ", CS" in st_en:
        st_fr = st_fr.replace(", CS",", site combiné")
        st_fr = st_fr.replace(" CS",", site combiné")
    st_fr = st_fr.replace(" (site combiné)",", site combiné")
    return st_fr


def rule_azygos(st_en, st_fr):
    if "mineur azygos" in st_fr:
        st_fr = st_fr.replace("du mineur azygos","de la petite veine azygos")
        st_fr = st_fr.replace("d'un mineur azygos","d'une petite veine azygos")
        st_fr = st_fr.replace("mineur azygos","la petite veine azygos")
        st_fr = st_fr.replace("petit azygos","la petite veine azygos")
    if "du petit azygos" in st_fr:
        st_fr = st_fr.replace("du petit azygos","de la petite veine azygos")
    return st_fr

def rule_compatible(st_en, st_fr):
    if "compatible with age and history" in st_en:
        st_fr = st_fr.replace("compatible avec l'âge et l'histoire","cohérent avec l'âge et les antécédents")

    return st_fr

def rule_female(st_en, st_fr):
    if "female individual" in st_en or "Female individual" in st_en:
        st_fr = st_fr.replace("individu féminin","individu de sexe féminin")

    if ", female" in st_en:
        st_fr = st_fr.replace("chez la femme",", de la femme")

    # if st_en=="female" or st_en=="Female":
    #     st_fr = st_fr.replace("mâle","femme")
    #     st_fr = st_fr.replace("Mâle","femme")

    if "féminin" in st_en and "female individual" not in st_fr:
        st_fr = st_fr.replace(" féminine "," de la femme ")
        st_fr = st_fr.replace(" féminine"," de la femme")
        st_fr = st_fr.replace(" féminin "," de la femme ")
        st_fr = st_fr.replace("féminin","de la femme")

    if " féminin " in st_fr and "female individual" not in st_fr:
        st_fr = st_fr.replace(" féminin "," de la femme ")

    if " féminin" in st_fr and "female individual" not in st_fr:
        st_fr = st_fr.replace(" féminin"," de la femme")

    if " féminins" in st_fr and "female individual" not in st_fr:
        st_fr = st_fr.replace(" féminins"," de la femme")

    return st_fr


def rule_male(st_en, st_fr):

    if "male individual" in st_en or "Male individual" in st_en:
        st_fr = st_fr.replace("individu mâle","individu de sexe masculin")

    if st_en=="male" or st_en=="Male":
        st_fr = st_fr.replace("mâle","homme")
        st_fr = st_fr.replace("Mâle","homme")

    # if " male " in st_en or " male" in st_en or " male":
    if "masculine" in st_fr and "Male individual" not in st_en:
        st_fr = st_fr.replace(" masculine "," de l'homme ")
        st_fr = st_fr.replace(" masculine"," de l'homme")
        st_fr = st_fr.replace("masculine"," de l'homme ")

    if "masculin" in st_fr and "Male individual" not in st_en:
        st_fr = st_fr.replace(" masculin "," de l'homme ")
        st_fr = st_fr.replace("masculin","de l'homme")

    if " mâle " in st_fr:
        st_fr = st_fr.replace(" mâle "," de l'homme ")

    if " mâles " in st_fr:
        st_fr = st_fr.replace(" mâles "," de l'homme ")

    if " mâle" in st_fr:
        st_fr = st_fr.replace(" mâle"," de l'homme")

    if " mâles" in st_fr:
        st_fr = st_fr.replace(" mâles"," de l'homme")

    return st_fr

def rule_male_to_female(st_en, st_fr):
    if ", male to female sex-change" in st_en:
        st_fr = st_fr.replace(", changement de sexe homme-femme", ", chirurgie de confirmation de genre homme vers femme")
    if ", female to male sex-change" in st_en:
        st_fr = st_fr.replace(", changement de sexe femme-homme",", chirurgie de confirmation de genre femme vers homme")
    return st_fr

def rule_atrial(st_en, st_fr):
    if "atrial" in st_en:
        st_fr = st_fr.replace("de l'oreillette","auriculaire")
    return st_fr

def rule_fallopian(st_en, st_fr, type_label):
    if "fallopian tube" in st_en or "Fallopian tube" in st_en:
        if "prefLabel" in type_label:
            st_fr = st_fr.replace("trompe de fallope","salpinx")
            st_fr = st_fr.replace("trompes de fallope","salpinx")
        else:
            st_fr = st_fr.replace("trompes de fallope","trompes de Fallope")
            st_fr = st_fr.replace("trompe de fallope","trompe de Fallope")

    return st_fr

def rule_lens(st_en, st_fr):
    if "lens" in st_en or "Lens" in st_en:
        st_fr = st_fr.replace("lentille","cristallin")
        st_fr = st_fr.replace("verre","cristallin")
    return st_fr

def rule_fluid(st_en, st_fr):
    if "in portion of fluid" in st_en:
        st_fr = st_fr.replace("en portion de liquide","dans une partie du liquide")
        st_fr = st_fr.replace("dans une portion de liquide","dans une partie du liquide")
        st_fr = st_fr.replace("d'une portion de liquide","dans une partie du liquide")
        st_fr = st_fr.replace("d'une portion liquidienne","dans une partie du liquide")
    return st_fr


def run_tests():

    st_fr = "aiguë"
    print(or1_orthographe_replace(st_fr))

    # st_fr = "je vendrai"
    # print(general_ortho(st_fr))
    #
    # st_fr = "je cède"
    # print(general_ortho(st_fr))
    #
    # st_fr = "je céderai"
    # ## right je cèderai
    # print(general_ortho(st_fr))

    st_fr = "je céderai"
    print(or2_accent(st_fr))

    st_fr = "receler"
    print(or2_accent(st_fr))

    st_fr = "événement"
    # print(general_ortho(st_fr))
    print(or2_accent(st_fr))

    st_fr = "post-opératoire"
    print(or3_or4_or5_hyphen(st_fr))

    st_fr = "fracture de Mason type ii"
    print(sc1_roman_uppercase(st_fr))

    st_en = "Seizure free > 12 months (finding)"
    st_fr = "Épilepsie libre > 12 mois (résultat)"
    print(sc6_replace_comp_symbols(st_fr))

    sent = 'PUL (body structure)'
    sent_fr = 'pUL (structure du corps)'
    print(ss1_ss2_rule(sent, sent_fr))

    sent = 'Lymphoma Cell to Total Cell Ratio Measurement'
    sent_fr = "Mesure du ratio lymphome-cellules totales"
    print(ss1_ss2_rule(sent, sent_fr))

    # st_fr = "Michael Jackson aime manger chez McDonalds"
    # print(general_ortho(st_fr))

    # st_fr = "je cèderai"
    # print(general_ortho(st_fr))

    st_en = "Structure of endometrial glandular cell (cell structure)"
    st_fr = "cellule glandulaire de l'endomètre, sai"
    print(replace_with_right_trans(st_en, st_fr))

    st_en = "Structure of submental space (body structure)"
    st_fr = "espace sous-mentonnier, sai"
    print(replace_with_right_trans(st_en,st_fr))

    st_en = "fever greater than 38 Celsius"
    st_fr = "fièvre supérieure à 38 Celsius"
    print(um2_temperature(st_fr))

    st_en = "Chromosomal disorder (disorder)"
    st_fr = "trouble chromosomique (trouble)"
    print(disorder_fix(st_en,st_fr))

    st_en = "Occupational lung disorder"
    st_fr = "trouble pulmonaire professionnel"
    print(disorder_fix(st_en,st_fr))

    st_fr = 'anastomose de Roux-en-Y (anomalie morphologique)'
    print(or3_or4_or5_hyphen(st_fr))

    st_fr = 'rosette de Homer-Wright'
    print(or3_or4_or5_hyphen(st_fr))

    st_en = 'Entire body region (body structure)'
    st_fr = 'région corporelle dans son ensemble (structure du corps)'
    print(ss1_ss2_rule(st_en, sent_fr))
    # detect_acronym(st_en, st_fr)

    st_en = "Nasal sinus structure (body structure)"
    st_fr = "sinus nasal, sai"
    print(ss1_ss2_rule(st_en, st_fr))

    st_en = "BRA - Branch retinal artery"
    st_fr = "bRA -branche de l'artère rétinienne"
    print(detect_acronym(st_en, st_fr))

    # st_en = "30 to 39 percent of body surface (body structure)"
    # st_fr = "sigma"
    #
    # st_en = "80 to 89 percent of body surface (body structure)"
    # st_fr = ""

    ## correct sigma

    st_en = "Colliquative necrosis"
    st_fr = "nécrose COLLIQUATIVE"
    print(detect_acronym(st_en, st_fr))

    st_en = "Nucleus pulposus, T8-T9"
    st_fr = "nucléus pulposus T8-T9"
    print(detect_acronym(st_en, st_fr))


def snomed_testing():

    bs2_dict = {}
    df_bs = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="bs2")
    df_bs = df_bs.fillna('')
    for index, row in df_bs.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation_prefLabel'])
        if corr!="":
            bs2_dict[err] = corr

    bs3_dict = {}
    df_bs = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="bs3")
    df_bs = df_bs.fillna('')
    for index, row in df_bs.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation_prefLabel'])
        if corr!="":
            bs3_dict[err] = corr

    hyphen_dict_pref = {}
    hyphen_dict_alt = {}
    df_hyphen = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="Hyphen")
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
    df_ar = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="ar7")
    df_ar = df_ar.fillna('')
    for index, row in df_ar.iterrows():
        err = str(row['erroneous_translation'])
        eng = str(row['english_term'])
        corr = str(row['correct_translation_prefLabel'])
        corr_alt = str(row['correct_translation_altLabel'])

        if err!="" and corr!="":
            ar7_dict_pref[err] = corr
        if err!="" and corr_alt!="":
            ar7_dict_alt[err] = corr_alt
        if err!="" and eng!="":
            ar7_dict_eng[err] = eng

    ll1_dict = {}
    df_ll1 = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="ll1")
    df_ll1 = df_ll1.fillna('')
    for index, row in df_ll1.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation'])
        if corr!="":
            ll1_dict[err] = corr

    custom_dict = {}
    df_custom = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="custom")
    df_custom = df_custom.fillna('')
    for index, row in df_custom.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation'])
        if corr!="":
            custom_dict[err] = corr

    to_check = [['BO - Buccal-occlusal',
                'BO - bucco-occlusal',
                'BO - bucco-occlusal'],

            ['Acute myeloid leukaemia, CBF-beta/MYH11',
            'Leucémie myéloïde aiguë, CBF-bêta / MYH11',
            'leucémie myéloïde aigüe, CBF-bêta/MYH11'],

            ['Structure of left bony anterior semicircular canal',
            'canal semi-circulaire antérieur osseux gauche',
            'canal semi-circulaire antérieur osseux gauche'],

            ['Noninvasive carcinoma ex pleomorphic adenoma',
            'carcinome non invasif ex-adénome pléomorphe',
            'carcinome non invasif ex-adénome pléomorphe'],

            ['Dorsal subaponeurotic space',
            'espace sous-aponévrotique dorsal',
            'espace sous-aponévrotique dorsal'],

            ['Entire ossicle of ear',
            "ossicule entier de l'oreille",
            "ossicule entier de l'oreille"],

            ['Cell positive for CD16 antigen and CD56 antigen',
            "cellules positives pour l'antigène CD16 et l'antigène CD56",
            "cellule positive pour les antigènes CD16 et CD56"],

            ['Body structure',
            'structures du corps',
            'structure corporelle'],

            ['Body structure, altered from its original anatomical structure',
            'structure du corps, altérée de sa structure anatomique originale',
            'structure corporelle, altérée de sa structure anatomique originale'],

            ['Entire posterior cruciate ligament of knee joint',
            "ligament croisé postérieur entier de l'articulation du genou",
            "ligament croisé postérieur entier de l'articulation du genou"],

            ['Entire left finger','doigt gauche entier','doigt gauche entier'],

            ['All iliac crest bone marrow','toute moelle osseuse de la crête iliaque.','toute la mœlle osseuse de la crête iliaque'],

            ['Adnexal naevus','naevus annexiel','nævus annexiel'],

            ['Eye', 'oeil', 'œil'],

            ['DI - Distal-incisal',	'DI-Distal-incisal', 'DI - Distal-incisal'],

            ['DO - Distal-occlusal','dO - distal-occlusal','DO - distal-occlusal'],

            ['MOD - Mesial-occlusal-distal', 'MOD - mésio-occlusal-distal', 'MOD - mésio-occlusal-distal'],

            ['Auditory ossicle', 'osselets auditifs', 'osselet auditif'],

            ["Structure of clivus ossis sphenoidalis", "structure du clivus de l'os sphénoïde", "structure du clivus de l'os sphénoïde"],

            ["PEComa, benign", "PEComa bénin", "PEComa bénin"],

            ['Entire semispinalis thoracis', 'altLabel', 'semispinalis thoracis entier', 'musculus semispinalis thoracis entier'],

            ['Both forefeet', 'altLabel', 'les deux pieds.', 'les deux mains'],

            ['Population of all isolated head spermatozoa in portion of fluid',
             'population de tous les spermatozoïdes isolés de la tête en portion de liquide.',
             'population de toutes les têtes de spermatozoïdes isolées dans une partie du liquide'],

             ['Currant-jelly blood clot', 'caillot sanguin cassis-gelée', 'caillot sanguin'],

             ['All bone marrow of radius and ulna', 'toute mœlle osseuse du radius et du cubitus', "toute la mœlle osseuse du radius et de l'ulna"],

             ['Anus, rectum and sigmoid colon, CS',	'anus, rectum et côlon sigmoïde, CS', 'anus rectum et côlon sigmoïde, site combiné'],

             ['Male structure', 'structure masculine', "de l'homme"],

             ['CD4+HLA-DR+ T Lymphocyte', 'Lymphocyte cD4 + HLA-DR + T', 'lymphocyte CD4+HLA-DR+ T']]

    for t in to_check:
        if len(t)==3:
            st_en = t[0]
            st_fr = t[1]
            st_correct = t[2]
            prop = ""

        else:
            st_en = t[0]
            prop = t[1]
            st_fr = t[2]
            st_correct = t[3]

        bs_tag = "bs2,bs3"
        # st_fr = replace_1st_letter_with_lower(st_en, st_fr)

        st_fr = st_fr.strip(".")
        st_fr = st_fr.replace(" / ","/")
        st_fr = st_fr.replace("( ","(")
        st_fr = st_fr.replace(" )",")")
        st_fr = st_fr.replace(" ; ",";")

        st_fr = st_fr.replace(" +", "+")

        if "+" in st_fr:
            st_fr_new = ""
            positions = [pos for pos, char in enumerate(st_fr) if char == "+"]

            for pos in positions[0:len(positions)-1]:
                if st_fr[pos+1]==" ":
                    st_fr = st_fr[:pos+1] + st_fr[pos+2:]


        st_fr = replace_1st_letter_with_lower(st_en,st_fr)

        st_fr = or2_accent(st_fr)
        # print(st_fr)

        st_fr = or3_or4_or5_hyphen(st_fr,hyphen_dict_pref, hyphen_dict_alt, prop)
        # print(st_fr)
        st_fr = sc1_roman_uppercase(st_fr)
        st_fr = sc6_replace_comp_symbols(st_fr)

        st_fr = ss1_ss2_rule(st_en, st_fr)

        st_fr = or1_orthographe_replace(st_fr)

        st_fr = bs1_replace_with_right_trans(st_en,st_fr)

        if "bs2" in bs_tag:
            st_fr = bs2_replace(st_en,st_fr,bs2_dict)

        if st_fr not in ar7_dict_pref and st_fr not in ar7_dict_alt:
            if "bs3" in bs_tag:

                if "all" not in st_en:
                    st_fr = bs3_replace(st_en,st_fr,bs3_dict, prop)

                if "pref" in prop and "structure" in st_en and sep=="SEP-S":
                    st_fr = st_fr.replace("structure de","")
                    st_fr = st_fr.replace("structure du","")
                    st_fr = st_fr.replace("structures","")
                    st_fr = st_fr.replace("structure","")

        st_fr = um2_temperature(st_fr)
        st_fr = disorder_fix(st_en,st_fr)

        st_fr = ll1_replace(st_fr, ll1_dict)
        # st_fr = check_for_mg(st_fr)

        st_fr = gr1_replace_greek_letter_with_long(st_fr)

        st_fr = ar7_plural_if_both_or_all(st_en, st_fr, ar7_dict_pref, ar7_dict_alt, ar7_dict_eng, prop)

        st_fr = ar2_remove_article_from_start(st_fr)

        st_fr = custom_replace(st_fr, custom_dict)

        st_fr = detect_acronym(st_en, st_fr)

        st_fr = rule_CS(st_en, st_fr)
        st_fr = rule_azygos(st_en, st_fr)
        st_fr = rule_lens(st_en, st_fr)
        st_fr = rule_male(st_en, st_fr)
        st_fr = rule_female(st_en, st_fr)
        st_fr = rule_male_to_female(st_en, st_fr)
        # st_fr = rule_fallopian(st_en, st_fr, prop)
        st_fr = rule_fluid(st_en, st_fr)
        st_fr = rule_compatible(st_en, st_fr)
        st_fr = rule_atrial(st_en,st_fr)



        if st_fr==st_correct:
            print('Done')
        else:
            print('Error: '+st_en+" | "+st_fr+" | "+st_correct)



def fix_snomed_rules():

    # mypath = '../../results_snomed_2022_new'
    # onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #
    # for onlyfile in onlyfiles:
    #     if onlyfile.endswith("csv"):
    #         print(onlyfile+"\n")

    bs2_dict = {}
    df_bs = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="bs2")
    df_bs = df_bs.fillna('')
    for index, row in df_bs.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation_prefLabel'])
        if corr!="":
            bs2_dict[err] = corr

    bs3_dict = {}
    df_bs = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="bs3")
    df_bs = df_bs.fillna('')
    for index, row in df_bs.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation_prefLabel'])
        if corr!="":
            bs3_dict[err] = corr

    hyphen_dict_pref = {}
    hyphen_dict_alt = {}
    df_hyphen = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="Hyphen")
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
    df_ar = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="ar7")
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
    df_ll1 = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="ll1")
    df_ll1 = df_ll1.fillna('')
    for index, row in df_ll1.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation'])
        if corr!="":
            ll1_dict[err] = corr

    custom_dict = {}
    df_custom = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="custom")
    df_custom = df_custom.fillna('')
    for index, row in df_custom.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation'])
        if corr!="":
            custom_dict[err] = corr

    df = pd.read_csv("/Users/konstantinosskianis/Documents/icd11-translation/snomed-2022/123037004_Body_structure-1.csv",sep='|')
    #
    # path = "../../results_snomed_2022_after_rules/snomed_2022_english_terms_cased.txt"
    # f = open(path,"r")
    # lines_en = f.readlines()
    # f.close()
    #
    # fr_to_tran = open("../../results_snomed_2022_after_rules/123037004_Body_structure-1_translated_cased.fr","r")
    # lines_fr = fr_to_tran.readlines()
    # fr_to_tran.close()

    df_bs = pd.read_csv("../../../snomed-2022/123037004_Body_structure-1.csv",sep="|")

    df_sep = pd.read_csv("../../../snomed-2022/SEP.csv",sep="|")

    nlp_en = spacy.load("en_core_web_trf")

    df_old = pd.read_csv("/Users/konstantinosskianis/Documents/icd11-translation/extension-2021/results_snomed_2022_after_rules/123037004_Body_structure_translated_RULES_NEW.csv",sep="|")

    df_all = pd.read_excel("/Users/konstantinosskianis/Documents/icd11-translation/extension-2021/results_snomed_2022_after_rules/SNOMED_123037004_body_structure_27-10-2022.xlsx")

    ffr = open("../../results_snomed_2022_after_rules/123037004_Body_structure.csv_translated_all.csv_RULES_2-2-2023_PROPN_only.csv","w")
    ffr.write("URI|PROPERTY|ENGLISH LABEL|TRANSLATION|FIXED\n")
    fr = open("../../results_snomed_2022_after_rules/123037004_Body_structure.csv_translated_all.csv_RULES_2-2-2023_changed_PROPN_only.csv","w")

    changed = 0
    counter = 0
    # for i, line in enumerate(tqdm(lines)):
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        # line = line.split("|")
        # st_en = line[2].replace("\n","").strip()
        # st_fr_orig = line[3].replace("\n","").strip()
        # st_fr = line[3].replace("\n","").strip()

        uri = str(row['uri'])
        prop = str(row['property'])
        st_en = str(row['en_term']).replace("\n","").strip()
        rules = str(row['rules']).split(",")
        # st_en_cased = lines_en[index].replace("\n","")

        row_sep = df_sep.loc[df_sep['URI'] == uri]
        try:
            sep = str(row_sep['SEP'].item())
        except:
            pass
        # import ipdb; ipdb.set_trace()
        # print(sep)
        st_fr = ""
        st_fr_orig = ""

        # cat_row = df_all.loc[df_all['en_term'] == st_en].squeeze()
        trans_row_all = df_all.iloc[index]
        doc_en = nlp_en(st_en)
        trans_row = df_old.iloc[index]
        st_fr_old = str(trans_row['fixed'])
        st_fr_orig = str(trans_row_all['TRANSLATION'])
        st_fr = str(trans_row_all['TRANSLATION'])
        # import ipdb; ipdb.set_trace()

        bs_tag = str(df_bs.iloc[index]['rules'])

        if doc_en[0].pos_=="PROPN": #and doc_en[0].dep_=="poss":
            st_fr = st_fr_old

        ## this is to catch missing words
        # if len(st_fr_old.split())>len(st_fr.split()):
        #     st_fr = st_fr_old

        # import ipdb; ipdb.set_trace()
        # if not cat_row.empty:
        #     if str(cat_row['uri'])==uri and str(cat_row['property'])==prop:
        #         st_fr_orig = str(cat_row['orig_translation'])
        #         st_fr = str(cat_row['fixed_translation_after_rules'])
        # else:
        #     cat_row = df_all.loc[df_all['en_term'] == st_en.split("(")[0].strip()].squeeze()
        #     # import ipdb; ipdb.set_trace()
        #     if str(cat_row['uri'])==uri and str(cat_row['property'])==prop:
        #         st_fr_orig = str(cat_row['orig_translation'])
        #         st_fr = str(cat_row['fixed_translation_after_rules'])

        # if st_fr=="":
        # st_fr_orig = lines_fr[counter].replace("\n","")
        # st_fr = lines_fr[counter].replace("\n","")
        # st_fr_orig = lines_fr[index].replace("\n","")
        # st_fr = lines_fr[index].replace("\n","")
        counter += 1

        st_fr = st_fr.strip(".")
        st_fr = st_fr.replace(" / ","/")
        st_fr = st_fr.replace("( ","(")
        st_fr = st_fr.replace(" )",")")
        st_fr = st_fr.replace(" ; ",";")

        st_fr = st_fr.replace(" +", "+")
        # if "+" in st_fr:
        #     st_fr_new = ""
        #     positions = [pos for pos, char in enumerate(st_fr) if char == "+"]
        #
        #     if len(positions)>1:
        #         for pos in positions[0:len(positions)-1]:
        #             if st_fr[pos+1]==" ":
        #                 st_fr_new = st_fr[:pos+1] + st_fr[pos+2:]
        #
        #         st_en_spl = st_en.split()
        #         for st in st_fr_new.split():
        #             if "+" in st:
        #                 if st not in st_en_spl:
        #                     st_fr_new = st_fr
        #
        #         st_fr = st_fr_new

        if "+" in st_fr:
            st_fr_new = ""
            positions = [pos for pos, char in enumerate(st_fr) if char == "+"]

            if len(positions)>=1:
                for pos in positions[0:len(positions)-1]:
                    if st_fr[pos+1]==" ":
                        st_fr = st_fr[:pos+1] + st_fr[pos+2:]

        # print(st_en)
        # print(st_fr)
        st_fr = replace_1st_letter_with_lower(st_en,st_fr)

        st_fr = or2_accent(st_fr)
        st_fr = or3_or4_or5_hyphen(st_fr,hyphen_dict_pref, hyphen_dict_alt, prop)

        st_fr = sc1_roman_uppercase(st_fr)
        st_fr = sc6_replace_comp_symbols(st_fr)

        st_fr = ss1_ss2_rule(st_en, st_fr)

        st_fr = or1_orthographe_replace(st_fr)

        st_fr = bs1_replace_with_right_trans(st_en,st_fr)

        if "bs2" in bs_tag:
            st_fr = bs2_replace(st_en,st_fr,bs2_dict)

        if st_fr not in ar7_dict_pref and st_fr not in ar7_dict_alt:
            if "bs3" in bs_tag:

                if "all" not in st_en and "All" not in st_en:
                    st_fr = bs3_replace(st_en,st_fr,bs3_dict, prop)

                if "pref" in prop and "structure" in st_en and sep=="SEP-S":
                    st_fr = st_fr.replace("structure de","")
                    st_fr = st_fr.replace("structure du","")
                    st_fr = st_fr.replace("structures","")
                    st_fr = st_fr.replace("structure","")
                    st_fr = st_fr.replace("os de l'os","os")

                # if "alt" in prop and "structure" in st_en and sep=="SEP-S":
                #
                # if " part " in st_en and sep=="SEP-P":

        st_fr = bs5_replace(st_fr)
        st_fr = bs6_replace(st_fr)
        st_fr = bs8_replace(st_fr, prop)

        st_fr = um2_temperature(st_fr)
        st_fr = disorder_fix(st_en,st_fr)

        st_fr = ll1_replace(st_fr, ll1_dict)
        # st_fr = check_for_mg(st_fr)

        st_fr = gr1_replace_greek_letter_with_long(st_fr)

        st_fr = ar7_plural_if_both_or_all(st_en, st_fr, ar7_dict_pref, ar7_dict_alt, ar7_dict_eng, prop)

        st_fr = ar2_remove_article_from_start(st_fr)

        st_fr = custom_replace(st_fr, custom_dict)

        st_fr = detect_acronym(st_en, st_fr)

        st_fr = rule_CS(st_en, st_fr)
        st_fr = rule_azygos(st_en, st_fr)
        st_fr = rule_lens(st_en, st_fr)
        st_fr = rule_male(st_en, st_fr)
        st_fr = rule_female(st_en, st_fr)
        st_fr = rule_male_to_female(st_en, st_fr)
        st_fr = rule_fallopian(st_en, st_fr, prop)
        st_fr = rule_fluid(st_en, st_fr)
        st_fr = rule_compatible(st_en, st_fr)
        st_fr = rule_atrial(st_en,st_fr)

        # st_fr = or1_orthographe_replace(st_fr)
        # st_fr = ll1_replace(st_fr, ll1_dict)

        # if not st_en.endswith("."):
        #     st_fr = st_fr.strip(".")

        if st_fr!=st_fr_orig:
            changed += 1
            fr.write(st_fr_orig+"|"+st_fr+"\n")

        st_fr = st_fr.strip()

        # ffr.write(line[0]+"|"+line[1]+"|"+line[2]+"|"+st_fr_orig+"|"+st_fr+"\n")
        ffr.write(uri+"|"+prop+"|"+st_en+"|"+st_fr_orig+"|"+st_fr+"\n")
        # ffr.write(st_fr+"\n")

    fr.close()
    ffr.close()

    print(str(changed))


def fix_ncit_rules():
    # import pluralizefr
    # import inflect
    # p = inflect.engine()

    acronyms = {'DNA': 'ADN', 'mtDNA': 'ADNmt', 'RNA': 'ARN', 'tRNA': 'ARNt', 'mRNA': 'ARNm', 'rRNA': 'ARNr',
                'PCR': 'PCR', 'rtPCR': 'rtPCR', 'IgE': 'IgE', 'IgG': 'IgG', 'IgM': 'IgM', 'IgA': 'IgA'}

    df = pd.read_excel('../../data/NCIT2022_àtraduire.xlsx', engine='openpyxl', na_values='')
    #
    # f = open("../../results/ncit_translation_orig.txt","r", encoding='utf8')
    # lines_orig = f.readlines()
    # f.close()

    f = open("../../results/NCIT_english_terms_cased.txt","r", encoding='utf8')
    lines_en = f.readlines()
    f.close()

    f = open("../../results/NCIT_translated_cased.fr","r", encoding='utf8')
    lines = f.readlines()
    f.close()

    # fr = open("../../results/NCIT2022_translated_FIXED_NEW_changed.fr","w",encoding='utf8')

    f = open("../../results/NCIT2022_translated_FIXED_CASED.fr","w",encoding='utf8')
    # f = open("../../results/ncit_results_similarity_FIXED_NEW.txt","w",encoding='utf8')
    # for line in lines:
    counter = 0
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    # for ind, line in enumerate(lines):
        # line_all = line.replace("\n","").split("|").copy()
        line = lines[index].strip()
        # line = line_all[1].replace("\n","")
        # line_orig = line.strip()
        # line_orig = lines_orig[ind].replace("\n","")
        # line = line.replace(" - ","-")
        line = line.replace(" / ","/")
        line = line.replace("~ ","~")
        # prev_line = line.split().copy()
        line_s = line.split().copy()
        line_en_cased = str(row['Valeur']).strip()
        line_en = lines_en[index].strip()

        if row['Type']=="def":
            line = line[0].upper() + "".join(line[1:])
        # else:
        #     ## FIX case
        #     line_s = line.split()
        #
        #     ### check extreme cases
        #     line_en_s = line_en.split()
        #
        #     if not line_s[0].isupper():
        #         line = line_s[0].lower() + " " + " ".join(line_s[1:])
        #
        #         if line_en_s[0]==line_s[0]:
        #             if line_en_s[0].strip("s").isupper():
        #                 line = prev_line[0] + " " + " ".join(line_s[1:])
        #
        #         if line_en_s[0].split("-")[0]==line_s[0].split("-")[0]:
        #             if line_en_s[0].split("-")[0].isupper():
        #                 line = prev_line[0] + " " + " ".join(line_s[1:])

            ## FIX singular
            # single = True
            # for word in line_en.split():
            #     if p.singular_noun(word):
            #         single = False
            #
            # single_fr = True
            # if single:
            #     for word in line.split():
            #         if singular_or_plural_fr(word)=='plural':
            #             single_fr = False
            #
            #     if not single_fr:
            #         sent = ""
            #         for word in line.split():
            #             sent += " " + pluralizefr.singularize(word)
            #
            #         line = sent.strip()

        for l in line_en.split():
            if l in acronyms:
                for ls in line.split():
                    if ls.lower()==l.lower():
                        line = line.replace(ls, acronyms[l])

        # if len(line_s)==1:
        #     line = line_orig

        # if line!=line_orig:
            # fr.write(line_orig+"|"+line+"\n")
            # counter += 1

        f.write(line_en+"|"+line_en_cased+"|"+line+"\n")
        # f.write(line+"\n")
    f.close()
    # fr.close()
    # print("Changed: "+str(counter))


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


def fix_editorial_snomed_rules():
    # import OS module
    import os

    # Get the list of all files and directories
    path = "snomed_fr"
    dir_list = os.listdir(path)

    for file in dir_list:

        f = open("snomed_en/"+file.strip(".fr")+".txt","r")
        lines_en = f.readlines()
        f.close()

        f = open(path+"/"+file,"r")
        lines = f.readlines()
        f.close()

        f = open("snomed_fr_rules/"+file,"w")
        for i, line in enumerate(lines):
            st_en = lines_en[i].strip("\n")
            st_fr = line.strip("\n")

            # print(st_en)
            # print(st_fr)

            st_fr = st_fr.strip(".")
            st_fr = st_fr.replace(" / ","/")
            st_fr = st_fr.replace("( ","(")
            st_fr = st_fr.replace(" )",")")
            st_fr = st_fr.replace(" ; ",";")
            st_fr = st_fr.replace(" +", "+")

            # st_fr = replace_1st_letter_with_lower(st_en,st_fr)

            # st_fr = or2_accent(st_fr)
            # st_fr = or3_or4_or5_hyphen(st_fr,hyphen_dict_pref, hyphen_dict_alt, prop)

            st_fr = sc1_roman_uppercase(st_fr)
            # st_fr = sc6_replace_comp_symbols(st_fr)

            # st_fr = ss1_ss2_rule(st_en, st_fr)

            st_fr = or1_orthographe_replace(st_fr)

            st_fr = bs1_replace_with_right_trans(st_en,st_fr)

            st_fr = um2_temperature(st_fr)
            st_fr = disorder_fix(st_en,st_fr)

            # st_fr = ll1_replace(st_fr, ll1_dict)
            # st_fr = check_for_mg(st_fr)

            st_fr = gr1_replace_greek_letter_with_long(st_fr)

            # st_fr = ar7_plural_if_both_or_all(st_en, st_fr, ar7_dict_pref, ar7_dict_alt, ar7_dict_eng, prop)

            st_fr = ar2_remove_article_from_start(st_fr)

            # st_fr = custom_replace(st_fr, custom_dict)

            st_fr = detect_acronym(st_en, st_fr)

            st_fr = rule_CS(st_en, st_fr)
            st_fr = rule_azygos(st_en, st_fr)
            st_fr = rule_lens(st_en, st_fr)
            st_fr = rule_male(st_en, st_fr)
            st_fr = rule_female(st_en, st_fr)
            st_fr = rule_male_to_female(st_en, st_fr)
            # st_fr = rule_fallopian(st_en, st_fr, prop)
            st_fr = rule_fluid(st_en, st_fr)
            st_fr = rule_compatible(st_en, st_fr)
            st_fr = rule_atrial(st_en,st_fr)

            # st_fr = or1_orthographe_replace(st_fr)
            # st_fr = ll1_replace(st_fr, ll1_dict)

            f.write(st_fr+"\n")

        f.close()


def fix_ncit_labelsyn_rules_2023():
    df = pd.read_excel('../../results_ncit_2023/2212d_ncit_labelsyn_TRANSLATION_FINAL.xlsx')
    # df = df.replace(np.nan, '', regex=True)

    f = open("../../results_ncit_2023/2212d_ncit_labelsyn_TRANSLATION_with_rules.txt","w")
    # fnan = open("../../results_ncit_2023/labelsyn_not_translated.txt","w")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        st_en = str(row['terms']).replace("\n","")

        st_fr = str(row['Translation']).replace("\n","")
        uri = str(row['uri']).replace("\n","")

        # print(st_en)
        # print(st_fr)

        # st_fr = st_fr.strip(".")
        # st_fr = st_fr.replace(" / ","/")
        # st_fr = st_fr.replace("( ","(")
        # st_fr = st_fr.replace(" )",")")
        # st_fr = st_fr.replace(" ; ",";")
        # st_fr = st_fr.replace(" +", "+")

        st_fr = replace_1st_letter_with_lower(st_en,st_fr)

        # st_fr = or2_accent(st_fr)
        # st_fr = or3_or4_or5_hyphen(st_fr,hyphen_dict_pref, hyphen_dict_alt, prop)

        # st_fr = sc1_roman_uppercase(st_fr)
        # st_fr = sc6_replace_comp_symbols(st_fr)

        # st_fr = ss1_ss2_rule(st_en, st_fr)

        # st_fr = or1_orthographe_replace(st_fr)

        # st_fr = bs1_replace_with_right_trans(st_en,st_fr)

        # st_fr = um2_temperature(st_fr)
        st_fr = disorder_fix(st_en,st_fr)

        # st_fr = ll1_replace(st_fr, ll1_dict)
        # st_fr = check_for_mg(st_fr)

        # st_fr = gr1_replace_greek_letter_with_long(st_fr)

        # st_fr = ar7_plural_if_both_or_all(st_en, st_fr, ar7_dict_pref, ar7_dict_alt, ar7_dict_eng, prop)

        # st_fr = ar2_remove_article_from_start(st_fr)

        # st_fr = custom_replace(st_fr, custom_dict)

        st_fr = detect_acronym(st_en, st_fr)

        # st_fr = rule_CS(st_en, st_fr)
        # st_fr = rule_azygos(st_en, st_fr)
        # st_fr = rule_lens(st_en, st_fr)
        # st_fr = rule_male(st_en, st_fr)
        # st_fr = rule_female(st_en, st_fr)
        # st_fr = rule_male_to_female(st_en, st_fr)
        # st_fr = rule_fallopian(st_en, st_fr, prop)
        # st_fr = rule_fluid(st_en, st_fr)
        # st_fr = rule_compatible(st_en, st_fr)
        # st_fr = rule_atrial(st_en,st_fr)

        f.write(st_fr+"\n")
    f.close()
    # fnan.close()

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


def fix_environ_snomed_rules():



    # df = pd.read_csv("/Users/konstantinosskianis/Documents/icd11-translation/snomed-2022/123037004_Body_structure-1.csv",sep='|')
    # df = pd.read_csv("/Users/konstantinosskianis/Documents/icd11-translation/snomed-2022/Environnment_geographical_location.csv",sep='|',encoding='utf-8')

    df = pd.read_excel("/Users/konstantinosskianis/Documents/icd11-translation/extension-2021/data/feedback_environment (1).xlsx")
    df = df.fillna('')

    # path = "../../results_snomed_2022_after_rules/snomed_2022_english_terms_cased.txt"
    # f = open(path,"r")
    # lines_en = f.readlines()
    # f.close()

    fr_to_tran = open("../../results_snomed_2022_after_rules/multiple_results_scores.txt","r")
    lines_fr = fr_to_tran.readlines()
    fr_to_tran.close()

    fr_to_tran = open("../../results_snomed_2022_after_rules/multiple_results_lower_scores.txt","r")
    lines_fr_lower = fr_to_tran.readlines()
    fr_to_tran.close()

    df_sep = pd.read_csv("../../../snomed-2022/SEP.csv",sep="|")

    nlp_en = spacy.load("en_core_web_trf")

    ffr = open("../../results_snomed_2022_after_rules/environ_translated_RULES_31-3-2023.csv","w")
    ffr.write("URI|PROPERTY|ENGLISH LABEL|TRANSLATION\n")
    # fr = open("../../results_snomed_2022_after_rules/123037004_Body_structure.csv_translated_all.csv_RULES_2-2-2023_changed_PROPN_only.csv","w")

    changed = 0
    counter = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        uri = str(row['URI'])
        prop = str(row['PROPERTY'])
        st_en = str(row['ENGLISH']).replace("\n","").strip()
        # rules = str(row['rules']).split(",")
        # st_en_cased = lines_en[index].replace("\n","")

        row_sep = df_sep.loc[df_sep['URI'] == uri]
        try:
            sep = str(row_sep['SEP'].item())
        except:
            pass
        # import ipdb; ipdb.set_trace()
        # print(sep)

        doc_en = nlp_en(st_en)

        # st_fr = ""
        st_fr_orig = lines_fr[index].strip()
        st_fr_lower = lines_fr_lower[index].strip()
        # import ipdb; ipdb.set_trace()

        # bs_tag = str(df_bs.iloc[index]['rules'])

        st_fr = st_fr_lower
        if doc_en[0].pos_=="PROPN": #and doc_en[0].dep_=="poss":
            st_fr = st_fr_orig

        # for word in doc_en.ents:
        # import ipdb; ipdb.set_trace()
        if len(doc_en.ents)>0:
            # print(st_fr_orig)
            if st_en.split()[0]==doc_en.ents[0].text:
                if doc_en.ents[0].label_=="ORG" or doc_en.ents[0].label_=="GPE":
                    st_fr = st_fr_orig
                    print(st_fr+" "+doc_en.ents[0].label_)

        counter += 1

        st_fr = st_fr.strip(".")
        st_fr = st_fr.replace(" / ","/")
        st_fr = st_fr.replace("( ","(")
        st_fr = st_fr.replace(" )",")")
        st_fr = st_fr.replace(" ; ",";")

        st_fr = st_fr.replace(" +", "+")

        if "+" in st_fr:
            st_fr_new = ""
            positions = [pos for pos, char in enumerate(st_fr) if char == "+"]

            if len(positions)>=1:
                for pos in positions[0:len(positions)-1]:
                    if st_fr[pos+1]==" ":
                        st_fr = st_fr[:pos+1] + st_fr[pos+2:]

        # print(st_en)
        # print(st_fr)
        st_fr = replace_1st_letter_with_lower(st_en,st_fr)

        st_fr = or2_accent(st_fr)
        st_fr = or3_or4_or5_hyphen(st_fr,hyphen_dict_pref, hyphen_dict_alt, prop)

        st_fr = sc1_roman_uppercase(st_fr)
        st_fr = sc6_replace_comp_symbols(st_fr)

        st_fr = ss1_ss2_rule(st_en, st_fr)

        st_fr = or1_orthographe_replace(st_fr)

        st_fr = bs1_replace_with_right_trans(st_en,st_fr)

        # if "bs2" in bs_tag:
        st_fr = bs2_replace(st_en,st_fr,bs2_dict)

        if st_fr not in ar7_dict_pref and st_fr not in ar7_dict_alt:
            # if "bs3" in bs_tag:

            if "all" not in st_en and "All" not in st_en:
                st_fr = bs3_replace(st_en,st_fr,bs3_dict, prop)

                # if "pref" in prop and "structure" in st_en and sep=="SEP-S":
                #     st_fr = st_fr.replace("structure de","")
                #     st_fr = st_fr.replace("structure du","")
                #     st_fr = st_fr.replace("structures","")
                #     st_fr = st_fr.replace("structure","")
                #     st_fr = st_fr.replace("os de l'os","os")

                # if "alt" in prop and "structure" in st_en and sep=="SEP-S":
                #
                # if " part " in st_en and sep=="SEP-P":

        st_fr = bs5_replace(st_fr)
        st_fr = bs6_replace(st_fr)
        st_fr = bs8_replace(st_fr, prop)

        st_fr = um2_temperature(st_fr)
        st_fr = disorder_fix(st_en,st_fr)

        st_fr = ll1_replace(st_fr, ll1_dict)
        # st_fr = check_for_mg(st_fr)

        st_fr = gr1_replace_greek_letter_with_long(st_fr)

        st_fr = ar7_plural_if_both_or_all(st_en, st_fr, ar7_dict_pref, ar7_dict_alt, ar7_dict_eng, prop)

        st_fr = ar2_remove_article_from_start(st_fr)

        st_fr = custom_replace(st_fr, custom_dict)

        st_fr = detect_acronym(st_en, st_fr)

        st_fr = rule_CS(st_en, st_fr)
        st_fr = rule_azygos(st_en, st_fr)
        st_fr = rule_lens(st_en, st_fr)
        st_fr = rule_male(st_en, st_fr)
        st_fr = rule_female(st_en, st_fr)
        st_fr = rule_male_to_female(st_en, st_fr)
        st_fr = rule_fallopian(st_en, st_fr, prop)
        st_fr = rule_fluid(st_en, st_fr)
        st_fr = rule_compatible(st_en, st_fr)
        st_fr = rule_atrial(st_en,st_fr)

        # if st_fr!=st_fr_orig:
        #     changed += 1
        #     fr.write(st_fr_orig+"|"+st_fr+"\n")

        st_fr = st_fr.strip()
        if st_en[-1]!="." and st_fr[-1]==".":
            st_fr = st_fr.strip(".")

        # ffr.write(line[0]+"|"+line[1]+"|"+line[2]+"|"+st_fr_orig+"|"+st_fr+"\n")
        ffr.write(uri+"|"+prop+"|"+st_en+"|"+st_fr_orig+"|"+st_fr+"\n")
        # ffr.write(st_fr+"\n")

    # fr.close()
    ffr.close()

    print(str(changed))


def environ_stats():
    from nltk.tokenize import word_tokenize
    from Levenshtein import distance
    cc = nltk.translate.bleu_score.SmoothingFunction()

    bs2_dict = {}
    df_bs = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="bs2")
    df_bs = df_bs.fillna('')
    for index, row in df_bs.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation_prefLabel'])
        if corr!="":
            bs2_dict[err] = corr

    bs3_dict = {}
    df_bs = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="bs3")
    df_bs = df_bs.fillna('')
    for index, row in df_bs.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation_prefLabel'])
        if corr!="":
            bs3_dict[err] = corr

    hyphen_dict_pref = {}
    hyphen_dict_alt = {}
    df_hyphen = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="Hyphen")
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
    df_ar = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="ar7")
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
    df_ll1 = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="ll1")
    df_ll1 = df_ll1.fillna('')
    for index, row in df_ll1.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation'])
        if corr!="":
            ll1_dict[err] = corr

    custom_dict = {}
    df_custom = pd.read_excel("../../../snomed-2022/Avancement_traduction_anatomie-1.xlsx", sheet_name="custom")
    df_custom = df_custom.fillna('')
    for index, row in df_custom.iterrows():
        err = str(row['erroneous_translation'])
        corr = str(row['correct_translation'])
        if corr!="":
            custom_dict[err] = corr

    # model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu', cache_folder='./')
    model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')

    df = pd.read_excel("/Users/konstantinosskianis/Documents/icd11-translation/extension-2021/data/feedback_environment (1).xlsx")
    df = df.fillna('')

    counter_ok = 0
    counter_l = 0

    fr_to_tran = open("../../results_snomed_2022_after_rules/multiple_results_scores.txt","r")
    lines_fr_upper = fr_to_tran.readlines()
    fr_to_tran.close()

    fr_to_tran = open("../../results_snomed_2022_after_rules/multiple_results_lower_scores.txt","r")
    lines_fr_lower = fr_to_tran.readlines()
    fr_to_tran.close()

    df_sep = pd.read_csv("../../../snomed-2022/SEP.csv",sep="|")

    nlp_en_trf = spacy.load("en_core_web_trf")
    nlp_en = spacy.load("en_core_web_lg")

    # f = open("environ_multiple_results.txt","r")
    # lines = f.readlines()
    # f.close()

    ffr = open("../../results_snomed_2022_after_rules/environ_translated_RULES_31-3-2023.csv","w")
    ffr.write("URI|PROPERTY|ENGLISH LABEL|TRANSLATION|POSTPROCESSING (RULES)|FEEDBACK|BLEU|FR SIMILARITY (tran vs feedback)|EN SIMILARITY (tran vs en)|Lev|Multiple\n")

    # f = open("environ_bleu.txt","w")
    # fe = open("environ_sim_eng.txt","w")
    # fr = open("environ_sim_fr.txt","w")
    # fl = open("environ_mult_scores.txt","w")
    # flev = open("environ_lev.txt","w")

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        # URI	PROPERTY	ENGLISH LABEL	POSTPROCESSING (RULES)	FEEDBACK
        uri = str(row['URI'])
        prop = str(row['PROPERTY'])
        st_en = str(row['ENGLISH LABEL']).replace("\n","").strip()
        st_fr_old = str(row['POSTPROCESSING (RULES)']).replace("\n","").strip()
        feedback = str(row['FEEDBACK']).replace("\n","").strip()

        doc_en = nlp_en(st_en)
        doc_en_trf = nlp_en_trf(st_en)

        # st_fr = ""
        st_fr_upper_all = lines_fr_upper[index].strip()
        st_fr_lower_all = lines_fr_lower[index].strip()

        st_fr_upper = st_fr_upper_all.split("|")[0].split("L:")[0].strip()
        st_fr_lower = st_fr_lower_all.split("|")[0].split("L:")[0].strip()
        # import ipdb; ipdb.set_trace()

        # bs_tag = str(df_bs.iloc[index]['rules'])

        st_fr = st_fr_lower
        multiple = st_fr_lower_all
        if doc_en[0].pos_=="PROPN" and doc_en_trf[0].pos_=="PROPN": #and doc_en[0].dep_=="poss":
            st_fr = st_fr_upper
            multiple = st_fr_upper_all

        # for word in doc_en.ents:
        # import ipdb; ipdb.set_trace()
        if len(doc_en_trf.ents)>0:
            # print(st_fr_orig)
            if st_en.split()[0]==doc_en_trf.ents[0].text:
                if doc_en_trf.ents[0].label_=="ORG" or doc_en_trf.ents[0].label_=="GPE":
                    st_fr = st_fr_upper
                    multiple = st_fr_upper_all
                    # print(st_fr+" "+doc_en.ents[0].label_)
        # import ipdb; ipdb.set_trace()
        # line = lines[index].strip()
        # line = line.split("|")

        st_fr_orig = st_fr

        st_fr = st_fr.strip(".")
        st_fr = st_fr.replace(" / ","/")
        st_fr = st_fr.replace("( ","(")
        st_fr = st_fr.replace(" )",")")
        st_fr = st_fr.replace(" ; ",";")

        st_fr = st_fr.replace(" +", "+")

        if "+" in st_fr:
            st_fr_new = ""
            positions = [pos for pos, char in enumerate(st_fr) if char == "+"]

            if len(positions)>=1:
                for pos in positions[0:len(positions)-1]:
                    if st_fr[pos+1]==" ":
                        st_fr = st_fr[:pos+1] + st_fr[pos+2:]

        st_fr = st_fr.strip()
        if st_en[-1]!="." and st_fr[-1]==".":
            st_fr = st_fr.strip(".")

        st_fr = replace_1st_letter_with_lower(st_en,st_fr)

        st_fr = or2_accent(st_fr)
        st_fr = or3_or4_or5_hyphen(st_fr,hyphen_dict_pref, hyphen_dict_alt, prop)

        st_fr = sc1_roman_uppercase(st_fr)
        st_fr = sc6_replace_comp_symbols(st_fr)

        st_fr = ss1_ss2_rule(st_en, st_fr)

        st_fr = or1_orthographe_replace(st_fr)

        st_fr = bs1_replace_with_right_trans(st_en,st_fr)

        # if "bs2" in bs_tag:
        st_fr = bs2_replace(st_en,st_fr,bs2_dict)

        if st_fr not in ar7_dict_pref and st_fr not in ar7_dict_alt:
            # if "bs3" in bs_tag:

            if "all" not in st_en and "All" not in st_en:
                st_fr = bs3_replace(st_en,st_fr,bs3_dict, prop)

        st_fr = bs5_replace(st_fr)
        st_fr = bs6_replace(st_fr)
        st_fr = bs8_replace(st_fr, prop)

        st_fr = um2_temperature(st_fr)
        st_fr = disorder_fix(st_en,st_fr)

        st_fr = ll1_replace(st_fr, ll1_dict)
        # st_fr = check_for_mg(st_fr)

        st_fr = gr1_replace_greek_letter_with_long(st_fr)

        st_fr = ar7_plural_if_both_or_all(st_en, st_fr, ar7_dict_pref, ar7_dict_alt, ar7_dict_eng, prop)

        st_fr = ar2_remove_article_from_start(st_fr)

        st_fr = custom_replace(st_fr, custom_dict)

        st_fr = detect_acronym(st_en, st_fr)

        st_fr = rule_CS(st_en, st_fr)
        st_fr = rule_azygos(st_en, st_fr)
        st_fr = rule_lens(st_en, st_fr)
        st_fr = rule_male(st_en, st_fr)
        st_fr = rule_female(st_en, st_fr)
        st_fr = rule_male_to_female(st_en, st_fr)
        st_fr = rule_fallopian(st_en, st_fr, prop)
        st_fr = rule_fluid(st_en, st_fr)
        st_fr = rule_compatible(st_en, st_fr)
        st_fr = rule_atrial(st_en,st_fr)

        hypothesis = word_tokenize(st_fr)
        corpus_embedding = model.encode(st_fr, convert_to_tensor=True)

        # txt = ""
        # for l in line:
        #     if l!="":
        #         query_embedding = model.encode(l, convert_to_tensor=True)
        #         cos_scores = util.cos_sim(query_embedding, corpus_embedding)[0]
        #         sim = cos_scores[0].item()
        #         if sim>1:
        #             sim = 1
        #         if sim>0.4:
        #             txt += l+"|"+str(sim).replace(".",",")+"|"
        # if txt!="":
        #     fl.write(txt.strip("|").strip()+"\n")

        reference = [word_tokenize(feedback)]

        query_embedding = model.encode(st_en, convert_to_tensor=True)

        cos_scores = util.cos_sim(query_embedding, corpus_embedding)[0]
        sim = cos_scores[0].item()

        if feedback=="OK":
            feedback = st_fr_old

        if feedback==st_fr:
            feedback="OK"
            reference = [word_tokenize(st_fr)]
            bleu_score_str = str(1.0)
            sim_fr = 1
            dist = 0
        else:
            dist = distance(st_fr, feedback)
            query_embedding = model.encode(feedback, convert_to_tensor=True)
            corpus_embedding = model.encode(st_fr, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, corpus_embedding)[0]
            sim_fr = cos_scores[0].item()

            if st_fr.lower()==feedback or feedback.lower()==st_fr:
                counter_l += 1
            bleu_score_str = str(float(nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, smoothing_function=cc.method2)))

        if feedback=="OK":
            counter_ok +=1

        if sim>0.97:
            sim=1
        if sim_fr>1:
            sim_fr=1

        ffr.write(uri+"|"+prop+"|"+st_en+"|"+st_fr_orig+"|"+st_fr+"|"+feedback+"|"+bleu_score_str+"|"+str(sim_fr)+"|"+str(sim)+"|"+str(dist)+"|"+str(multiple)+"\n")

    #     fe.write(str(sim)+'\n')
    #     fr.write(str(sim_fr)+'\n')
    #     f.write(bleu_score_str+"\n")
    #     flev.write(str(dist)+"\n")
    # fl.close()
    # fe.close()
    # fr.close()
    # f.close()
    ffr.close()
    print("Ok: "+str(counter_ok))
    print("Lower: "+str(counter_l))


if __name__=="__main__":

    # run_tests()
    # snomed_testing()
    # fix_snomed_rules()
    # fix_ncit_rules()
    # examine_casing()
    # fix_editorial_snomed_rules()
    # check_upper()
    # fix_ncit_labelsyn_rules_2023()
    # fix_environ_snomed_rules()
    environ_stats()
