import numpy as np

import string
import math
import pickle
import spacy
# from pattern.fr import singularize, pluralize
import nltk
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('french'))

# def singular_or_plural(token):
#     return 'plural' if pluralize(token) == token else 'singular'


def replace_1st_letter_with_upper(st):
    st = st[0].upper() + st[1:]
    return st


def replace_1st_letter_with_lower(st):
    st = st[0].lower() + "".join(st[1:])
    return st


def remove_space_after_acc(st):
    st = st.replace("' ","'")
    return st


def remove_space_after_symb(st):
    st = st.replace(" / ","/")
    return st


def replace_with_right_trans(st_en, st_fr):
    if st_en.endswith("(body structure)") and st_fr.endswith(", sai"):
        st_fr = st_fr.replace(", sai"," (structure du corps)")
    if st_en.endswith("(disorder)") and st_fr.endswith(", sai"):
        st_fr = st_fr.replace(", sai"," (désordre)")
    return st_fr


def ss2(st):
    # mod = spacy.load('fr_core_news_md', disable=['parser', 'textcat', 'ner', 'tagger'])
    # assert singular_or_plural("garçons") == 'plural'
    # assert singular_or_plural("chienne") == 'singular'

    return st


def ar2_remove_article_from_start(st):
    st = st.split()
    if st[0] in stopwords:
        st = st[1:]
    return " ".join(st)


def ab1_repl_words_with_periods_with_fulls(st):
    if "." in st:
        st = st.replace("","")
    return st


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

    for word, initial in greek_letters.items():
        st = st.replace(st.lower(), initial)
    return st


def or1_orthographe_replace(st):
    """
    """
    if "aigüe" in st:
        st = st.replace("aigüe", "aiguë")
    return st


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
    # import spacy

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

#
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

def run_tests():
    st_en = "Seizure free > 12 months (finding)"
    st_fr = "Épilepsie libre > 12 mois (résultat)"
    print(sc6_replace_comp_symbols(st_fr))

    st_en = "Structure of submental space (body structure)"
    st_fr = "Espace sous-mentonnier, sai"
    print(replace_with_right_trans(st_en,st_fr))

    st_en = "Fever greater than 38 Celsius"
    st_fr = "Fièvre supérieure à 38 Celsius"
    print(um2_temperature(st_fr))

    st_en = "Chromosomal disorder (disorder)"
    st_fr = "Trouble chromosomique (trouble)"
    print(disorder_fix(st_en,st_fr))

    st_en = "Occupational lung disorder"
    st_fr = "Trouble pulmonaire professionnel"
    print(disorder_fix(st_en,st_fr))


# if __name__=="__main__":
#
#     mypath = '../../results_snomed_2022_new'
#     onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#
#     for onlyfile in onlyfiles:
#         if onlyfile.endswith("csv"):
#             print(onlyfile+"\n")
#
#             f = open(mypath+"/"+onlyfile,"r")
#             lines = f.readlines()
#             f.close()
#
#             ffr = open("../../results_snomed_2022_after_rules/"+onlyfile+"_RULES.csv","w")
#
#             for i, line in enumerate(tqdm(lines)):
#                 line = line.split("|")
#                 st_en = line[2].replace("\n","")
#                 st_fr = line[3].replace("\n","")
#                 st_fr = sc6_replace_comp_symbols(st_fr)
#
#                 st_fr = replace_with_right_trans(st_en,st_fr)
#
#                 st_fr = um2_temperature(st_fr)
#
#                 st_fr = disorder_fix(st_en,st_fr)
#
#                 # st_fr = check_for_mg(st_fr)
#
#                 st_fr = gr1_replace_greek_letter_with_long(st_fr)
#
#                 st_fr = ar2_remove_article_from_start(st_fr)
#
#                 st_fr = replace_1st_letter_with_lower(st_fr)
#
#                 ffr.write(line[0]+"|"+line[1]+"|"+line[2]+"|"+st_fr+"\n")
#
#             ffr.close()
