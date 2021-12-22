import numpy as np

import string
import math
import pickle


def um8_unit_replace(string):
    """Rule um8
    The letter μ meaning micro in a unit is replaced by the letter u.
    """
    if 'mol / L' in string:
        string = string.replace("μ","u")
    return string


def or1_orthographe_replace(string):
    """
    """
    string = string.replace("aigüe", "aiguë")
    return string


def check_for_mg(string):
    check = string.split("m")
    print(check)
    if check[len(check)-1] == "g":
        print("already in mg")
    else:
        step = string.split(" ")
        step2 = step[len(step)-1].split("mg")
        newstring = " ".join(step)
        step3 = step2[0].split("g")
        step4 = float(step3[0])
        newstring = newstring.replace(step2[0], str(step4*10000)+"mg")
        print(newstring)


def gr1_replace_greek_letter_with_long(string):
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
        string = string.replace(string.lower(), initial)
    return string


def disorder_fix():
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
    functions = ['Eating', 'sleep', 'Development']
    structures = ['Lung', 'rectal']
    if disorder in string:
        found = False
        for function in functions:
            if function in string:
                trans = "trouble"
                found = True
                break

        if found = False:
            for structure in structures:
                if structure in string:
                    trans = 'affection'
                    found = True
                    break

        if found = False:
            trans = 'anomalie'

    return string, trans


def pa2_disorder_fix_with_embeddings():
    """
    check if function or structure with topics from word embeddings or with distance to keyword
    """
