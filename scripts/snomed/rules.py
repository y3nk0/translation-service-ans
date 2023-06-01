from utils import *

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


# def ss1(st_en, st_fr):
#     st_all = st.split()
#     if len(st_all)==1:
#         if st_en[0].isupper():
#             st_fr = st_fr[0].upper()+st_fr[1:]
#         if st_en[0].islower():
#             st_fr = st_fr[0].lower()+st_fr[1:]
#         return st_fr
#     # else:
#     #     for st in st_all:
#     #         st_fr[]
#
#     return st

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

    # if st[0] in stopwords:
    #     st = st[1:]

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
