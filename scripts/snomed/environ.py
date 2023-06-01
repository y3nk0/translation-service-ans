from utils import *
from rules import *


def fix_environ_snomed_rules():
    from nltk.tokenize import word_tokenize
    from Levenshtein import distance
    cc = nltk.translate.bleu_score.SmoothingFunction()

    bs2_dict, bs3_dict, hyphen_dict_pref, hyphen_dict_alt, ar7_dict_pref, ar7_dict_alt, ar7_dict_eng, ll1_dict, custom_dict = get_dicts()


    # model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu', cache_folder='./')
    model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')

    df = pd.read_excel("../resources/feedback_environment_6-4-2023.xlsx")
    df = df.fillna('')

    counter_ok = 0
    counter_l = 0

    fr_to_tran = open("../resources/multiple_results_scores.txt","r")
    lines_fr_upper = fr_to_tran.readlines()
    fr_to_tran.close()

    fr_to_tran = open("../resources/multiple_results_lower_scores.txt","r")
    lines_fr_lower = fr_to_tran.readlines()
    fr_to_tran.close()

    df_sep = pd.read_csv("../resources/SEP.csv",sep="|")

    ## load both models to match ner entities
    nlp_en_trf = spacy.load("en_core_web_trf")
    nlp_en = spacy.load("en_core_web_lg")

    ffr = open("../resources/environ_translated_RULES_9-5-2023.csv","w")
    ffr.write("URI|PROPERTY|ENGLISH LABEL|TRANSLATION|POSTPROCESSING (RULES)|FEEDBACK|BLEU|FR SIMILARITY (tran vs feedback)|EN SIMILARITY (tran vs en)|Lev|Multiple\n")

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

    ffr.close()
    print("Ok: "+str(counter_ok))
    print("Lower: "+str(counter_l))



if __name__=="__main__":

    fix_environ_snomed_rules()
