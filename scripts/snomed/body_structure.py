def fix_body_structure_rules():

    bs2_dict, bs3_dict, hyphen_dict_pref, hyphen_dict_alt, ar7_dict_pref, ar7_dict_alt, ar7_dict_eng, ll1_dict, custom_dict = get_dicts()

    df = pd.read_csv("../../../snomed-2022/123037004_Body_structure-1.csv",sep='|')
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

    df_old = pd.read_csv("../../results_snomed_2022_after_rules/123037004_Body_structure_translated_RULES_NEW.csv",sep="|")

    df_all = pd.read_excel("../../results_snomed_2022_after_rules/SNOMED_123037004_body_structure_27-10-2022.xlsx")

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
