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

if __name__=="__main__":
    # fix_ncit_rules()
    fix_ncit_labelsyn_rules_2023()
