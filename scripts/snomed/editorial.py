import os

def fix_editorial_snomed_rules():

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


if __name__=="__main__":
    fix_editorial_snomed_rules()
