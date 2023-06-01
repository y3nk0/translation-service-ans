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

    bs2_dict, bs3_dict, hyphen_dict_pref, hyphen_dict_alt, ar7_dict_pref, ar7_dict_alt, ar7_dict_eng, ll1_dict, custom_dict = get_dicts()

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
