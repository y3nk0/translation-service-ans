from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-mpnet-base-v2')

passages = ["os de l'oreille dans son ensemble", "osselet d'une oreille", "ossicule de l'oreille", "osselet dans son ensemble",
            "osselon de l'oreille", "osselets d'oreille", "ossicule dans son ensemble", "osselet de l'oreille",
            "osselets d'une oreille", "osselets de l'oreille"]

query_embedding = model.encode('Entire ossicle of ear')
passage_embedding = model.encode(passages)

print("Similarity:", util.cos_sim(query_embedding, passage_embedding))

passages = ["toute moelle osseuse de crête iliaque.", "toute moelle osseuse de crête iliaque", "toute la crête iliaque médullaire",
            "toute la crête iliaque médullaire.", "toute moelle osseuse de la crête iliaque", "toute moelle osseuse crête iliaque",
            "toute moelle osseuse de la crête iliaque.", "moelle osseuse de la crête iliaque", "la moelle osseuse de la crête iliaque",
            "moelle osseuse de la crête iliaque."]

query_embedding = model.encode('all iliac crest bone marrow')
passage_embedding = model.encode(passages)

print("Similarity:", util.cos_sim(query_embedding, passage_embedding))
