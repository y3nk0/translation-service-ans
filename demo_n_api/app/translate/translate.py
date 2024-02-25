import os
from typing import List
import torch
import math
from nltk.tokenize import sent_tokenize
from transformers import pipeline

from .conf_processing import replace_with_right_trans, disorder_fix, ar2_remove_article_from_start, replace_1st_letter_with_lower

from flask import jsonify
from sentence_transformers import SentenceTransformer, util

import sys
# sys.path.insert(0,'..')
# from sequence_scorer import SequenceScorer

model_sim = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual', device='cpu')
#en2fr = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', source_lang='en', target_lang='fr', tokenizer='moses', bpe='subword_nmt', checkpoint_file='/home/dbnet/kostas/icd11/checkpoints/fconv_wmt_en_fr_medical_dicts_5th_round_2021/checkpoint_best.pt')
#en2fr = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', source_lang='en', target_lang='fr', tokenizer='moses', bpe='subword_nmt', checkpoint_file='/home/dbnet/kostas/icd11/round_3rd/1_checkpoint_best.pt')
#en2fr = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', source_lang='en', target_lang='fr', tokenizer='moses', bpe='subword_nmt', checkpoint_file='/home/dbnet/kostas/icd11/round_4th/2_checkpoint_best.pt')
#en2fr.eval()  # disable dropout


class Translator:
    def __init__(self, models_dir):
        # self.model = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', encoding='utf8', tokenizer='moses', bpe='subword_nmt', source_lang='en', target_lang='fr', checkpoint_file='E:/icd11/models/round_5th_checkpoint_best.pt')
        # self.model = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt', source_lang='en', target_lang='fr', checkpoint_file='E:/icd11/models/round_3rd_checkpoint_best.pt|E:/icd11/models/round_4th_checkpoint_best.pt|E:/icd11/models/round_5th_checkpoint_best.pt')
        self.model = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt', source_lang='en', target_lang='fr', checkpoint_file='/app/models/round_3rd_checkpoint_best.pt:/app/models/round_4th_checkpoint_best.pt:/app/models/round_5th_checkpoint_best.pt')
        self.model.eval()  # disable dropout
        # self.model.cuda()

        model_checkpoint = "Helsinki-NLP/opus-mt-fr-en"
        self.rev = pipeline("translation", model=model_checkpoint)

        self.models = {}
        self.models_dir = models_dir

    def get_supported_langs(self):
        routes = [x.split('-')[-2:] for x in os.listdir(self.models_dir)]
        return routes

    def load_model(self, trModel):
        if trModel=="fr":
            # from fairseq.models.fconv import FConvModel
            # en2fr = FConvModel.from_pretrained('E:/icd11/models/checkpoints/fconv_wmt_en_fr_medical_dicts_5th_round_reverse_2021/', checkpoint_file='checkpoint_best.pt', data_name_or_path='E:/icd11/models/data_bin_reverse_2021', source_lang='fr', target_lang='en', bpe='subword_nmt', bpe_codes='/home/dbnet/kostas/wmt14.en-fr.fconv-py/bpecodes')
            model_checkpoint = "Helsinki-NLP/opus-mt-fr-en"
            self.rev = pipeline("translation", model=model_checkpoint)
        elif trModel=="en":
            en2fr = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', tokenizer='moses', bpe='subword_nmt', source_lang='en', target_lang='fr', checkpoint_file='/app/models/round_5th_checkpoint_best.pt')
            en2fr.eval()
            # en2fr.cuda()
            self.model = en2fr
        else:
            en2fr = torch.hub.load('pytorch/fairseq', 'transformer.wmt14.en-fr', tokenizer='moses', bpe='fastbpe', source_lang='en', target_lang='fr', checkpoint_file='/app/models/transf/checkpoint_best.pt')
            en2fr.eval()
            # en2fr.cuda()
            self.model = en2fr
        return "1"

    def translate(self, source, target, text):
        route = f'{source}-{target}'
        if not self.models.get(route):
            success_code, message = self.load_model(route)
            if not success_code:
                return message

        batch = self.models[route][1].prepare_seq2seq_batch(src_texts=list([text]), return_tensors="pt")
        gen = self.models[route][0].generate(**batch)
        words: List[str] = self.models[route][1].batch_decode(gen, skip_special_tokens=True)
        return words

    def translate_reverse(self, text, multiple, apply_rules):
        trans = ""
        if len(text)>500:
            sents = text.split(". ")
            for sent in sents:
                translation = self.rev(sent)
                translation = translation[0]['translation_text']
                trans += translation.strip(".").strip() + ". "
        else:
            trans = self.rev(text)
            trans = trans[0]['translation_text']
        return trans

    def translate_fairseq(self, text, multiple, apply_rules, metric):
        score = 9999
        if len(text)<5000:
            texts = text.split("\n")
            trans = []
            scores = []
            for text in texts:
                if multiple=="True":
                    #translation = en2fr.translate(text, beam=5000)
                    en_toks = self.model.tokenize(text)
                    # Manually apply BPE:
                    en_bpe = self.model.apply_bpe(en_toks)
                    # assert en_bpe == 'H@@ ello world !'
                    # Manually binarize:
                    en_bin = self.model.binarize(en_bpe)
                    # Generate five translations with top-k sampling:
                    #fr_bin = en2fr.generate(en_bin, beam=20, nbest=1, sampling=True, sampling_topk=150)
                    fr_bin = self.model.generate(en_bin, stochastic_beam_search=True, beam=10, nbest=5, no_early_stopping=True, unnormalized=True, sampling_temperature=0.3)

                    # Convert one of the samples to a string and detokenize
                    translation = ""
                    translations = []
                    scored = {}
                    scored2 = {}
                    for ind, fr in enumerate(fr_bin):
                        fr_sample = fr_bin[ind]['tokens']
                        fr_bpe = self.model.string(fr_sample)
                        fr_toks = self.model.remove_bpe(fr_bpe)
                        fr = self.model.detokenize(fr_toks)
                        #translation += fr + "\n"
                        if metric=="True":
                            score = fr_bin[ind]['score'].item()
                            score = math.exp(score)
                            score = "{:.2f}".format(score)

                            # model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu', cache_folder='/')

                            corpus_embedding = model_sim.encode(text, convert_to_tensor=True, normalize_embeddings=True)
                            query_embedding = model_sim.encode(fr, convert_to_tensor=True, normalize_embeddings=True)
                            hits = util.cos_sim(query_embedding, corpus_embedding)
                            score2 = "{:.2f}".format(float(hits[0][0].item()))
                            if fr not in translations:
                                translations.append(fr+" l:"+score+" M:"+score2)
                                scored[fr] = float(score)
                                scored2[fr] = float(score2)

                        else:
                            if fr not in translations:
                                translations.append(fr)
                    # translations = list(set(translations))
                    new_trans = []
                    if metric=="True":
                        for w in sorted(scored2, key=scored2.get, reverse=True):
                            new_trans.append(w+" L:"+"{:.2f}".format(scored[w])+" M:" +"{:.2f}".format(scored2[w]))
                        translation = "\n".join(new_trans)
                    else:
                        translation = "\n".join(translations)



                else:
                    #translation = ""

                    if len(text)>700:
                        sents = sent_tokenize(text)
                        translation = ""
                        for sent in sents:
                            if len(sent)>500:
                                new_sents = sent.split(";")
                                for new_sent in new_sents:
                                    translation += self.model.translate(new_sent) + "; "

                            else:
                                translation += self.model.translate(sent) + " "
                        translation = translation.strip()
                    else:
                        # en = self.model.encode(text)
                        # output = self.model.generate(en, stochastic_beam_search=True, beam=10, nbest=5, no_early_stopping=True, unnormalized=True, sampling_temperature=0.3)
                        # translation = self.model.decode(output[0]['tokens'])

                        en_toks = self.model.tokenize(text)
                        # Manually apply BPE:
                        en_bpe = self.model.apply_bpe(en_toks)
                        en_bin = self.model.binarize(en_bpe)
                        fr_bin = self.model.generate(en_bin, stochastic_beam_search=True, beam=10, nbest=5, no_early_stopping=True, unnormalized=True, sampling_temperature=0.3)

                        # Convert one of the samples to a string and detokenize
                        translation = ""
                        translations = []
                        scored = {}
                        scored2 = {}
                        for ind, fr in enumerate(fr_bin):
                            fr_sample = fr_bin[ind]['tokens']
                            fr_bpe = self.model.string(fr_sample)
                            fr_toks = self.model.remove_bpe(fr_bpe)
                            fr = self.model.detokenize(fr_toks)
                            #translation += fr + "\n"

                            score = fr_bin[ind]['score'].item()
                            score = math.exp(score)
                            score = "{:.2f}".format(score)

                            corpus_embedding = model_sim.encode(text, convert_to_tensor=True, normalize_embeddings=True)
                            query_embedding = model_sim.encode(fr, convert_to_tensor=True, normalize_embeddings=True)
                            hits = util.cos_sim(query_embedding, corpus_embedding)
                            score2 = "{:.2f}".format(float(hits[0][0].item()))
                            if fr not in translations:
                                translations.append(fr+" l:"+score+" M:"+score2)
                                scored[fr] = float(score)
                                scored2[fr] = float(score2)

                        sorted_scored2 = sorted(scored2, key=scored2.get, reverse=True)
                        best_translation_m = sorted_scored2[0]

                        sorted_scored = sorted(scored, key=scored.get, reverse=True)
                        best_translation_l = sorted_scored[0]
                        translation = best_translation_l

                        # if scored2[best_translation_m]<1:
                        #     if scored2[best_translation_m]>scored2[best_translation_l]:
                        #         translation = best_translation_m

                        if metric=="True":

                            # en_toks = self.model.tokenize(text)
                            # # Manually apply BPE:
                            # en_bpe = self.model.apply_bpe(en_toks)
                            # # assert en_bpe == 'H@@ ello world !'
                            # # Manually binarize:
                            # en_bin = self.model.binarize(en_bpe)
                            # # Generate five translations with top-k sampling:
                            # #fr_bin = en2fr.generate(en_bin, beam=20, nbest=1, sampling=True, sampling_topk=150)
                            # # fr_bin = self.model.generate(en_bin, stochastic_beam_search=True)
                            # fr_bin = self.model.generate(en_bin, stochastic_beam_search=True, beam=10, nbest=5, no_early_stopping=True, unnormalized=True, sampling_temperature=0.3)
                            # score = fr_bin[0]['score'].item()
                            # score = math.exp(score)
                            # score = "{:.2f}".format(score)
                            #
                            # corpus_embedding = model_sim.encode(text, convert_to_tensor=True, normalize_embeddings=True)
                            # query_embedding = model_sim.encode(translation, convert_to_tensor=True, normalize_embeddings=True)
                            # hits = util.cos_sim(query_embedding, corpus_embedding)
                            # score2 = "{:.2f}".format(float(hits[0][0].item()))
                            score = scored[translation]
                            score2 = scored2[translation]

                if apply_rules=="True":
                    # translation = sc6_replace_comp_symbols(translation)
                    translation = replace_with_right_trans(text,translation)
                    # translation = um2_temperature(translation)
                    translation = disorder_fix(text,translation)
                    # translation = gr1_replace_greek_letter_with_long(translation)
                    translation = ar2_remove_article_from_start(translation)
                    translation = replace_1st_letter_with_lower(translation)

                trans.append(translation)
            translation = "\n".join(trans)
        else:
            translation = "Text too long to translate."
            #translation += tr + "\n"

        if metric=="True":
            return translation, score, score2
        return translation
