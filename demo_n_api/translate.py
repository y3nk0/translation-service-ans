#from transformers import MarianTokenizer, MarianMTModel
import os
from typing import List
import torch
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

#en2fr = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', source_lang='en', target_lang='fr', tokenizer='moses', bpe='subword_nmt', checkpoint_file='/home/dbnet/kostas/icd11/checkpoints/fconv_wmt_en_fr_medical_dicts_5th_round_2021/checkpoint_best.pt')
#en2fr = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', source_lang='en', target_lang='fr', tokenizer='moses', bpe='subword_nmt', checkpoint_file='/home/dbnet/kostas/icd11/round_3rd/1_checkpoint_best.pt')
#en2fr = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', source_lang='en', target_lang='fr', tokenizer='moses', bpe='subword_nmt', checkpoint_file='/home/dbnet/kostas/icd11/round_4th/2_checkpoint_best.pt')
#en2fr.eval()  # disable dropout

class Translator():
    def __init__(self, models_dir):
        en2fr = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', source_lang='en', target_lang='fr', tokenizer='moses', bpe='subword_nmt', checkpoint_file='/home/dbnet/kostas/icd11/checkpoints/fconv_wmt_en_fr_medical_dicts_5th_round_2021/checkpoint_best.pt')
        #en2fr = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', source_lang='en', target_lang='fr', tokenizer='moses', bpe='subword_nmt', checkpoint_file='/home/dbnet/kostas/icd11/round_3rd/1_checkpoint_best.pt')
        #en2fr = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', source_lang='en', target_lang='fr', tokenizer='moses', bpe='subword_nmt', checkpoint_file='/home/dbnet/kostas/icd11/round_4th/2_checkpoint_best.pt')
        en2fr.eval()

        self.model = en2fr
        self.models = {}
        self.models_dir = models_dir

    def get_supported_langs(self):
        routes = [x.split('-')[-2:] for x in os.listdir(self.models_dir)]
        return routes

    def load_model(self, trModel):
        #model = f'opus-mt-{route}'
        #path = os.path.join(self.models_dir,model)
        #try:
        #    model = MarianMTModel.from_pretrained(path)
        #    tok = MarianTokenizer.from_pretrained(path)
        #except:
        #    return 0,f"Make sure you have downloaded model for {route} translation"
        #self.models[route] = (model,tok)
        #return 1,f"Successfully loaded model for {route} transation"
        if trModel=="fr":
            from fairseq.models.fconv import FConvModel
            en2fr = FConvModel.from_pretrained('/home/dbnet/kostas/icd11/checkpoints/fconv_wmt_en_fr_medical_dicts_5th_round_reverse_2021/', checkpoint_file='checkpoint_best.pt', data_name_or_path='/home/dbnet/kostas/icd11/data_bin_reverse_2021', source_lang='fr', target_lang='en', bpe='subword_nmt', bpe_codes='/home/dbnet/kostas/wmt14.en-fr.fconv-py/bpecodes')
        else:
            en2fr = torch.hub.load('pytorch/fairseq', 'conv.wmt14.en-fr', source_lang='en', target_lang='fr', tokenizer='moses', bpe='subword_nmt', checkpoint_file='/home/dbnet/kostas/icd11/checkpoints/fconv_wmt_en_fr_medical_dicts_5th_round_2021/checkpoint_best.pt')
        en2fr.eval()
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

    def translate_fairseq(self, text, multiple):
        
        if multiple=="true":
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
            for ind, fr in enumerate(fr_bin):
                fr_sample = fr_bin[ind]['tokens']
                fr_bpe = self.model.string(fr_sample)
                fr_toks = self.model.remove_bpe(fr_bpe)
                fr = self.model.detokenize(fr_toks)
                #translation += fr + "\n"
                translations.append(fr)
            translations = list(set(translations))
            
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
                #for beam in [5,10,100,200,1000,5000]:
                translation = self.model.translate(text)
            #translation += tr + "\n"

        return translation
