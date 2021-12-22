#!/bin/bash
./prepare_en_fr.sh

FAIRSEQ=~/Documents/icd/fairseq
PRETRAINED_MODEL=~/Documents/icd/wmt14.en-fr.fconv-py

SEED=1

EXP_NAME=fine-tune

SRC=en
TRG=fr

SRC_VOCAB=$PRETRAINED_MODEL/dict.$SRC.txt
TRG_VOCAB=$PRETRAINED_MODEL/dict.$TRG.txt

PRETRAINED_MODEL_FILE=$PRETRAINED_MODEL/model.pt

CORPUS_DIR=~/Documents/icd/french/data_2021
DATA_DIR=~/Documents/icd/french/data_bin_2021_PATR

TRAIN_PREFIX=$CORPUS_DIR/train
DEV_PREFIX=$CORPUS_DIR/valid

mkdir -p $CORPUS_DIR
mkdir -p $DATA_DIR

######################################
# Preprocessing
######################################
CUDA_VISIBLE_DEVICES=0 fairseq-preprocess \
    --source-lang $SRC \
    --target-lang $TRG \
    --trainpref $TRAIN_PREFIX \
    --validpref $DEV_PREFIX \
    --destdir $DATA_DIR \
    --srcdict $SRC_VOCAB \
    --tgtdict $TRG_VOCAB \
    --workers `nproc` \


######################################
# Training
######################################
CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_DIR \
    --restore-file $PRETRAINED_MODEL_FILE \
    --lr 0.5 --clip-norm 0.1 --dropout 0.1 --max-tokens 3000 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --lr-scheduler fixed --force-anneal 50 \
    --arch fconv_wmt_en_fr \
    --reset-optimizer \
    --skip-invalid-size-inputs-valid-test \
    --save-dir checkpoints/fconv_wmt_en_fr_medical_dicts_NEW_VALID_2021_PATR


from fairseq.models.fconv import FConvModel
fr2en = FConvModel.from_pretrained(
  '/path/to/checkpoints',
  checkpoint_file='checkpoint_best.pt',
  data_name_or_path='data-bin/wmt17_zh_en_full',
  bpe='subword_nmt',
  bpe_codes='data-bin/wmt17_zh_en_full/zh.code'
)
fr2en.translate('你好 世界')

######################################
# Averaging
######################################
#rm -rf $MODEL_DIR/average
#mkdir -p $MODEL_DIR/average
#python3 $FAIRSEQ/scripts/average_checkpoints.py --inputs $MODEL_DIR --output $MODEL_DIR/average/average.pt --num-update-checkpoints 8

