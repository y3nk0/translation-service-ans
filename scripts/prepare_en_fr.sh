SCRIPTS=../mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=../subword-nmt/subword_nmt
BPE_TOKENS=50000

src=en
tgt=fr
lang=en-fr
tmp=tmp
orig=orig

mkdir -p $tmp $prep

#test=~/Documents/icd/final_ICD11.en
#cat $test | perl $TOKENIZER -threads 8 -a -l en >> test.$lang.tok.en

#cd $orig

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    #cat 'data/UFAL_medical_shuffled/medical_UFAL'.$l | \
    cat 'data_2021/good_training_2021+PATR'.$l | \
    #cat 'data/final_training_NO_TEST_IN'.$l | \
        perl $NORM_PUNC $l | \
        perl $REM_NON_PRINT_CHAR | \
        perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    #cat 'data/icd_10'.$l | \
	#perl $NORM_PUNC $l | \
	#perl $REM_NON_PRINT_CHAR | \
	#perl $TOKENIZER -threads 8 -a -l $l >> $tmp/valid.tags.$lang.tok.$l
done

echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%500 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk '{if (NR%500 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done

TRAIN=$tmp/train.fr-en
#BPE_CODE=$prep/code
BPE_CODE=../wmt14.en-fr.fconv-py/bpecodes

rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

#echo "learn_bpe.py on ${TRAIN}..."
#python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

#python $BPEROOT/apply_bpe.py -c $BPE_CODE < test.$lang.tok.en > data/bpe.test

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt data_2021/train 1 5000
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt data_2021/valid 1 5000

