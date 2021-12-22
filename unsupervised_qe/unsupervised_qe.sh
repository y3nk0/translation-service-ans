## Preprocess the input data
for LANG in $SRC_LANG $TGT_LANG; do
  perl $MOSES_DECODER/scripts/tokenizer/tokenizer.perl -threads 80 -a -l $LANG < $INPUT.$LANG > $TMP/preprocessed.tok.$LANG
  python $BPE_ROOT/apply_bpe.py -c ${BPE} < $TMP/preprocessed.tok.$LANG > $TMP/preprocessed.tok.bpe.$LANG
done

## Binarize the data for faster translation:
fairseq-preprocess --srcdict $MODEL_DIR/dict.$SRC_LANG.txt --tgtdict $MODEL_DIR/dict.$TGT_LANG.txt --source-lang ${SRC_LANG} --target-lang ${TGT_LANG} --testpref $TMP/preprocessed.tok.bpe --destdir $TMP/bin --workers 4

## Translate
CUDA_VISIBLE_DEVICES=$GPU fairseq-generate $TMP/bin --path ${MODEL_DIR}/${SRC_LANG}-${TGT_LANG}.pt --beam 5 --source-lang $SRC_LANG --target-lang $TGT_LANG --no-progress-bar --unkpen 5 > $TMP/fairseq.out
grep ^H $TMP/fairseq.out | cut -d- -f2- | sort -n | cut -f3- > $TMP/mt.out


## Post-process
sed -r 's/(@@ )| (@@ ?$)//g' < $TMP/mt.out | perl $MOSES_DECODER/scripts/tokenizer/detokenizer.perl
-l $TGT_LANG > $OUTPUT_DIR/mt.out


### Produce uncertainty estimates

#Scoring
#Make temporary files to store the translations repeated N times.
python ${SCRIPTS}/scripts/uncertainty/repeat_lines.py -i $TMP/preprocessed.tok.bpe.$SRC_LANG -n $DROPOUT_N -o $TMP/repeated.$SRC_LANG
python ${SCRIPTS}/scripts/uncertainty/repeat_lines.py -i $TMP/mt.out -n $DROPOUT_N -o $TMP/repeated.$TGT_LANG

fairseq-preprocess --srcdict ${MODEL_DIR}/dict.${SRC_LANG}.txt $TGT_DIC --source-lang ${SRC_LANG} --target-lang ${TGT_LANG} --testpref ${TMP}/repeated --destdir ${TMP}/bin-repeated


## Produce model scores for the generated translations using --retain-dropout option to apply dropout at inference time:
CUDA_VISIBLE_DEVICES=${GPU} fairseq-generate ${TMP}/bin-repeated --path ${MODEL_DIR}/${LP}.pt --beam 5 --source-lang $SRC_LANG --target-lang $TGT_LANG --no-progress-bar --unkpen 5 --score-reference --retain-dropout --retain-dropout-modules '["TransformerModel","TransformerEncoder","TransformerDecoder","TransformerEncoderLayer"]' TransformerDecoderLayer --seed 46 > $TMP/dropout.scoring.out

grep ^H $TMP/dropout.scoring.out | cut -d- -f2- | sort -n | cut -f2 > $TMP/dropout.scores


## Compute the mean of the resulting output distribution:
python $SCRIPTS/scripts/uncertainty/aggregate_scores.py -i $TMP/dropout.scores -o $OUTPUT_DIR/dropout.scores.mean -n $DROPOUT_N