## Generation
## Produce multiple translation hypotheses for the same source using --retain-dropout option:

CUDA_VISIBLE_DEVICES=${GPU} fairseq-generate ${TMP}/bin-repeated --path ${MODEL_DIR}/${LP}.pt --beam 5 --source-lang $SRC_LANG --target-lang $TGT_LANG --no-progress-bar --retain-dropout --unkpen 5 --retain-dropout-modules TransformerModel TransformerEncoder TransformerDecoder TransformerEncoderLayer TransformerDecoderLayer --seed 46 > $TMP/dropout.generation.out

grep ^H $TMP/dropout.generation.out | cut -d- -f2- | sort -n | cut -f3- > $TMP/dropout.hypotheses_

sed -r 's/(@@ )| (@@ ?$)//g' < $TMP/dropout.hypotheses_ | perl $MOSES_DECODER/scripts/tokenizer/detokenizer.perl -l $TGT_LANG > $TMP/dropout.hypotheses

## Compute similarity between multiple hypotheses corresponding to the same source sentence using Meteor evaluation metric:
python meteor.py -i $TMP/dropout.hypotheses -m <path_to_meteor_installation> -n $DROPOUT_N -o $OUTPUT_DIR/dropout.gen.sim.meteor