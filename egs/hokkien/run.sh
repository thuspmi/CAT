#!/bin/bash

# Copyright 2018-2019 Tsinghua University, Author: Keyu An
# Apache 2.0.
# This script implements CTC-CRF training on Aishell dataset.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh

export LANG=C.UTF-8
tm_fm=t"+%Y-%m-%d %H:%M:%S"
stage=1
dir=`pwd -P`

hokkien_wav_dir=/mnt/nas_workspace2/spmiFeature/data_16k/hokkien/wav
hokkien_trans_dir=/mnt/nas_workspace2/spmiFeature/data_16k/hokkien/transcript
hokkien_lang_dir=data/lang_TLG_hokkien_3gram_0_0_0


gpu_batch_size=64 
feature_size=129 
output_unit=218 
hdim=320
layers=6
dropout=0.5  
origin_lr=0.001
stop_lr=0.00001
dist_url='tcp://localhost:23457'  
world_size=4
start_rank=0
chunk_size=40
jitter_range=10
cate=31
triple_speed = false


if [ ! -d ctc-crf ]; then
    ln -s ../../scripts/ctc-crf ctc-crf
fi

if [ ! -d steps ]; then
    ln -s $KALDI_ROOT/egs/wsj/s5/steps steps
fi

if [ ! -d utils ]; then
    ln -s $KALDI_ROOT/egs/wsj/s5/utils utils
fi

if [ $stage -le 1 ]; then
  echo "$(date "$tm_fmt")--->Data Preparation and FST Construction"
  # # Use the same datap prepatation script from Kaldi
  local/data_prep.sh $hokkien_wav_dir $hokkien_trans_dir/hokkien.txt || exit 1;
fi

if [ $triple_speed == true ]; then
  data_tr=train_sp
  data_cv=dev_sp
else
  data_tr=train
  data_cv=dev
fi

f [ $stage -le 2 ]; then
  echo "$(date "$tm_fmt")--->FBank Feature Generation"
  # Split the whole training data into training (95%) and cross-validation (5%) sets
  if [ $triple_speed == true ]; then
    utils/data/perturb_data_dir_speed_3way.sh data/train data/$data_tr
    utils/data/perturb_data_dir_speed_3way.sh data/dev data/$data_cv
    echo "$(date "$tm_fmt")---> preparing directo<F10>ry for speed-perturbed data done"
  fi

  # Generate the fbank features; by default 40-dimensional fbanks on each frame
  fbankdir=fbank
  for set in $data_tr $data_cv; do
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj $nj data/$set exp/make_fbank_pitch/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank_pitch/$set $fbankdir || exit 1;
  done
fi

if [ $stage -le 3 ]; then
  python3 ctc-crf/prep_ctc_trans.py $lang_dir/lexicon_numbers.txt data/$data_tr/text "<UNK>" > data/$data_tr/text_number
  python3 ctc-crf/prep_ctc_trans.py $lang_dir/lexicon_numbers.txt data/$data_cv/text "<UNK>" > data/$data_cv/text_number
  echo "$(date "$tm_fmt")--->convert text_number finished"

  # prepare denominator
  python3 ctc-crf/prep_ctc_trans.py $lang_dir/lexicon_numbers.txt data/train/text "<UNK>" > data/train/text_number
  cat data/train/text_number | sort -k 2 | uniq -f 1 > data/train/unique_text_number
  mkdir -p data/den_meta
  chain-est-phone-lm ark:data/train/unique_text_number data/den_meta/phone_lm.fst
  python3  ctc-crf/ctc_token_fst_corrected.py den $lang_dir/tokens.txt | fstcompile | fstarcsort --sort_type=olabel > data/den_meta/T_den.fst
  fstcompose data/den_meta/T_den.fst data/den_meta/phone_lm.fst > data/den_meta/den_lm.fst
  echo "$(date "$tm_fmt")--->prepare denominator finished"
  
fi

if [ $stage -le 4 ]; then 
  ../../src/ctc_crf/path_weight/build/path_weight data/$data_tr/text_number data/den_meta/phone_lm.fst > data/$data_tr/weight
  ../../src/ctc_crf/path_weight/build/path_weight data/$data_cv/text_number data/den_meta/phone_lm.fst > data/$data_cv/weight
  echo "$(date "$tm_fmt")--->prepare weight finished"

  feats_tr="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:data/$data_tr/utt2spk scp:data/$data_tr/cmvn.scp scp:data/$data_tr/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
  #echo "$(date "$tm_fmt")--->$feats_tr"
  feats_cv="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:data/$data_cv/utt2spk scp:data/$data_cv/cmvn.scp scp:data/$data_cv/feats.scp ark:- \
      | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"

  ark_dir=data/all_ark
  mkdir -p $ark_dir
  copy-feats "$feats_tr" "ark,scp:$ark_dir/tr.ark,$ark_dir/tr.scp" || exit 1
  copy-feats "$feats_cv" "ark,scp:$ark_dir/cv.ark,$ark_dir/cv.scp" || exit 1

  echo "$(date "$tm_fmt")--->copy -feats finished"
  mkdir -p data/pkl
  python3 ctc-crf/convert_to_pickle.py data/all_ark/cv.scp data/$data_cv/text_number data/$data_cv/weight data/pkl/cv.pkl || exit 1
  python3 ctc-crf/convert_to_pickle.py data/all_ark/tr.scp data/$data_tr/text_number data/$data_tr/weight data/pkl/tr.pkl || exit 1
  
  mkdir -p data/pkl/cv
  mkdir -p data/pkl/tr
  python3 ctc-crf/convert_to_pickle_chunk.py data/all_ark/cv.scp data/$data_cv/text_number data/$data_cv/weight $chunk_size data/pkl/cv || exit 1
  python3 ctc-crf/convert_to_pickle_chunk.py data/all_ark/tr.scp data/$data_tr/text_number data/$data_tr/weight $chunk_size data/pkl/tr || exit 1
fi

if [ $stage -le 5 ]; then
    echo "$(date "$tm_fmt")--->nn training RegModel."
    python3 ctc-crf/train_dist.py --model='model_blstm'  --tr_data_path=data/pkl/tr.pkl --dev_data_path=data/pkl/cv.pkl \
      --den_lm_fst_path=data/den_meta/den_lm.fst \
      --data_loader_workers=0  --gpu_batch_size=$gpu_batch_size --feature_size=$feature_size \
      --output_unit=$output_unit --layers=$layers --hdim=$hdim  --dropout=$dropout  --origin_lr=$origin_lr --stop_lr=$stop_lr \
      --dist_url=$dist_url --world_size=$world_size --start_rank=$start_rank  || exit 1
fi

if [ $stage -le 6 ]; then
  fbankdir=fbank
  for set in test; do
    steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj $nj data/$set exp/make_fbank_pitch/$set $fbankdir || exit 1;
    utils/fix_data_dir.sh data/$set || exit;
    steps/compute_cmvn_stats.sh data/$set exp/make_fbank_pitch/$set $fbankdir || exit 1;
  done
fi

if [ $stage -le 7 ]; then
  for set in test; do
    data_test=data/$set
    feats_test="ark,s,cs:apply-cmvn --norm-vars=true --utt2spk=ark:$data_test/utt2spk scp:$data_test/cmvn.scp scp:$data_test/feats.scp ark:- \
       | add-deltas ark:- ark:- | subsample-feats --n=3 ark:- ark:- |"
    mkdir -p data/${set}_data
    copy-feats "$feats_test" "ark,scp:data/${set}_data/${set}.ark,data/${set}_data/${set}.scp" ||exit 1;
  done
fi

if [ $stage -le 8 ]; then
  for set in test; do
    mkdir -p exp/decode_$set/ark
    #if the train.py was not set ctc_crf the model will be stored in models/ctc or rather it will be stored in models/ctc_crf
    #the -data_path should be changed depend on the train type
    python3 ctc-crf/calculate_logits.py  --nj=$nj --input_scp=data/${set}_data/${set}.scp \
    --feature_size=$feature_size --hdim=$hdim --layers=$layers --model=$dir/models/best_model --output_unit=$output_unit --dropout=$dropout \
    --output_dir=exp/decode_${set}/ark || exit 1;
  done
fi

if [ $stage -le 9 ]; then
  # now for decode
  acwt=1.0
  for set in test; do
      bash local/decode.sh $acwt $set $lang_dir $nj  || exit 1;
      grep WER exp/decode_${set}/lattice/cer_* | utils/best_wer.sh || exit 1;
  done
fi

if [ $stage -le 10 ]; then
    python3 ctc-crf/convert_to_regmodel.py --input_model=models/best_model --output_model=models/best_model_reg --layers=$layers || exit 1;
    echo "$(date "$tm_fmt")--->nn training chunk_twin_context."
    python3 ctc-crf/train_twin_context.py --model='chunk_twin_context'  --tr_data_path=data/pkl/tr --dev_data_path=data/pkl/cv \
      --den_lm_fst_path=data/den_meta/den_lm.fst \
      --regmodel_checkpoint=models/best_model_reg  \
      --data_loader_workers=0  --gpu_batch_size=$gpu_batch_size --feature_size=$feature_size \
      --output_unit=$output_unit --layers=$layers --hdim=$hdim  --dropout=$dropout  --origin_lr=$origin_lr --stop_lr=$stop_lr \
	  --default_chunk_size=$chunk_size --jitter_range=$jitter_range  --cate=$cate  || exit 1;

fi

if [ $stage -le 11 ]; then
  for set in test; do
    mkdir -p exp/decode_$set/ark
    #if the train.py was not set ctc_crf the model will be stored in models/ctc or rather it will be stored in models/ctc_crf
    #the -data_path should be changed depend on the train type
    python3 ctc-crf/calculate_logits_chunk_context.py  --nj=$nj --input_scp=data/${set}_data/${set}.scp \
      --feature_size=$feature_size --hdim=$hdim  --model=$dir/models_chunk_twin_context/best_model --output_unit=$output_unit --dropout=$dropout \
      --output_dir=exp/decode_${set}/ark || exit 1;
  done
fi  

if [ $stage -le 12 ]; then  
  # now for decode
  acwt=1.0
  for set in test; do
      bash local/decode.sh $acwt $set $lang_dir $nj || exit 1;
      grep WER  exp/decode_${set}/lattice/cer_* | utils/best_wer.sh || exit 1;
  done
fi

