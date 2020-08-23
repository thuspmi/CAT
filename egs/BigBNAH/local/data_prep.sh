#!/bin/bash

# Copyright 2017 Xingyu Na
# Apache 2.0

. ./path.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <audio-path> <text-path>"
  exit 1;
fi

audio_dir=$1
text=$2

train_dir=data/local/train
dev_dir=data/local/dev
CNTVHNTest_dir=data/local/CNTVHNTest
KinglineTest_dir=data/local/KinglineTest
tmp_dir=data/local/tmp

mkdir -p $train_dir
mkdir -p $dev_dir
mkdir -p $CNTVHNTest_dir
mkdir -p $KinglineTest_dir
mkdir -p $tmp_dir

# data directory check
if [ ! -d $audio_dir ] || [ ! -f $text ]; then
  echo "Error: $0 requires two directory arguments"
  exit 1;
fi

# find wav audio file for train, dev and test resp.
find $audio_dir -iname "*.wav" > $tmp_dir/wav.flist
n=`cat $tmp_dir/wav.flist | wc -l`
#[ $n -ne 141925 ] && \
# echo Warning: expected 141925 data data files, found $n

grep -i "wav/train" $tmp_dir/wav.flist > $train_dir/wav.flist || exit 1;
grep -i "wav/dev" $tmp_dir/wav.flist > $dev_dir/wav.flist || exit 1;
grep -i "wav/CNTVHNTest" $tmp_dir/wav.flist > $CNTVHNTest_dir/wav.flist || exit 1;
grep -i "wav/KinglineTest" $tmp_dir/wav.flist > $KinglineTest_dir/wav.flist || exit 1;

rm -r $tmp_dir

# Transcriptions preparation
for dir in $train_dir $dev_dir $CNTVHNTest_dir $KinglineTest_dir; do
  echo Preparing $dir transcriptions
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{i=NF-1;printf("%s %s\n",$NF,$i)}' > $dir/utt2spk_all
  paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
  utils/filter_scp.pl -f 1 $dir/utt.list $text > $dir/transcripts.txt
  awk '{print $1}' $dir/transcripts.txt > $dir/utt.list
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/utt2spk_all | sort -u > $dir/utt2spk
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
  sort -u $dir/transcripts.txt > $dir/text
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
done

mkdir -p data/train data/dev data/CNTVHNTest data/KinglineTest

for f in spk2utt utt2spk wav.scp text; do
  cp $train_dir/$f data/train/$f || exit 1;
  cp $dev_dir/$f data/dev/$f || exit 1;
  cp $CNTVHNTest_dir/$f data/CNTVHNTest/$f || exit 1;
  cp $KinglineTest_dir/$f data/KinglineTest/$f || exit 1;
done

echo "$0: BigBNAH data preparation succeeded"
exit 0;
