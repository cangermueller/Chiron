#!/usr/bin/env bash

set -e
shopt -s extglob

check=1
function run {
  local cmd=$@
  echo
  echo "#################################"
  echo $cmd
  echo "#################################"
  eval $cmd
  if [ $check -ne 0 -a $? -ne 0 ]; then
    1>&2 echo "Command failed!"
    exit 1
  fi
}

data_dir="../data"

function train {
  local model_name=${1:-"model1"}

  out_dir="./$model_name"
  run "rm -rf $out_dir && mkdir -p $out_dir"
  cmd="chiron train
    --data_dir $data_dir
    --log_dir $out_dir/log
    --cache_dir $out_dir/cache
    --model_name $model_name
    --batch_size 32
    --max_samples 1000
  "
  run $cmd
}

train
