# Usage: ./spot_train_eval.sh <GPU> <outfile_name.txt> </path/to/config.yaml> [<config_argument> ...]
# - <config_argument> has the form key1.key2.key3...=value, eg. dataset.name='new name'

# TODO:
# - run a parameter sweep
# - handle the THUMOS and i-5O datasets

export CUDA_VISIBLE_DEVICES="$1"

#echo "$0" "$@" | tee "output/experiments/${2}" # record the command used to run this script.

{ time python spot_train.py "${@:3}" ; } 2>&1 | tee -a "output/experiments/${2}"
{ time python spot_inference.py "${@:3}" ; } 2>&1 | tee -a "output/experiments/${2}"
{ time python eval.py "${@:3}" ; } 2>&1 | tee -a "output/experiments/${2}"
