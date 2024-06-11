# Usage: ./spot_train_eval.sh <outfile_name.txt> [<config_argument> ...]

# TODO:
# - specify the GPU to use (presently in the config you can only specify the number)
# - Permit the specification of modifications to the config (but not by creating a new file)
# - run a parameter sweep
# - handle the THUMOS and i-5O datasets

export CUDA_VISIBLE_DEVICES=1

echo "$0" "$@" | tee "output/experiments/${1}" # record the command used to run this script.

{ time python spot_train.py "${@:2}" ; } 2>&1 | tee -a "output/experiments/${1}"
{ time python spot_inference.py "${@:2}" ; } 2>&1 | tee -a "output/experiments/${1}"
{ time python eval.py "${@:2}" ; } 2>&1 | tee -a "output/experiments/${1}"
