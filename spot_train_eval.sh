# Usage: ./spot_train_eval.sh <outfile_name.txt>

echo "$0" "$@" | tee -a "output/experiments/${1}" # record the command used to run this script.

{ time python spot_train.py ; } 2>&1 | tee -a "output/experiments/${1}"
{ time python spot_inference.py ; } 2>&1 | tee -a "output/experiments/${1}"
{ time python eval.py ; } 2>&1 | tee -a "output/experiments/${1}"
