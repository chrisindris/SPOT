# Usage: ./spot_train_eval.sh <outfile_name.txt>
{ time python spot_train.py ; } 2>&1 | tee -a outfile.txt
{ time python spot_inference.py ; } 2>&1 | tee -a outfile.txt
{ time python eval.py ; } 2>&1 | tee -a outfile.txt
