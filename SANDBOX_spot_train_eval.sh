import sys
# Usage: ./spot_train_eval.sh <outfile_name.txt>

# TODO:
# - specify the GPU to use (presently in the config you can only specify the number)
# - Permit the specification of modifications to the config (but not by creating a new file)
# - run a parameter sweep
# - handle the THUMOS and i-5O datasets

echo "$0" "${@:2}"

#time python utils/arguments.py "${@:2:}"

