# Usage: ./spot_train_eval.sh <GPU> <outfile_name.txt> </path/to/config.yaml> [<config_argument> ...]
# - <config_argument> has the form key1.key2.key3...=value, eg. dataset.name='new name'

# TODO:
# - specify the GPU to use (presently in the config you can only specify the number) 
# - run a parameter sweep
# - handle the THUMOS and i-5O datasets

export CUDA_VISIBLE_DEVICES="$1"

echo $CUDA_VISIBLE_DEVICES
echo "$0" "$@" # record the command used to run this script.

python SANDBOX_spot_train.py "${@:3}"
