# Usage: ./spot_train_eval.sh <GPU> <outfile_name.txt> </path/to/config.yaml> [<config_argument> ...]
# - <config_argument> has the form key1.key2.key3...=value, eg. dataset.name='new name'

# TODO:
# - specify the GPU to use (presently in the config you can only specify the number) 
# - run a parameter sweep
# - handle the THUMOS and i-5O datasets

declare -a dataset_name=("ANet1" "ANet2")
declare -a dataset_training_unlabel_percent=(0.0, 0.1)

gpu=$1

if (( gpu == 0 )); then
  output_pth="./output/"

  for x in {0..1},{0..1}
  do
    IFS=',' read -r a b < <(echo $x)
    #echo $a $b

    echo "./spot_train_eval.sh "$gpu" latest_trial-training_bloss_new-toploss-name_${dataset_name[$a]}-unlabel_percent_${dataset_training_unlabel_percent[$b]}.txt ./configs/anet.yaml \
      dataset.name=${dataset_name[$a]} \
      dataset.training.unlabel_percent=${dataset_training_unlabel_percent[$b]} \
      dataset.testing.unlabel_percent=0.0"

  done 

else
  output_pth="./output_2/"

fi


#python SANDBOX_spot_train.py "${@:3}"
