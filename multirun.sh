# Usage: ./multirun.sh <GPU-index>

# ./spot_train_eval.sh 0 TRIAL_0.9_parallel.txt ./configs/anet.yaml \
  # dataset.training.unlabel_percent=0.9 \
  # dataset.testing.unlabel_percent=0.9 \
  # dataset.training.output_path="./output_2/" \
  # dataset.training.output_path="./output_2/" \
  # training.checkpoint_path="./output_2/"

# ./spot_train_eval.sh 1 TRIAL_0.8_parallel.txt ./configs/anet.yaml \
  # dataset.training.unlabel_percent=0.8 \
  # dataset.testing.unlabel_percent=0.8 \
  # dataset.training.output_path="./output_2/" \
  # dataset.training.output_path="./output_2/" \
  # training.checkpoint_path="./output_2/"

gpu=$1

if (( gpu == 0 )); then
  output_pth="./output/"

  

else
  output_pth="./output_2/"
 
  ./spot_train_eval.sh "$gpu" latest_trial-training_bloss_new-toploss-unlabel_percent_0.4-max_epoch_30.txt ./configs/anet.yaml \
    dataset.training.unlabel_percent=0.4 \
    dataset.testing.unlabel_percent=0.4 \
    training.max_epoch=30 \
    dataset.training.output_path=$output_pth \
    dataset.training.output_path=$output_pth \
    training.checkpoint_path=$output_pth

  ./spot_train_eval.sh "$gpu" latest_trial-training_bloss_new-toploss-unlabel_percent_0.3-max_epoch_30.txt ./configs/anet.yaml \
    dataset.training.unlabel_percent=0.3 \
    dataset.testing.unlabel_percent=0.3 \
    training.max_epoch=30 \
    dataset.training.output_path=$output_pth \
    dataset.training.output_path=$output_pth \
    training.checkpoint_path=$output_pth

  ./spot_train_eval.sh "$gpu" latest_trial-training_bloss_new-toploss-unlabel_percent_0.2-max_epoch_30.txt ./configs/anet.yaml \
    dataset.training.unlabel_percent=0.2 \
    dataset.testing.unlabel_percent=0.2 \
    training.max_epoch=30 \
    dataset.training.output_path=$output_pth \
    dataset.training.output_path=$output_pth \
    training.checkpoint_path=$output_pth

  ./spot_train_eval.sh "$gpu" latest_trial-training_bloss_new-toploss-unlabel_percent_0.1-max_epoch_30.txt ./configs/anet.yaml \
    dataset.training.unlabel_percent=0.1 \
    dataset.testing.unlabel_percent=0.1 \
    training.max_epoch=30 \
    dataset.training.output_path=$output_pth \
    dataset.training.output_path=$output_pth \
    training.checkpoint_path=$output_pth

fi
