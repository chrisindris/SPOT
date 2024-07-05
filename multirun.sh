# Usage: ./multirun.sh <GPU-index>
# Purpose: run programs in series (per gpu) and parallel (across GPUs)

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

  ./spot_train_eval.sh "$gpu" TRIAL_scheduler-check.txt ./configs/anet.yaml \
    pretraining.warmup_epoch=1 \
    pretraining.consecutive_warmup_epochs=1 \
    training.max_epoch=1 \
    training.consecutive_train_epochs=1 \
    dataset.training.output_path=$output_pth \
    dataset.testing.output_path=$output_pth \
    training.checkpoint_path=$output_pth

else
  output_pth="./output_2/"
 
  ./spot_train_eval.sh "$gpu" latest_trial-training_bloss_new-toploss-unlabel_percent_0.4-max_epoch_20.txt ./configs/anet.yaml \
    dataset.training.unlabel_percent=0.4 \
    dataset.testing.unlabel_percent=0.4 \
    pretraining.warmup_epoch=18 \
    training.max_epoch=20 \
    dataset.training.output_path=$output_pth \
    dataset.testing.output_path=$output_pth \
    training.checkpoint_path=$output_pth

  ./spot_train_eval.sh "$gpu" latest_trial-training_bloss_new-toploss-unlabel_percent_0.3-max_epoch_20.txt ./configs/anet.yaml \
    dataset.training.unlabel_percent=0.3 \
    dataset.testing.unlabel_percent=0.3 \
    pretraining.warmup_epoch=18 \
    training.max_epoch=20 \
    dataset.training.output_path=$output_pth \
    dataset.testing.output_path=$output_pth \
    training.checkpoint_path=$output_pth

  ./spot_train_eval.sh "$gpu" latest_trial-training_bloss_new-toploss-unlabel_percent_0.2-max_epoch_20.txt ./configs/anet.yaml \
    dataset.training.unlabel_percent=0.2 \
    dataset.testing.unlabel_percent=0.2 \
    pretraining.warmup_epoch=18 \
    training.max_epoch=20 \
    dataset.training.output_path=$output_pth \
    dataset.testing.output_path=$output_pth \
    training.checkpoint_path=$output_pth

  ./spot_train_eval.sh "$gpu" latest_trial-training_bloss_new-toploss-unlabel_percent_0.1-max_epoch_20.txt ./configs/anet.yaml \
    dataset.training.unlabel_percent=0.1 \
    dataset.testing.unlabel_percent=0.1 \
    pretraining.warmup_epoch=18 \
    training.max_epoch=20 \
    dataset.training.output_path=$output_pth \
    dataset.testing.output_path=$output_pth \
    training.checkpoint_path=$output_pth

fi
