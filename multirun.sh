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

  ./spot_train_eval.sh "$gpu" THUMOS_2.txt ./configs/thumos.yaml \
    pretraining.warmup_epoch=1 \
    pretraining.consecutive_warmup_epochs=1 \
    training.max_epoch=1 \
    training.consecutive_train_epochs=1 \
    dataset.training.output_path=$output_pth \
    dataset.testing.output_path=$output_pth \
    training.checkpoint_path=$output_pth

else
  output_pth="./output_2/"
 
  ./spot_train_eval.sh "$gpu" THUMOS_2.txt ./configs/thumos.yaml \
    pretraining.warmup_epoch=10 \
    pretraining.consecutive_warmup_epochs=10 \
    training.max_epoch=10 \
    training.consecutive_train_epochs=10 \
    dataset.training.output_path=$output_pth \
    dataset.testing.output_path=$output_pth \
    training.checkpoint_path=$output_pth

fi
