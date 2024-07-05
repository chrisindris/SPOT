# Usage: ./parameter_sweep.sh <GPU>

# TODO: I can list all of the possible variables here, I don't need to use them all
#declare -a dataset_name=("anet" "thumos14as" "i5O")
#declare -a dataset_training_num_frame=(16 19) # default, and it seems like they come in gaps of 3, so 16 + 3 = 19 is my hack to get the distance between them
#declare -a dataset_training_unlabel_percent=(0.0 0.1)
declare -a model_embedding_head=(1 4)
declare -a training_step=(5 10)
declare -a training_gamma=(0.2 0.8)
declare -a training_loss_balance=(0.1 0.5 0.9)
declare -a loss_lambda_2=(0.1 0.4 0.9)
#declare -a testing_mask_thresh=([0.1] [0.1,0.2]) # When specifying a list, no spaces in the list ([1,2] not [1, 2])

gpu=$1

if (( gpu == 0 )); then
  output_pth="./output/"
 
  for x in {0..0},{0..1},{0..1},{0..2},{0..2} # Ensure that only the used variables are listed here. Format: {0..X},{0..Y} ... where X and Y are the number of possible hyperparameter values (minus 1)
    do
      IFS=',' read -r eh s g lb l2 < <(echo $x) # Ensure that only the used variables are listed here.
      #unset IFS

      #echo $eh $s $g $lb $l2

      EH=${model_embedding_head[$eh]}
      S=${training_step[$s]}
      G=${training_gamma[$g]}
      LB=${training_loss_balance[$lb]}
      L2=${loss_lambda_2[$l2]}

      #echo $EH $S $G $LB $L2
      outfile="sweep_eh-${EH}-s_${S}-g_${G}-lb_${LB}-l2_${L2}.txt"
      
      if [ ! -e "./output/experiments/$outfile" ]; then
        # Ensure that only used variables are listed in the output file name.
        ./spot_train_eval.sh "$gpu" "$outfile" ./configs/anet.yaml \
          model.embedding_head="$EH" \
          training.step="$S" \
          training.gamma="$G" \
          training.loss_balance="$LB" \
          loss.lambda_2="$L2" \
          dataset.training.output_path=$output_pth \
          dataset.testing.output_path=$output_pth \
          training.checkpoint_path=$output_pth
          # to cancel out a line, precede it with \ and #, for example:
          # \ #dataset.training ...
      fi 

    done 

else
  output_pth="./output_2/"


  for x in {1..1},{0..1},{0..1},{0..2},{0..2} # Ensure that only the used variables are listed here. Format: {0..X},{0..Y} ... where X and Y are the number of possible hyperparameter values (minus 1)
    do
      IFS=',' read -r eh s g lb l2 < <(echo $x) # Ensure that only the used variables are listed here.
      #unset IFS

      #echo $eh $s $g $lb $l2

      EH=${model_embedding_head[$eh]}
      S=${training_step[$s]}
      G=${training_gamma[$g]}
      LB=${training_loss_balance[$lb]}
      L2=${loss_lambda_2[$l2]}

      #echo $EH $S $G $LB $L2
      outfile="sweep_eh-${EH}-s_${S}-g_${G}-lb_${LB}-l2_${L2}.txt"

      if [ ! -e "./output/experiments/$outfile" ]; then
        # Ensure that only used variables are listed in the output file name.
        ./spot_train_eval.sh "$gpu" "$outfile" ./configs/anet.yaml \
          model.embedding_head="$EH" \
          training.step="$S" \
          training.gamma="$G" \
          training.loss_balance="$LB" \
          loss.lambda_2="$L2" \
          dataset.training.output_path=$output_pth \
          dataset.testing.output_path=$output_pth \
          training.checkpoint_path=$output_pth
          # to cancel out a line, precede it with \ and #, for example:
          # \ #dataset.training ...
      fi

    done 


fi
