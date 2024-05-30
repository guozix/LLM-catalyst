export TASK_NAME=superglue
export DATASET_NAME=boolq
export CUDA_VISIBLE_DEVICES=$1

bs=32
lr=7e-3
dropout=0.1
psl=8
epoch=100

runid=$2

for seed in 1 2 3
do
python3 run_rounds.py \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir checkpoints/PT-$DATASET_NAME-roberta-$runid/-seed${seed}/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed ${seed} \
  --save_strategy no \
  --evaluation_strategy epoch \
  --prompt
  # --prefix
done

# bash run_script/run_boolq_roberta.sh 4 psl8
