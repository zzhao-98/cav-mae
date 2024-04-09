


set -x
# comment this line if not running on sls cluster
# . /data/sls/scratch/share-201907/slstoolchainrc
# source ~/.bashrc


model=cav-mae-ft
ftmode=multimodal # or audioonly or videoonly

# you can replace with any checkpoint you want, but by default, we use cav-mae-scale++
cur_dir=$(pwd)
wget -nc https://www.dropbox.com/s/l5t5geufdy3qvnv/audio_model.21.pth?dl=1 -O cav-mae-scale++.pth
pretrain_path=${cur_dir}/cav-mae-scale++.pth

freeze_base=False
head_lr=50 # newly initialized ft layers uses 50 times larger than the base lr

# bal=bal
lr=1e-5
epoch=100
lrscheduler_start=5
lrscheduler_decay=0.95
lrscheduler_step=1
wa=True
wa_start=1
wa_end=10
lr_adapt=False
dataset_mean=0.00
dataset_std=0.00
target_length=1024
noise=True
freqm=48
timem=192
mixup=0.5
batch_size=128
label_smooth=0.1

dataset=audioset
tr_data=/home/artmed/PycharmProjects/cav-mae/zzh_code/FOS_train_dataset.json
te_data=/home/artmed/PycharmProjects/cav-mae/zzh_code/FOS_validation_dataset.json
label_csv=/home/artmed/PycharmProjects/cav-mae/zzh_code/FOS_used_label.csv

exp_dir=./exp/testmae01-full-${model}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-${lrscheduler_step}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-no_mean_std_r3
mkdir -p $exp_dir

python ../src/run_cavmae_ft.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class 13 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} \
--label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--loss BCE --metrics mAP --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--pretrain_path ${pretrain_path} --ftmode ${ftmode} \
--freeze_base ${freeze_base} --head_lr ${head_lr} \
--num-workers 32