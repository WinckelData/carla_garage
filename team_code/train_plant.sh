#!/bin/bash
#SBATCH --job-name=plant_v08_default_lav
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=72
#SBATCH --nodes=1 
#SBATCH --time=3-00:00 
#SBATCH --gres=gpu:8 
#SBATCH --mem=256G 
#SBATCH --output=/mnt/qb/work/geiger/gwb710/carla_garage/training_logdir/slurm/plant_v08_default_lav.out 
#SBATCH --error=/mnt/qb/work/geiger/gwb710/carla_garage/training_logdir/slurm/plant_v08_default_lav.err 
#SBATCH --partition=gpu-2080ti

# print info about current job
scontrol show job $SLURM_JOB_ID

pwd
export CARLA_ROOT=/mnt/qb/work/geiger/gwb710/carla_garage/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/qb/work/geiger/gwb710/software/anaconda3/lib

export OMP_NUM_THREADS=72
export OPENBLAS_NUM_THREADS=1
# Set the value of OMP_NUM_THREADS to (cpus / gpus)
# v07: /mnt/qb/geiger/bjaeger25/data/til_dataset_v07_2023_01_18
# v08: /mnt/qb/work2/geiger0/bjaeger25/datasets/hb_dataset_v08_2023_05_10
# for lav: --setting 02_05_withheld
# for longest6: --setting all
srun torchrun --nnodes=1 --nproc_per_node=8 --max_restarts=1 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
train.py --id plant_v08_default_lav --batch_size 64 --setting 02_05_withheld \
--root_dir /mnt/qb/work2/geiger0/bjaeger25/datasets/hb_dataset_v08_2023_05_10 \
--epochs 47 --lr 0.0001 --logdir /mnt/qb/work/geiger/gwb710/carla_garage/training_logdir/ --sync_batch_norm 0 --zero_redundancy_optimizer 1 \
--use_disk_cache 1 --val_every 2 --use_cosine_schedule 0 --use_plant 1 --augment 1 --learn_origin 1 --use_focal_loss 1 \
--use_controller_input_prediction 1 --use_velocity 1 --detect_boxes 0 --train_sampling_rate 5 --use_grad_clip 1 --use_color_aug 0 \
--use_wp_gru 1 --use_semantic 0 --use_bev_semantic 0 --use_depth 0 --num_repetitions 3 --continue_epoch 0 --smooth_route 1 \
--schedule_reduce_epoch_01 45 --schedule_reduce_epoch_02 50 --num_lidar_hits_for_detection -999 --use_speed_weights 1 --max_num_bbs 40 \
--estimate_class_distributions 0 --use_optim_groups 1 --weight_decay 0.1 --cpu_cores 72