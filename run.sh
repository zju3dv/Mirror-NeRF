MODE=$1
GPUS=$2

MODEL_TYPE="nerf"
# MODEL_TYPE="nerf_tcnn"  # For speedup


# # for datasets of synthetic scenes
DATASET_NAME=blender
BOUND=6

DATASET=livingroom
TRAIN_SKIP_STEP=2
VAL_IDX=9
NEAR=0.05
FAR=8.0
IMAGE_W=400
IMAGE_H=300

# DATASET=washroom
# TRAIN_SKIP_STEP=2
# VAL_IDX=7
# NEAR=0.05
# FAR=6.0
# IMAGE_W=400
# IMAGE_H=400

# DATASET=office
# TRAIN_SKIP_STEP=3
# VAL_IDX=14
# NEAR=0.05
# FAR=6.0
# IMAGE_W=400
# IMAGE_H=400


# # for datasets of real scenes
# DATASET_NAME=real_arkit

# DATASET=discussion_room
# TRAIN_SKIP_STEP=2
# VAL_IDX=0
# NEAR=0.05
# FAR=6.0
# IMAGE_W=480
# IMAGE_H=360
# BOUND=6

# DATASET=market
# TRAIN_SKIP_STEP=1
# VAL_IDX=0
# NEAR=0.05
# FAR=10.0
# IMAGE_W=480
# IMAGE_H=360
# BOUND=8

# DATASET=lounge
# TRAIN_SKIP_STEP=1
# VAL_IDX=0
# NEAR=0.05
# FAR=8.0
# IMAGE_W=480
# IMAGE_H=360
# BOUND=6

if [[ $DATASET_NAME == "blender" ]]; then
DATASET_DIR=../datasets/synthetic/${DATASET}
elif [[ $DATASET_NAME == "real_arkit" ]]; then
DATASET_DIR=../datasets/real/${DATASET}
fi

EXP=${DATASET_NAME}_${DATASET}_skip${TRAIN_SKIP_STEP}_res${IMAGE_W}_${MODEL_TYPE}

# # For loading a pretrained model or checkpoint:
# LOG=xxx
# CKPT_PATH=logs/$LOG/last.ckpt
# if [[ $CKPT_PATH == *"tcnn"* ]]; then
# MODEL_TYPE="nerf_tcnn"
# fi

# # For loading a pretrained model of substituted scene (radiance field):
# SUBSTITUTION_LOG=xxx
# SUBSTITUTION_CKPT_PATH=logs/$SUBSTITUTION_LOG/last.ckpt

# # For loading a pretrained model of placed object (radiance field):
# OBJ_CKPT_PATH=D-NeRF/logs/jumpingjacks/800000.tar

if [[ $MODEL_TYPE == "nerf" ]]; then
SCALE_FACTOR=$BOUND
elif [[ $MODEL_TYPE == "nerf_tcnn" ]]; then
SCALE_FACTOR=1
fi


# Novel View Synthesis (Evaluation)
if [ $MODE = 1 ]; then
SPLIT=test
# SPLIT=test_challenging
# SPLIT=test_toward_mirror
CUDA_VISIBLE_DEVICES=$GPUS python eval.py \
   --val_idx $VAL_IDX \
   --split $SPLIT \
   --max_recursive_level 2 \
   --root_dir $DATASET_DIR \
   --near $NEAR \
   --far $FAR \
   --scale_factor $SCALE_FACTOR \
   --dataset_name $DATASET_NAME --exp_name ${SPLIT}_$LOG \
   --img_wh $IMAGE_W $IMAGE_H --N_importance 64 --ckpt_path $CKPT_PATH \
   --bound $BOUND \
   --model_type $MODEL_TYPE \
   --predict_normal \
   --predict_mirror_mask \
   --trace_secondary_rays \
   --chunk 16384

# extract mesh
elif [ $MODE = 2 ]; then
CUDA_VISIBLE_DEVICES=$GPUS python extract_color_mesh.py \
   --root_dir $DATASET_DIR \
   --near $NEAR \
   --far $FAR \
   --scale_factor $SCALE_FACTOR \
   --dataset_name $DATASET_NAME --exp_name $LOG \
   --img_wh $IMAGE_W $IMAGE_H --N_importance 64 --ckpt_path $CKPT_PATH \
   --bound $BOUND \
   --model_type $MODEL_TYPE \
   --predict_normal \
   --predict_mirror_mask \
   --trace_secondary_rays \
   --x_range -0.15 0.15 \
   --y_range -0.15 0.15 \
   --z_range -0.15 0.15

# application - place_new_mirror
elif [ $MODE = 3 ]; then
SPLIT=test
# SPLIT=test_toward_mirror
PLANE_POS=plane_x
# PLANE_POS=plane_y
CUDA_VISIBLE_DEVICES=$GPUS python eval.py \
   --val_idx $VAL_IDX \
   --split $SPLIT \
   --max_recursive_level 50 \
   --app_place_new_mirror \
   --plane_pos $PLANE_POS \
   --root_dir $DATASET_DIR \
   --near $NEAR \
   --far $FAR \
   --scale_factor $SCALE_FACTOR \
   --dataset_name $DATASET_NAME --exp_name app_place_new_mirror_${PLANE_POS}_${SPLIT}_$LOG \
   --img_wh $IMAGE_W $IMAGE_H --N_importance 64 --ckpt_path $CKPT_PATH \
   --bound $BOUND \
   --model_type $MODEL_TYPE \
   --predict_normal \
   --predict_mirror_mask \
   --trace_secondary_rays \
   --chunk 16384

# application - reflect_newly_placed_objects
elif [ $MODE = 4 ]; then
# SPLIT=test
SPLIT=test_toward_mirror
CUDA_VISIBLE_DEVICES=$GPUS python eval.py \
   --val_idx $VAL_IDX \
   --split $SPLIT \
   --app_reflect_newly_placed_objects \
   --obj_ckpt_path $OBJ_CKPT_PATH \
   --root_dir $DATASET_DIR \
   --near $NEAR \
   --far $FAR \
   --scale_factor $SCALE_FACTOR \
   --dataset_name $DATASET_NAME --exp_name reflect_newly_placed_objects_${SPLIT}_$LOG \
   --img_wh $IMAGE_W $IMAGE_H --N_importance 64 --ckpt_path $CKPT_PATH \
   --bound $BOUND \
   --model_type $MODEL_TYPE \
   --predict_normal \
   --predict_mirror_mask \
   --trace_secondary_rays \
   --chunk 16384

# application - control_mirror_roughness
elif [ $MODE = 5 ]; then
trace_ray_times=64
normal_noise_std=0.0025
# SPLIT=test
SPLIT=test_toward_mirror
CUDA_VISIBLE_DEVICES=$GPUS python eval.py \
   --val_idx $VAL_IDX \
   --split $SPLIT \
   --app_control_mirror_roughness \
   --trace_ray_times $trace_ray_times \
   --normal_noise_std $normal_noise_std \
   --root_dir $DATASET_DIR \
   --near $NEAR \
   --far $FAR \
   --scale_factor $SCALE_FACTOR \
   --dataset_name $DATASET_NAME --exp_name app_control_mirror_roughness_Trace${trace_ray_times}_std${normal_noise_std}_${SPLIT}_$LOG \
   --img_wh $IMAGE_W $IMAGE_H --N_importance 64 --ckpt_path $CKPT_PATH \
   --bound $BOUND \
   --model_type $MODEL_TYPE \
   --predict_normal \
   --predict_mirror_mask \
   --trace_secondary_rays \
   --chunk 16384
# application - control_mirror_roughness - normal_noise_std_changes
elif [ $MODE = 52 ]; then
trace_ray_times=64
normal_noise_std=0.01
# SPLIT=test
SPLIT=test_toward_mirror
CUDA_VISIBLE_DEVICES=$GPUS python eval.py \
   --val_idx $VAL_IDX \
   --split $SPLIT \
   --app_control_mirror_roughness \
   --trace_ray_times $trace_ray_times \
   --normal_noise_std $normal_noise_std \
   --normal_noise_std_changes \
   --root_dir $DATASET_DIR \
   --near $NEAR \
   --far $FAR \
   --scale_factor $SCALE_FACTOR \
   --dataset_name $DATASET_NAME --exp_name app_control_mirror_roughness_Trace${trace_ray_times}_std${normal_noise_std}_change_${SPLIT}_$LOG \
   --img_wh $IMAGE_W $IMAGE_H --N_importance 64 --ckpt_path $CKPT_PATH \
   --bound $BOUND \
   --model_type $MODEL_TYPE \
   --predict_normal \
   --predict_mirror_mask \
   --trace_secondary_rays \
   --chunk 16384

# application - reflection_substitution
elif [ $MODE = 6 ]; then
# SPLIT=test
SPLIT=test_toward_mirror
CUDA_VISIBLE_DEVICES=$GPUS python eval.py \
   --val_idx $VAL_IDX \
   --split $SPLIT \
   --app_reflection_substitution \
   --substitution_ckpt_path $SUBSTITUTION_CKPT_PATH \
   --root_dir $DATASET_DIR \
   --near $NEAR \
   --far $FAR \
   --scale_factor $SCALE_FACTOR \
   --dataset_name $DATASET_NAME --exp_name reflection_substitution_${SPLIT}_${LOG}_livingroom \
   --img_wh $IMAGE_W $IMAGE_H --N_importance 64 --ckpt_path $CKPT_PATH \
   --bound $BOUND \
   --model_type $MODEL_TYPE \
   --predict_normal \
   --predict_mirror_mask \
   --trace_secondary_rays \
   --chunk 16384

# train
else
CUDA_VISIBLE_DEVICES=$GPUS python train.py \
   --dataset_name $DATASET_NAME \
   --root_dir $DATASET_DIR \
   --near $NEAR \
   --far $FAR \
   --scale_factor $SCALE_FACTOR \
   --N_importance 64 --img_wh $IMAGE_W $IMAGE_H --noise_std 1 \
   --num_epochs 30 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
   --exp_name $EXP \
   --bound $BOUND \
   --model_type $MODEL_TYPE \
   --predict_normal \
   --predict_mirror_mask \
   --trace_secondary_rays \
   --train_geometry_stage \
   --use_plane_consistent_loss \
   --val_idx $VAL_IDX \
   --train_skip_step $TRAIN_SKIP_STEP \
   --chunk 8192 \
   --only_trace_rays_in_mirrors
fi

# Usage: bash run.sh {MODE} {GPU_ID}
# e.g. Train: bash run.sh 0 0
