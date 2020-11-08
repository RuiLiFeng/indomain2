#!bin/bash

set -x
set -e

BOUNDARY_PATH=$1
aug='/'

for boundary in `ls $BOUNDARY_PATH/`
    do
      boundary_path=${BOUNDARY_PATH}${aug}${boundary}
      echo $score_path
      python manipulate.py \
        styleganinv_face_256.pkl \
        results/inversion/img_list_celeba \
        ${boundary_path} \
        -o mani_res_all \
        --gpu_id 4 \
        --step 11 \
        --start_distance -2 \
        --end_distance 2
    done