#Running Scene classification for 8 bits UCMerged
#python geochat/eval/batch_geochat_scene.py --model-path MBZUAI/geochat-7B --question-file datasets/GeoChat-Bench/UCmerced.jsonl --answers-file datasets/UCMerced/ans_8bit.json --image-folder datasets/UCMerced/Images/ --load-8bit | tee UC8bits.txt


#Running Scene classification for 8 bits AID
#python geochat/eval/batch_geochat_scene.py --model-path MBZUAI/geochat-7B --question-file datasets/GeoChat-Bench/aid.jsonl --answers-file datasets/AID/ans_8bit.json --image-folder datasets/AID/ --load-8bit | tee AID8bits.txt


#Region Caption 8 bits
#python geochat/eval/batch_geochat_grounding.py --model-path MBZUAI/geochat-7B --question-file datasets/GeoChat-Bench/region_captioning.jsonl --answers-file datasets/rc8bits.json --image-folder datasets/GeoChat_Instruct/ --load-8bit | tee rc8bits.txt


# Refering (Visual Grounding) 8 bits
#python geochat/eval/batch_geochat_referring.py --model-path MBZUAI/geochat-7B --question-file datasets/GeoChat-Bench/referring.jsonl --answers-file datasets/vg8bits.json --image-folder datasets/GeoChat_Instruct/ --load-8bit | tee vg8bits.txt


#python geochat/dataset/rc_metrics.py --answers-file rc8bits.json --output-file final_rc8bits.json | tee final_rc8bits.json
#python geochat/dataset/vg_metrics.py --answers-file vg8bits.json --output-file final_vg8bits.json | tee final_vg8bits.json

#Some of the others were done on the terminal since we were checking if they were working and never added them to the scripts
