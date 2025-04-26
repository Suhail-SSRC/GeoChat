#Visual Question Answer 4 Bits QVLM
#python geochat/eval/batch_geochatq_vqa.py --model-path MBZUAI/geochat-7B --question-file datasets/GeoChat-Bench/lbern_small.jsonl --answers-file datasets/LBERN/ansQVLM_4bit.json --image-folder-calibrate datasets/LBERN/train/ --question-file-calibrate datasets/LBERN/lbern_train.json --load-4bit

#python datasets/LBERN/evaluate_vqa.py --predictions-file datasets/LBERN/ans_4bit.json --ground-truth-file datasets/LBERN/LR_split_test_answers.json --questions-file datasets/LBERN/LR_split_test_questions.json --output-file datasets/LBERN/overall_results_4bit.json --output-detailed datasets/LBERN/detailed_results_4bit.json


#4 bits
#python geochat/eval/batch_geochat_scene.py --model-path MBZUAI/geochat-7B --question-file datasets/GeoChat-Bench/UCmerced.jsonl --answers-file datasets/UCMerced/ansQVLM_4bit.json --image-folder-calibrate datasets/UCMerced/Images/train --question-file-calibrate datasets/UCmerced/UCmerced_train.json  --load-4bit | tee UC4bits.txt


#Scene Calssification 4 bits
#python geochat/eval/batch_geochat_scene.py --model-path MBZUAI/geochat-7B --question-file datasets/GeoChat-Bench/aid.jsonl --answers-file datasets/AID/ansQVLM_4bit.json --image-folder-calibrate datasets/AID/train/ --question-file-calibrate datasets/LBERN/lbern_train.json --load-4bit | tee AID4bits.txt


#Region Caption 4 bit
#python geochat/eval/batch_geochat_grounding.py --model-path MBZUAI/geochat-7B --question-file datasets/GeoChat-Bench/region_captioning.jsonl --answers-file datasets/rc4bits.json --image-folder datasets/GeoChat_Instruct/ --load-4bit | tee rc4bits.txt

# Refering (Visual Grounding)

# 4bits
#python geochat/eval/batch_geochat_referring.py --model-path MBZUAI/geochat-7B --question-file datasets/GeoChat-Bench/referring.jsonl --answers-file datasets/vg4bits.json --image-folder datasets/GeoChat_Instruct/ --load-4bit | tee vg4bits.txt


#python geochat/dataset/rc_metrics.py --answers-file rc4bits.json --output-file final_rc4bits.json | tee final_rc4bits.json
#python geochat/dataset/vg_metrics.py --answers-file vg4bits.json --output-file final_vg4bits.json | tee final_vg4bits.json
