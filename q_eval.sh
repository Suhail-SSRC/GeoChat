#Visual Question Answer 8 Bits QVLM
python geochat/eval/batch_geochatq_vqa.py --model-path MBZUAI/geochat-7B --question-file datasets/GeoChat-Bench/lbern_small.jsonl --answers-file datasets/LBERN/ansQVLM_4bit.json --image-folder-calibrate datasets/LBERN/train/ --question-file-calibrate datasets/LBERN/lbern_train.json --load-4bit
