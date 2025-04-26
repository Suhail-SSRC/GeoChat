import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from geochat.conversation import conv_templates, SeparatorStyle, Chat
from geochat.model.builder import load_pretrained_model
from geochat.utils import disable_torch_init
from geochat.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import random
import numpy as np

from custom_bitsandbytes.bitsandbytes.quantization_utils.quant_modules import QuantAct

# ---------------- Utility Functions for Chunking ----------------

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# ---------------- Calibration Function ----------------

def run_calibrate(args, tokenizer, model, image_processor):
    print('\n==> start calibrate')
    # Set calibration flag for all QuantAct modules
    for name, module in model.named_modules():
        if isinstance(module, QuantAct):
            module.set_calibrate(calibrate=True)
    
    # Load calibration questions (assumed to be in JSON format)
    calib_questions = json.load(open(os.path.expanduser(args.question_file_calibrate), "r"))
    np.random.seed(0)
    np.random.shuffle(calib_questions)
    num_of_sample = 16 * args.num_chunks
    calibrate_images = 8  # process this many images for the calibration phase
    search_flag = 0
    calib_questions = get_chunk(calib_questions[:num_of_sample], args.num_chunks, args.chunk_idx)

    # for i, line in enumerate(tqdm(calib_questions)):
    #     idx = line["id"]
    #     question = line['conversations'][0]
    #     gt_ans = line["conversations"][1]
    #     qs = question['value'].replace('<image>', '').strip()
    #     cur_prompt = qs

    #     if 'image' in line:
    #         image_file = line["image"]
    #         image_path = os.path.join(args.image_folder_calibrate, image_file)
    #         try:
    #             image = Image.open(image_path)
    #         except Exception as e:
    #             print(f"Skipping image {image_path}: {e}")
    #             continue
    #         image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    #         images = image_tensor.unsqueeze(0).half().cuda()
    #         # Optionally, you can store image sizes if needed:
    #         # image_sizes = [image.size]
    #         if getattr(model.config, 'mm_use_im_start_end', False):
    #             qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    #         else:
    #             qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    #         cur_prompt = '<image>' + '\n' + cur_prompt
    #     else:
    #         images = None
    for i in tqdm(range(0,len(calib_questions),args.batch_size)):
            input_batch=[]
            input_image_batch=[]
            count=i
            image_folder=[]     
            batch_end = min(i + args.batch_size, len(calib_questions))

                
            for j in range(i,batch_end):
                image_file=calib_questions[j]['image']
                qs=calib_questions[j]['text']
                
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                input_batch.append(input_ids)

                image = Image.open(os.path.join(args.image_folder, image_file))

                image_folder.append(image)

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
            if search_flag > 0 and image is None:
                print("skip")
                continue

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )
            
            if search_flag == 1:
                search_flag += 1
            elif search_flag == 2:
                break
            if i == calibrate_images - 1:
                for name, module in model.named_modules():
                    if isinstance(module, QuantAct):
                        module.set_search(search=True)
                print('==> searching!')
                search_flag += 1

    # Turn off calibration after processing
    for name, module in model.named_modules():
        if isinstance(module, QuantAct):
            module.set_calibrate(calibrate=False)
    print('==> end calibrate')

# ---------------- Evaluation Function ----------------

def eval_model(args):
    # Disable torch initialization overhead
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # Load model, tokenizer, and image processor
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, args.load_8bit, args.load_4bit
    )
    
    # Run calibration to set quantization ranges using QuantAct modules
    run_calibrate(args, tokenizer, model, image_processor)
    
    # Load evaluation questions (assumed to be a JSON file with a list of samples)
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    print(f"Printing first question for sanity check:\n{questions[0]}")
    
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    for i in tqdm(range(0, len(questions), args.batch_size)):
        input_batch = []
        image_folder = []
        batch_end = min(i + args.batch_size, len(questions))
        count = i
        
        for j in range(i, batch_end):
            # For evaluation, each sample contains an image and a text prompt.
            image_file = questions[j]['image']
            qs = questions[j]['text']
            
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            input_batch.append(input_ids)
            
            # Load the evaluation image from the evaluation image folder.
            image_path = os.path.join(args.image_folder, image_file)
            image = Image.open(image_path)
            image_folder.append(image)
            
            # Set up stopping criteria (if needed)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        max_length = max(tensor.size(1) for tensor in input_batch)
        final_input_list = [
            torch.cat((torch.zeros((1, max_length - tensor.size(1)), dtype=tensor.dtype, device=tensor.get_device()), tensor), dim=1)
            for tensor in input_batch
        ]
        final_input_tensors = torch.cat(final_input_list, dim=0)
        
        image_tensor_batch = image_processor.preprocess(
            image_folder, crop_size={'height': 504, 'width': 504},
            size={'shortest_edge': 504}, return_tensors='pt'
        )['pixel_values']
        
        with torch.inference_mode():
            output_ids = model.generate(
                final_input_tensors,
                images=image_tensor_batch.half().cuda(),
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=1,
                max_new_tokens=256,
                length_penalty=2.0,
                use_cache=True
            )
        
        input_token_len = final_input_tensors.shape[1]
        n_diff_input_output = (final_input_tensors != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids differ from the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        
        for k in range(0, len(final_input_list)):
            output = outputs[k].strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "question_id": questions[count]["question_id"],
                "image_id": questions[count]["image"],
                "answer": output,
            }) + "\n")
            count += 1
            ans_file.flush()
    ans_file.close()

# ---------------- Main ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")  # Evaluation images folder
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    
    # Calibration data
    parser.add_argument("--image-folder-calibrate", type=str, default="")  # Calibration images folder
    parser.add_argument("--question-file-calibrate", type=str, default="tables/question_train.json")
    
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    
    args = parser.parse_args()
    eval_model(args)
