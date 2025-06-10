import torch
import numpy as np
import argparse
import time
import PIL
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import os
from tqdm import tqdm
from chameleon.inference.image_tokenizer import ImageTokenizer
from VQA_Eval.conversation import conv_templates
from threading import Thread
from T2I_Eval.genaibench_generation import sample

IMAGE_TOKEN_INDEX = -200

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def main(args):
    temperature = args.temperature
    guidance_scale = args.cfg
    top_K = args.TopK
    top_P = args.TopP
    image_save_pth = args.save_path
    if not os.path.exists(image_save_pth):
        os.makedirs(image_save_pth)
    
    assert temperature <= 1.0
    assert top_K <= 8192
    assert top_P <= 1.0

    model_id = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    ori_vocabe_size = len(tokenizer)

    vqllm = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_8bit,
    )
    if not args.load_8bit:
        vqllm = vqllm.to('cuda')
    
    vqgan_cfg_path = "chameleon/vqgan.yaml"
    vqgan_ckpt_path = "chameleon/vqgan.ckpt"
    image_tokenizer = ImageTokenizer(cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda:0")

    # 입력 이미지 처리
    image = Image.open(args.image_path).convert('RGB')
    pad_image = expand2square(image, (122, 116, 104))
    input_image = pad_image.resize((512, 512), PIL.Image.LANCZOS)
    
    with torch.no_grad():
        # 입력 이미지를 토큰으로 변환
        input_vq_code = image_tokenizer.img_tokens_from_pil(input_image)
        input_vqcode = input_vq_code.cpu() + len(tokenizer)

        # 프롬프트 준비 (이미지 + 텍스트)
        qs = args.prompt
        qs = '<boi><image><eoi>' + '\n' + qs + ' Generate a new image based on this.'
        
        conv = conv_templates['gemma'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        print("Input prompt:")
        print(prompt)
        print("-" * 50)
        # 🔧 디버깅 정보 추가
        print(f"Input image VQ code shape: {input_vq_code.shape}")
        print(f"Input image VQ code range: {input_vq_code.min()} ~ {input_vq_code.max()}")
        print("-" * 50)

        # 텍스트와 이미지 토큰 결합
        text_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        num_images = (text_ids == IMAGE_TOKEN_INDEX).sum()
        image_token_indices = [-1] + torch.where(text_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [text_ids.shape[0]]
        
        cur_input_ids = []
        for i in range(num_images + 1):
            cur_input_ids.append(text_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            if i < num_images:
                cur_input_ids.append(input_vqcode)
        
        input_ids = torch.cat(cur_input_ids, dim=0)
        print(f"Total input sequence length: {input_ids.shape[0]}")
        
        # 배치 처리를 위해 확장
        batch_size = args.num_samples
        input_ids_batch = input_ids.unsqueeze(0).repeat(batch_size, 1)
        
        # 🔧 CFG 사용 시 텐서 크기 불일치 문제 해결
        if guidance_scale > 1:
            uncond_prompt = '<unconditional><boi>'
            uncond_ids = tokenizer(uncond_prompt, return_tensors="pt").input_ids.squeeze(0)
            
            # 길이 맞추기 위한 패딩 (핵심 수정!)
            target_length = input_ids.shape[0]
            current_length = uncond_ids.shape[0]
            
            print(f"Conditional length: {target_length}, Unconditional length: {current_length}")
            
            if current_length < target_length:
                # 부족한 만큼 패딩 토큰으로 채우기
                padding_length = target_length - current_length
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                padding = torch.full((padding_length,), pad_token_id, dtype=uncond_ids.dtype)
                uncond_ids = torch.cat([uncond_ids, padding])
                print(f"Added {padding_length} padding tokens to unconditional input")
            elif current_length > target_length:
                # 너무 길면 자르기
                uncond_ids = uncond_ids[:target_length]
                print(f"Truncated unconditional input to {target_length} tokens")
            
            uncond_batch = uncond_ids.unsqueeze(0).repeat(batch_size, 1)
            
            # 🔧 Attention mask 생성
            # 조건부 attention mask (모든 토큰에 attention)
            cond_attention_mask = torch.ones_like(input_ids_batch, dtype=torch.bool)
            # 무조건부 attention mask (패딩 부분은 False)
            uncond_attention_mask = torch.ones_like(uncond_batch, dtype=torch.bool)
            if current_length < target_length:
                uncond_attention_mask[:, current_length:] = False
            
            # 조건부와 무조건부 입력 결합
            full_input_ids = torch.cat([input_ids_batch, uncond_batch], dim=0).to("cuda:0")
            attention_mask = torch.cat([cond_attention_mask, uncond_attention_mask], dim=0).to("cuda:0")
        else:
            full_input_ids = input_ids_batch.to("cuda:0")
            attention_mask = torch.ones_like(input_ids_batch, dtype=torch.bool).to("cuda:0")

        # 이미지 생성 루프
        sampling_kwargs = {'temperature': temperature, 'top_k': top_K, 'top_p': top_P, 'sample_logits': True}
        cur_len = full_input_ids.shape[1]
        model_kwargs = {'use_cache': True, 'attention_mask': attention_mask}
        model_kwargs["cache_position"] = torch.arange(cur_len, device=full_input_ids.device)

        pred_tokens = []
        print("Generating new image...")
        print(f"Full input shape: {full_input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print("-" * 50)
        
        for i in tqdm(range(1024)):
            model_inputs = vqllm.prepare_inputs_for_generation(full_input_ids, **model_kwargs)
            
            outputs = vqllm(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[:, -1:, :]
            
            if guidance_scale > 1:
                cond_logits, uncond_logits = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0)
                cfg_logits = uncond_logits + (cond_logits - uncond_logits) * guidance_scale
                half_next_token, _ = sample(cfg_logits, **sampling_kwargs)
                pred_tokens.append(half_next_token)
                next_token = torch.cat([half_next_token, half_next_token])
                
                # attention mask도 확장
                new_attention = torch.ones((attention_mask.shape[0], 1), dtype=torch.bool, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, new_attention], dim=1)
            else:
                next_token, next_prob = sample(next_token_logits, **sampling_kwargs)
                pred_tokens.append(next_token)
                
                # attention mask도 확장
                new_attention = torch.ones((attention_mask.shape[0], 1), dtype=torch.bool, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, new_attention], dim=1)

            # 다음 스텝을 위한 업데이트
            full_input_ids = torch.cat([full_input_ids, next_token], dim=-1)
            model_kwargs['attention_mask'] = attention_mask
            model_kwargs = vqllm._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=vqllm.config.is_encoder_decoder,
            )

        # 생성된 토큰을 이미지로 변환
        image_vq_id = torch.cat(pred_tokens, dim=1) - ori_vocabe_size
        image_vq_id = torch.clamp(image_vq_id, min=0, max=8191)
        
        generated_image_list = []
        print(f"Saving {batch_size} generated images...")
        
        # 🔧 기존 파일 번호 체크해서 이어서 저장
        def get_next_file_index(save_path):
            if not os.path.exists(save_path):
                return 0
            
            existing_files = [f for f in os.listdir(save_path) if f.startswith('sample_') and f.endswith('.jpg')]
            if not existing_files:
                return 0
            
            # sample_숫자.jpg에서 숫자 추출
            indices = []
            for f in existing_files:
                try:
                    # sample_3.jpg -> 3 추출
                    num = int(f.replace('sample_', '').replace('.jpg', '').replace('_failed', ''))
                    indices.append(num)
                except:
                    continue
            
            return max(indices) + 1 if indices else 0
        
        start_index = get_next_file_index(image_save_pth)
        print(f"Starting from index: {start_index}")
        
        # 🔧 더 안전한 파일 저장
        for index, generate_id in enumerate(image_vq_id):
            file_index = start_index + index
            try:
                rec_img = image_tokenizer.pil_from_img_toks(generate_id)
                generated_image_list.append(rec_img)
                rec_img.save(f'{image_save_pth}/sample_{file_index}.jpg')
                print(f"Saved: {image_save_pth}/sample_{file_index}.jpg")
            except Exception as e:
                print(f"Error generating image {file_index}: {e}")
                # 실패한 경우 빈 이미지라도 저장
                try:
                    empty_img = Image.new('RGB', (512, 512), (128, 128, 128))
                    empty_img.save(f'{image_save_pth}/sample_{file_index}_failed.jpg')
                    print(f"Saved placeholder for failed image: {image_save_pth}/sample_{file_index}_failed.jpg")
                except:
                    pass

        print("Image generation completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image-to-Image generation with text prompt')
    parser.add_argument('--model_path', type=str, default='Junfeng5/Liquid_V1_7B', help='model path')
    parser.add_argument('--image_path', type=str, required=True, help='input image path')
    parser.add_argument('--prompt', type=str, required=True, help='text prompt for image transformation')
    parser.add_argument('--save_path', type=str, default='samples/it2i', help='save path for generated images')
    parser.add_argument('--load_8bit', action='store_true', default=False, help='use 8bit to save memory')
    parser.add_argument('--cfg', type=float, default=7.0, help='Classifier-Free Guidance scale')
    parser.add_argument('--TopP', type=float, default=0.96, help='Top P, max=1.0')
    parser.add_argument('--TopK', type=int, default=4096, help='Top K, max=8192')
    parser.add_argument('--temperature', type=float, default=0.99, help='sampling temperature, max=1.0')
    parser.add_argument('--num_samples', type=int, default=4, help='number of images to generate')

    args = parser.parse_args()
    main(args)