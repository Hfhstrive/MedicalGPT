from PIL import Image
import re
from collections import defaultdict
import os
import glob
import json
import argparse
import cv2
import random
import numpy as np
from ipdb import set_trace
from tqdm import tqdm

from transformers import AutoModelForImageTextToText, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel


def crop_invalid_region(img, padding=[0, 0, 0, 0], padding_dynamic=True, DEBUG=False, ignore_square=False):
    # padding parameters: [x_left_padding, x_right_padding, y_top_padding, y_bottom_padding]
    assert min(padding) >= 0 and max(padding) <= 1

    if ignore_square and 0.8 <= img.shape[0] / img.shape[1] <= 1.2:
        return img, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_size = min(img.shape[0], img.shape[1]) // 100
    if (blur_size % 2) == 0:
        blur_size += 1
    gray_blur = cv2.medianBlur(gray, blur_size)

    pixel_gap = 5
    t1 = abs(img[:, :, 0].astype('int32') - img[:, :, 1].astype('int32'))
    t2 = abs(img[:, :, 1].astype('int32') - img[:, :, 2].astype('int32'))
    t3 = abs(img[:, :, 0].astype('int32') - img[:, :, 2].astype('int32'))
    t1[t1 < pixel_gap] = 0
    t1[t1 >= pixel_gap] = 1
    t2[t2 < pixel_gap] = 0
    t2[t2 >= pixel_gap] = 1
    t3[t3 < pixel_gap] = 0
    t3[t3 >= pixel_gap] = 1
    gray_mask = (t1 + t2 + t3).astype('uint8')
    gray_mask[gray_mask > 0] = 1
    element_size = min(img.shape[0], img.shape[1]) // 200
    if (element_size % 2) == 0:
        element_size += 1
    element_struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (element_size, element_size))
    gray_mask = cv2.erode(cv2.dilate(gray_mask, element_struct), element_struct)  # 腐蚀

    # original img is gray image
    if gray_mask.max() == 0:
        gray_mask = np.ones(gray_blur.shape).astype('uint8')

    enhance_blur = gray_blur.copy()
    if gray_mask.mean() < 0.6:
        enhance_blur = np.multiply(gray_blur, gray_mask + 0.5)
        enhance_blur[enhance_blur > 255] = 255
        enhance_blur[enhance_blur < 20] = 0
        enhance_blur = enhance_blur.astype('uint8')

    threshold = 30.0
    if enhance_blur.min() > threshold / 3:
        threshold *= enhance_blur.min() / 10
    if enhance_blur.mean() > threshold * 3:
        threshold *= enhance_blur.mean() * 0.75
    if enhance_blur.max() < threshold * 3:
        threshold *= enhance_blur.max() / 255
    if enhance_blur.mean() < threshold:
        threshold = enhance_blur.mean() * 0.5

    (_, mask) = cv2.threshold(enhance_blur, threshold, 255.0, cv2.THRESH_BINARY)

    (contours_ori, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours_ori, key=lambda contour: len(contour), reverse=True)

    satisfy = False
    for contour in contours:
        roi = cv2.boundingRect(contour)
        if 0.7 < roi[2] / roi[3] < 1.5 and min(roi[2], roi[3]) > (min(img.shape[0], img.shape[1]) / 2):
            satisfy = True
            break

    if not satisfy:
        if DEBUG:
            print('max: {}   min: {}   mean: {}   threshold: {}'.format(enhance_blur.max(), enhance_blur.min(), enhance_blur.mean(), threshold))
            cv2.imshow('img', img)
            cv2.imshow('gray_blur', gray_blur)
            cv2.imshow('gray_mask', gray_mask * 255)
            cv2.imshow('enhance_blur', enhance_blur)
            cv2.imshow('mask', mask)
            cv2.waitKey(0)
        return img, None

    stitched = img.copy()
    if max(padding) > 0:
        weight = roi[2]
        height = roi[3]
        if padding_dynamic:
            padding_factor = [random.uniform(0, padding[0]), random.uniform(0, padding[1]), random.uniform(0, padding[2]), random.uniform(0, padding[3])]
        else:
            padding_factor = padding
        x_left_padding = weight * padding_factor[0]
        x_right_padding = weight * padding_factor[1]
        y_top_padding = height * padding_factor[2]
        y_bottom_padding = height * padding_factor[3]
        roi_x1, roi_y1, roi_w, roi_h = list(roi)
        roi_x2 = roi_x1 + roi_w
        roi_y2 = roi_y1 + roi_h

        new_x1 = 0 if (roi_x1 - x_left_padding) < 0 else int(roi_x1 - x_left_padding)
        new_x2 = stitched.shape[1] if (roi_x2 + x_right_padding) > stitched.shape[1] else int(roi_x2 + x_right_padding)
        new_y1 = 0 if (roi_y1 - y_top_padding) < 0 else int(roi_y1 - y_top_padding)
        new_y2 = stitched.shape[0] if (roi_y2 + y_bottom_padding) > stitched.shape[0] else int(roi_y2 + y_bottom_padding)
        roi = tuple([new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1])
    stitched = stitched[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    return stitched, roi


def vlm_inference_for_case(model, processor, img_paths, case_name, args):
    """对单个病例的图像进行 VLM 推理
    
    Args:
        model: VLM 模型
        processor: VLM 处理器
        img_paths: 该病例的所有图像路径列表
        case_name: 病例名称
        args: 命令行参数
    
    Returns:
        vlm_describe: {图像名: 描述}
    """
    vlm_describe = {}
    tmp_crop_dir = None
    
    if args.crop_ai:
        tmp_crop_dir = os.path.join(args.save_dir, f'{case_name}_crops')
        os.makedirs(tmp_crop_dir, exist_ok=True)

    for image_path in tqdm(img_paths, desc=f'VLM inference for {case_name}'):
        try:
            image_for_model = image_path
            
            # AI 裁剪（如果启用）
            if args.crop_ai:
                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        image_for_model = image_path
                    else:
                        cropped, roi = crop_invalid_region(img, padding=[0.03, 0.03, 0.05, 0.05], 
                                                          padding_dynamic=True, ignore_square=False)
                        if roi is not None:
                            tmp_path = os.path.join(tmp_crop_dir, os.path.basename(image_path))
                            cv2.imwrite(tmp_path, cropped)
                            image_for_model = tmp_path
                except Exception as e:
                    print(f'Warning: Failed to crop {image_path}: {e}')
                    image_for_model = image_path

            # VLM 推理
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_for_model},
                        {"type": "text", "text": "请根据消化内镜诊治标准，简洁地用一句话描述图像中的病变及黏膜特征。"},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            model_desc = output_texts[0].strip() if output_texts else ''
        except Exception as e:
            print(f'Warning: VLM inference failed for {image_path}: {e}')
            model_desc = ''

        image_name = os.path.basename(image_path)
        vlm_describe[image_name] = model_desc

    return vlm_describe


def average_hash(path, hash_size=8):
    try:
        img = Image.open(path).convert('L').resize((hash_size, hash_size), Image.BILINEAR)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        bits = ''.join('1' if p > avg else '0' for p in pixels)
        return int(bits, 2)
    except Exception:
        return None


def hamming(a, b):
    if a is None or b is None:
        return 64
    x = a ^ b
    return x.bit_count()


# 分组函数
def locate_group(loc_text):
    if not loc_text:
        return '无效'
    for key in ['食管', '胃体', '贲门', '胃角', '胃底']:
        if key in loc_text:
            return key
    if '胃窦' in loc_text or '幽门' in loc_text:
        return '胃窦'
    if '十二指肠' in loc_text:
        return '十二指肠'
    if loc_text == '胃':
        return '胃体'
    return '无效'


def select_representatives(img_dir, filenames, case_info, hash_threshold=5, group_images=3):
    """聚类并选择代表图像。

    规则：
      1) 排除质量不为 '正常质量' 或 体内外不为 '体内' 的图片
      2) 根据部位将图像分为 8 组：食管、胃体、贲门、胃角、胃底、胃窦（幽门）、十二指肠、无效
      3) 每组通过哈希去重并最多保留 3 张，优先选择：
         a) 有 det 描述的图片（det 作为 combined_desc）
         b) det 为空时使用 vlm 描述
         c) 优先保持描述多样性，若描述重复则补充不同分辨率的图片
    """
    # 构建 image -> info 映射，case_info 结构为 {case_no: [ {img: {...}}, ... ]}
    info_map = {}
    if isinstance(case_info, dict):
        for v in case_info.values():
            if isinstance(v, list):
                for entry in v:
                    if isinstance(entry, dict):
                        for name, info in entry.items():
                            info_map[name] = info

    # 收集候选图片并过滤质量/体内外
    items = []
    for fn in filenames:
        path = os.path.join(img_dir, fn)
        name = os.path.basename(path)
        info = info_map.get(name, {})
        quality = info.get('质量', '')
        vitro = info.get('体内外', '')
        if quality != '正常质量' or vitro != '体内':
            continue

        det_text = info['det']
        vlm_text = info['vlm']
        loc = info['位置']
        loc_desc = ''
        if len(loc) > 0:
            loc_desc = f'当前部位为{loc}。'
        combined = loc_desc + det_text if det_text else loc_desc + vlm_text
        items.append({
            'path': path,
            'name': name,
            'hash': average_hash(path),
            'loc': loc,
            'det_present': bool(det_text),
            'combined': combined.strip(),
        })

    if not items:
        return []

    groups = defaultdict(list)
    for it in items:
        grp = locate_group(it['loc'])
        groups[grp].append(it)

    reps_selected = []

    # 处理每个组：基于 hash 聚类，每簇选出最优候选，最后选最多 3 张
    for grp_name in ['食管', '胃体', '贲门', '胃角', '胃底', '胃窦', '十二指肠', '无效']:
        group_items = groups.get(grp_name, [])
        if not group_items:
            continue

        # 基于 hash 做简单聚类
        clusters = []
        for it in group_items:
            placed = False
            for cl in clusters:
                rep_h = cl[0]['hash']
                if it['hash'] is None or rep_h is None:
                    continue
                if hamming(it['hash'], rep_h) <= hash_threshold:
                    cl.append(it)
                    placed = True
                    break
            if not placed:
                clusters.append([it])

        # 每个簇选出最佳候选（优先 det_present，再按 combined 长度）
        cluster_candidates = []
        for cl in clusters:
            best = None
            best_score = (-1, 0)
            for it in cl:
                score = (1 if it['det_present'] else 0, len(it['combined'] or ''))
                if score > best_score:
                    best = it
                    best_score = score
            cluster_candidates.append({'best': best, 'cluster': cl, 'score': best_score})

        # 按优先级排序并尝试挑选最多 group_images 张，保证描述多样性
        cluster_candidates.sort(key=lambda x: (x['score'][0], x['score'][1]), reverse=True)
        selected = []
        seen_desc = set()
        for ci in cluster_candidates:
            if len(selected) >= group_images:
                break
            cand = ci['best']
            desc = (cand['combined'] or '').strip()
            if desc and desc in seen_desc:
                continue
            selected.append(cand)
            if desc:
                seen_desc.add(desc)

        # 若不足 5 张，从各簇中补充不同描述的图片
        if len(selected) < group_images:
            for ci in cluster_candidates:
                if len(selected) >= group_images:
                    break
                for it in ci['cluster']:
                    if it in selected:
                        continue
                    desc = (it['combined'] or '').strip()
                    if desc and desc in seen_desc:
                        continue
                    selected.append(it)
                    if desc:
                        seen_desc.add(desc)
                    break

        # 最后仍不足时，按分辨率补齐
        # if len(selected) < args.group_images:
        #     remaining = [it for cl in clusters for it in cl if it not in selected]
        #     def area(it):
        #         try:
        #             with Image.open(it['path']) as im:
        #                 w, h = im.size
        #                 return w * h
        #         except Exception:
        #             try:
        #                 return os.path.getsize(it['path'])
        #             except Exception:
        #                 return 0
        #     remaining.sort(key=lambda it: area(it), reverse=True)
        #     for it in remaining:
        #         if len(selected) >= args.group_images:
        #             break
        #         selected.append(it)

        reps_selected.extend(selected)

    return reps_selected


def generate_det_description(det):
    # # 按类别分组存储

    # det = [
    #     {'类别': '反流性食管炎', '面积': '0.01', 'LA分级': 'LA-A'},
    #     {'类别': '反流性食管炎', '面积': '0.02', 'LA分级': 'LA-B'},
    #     {'类别': '胃癌', '面积': '0.24'},
    #     {'类别': '胃息肉', '面积': '0.07'},
    #     {'类别': '胃溃疡', '面积': '0.12', 'ahs分期': 'A1', 'Forrest分级': 'Ia'},
    #     {'类别': '胃息肉', '面积': '0.07'},
    #     {'类别': '胃溃疡', '面积': '0.12', 'ahs分期': 'A3', 'Forrest分级': 'IIa'},
    # ]

    category_data = defaultdict(lambda: {'count': 0, 'items': []})

    for item in det:
        category = item['类别']

        # 特殊处理：只记一次，且跳过后续的
        if category in ['食管静脉曲张', '反流性食管炎', '霉菌性食管炎', 'Barrett食管']:
            if category_data[category]['count'] == 0:
                category_data[category]['count'] = 1
                # 收集除了'类别'和'面积'外的其他字段
                extra_keys = [k for k in item.keys() if k not in ['类别', '面积']]
                if extra_keys:
                    extra_dict = {key: item[key] for key in extra_keys}
                    category_data[category]['items'].append(extra_dict)
                else:
                    category_data[category]['items'].append({})
            continue

        category_data[category]['count'] += 1

        # 收集除了'类别'和'面积'外的其他字段
        extra_keys = [k for k in item.keys() if k not in ['类别', '面积']]
        if extra_keys:
            extra_dict = {key: item[key] for key in extra_keys}
            category_data[category]['items'].append(extra_dict)
        else:
            category_data[category]['items'].append({})  # 无额外字段时添加空字典

    # 生成描述
    descriptions = ''
    for category, data in category_data.items():
        count = data['count']
        items = data['items']

        if count == 1:
            # 单个病变，直接输出
            if items[0]:
                extra_parts = [f"{key}为{value}" for key, value in items[0].items()]
                extra_str = '，' + '，'.join(extra_parts)
            else:
                extra_str = ''
            descriptions += f"识别到1处{category}{extra_str}。"
        else:
            # 多个病变，需要分别列出
            item_descs = []
            for idx, extra_dict in enumerate(items, 1):
                if extra_dict:
                    extra_parts = [f"{key}为{value}" for key, value in extra_dict.items()]
                    extra_str = '，'.join(extra_parts)
                    item_descs.append(f"1个病变{extra_str}")

            # 合并描述，用逗号分隔
            if len(item_descs) > 0:
                items_str = '，'.join(item_descs)
                descriptions += f"识别到{count}处{category}，{items_str}。"
            else:
                descriptions += f"识别到{count}处{category}。"

    return descriptions


def merge_det_vlm(img_path, imgs, det_data, vlm_data):
    """合并 det_data 与 vlm_data，返回以病例号/图像名为 key 的字典。

    每个 entry 包含：
      - 图像质量
      - 体内外
      - 部位
      - det_describe： 前置模型的描述
      - vlm_describe： vlm模型的描述
    """
    case_no = img_path.split('/')[-2]
    case_info = {
        case_no: []
    }

    for img in imgs:
        det_entry = det_data.get(img) if isinstance(det_data, dict) else None
        vlm_entry = vlm_data.get(img) if isinstance(vlm_data, dict) else None

        loc = ''
        quality = ''
        vitro = ''
        status_scribe = ''
        det_scribe = ''
        vlm_describe = ''

        det_flag = False
        if det_entry is not None:
            loc = det_entry['位置'] if det_entry['位置'] != '未识别' else ''
            status = det_entry['状态']
            light = '白光' if status['光源'] == 'WLI' else status['光源']
            mag = '放大' if '非放大' not in status['放大'] else '非放大'
            quality = status['质量']
            vitro = status['体内外']
            # 状态描述
            status_scribe = f"在{light}{mag}下观察，"
            if status['染色'] != '无染色':
                status_scribe += f"存在{status['染色']}，"
            if status['手术帽'] != '无手术帽':
                status_scribe += f"{status['手术帽']}，"
            if status['器械'] != '无器械':
                status_scribe += f"{status['器械']}，"
            det_scribe += status_scribe
            # 检测结果
            if len(det_entry['检测']) != 0:
                det_flag = True
                det_scribe += generate_det_description(det_entry['检测'])
            assert not (len(det_entry['ipcl']) != 0 and len(det_entry['萎缩']) != 0)
            if len(det_entry['ipcl']) != 0:
                det_flag = True
                det_scribe += f"IPCL呈{det_entry['ipcl']}型。"
            if len(det_entry['萎缩']) != 0:
                det_flag = True
                det_scribe += f"背景黏膜存在萎缩。"

        if vlm_entry is not None:
            des_idx = 0
            if '部位' in vlm_entry:
                des_idx += 1
                if loc == '':
                    loc = re.search(r'部位为([^。]+)', vlm_entry).group(1)
            if '光源' in vlm_entry:
                des_idx += 1
                if len(status_scribe) == '':
                    light_vlm = re.search(r'光源为([^。]+)', vlm_entry).group(1)
                    status_scribe = f"在{light_vlm}下观察，"

            vlm_describe = status_scribe + '。'.join(vlm_entry.split('。')[des_idx:])

        entry = {
            img: {
                '质量': quality,
                '体内外': vitro,
                '位置': loc,
                'det': det_scribe if det_flag else '',
                'vlm': vlm_describe if vlm_describe.endswith('。') else vlm_describe + '。',
            }
        }
        case_info[case_no].append(entry)

    save_path = os.path.join(args.save_dir, 'merge_describe.jsonl')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(case_info, ensure_ascii=False, indent=4))
    return case_info


def batch_generate_answer(
        sentences,
        model,
        tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.0,
        stop_str="</s>",
):
    """Generate answers from prompts in batch mode"""
    generated_texts = []
    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True if temperature > 0.0 else False,
        repetition_penalty=repetition_penalty,
    )

    # Construct batch prompts
    prompts = []
    for s in sentences:
        # Construct dialogue message format
        messages = [{"role": "user", "content": s}]

        # Generate prompt using apply_chat_template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    inputs_tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs_tokens['input_ids'].to(model.device)
    attention_mask = inputs_tokens['attention_mask'].to(model.device)
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
    for gen_sequence in outputs:
        prompt_len = len(input_ids[0])
        gen_sequence = gen_sequence[prompt_len:]
        gen_text = tokenizer.decode(gen_sequence, skip_special_tokens=True)
        pos = gen_text.find(stop_str)
        if pos != -1:
            gen_text = gen_text[:pos]
        gen_text = gen_text.strip()
        generated_texts.append(gen_text)

    return generated_texts


def llm_inference(examples):
    # load model + tokenizer
    print('Loading LLM model...')
    base_model = AutoModelForCausalLM.from_pretrained(args.llm, dtype="auto", device_map="auto")
    base_model.generation_config = GenerationConfig.from_pretrained(args.llm, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, args.lora_llm, torch_dtype="auto", device_map='auto')
    model.eval()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm, trust_remote_code=True, padding_side='left')

    # Prepare examples as a list if it's a single string
    if isinstance(examples, str):
        examples = [examples]

    # Generate answers
    responses = batch_generate_answer(
        examples,
        model,
        tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.0,
        stop_str="</s>",
    )

    return responses


def main(args):
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Load detection data (global)
    det_data = {}
    if os.path.exists(args.det_path):
        try:
            with open(args.det_path, 'r', encoding='utf-8') as f:
                det_data = json.load(f)
        except Exception as e:
            print(f'Warning: Failed to load det_data: {e}')
            det_data = {}

    # Load VLM and processor once
    print('Loading VLM model...')
    vlm_model = AutoModelForImageTextToText.from_pretrained(args.vlm, dtype="auto", device_map="auto")
    vlm_model = PeftModel.from_pretrained(vlm_model, args.lora_vlm)
    vlm_processor = AutoProcessor.from_pretrained(args.vlm)
    vlm_model.eval()

    # Process each case
    cases = sorted([d for d in os.listdir(args.ori_path) 
                   if os.path.isdir(os.path.join(args.ori_path, d))])
    
    for case_idx, case in enumerate(cases, 1):
        print(f'\n[{case_idx}/{len(cases)}] Processing case: {case}')
        
        case_dir = os.path.join(args.ori_path, case)
        dir_name = 'images' if args.crop_ai else 'images_crop'
        img_path = os.path.join(case_dir, dir_name)
        
        # Skip if image directory doesn't exist
        if not os.path.isdir(img_path):
            print(f'  Skipping: {dir_name} directory not found')
            continue
        
        # Get image list
        support_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        imgs = sorted([f for f in os.listdir(img_path) 
                      if os.path.splitext(f)[1].lower() in support_ext])
        
        # Skip if too few images
        if len(imgs) <= args.min_images:
            print(f'  Skipping: Only {len(imgs)} images (min required: {args.min_images})')
            continue
        
        print(f'  Found {len(imgs)} images')
        
        # Step 1: VLM inference for this case
        print(f'  Step 1: Running VLM inference...')
        img_full_paths = [os.path.join(img_path, img) for img in imgs]
        vlm_data = vlm_inference_for_case(vlm_model, vlm_processor, img_full_paths, case, args)
        
        # Step 2: Merge detection and VLM data
        print(f'  Step 2: Merging VLM and detection data...')
        merge_info = merge_det_vlm(img_path, imgs, det_data, vlm_data)
        
        # Step 3: Select representative images
        print(f'  Step 3: Selecting representative images...')
        reps = select_representatives(img_path, imgs, merge_info, 
                                     hash_threshold=args.hash_threshold,
                                     group_images=args.group_images)
        
        if not reps:
            print(f'  Warning: No representative images selected')
            continue
        
        # Step 4: Group by anatomical location
        print(f'  Step 4: Grouping images by location...')
        groups_order = ['食管', '胃体', '贲门', '胃角', '胃底', '胃窦', '十二指肠', '无效']
        grouped = {g: [] for g in groups_order}
        for it in reps:
            grp = locate_group(it['loc'])
            if grp not in grouped:
                grouped['无效'].append(it)
            else:
                grouped[grp].append(it)

        # Build case info string
        grouped_parts = []
        for g in groups_order:
            items_g = grouped.get(g, [])
            if not items_g:
                continue
            part = f"{g}下有{len(items_g)}张图像："
            descs = []
            for idx, it in enumerate(items_g, 1):
                combined = it.get('combined', '').strip()
                descs.append(f"第{idx}张图像:{combined}")
            part += ''.join(descs)
            grouped_parts.append(part)

        case_info = '\n'.join(grouped_parts)
        user_info = f'该上消化道内镜下的病例套图中, 其模型识别的特征及病变如下所示，请帮我生成内镜报告。其模型详细结果如下: {case_info}'

        # Step 5: Generate diagnostic report using LLM
        print(f'  Step 5: Generating LLM report...')
        responses = llm_inference(user_info)
        
        # Step 6: Save results
        if responses:
            report = responses[0]
            report_path = os.path.join(args.save_dir, f'{case}_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f'  ✓ Report saved to {report_path}')
        else:
            print(f'  Warning: No LLM response generated')
    
    print(f'\n✓ All cases processed. Results saved to {args.save_dir}')


if __name__ == '__main__':
    DEFAULT_VLM = 'Qwen/Qwen3-VL-2B-Instruct'
    DEFAULT_LORA_VLM = '/home/inno/code/VLM/Qwen3-VL/qwen-vl-finetune/output/V2/lora_qwen3_2b_r64_alpha128_dropout0.05_zero2_448_768_freeze_vision-mlp_lr1e-4/checkpoint-320/'
    DEFAULT_LLM = 'Qwen/Qwen3-4B-Instruct-2507'
    DEFAULT_LORA_LLM = "/home/inno/code/LLM/MedicalGPT/outputs-sft-qwen3-4b-v2"

    parser = argparse.ArgumentParser(description='Medical endoscopy report generation pipeline')
    
    # VLM parameters
    parser.add_argument('--vlm', type=str, default=DEFAULT_VLM, 
                        help='Vision-Language Model name or path')
    parser.add_argument('--lora_vlm', type=str, default=DEFAULT_LORA_VLM, 
                        help='LoRA adapter path for VLM')
    
    # LLM parameters
    parser.add_argument('--llm', type=str, default=DEFAULT_LLM, 
                        help='Large Language Model name or path')
    parser.add_argument('--lora_llm', type=str, default=DEFAULT_LORA_LLM,
                        help='LoRA adapter path for LLM')
    
    # Data parameters
    parser.add_argument('--ori_path', type=str,
                        default='/media/inno/VLM/D1_images_and_reports_which_have_video_20250316/胃镜/',
                        help='Root path of original data')
    parser.add_argument('--det_path', type=str,
                        default='/media/inno/VLM/D1_images_and_reports_which_have_video_20250316/base/det/胃镜_V3.json',
                        help='Detection results JSON file')
    parser.add_argument('--save_dir', type=str,
                        default='/media/inno/VLM/D1_images_and_reports_which_have_video_20250316/MedicalGPT/V3/',
                        help='Output directory for results')
    
    # Processing parameters
    parser.add_argument('--crop_ai', action='store_true', 
                        help='Enable AI-based image cropping')
    parser.add_argument('--hash_threshold', type=int, default=5, 
                        help='Hamming distance threshold for image deduplication')
    parser.add_argument('--group_images', type=int, default=5, 
                        help='Max number of representative images per anatomical location')
    parser.add_argument('--min_images', type=int, default=5, 
                        help='Minimum number of images per case to process')
    
    args = parser.parse_args()
    main(args)
