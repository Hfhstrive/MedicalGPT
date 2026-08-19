import argparse
import copy
import glob
import json
import os
import re
import ast
import time
from ipdb import set_trace
from typing import List, Tuple

template_process = {
    '慢性浅表性胃炎': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜光滑，黏液湖清。",
        "胃体": "皱襞走向规则，黏膜光滑，血管纹理清晰。",
        "胃角": "形态完整，黏膜光滑。",
        "胃窦": "黏膜光滑，红白相间，以红为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
    '萎缩性胃炎': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜光滑，黏液湖清。",
        "胃体": "皱襞走向规则，黏膜光滑，血管纹理清晰。",
        "胃角": "形态完整，黏膜光滑。",
        "胃窦": "黏膜光滑，红白相间，以白为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
    '萎缩性胃炎C1': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜光滑，黏液湖清。",
        "胃体": "皱襞走向规则，黏膜光滑，血管纹理清晰。",
        "胃角": "形态完整，黏膜光滑。",
        "胃窦": "黏膜光滑，红白相间，以白为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
    '萎缩性胃炎C2': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜光滑，黏液湖清。",
        "胃体": "皱襞走向规则，下部小弯侧黏膜红白相间，以白为主，血管透见，蠕动正常。",
        "胃角": "黏膜变薄，血管纹理显露。",
        "胃窦": "黏膜光滑，红白相间，以白为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
    '萎缩性胃炎C3': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜光滑，黏液湖清。",
        "胃体": "皱襞走向规则，小弯黏膜地图样发红，色调逆转，延及近贲门。",
        "胃角": "黏膜变薄，血管纹理显露。",
        "胃窦": "黏膜光滑，红白相间，以白为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
    '萎缩性胃炎O1': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜光滑，黏液湖清。",
        "胃体": "皱襞走向规则，贲门下及小弯侧黏膜菲薄，红白相间，以白为主，血管透见。",
        "胃角": "黏膜菲薄，黏膜下血管透见。",
        "胃窦": "黏膜光滑，红白相间，以白为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
    '萎缩性胃炎O2': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜菲薄，黏膜下血管透见，黏液湖清。",
        "胃体": "皱襞走向规则，小弯黏膜地图样发红，色调逆转，延及近贲门。",
        "胃角": "黏膜菲薄，黏膜下血管透见。",
        "胃窦": "黏膜光滑，红白相间，以白为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
    '萎缩性胃炎O3': {
        "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
        "贲门": "贲门闭合良好，黏膜光滑。",
        "胃底": "黏膜菲薄，黏膜下血管透见，黏液湖清。",
        "胃体": "粘膜菲薄，黏膜下血管透见。",
        "胃角": "粘膜菲薄，黏膜下血管透见。",
        "胃窦": "黏膜光滑，红白相间，以白为主。",
        "幽门": "圆，通畅。",
        "十二指肠": "黏膜光滑，降段上部黏膜未见异常。",
    },
}

dup_list = [
    [['萎缩性胃炎O3'], ['萎缩性胃炎O2'], ['萎缩性胃炎O1'], ['萎缩性胃炎C3'], ['萎缩性胃炎C2'], ['萎缩性胃炎C1'], ['萎缩性胃炎'], ['慢性浅表性胃炎'], ['浅表性胃炎']],
    [['HP现症感染'], ['Hp现症感染', 'hp现症感染', '现症感染']],
    [['贲门黏膜病变', '胃体黏膜病变', '胃角黏膜病变', '胃窦黏膜病变', '幽门黏膜病变', '十二指肠黏膜病变', '胃底黏膜病变'], ['胃黏膜病变'], ['黏膜病变'], ['胃癌']],
    [['食管黏膜病变'], ['黏膜病变'], ['食管癌']],
    [['黄色瘤'], ['食管黄色瘤', '胃黄色瘤']],
    [['食管乳头状瘤'], ['食管黏膜隆起']],
    [['静脉瘤', '平滑肌瘤'], ['食管SMT'], ['食管黏膜下肿瘤'], ['食管黏膜隆起']],
    [['间质瘤', '异位胰腺'], ['胃SMT'], ['胃黏膜下肿瘤']],
    [['增生性息肉', '胃底腺息肉'], ['胃息肉'], ['胃黏膜隆起']],
    [['食管静脉曲张', '胃静脉曲张']],
]

# 只输出包含特定字眼的诊断结论
allowed_keywords = ['食管', '胃', '静脉', '癌', '隆起', '息肉', 'SMT', '瘤', '黏膜', '病变', '溃疡', '肠', '萎缩', '感染']


def load_asr_model(checkpoint: str, device: str = "cuda:0", hotwords_path: str = None):
    from funasr import AutoModel

    hotwords = None
    if hotwords_path and os.path.exists(hotwords_path):
        with open(hotwords_path, "r", encoding="utf-8") as f:
            hotwords = [line.strip() for line in f if line.strip()]

    # 非实时交互时，vad_model可不使用
    model_kwargs = {
        "model": checkpoint,
        "trust_remote_code": False,
        # vad_model="fsmn-vad",
        "device": device,
        "disable_update": True,
        "dtype": "fp16",
    }
    if hotwords:
        model_kwargs["hotwords"] = hotwords

    model = AutoModel(**model_kwargs)
    return model, hotwords


def asr_transcribe(model, wav_path: str) -> str:
    res = model.generate(input=[wav_path], cache={}, batch_size_s=0)
    return res[0].get("text", "")


def load_llm_model(base_model_name: str, lora_adapter_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model.eval()
    return tokenizer, model


def llm_process(tokenizer, model, info: str) -> Tuple[str, str, str]:
    questions = [
        # f"规范表达以下语句：{info}",
        # "提取语句中所涉及的结论性疾病？",
        # "根据[部位, 特征描述, 病变名称]格式拆分语句：",
        f"提取有效信息，规范表达以下语句：{info}",
        "提取语句中所涉及的结论性疾病？",
        "根据[部位, 特征描述, 病变名称]格式拆分语句，并专业化特征描述：",
    ]

    messages = []
    responses = []

    for i, q in enumerate(questions):
        # 将当前问题加入对话历史
        messages.append({"role": "user", "content": q})

        # 应用模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 推理
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,  # 选取最高概率, 保证可复现性
            temperature=None,  # 显式清除
            top_p=None,  # 显式清除
            top_k=None,  # 显式清除
        )

        # 解码
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)

        # 重要：将模型的回答加入对话历史，这样下一轮才能实现“多问题推理”
        messages.append({"role": "assistant", "content": response})
        responses.append(response.strip())

    return responses


def llm_summarize(tokenizer, model, texts: List[str]) -> str:
    """使用 LLM 将多条描述合并并简化为一句话（保留医学术语）。"""
    if not texts:
        return ""

    joined = "，".join([t for t in texts if t])
    prompt = f"一句话概括以下内容，使其更连贯：\n{joined}"

    # 使用 tokenizer 的模板方法（若有），否则直接发送 prompt
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=None,  # 显式清除
        top_p=None,  # 显式清除
        top_k=None,  # 显式清除
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    out = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return out


def normalize_disease_name(name: str) -> str:
    """归一化诊断疾病名称，去除‘型’字或做形式统一（如 萎缩性胃炎C2型 -> 萎缩性胃炎C2）"""
    name = name.strip()
    # 提取 C1-C3, O1-O3 等亚型
    m = re.search(r'([COco])\s*([123])', name)
    if m:
        prefix = m.group(1).upper()
        num = m.group(2)
        if any(keyword in name for keyword in ['萎缩', 'atrophy', 'c', 'o', 'C', 'O']):
            return f"萎缩性胃炎{prefix}{num}"
            
    if '慢性浅表' in name or '浅表性' in name:
        return "慢性浅表性胃炎"
    if '萎缩性胃炎' in name or '慢性萎缩性胃炎' in name:
        return "萎缩性胃炎"
        
    return name


def deduplicate_concl(conclusion):
    conclusion = [normalize_disease_name(item) for item in conclusion]
    # 1. 建立词到 “规则组ID” 和 “梯队层级” 的映射
    # 结构为: { 词: (规则组idx, 梯队idx) }
    word_to_tier = {}
    for group_idx, group in enumerate(dup_list):
        for tier_idx, tier in enumerate(group):
            for item in tier:
                word_to_tier[item] = (group_idx, tier_idx)

    # 2. 扫描 conclusion，找出【每个规则组】各自被触发的最高优先级（最小的梯队索引）
    # 结构为: { 规则组idx: 本次触发的最小梯队idx }
    group_highest_triggered = {}
    for item in conclusion:
        if item in word_to_tier:
            group_idx, tier_idx = word_to_tier[item]
            if group_idx not in group_highest_triggered:
                group_highest_triggered[group_idx] = tier_idx
            else:
                group_highest_triggered[group_idx] = min(group_highest_triggered[group_idx], tier_idx)

    # 3. 过滤与构建结果
    result = []
    for item in conclusion:
        if item in word_to_tier:
            group_idx, tier_idx = word_to_tier[item]
            # 核心判断：只有当该词的梯队 “等于” 它所属规则组本次触发的最高梯队时，才保留
            if tier_idx == group_highest_triggered[group_idx]:
                result.append(item)
            # 如果比最高触发梯队大（优先级低），则被干掉
        else:
            # 不在任何规则库中的词（如 HP感染），直接保留
            result.append(item)

    return list(dict.fromkeys(result)), group_highest_triggered


def analysis_result(responses: List[Tuple[str, str, str]], tokenizer, llm_model, out_path: str):
    standard, conclusion, split_sentences = responses[0], ast.literal_eval(responses[1]), ast.literal_eval(responses[2])

    loc_agg = {}  # 聚合每个部位的描述和病变，避免后续重复条目
    for split_sentence in split_sentences:
        loc, describe, disease = split_sentence
        if loc:
            ent = loc_agg.setdefault(loc, {"texts": [], "lesions": set(), "num": 0})
            ent["num"] += 1
            if describe:
                ent["texts"].append(describe)
            if disease:
                disease = normalize_disease_name(disease)
                if '癌' in disease:
                    disease = loc + '黏膜病变'
                ent["lesions"].add(disease)

    # 构建 llm_infos：每项为 (loc, merged_text, merged_lesions)
    llm_infos = []
    for loc, v in loc_agg.items():
        # 若合并数量 >1 且 LLM 可用，则调用 llm_summarize 进行简化
        if v.get("num", 0) > 1 and tokenizer is not None and llm_model is not None:
            merged_text = llm_summarize(tokenizer, llm_model, v["texts"])
        else:
            merged_text = "；".join(v["texts"]) if v["texts"] else ""
        merged_lesions = list(v["lesions"])
        conclusion.extend(merged_lesions)
        llm_infos.append((loc, merged_text, merged_lesions))

    report = generate_report(llm_infos, conclusion)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已保存到 {out_path}")

    return llm_infos, conclusion


def generate_report(llm_infos: List[Tuple[str, str, str]], conclusion: List[str]) -> dict:
    report_concl = ["慢性浅表性胃炎"]
    report_concl.extend(conclusion)
    dedup_report_concl, group_highest_triggered = deduplicate_concl(report_concl)

    dedup_report_concl = [item for item in dedup_report_concl if any(kw in item for kw in allowed_keywords)]

    base_gastritis = dup_list[0][group_highest_triggered[0]][0]
    base_report = {
        "检查过程": copy.deepcopy(template_process[base_gastritis]),
        "检查结果": '；'.join(dedup_report_concl),
    }

    for loc, norm_text, lesion in llm_infos:
        if loc in ['胃窦', '胃角', '胃体', '贲门']:
            has_atrophy = any('胃炎' in item for item in lesion)
            if not has_atrophy:
                norm_text = llm_summarize(tokenizer, llm_model, [base_report["检查过程"][loc], norm_text])
        base_report["检查过程"][loc] = norm_text.replace(f'{loc}', '')

    return base_report


def find_audio_files(wav_dir: str) -> List[str]:
    patterns = ["*.wav", "*.mp3"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(wav_dir, p)))
    return sorted(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch ASR(音频文件夹/文件，病例的整体描述) -> LLM -> Report")
    # parser.add_argument("--wav_dir", default="/media/inno/ASR/audio/test/0601/", help="音频文件夹路径")
    parser.add_argument("--wav_dir", default="/media/inno/LLM/胃镜/report/V5/test/audio/", help="音频文件夹路径")
    parser.add_argument("--asr_checkpoint", default="/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_gi_v3/", help="ASR模型路径")
    parser.add_argument("--hotwords", default="/media/inno/ASR/胃镜/gi_hotwords_v3.txt", help="ASR模型热词路径，可为空")
    # parser.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507", help="LLM基础模型")
    parser.add_argument("--base_model", default="/home/inno/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554/", help="LLM基础模型")
    parser.add_argument("--lora", default="/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v5-rank16/", help="LLM微调模型路径")
    parser.add_argument("--out", default="/media/inno/output/report/胃镜/test/", help="输出文件夹路径")
    args = parser.parse_args()

    start = time.time()
    # print("加载 ASR 模型...")
    asr_model, hotwords = load_asr_model(args.asr_checkpoint, hotwords_path=args.hotwords)
    time_1 = time.time()
    # print("加载 LLM 模型...")
    tokenizer, llm_model = load_llm_model(args.base_model, args.lora)
    time_2 = time.time()

    if os.path.isdir(args.wav_dir):
        wav_files = find_audio_files(args.wav_dir)
    elif os.path.isfile(args.wav_dir):
        wav_files = [args.wav_dir]
    else:
        wav_files = []

    if not wav_files:
        print(f"未找到音频文件: {args.wav_dir}")
    else:
        os.makedirs(args.out, exist_ok=True)
        print(f"找到 {len(wav_files)} 个待处理音频文件...")
        for i, wav_path in enumerate(wav_files):
            file_start = time.time()
            try:
                print(f"\n[{i+1}/{len(wav_files)}] 正在处理: {wav_path}")
                asr_text = asr_transcribe(asr_model, wav_path)
                print('ASR结果：\n', asr_text)
                time_3 = time.time()
                responses = llm_process(tokenizer, llm_model, asr_text)
                print('LLM结果：\n', responses)
                time_4 = time.time()
                
                # 保存的report.json改成对应媒体文件.json
                base_name = os.path.splitext(os.path.basename(wav_path))[0]
                out_path = os.path.join(args.out, f"{base_name}.json")
                
                # 保存 ASR 文本和 LLM 原始响应
                asr_llm_path = os.path.join(args.out, f"{base_name}_asr_llm.json")
                with open(asr_llm_path, "w", encoding="utf-8") as f:
                    json.dump({"asr_text": asr_text, "responses": responses}, f, ensure_ascii=False, indent=2)
                print(f"ASR和LLM结果已保存到 {asr_llm_path}")

                llm_infos, conclusion = analysis_result(responses, tokenizer, llm_model, out_path)
                file_end = time.time()
                print(f"处理完成，耗时: ASR推理={time_3 - file_start:.2f}s, LLM推理={time_4 - time_3:.2f}s, 后处理={file_end - time_4:.2f}s")
            except Exception as e:
                print(f"处理文件失败 {wav_path}: {e}")

    end = time.time()
    print(f"\nASR 加载耗时: {time_1 - start:.2f}s")
    print(f"LLM 加载耗时: {time_2 - time_1:.2f}s")
    print(f"总处理耗时: {end - start:.2f}s")
