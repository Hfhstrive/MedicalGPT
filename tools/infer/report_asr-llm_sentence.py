import argparse
import glob
import json
import os
import re
import ast
from ipdb import set_trace
from typing import List, Tuple


def load_asr_model(checkpoint: str, device: str = "cuda:0", hotwords_path: str = None):
    from funasr import AutoModel

    hotwords = None
    if hotwords_path and os.path.exists(hotwords_path):
        with open(hotwords_path, "r", encoding="utf-8") as f:
            hotwords = [line.strip() for line in f if line.strip()]

    if hotwords is None:
        model = AutoModel(
            model=checkpoint,
            vad_model="fsmn-vad",
            device=device,
            disable_update=True,
            dtype="fp16",
        )
    else:
        model = AutoModel(
            model=checkpoint,
            vad_model="fsmn-vad",
            hotwords=hotwords,
            device=device,
            disable_update=True,
            dtype="fp16",
        )
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
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model.eval()
    return tokenizer, model


def llm_process(tokenizer, model, info: str) -> Tuple[str, str, str]:
    # questions: normalized text, organ, disease
    questions = [
        # # V2
        # f"规范表达以下语句：{info}",
        # "描述的是哪个部位（'食管', '贲门', '胃体', '胃窦', '幽门', '胃角', '胃底', '十二指肠'）？",
        # "描述涉及哪些疾病？",
        # V3
        f"以下描述涉及哪些疾病：{info}",
        "这些疾病分别在哪个部位（'食管', '贲门', '胃体', '胃窦', '幽门', '胃角', '胃底', '十二指肠'）？",
        "对这些病变分别进行规范描述：",
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
            temperature=0.1,  # 医疗建议通常调低温度，增加确定性
            do_sample=True,
        )

        # 解码
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)

        # 重要：将模型的回答加入对话历史，这样下一轮才能实现“多问题推理”
        messages.append({"role": "assistant", "content": response})
        responses.append(response.strip())

    # 返回三项：规范化文本、部位、疾病
    # V2
    # describes, locations, lesions = [responses[0]], [responses[1]], [responses[2]]
    describes, locations, lesions = ast.literal_eval(responses[2]), ast.literal_eval(responses[1]), ast.literal_eval(responses[0])
    return describes, locations, lesions


def llm_summarize(tokenizer, model, texts: List[str]) -> str:
    """使用 LLM 将多条描述合并并简化为一句话（保留医学术语）。"""
    if not texts:
        return ""

    joined = "；".join([t for t in texts if t])
    prompt = f"一句话概况以下内容：\n{joined}"

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
        temperature=0.1,
        do_sample=False,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    out = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return out


def generate_report(llm_infos: List[Tuple[str, str, str]], tokenizer=None, model=None) -> dict:
    base_report = {
        "检查过程": {
            "食管": "食管黏膜光滑湿润，血管纹理清晰，可见清晰齿状线，NBI下未见明显异常茶色区。",
            "贲门": "贲门闭合良好，黏膜光滑。",
            "胃底": "黏膜光滑，黏液湖清亮。",
            "胃体": "皱襞走向规则，黏膜光滑，血管纹理清晰。",
            "胃角": "形态完整，黏膜光滑。",   # 黏膜变薄，血管纹理显露
            "胃窦": "黏膜光滑，红白相间，以白为主。",
            "幽门": "圆，通畅。",
            "十二指肠球部": "黏膜光滑，降段上部粘膜未见异常。",
        },
        "检查结果": ["慢性浅表性胃炎"],
    }

    # 按部位合并描述
    loc_map = {}
    for norm_text, loc, lesion in llm_infos:
        if loc:
            loc_map.setdefault(loc, []).append(norm_text)

        # if lesion:
        for les in lesion.split('、'):
            base_report["检查结果"].append(les)

    # 对每个部位，若有多段描述（用分号分隔）则合并并（如果有 LLM）进一步简化
    for loc, texts in loc_map.items():
        base_report["检查过程"][loc] = texts[0]

    # 将检查结果列表去重并以半角分号连接
    results = base_report.get("检查结果", [])
    if isinstance(results, list):
        seen = set()
        deduped = []
        for r in results:
            if r and r not in seen:
                deduped.append(r)
                seen.add(r)
        # 如果存在'萎缩性胃炎'，则移除较弱表述'慢性浅表性胃炎'
        if "萎缩性胃炎" in deduped and "慢性浅表性胃炎" in deduped:
            deduped = [x for x in deduped if x != "慢性浅表性胃炎"]
        base_report["检查结果"] = ";".join(deduped)

    return base_report


def find_audio_files(wav_dir: str) -> List[str]:
    patterns = ["*.wav", "*.mp3"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(wav_dir, p)))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Batch ASR(多个音频文件) -> LLM -> Report")
    # parser.add_argument("--wav_dir", default="/media/inno/ASR/audio/test/hfh/", help="音频文件路径")
    parser.add_argument("--wav_dir", default="/media/inno/ASR/audio/test/0513_2/", help="音频文件夹路径")
    parser.add_argument("--asr_checkpoint", default="/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v2/fun_asr_nano_2512_gi_v2/", help="ASR模型路径")
    parser.add_argument("--hotwords", default="/media/inno/ASR/gi_hotwords_v2.txt", help="ASR模型热词路径，可为空")
    # parser.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507", help="LLM基础模型")
    parser.add_argument("--base_model", default="/root/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554/", help="LLM基础模型")
    parser.add_argument("--lora", default="/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v3/", help="LLM微调模型路径")
    parser.add_argument("--out", default="/media/inno/ASR/audio/test/hfh/LLM_v1/report.json", help="输出文件")
    args = parser.parse_args()

    wav_dir = args.wav_dir
    if not os.path.isdir(wav_dir):
        raise SystemExit(f"wav_dir not found: {wav_dir}")

    print("加载 ASR 模型...")
    asr_model, hotwords = load_asr_model(args.asr_checkpoint, hotwords_path=args.hotwords)
    print("加载 LLM 模型（这可能需要一些时间）...")
    tokenizer, llm_model = load_llm_model(args.base_model, args.lora)

    audio_files = find_audio_files(wav_dir)
    if not audio_files:
        print("未在文件夹中找到音频文件。")
        return

    # 聚合每个部位的描述和病变，避免后续重复条目
    loc_agg = {}
    for p in audio_files:
        print(f"处理: {p}")
        try:
            asr_text = asr_transcribe(asr_model, p)
            describes, locations, diseases = llm_process(tokenizer, llm_model, asr_text)
            for describe, loc, disease in zip(describes, locations, diseases):
                if loc:
                    ent = loc_agg.setdefault(loc, {"texts": [], "lesions": set(), "num": 0})
                    ent["num"] += 1
                    if describe:
                        ent["texts"].append(describe)
                    if disease:
                        if '癌' in disease:
                            disease = loc + '黏膜病变'
                        ent["lesions"].add(disease)
        except Exception as e:
            print(f"处理文件失败 {p}: {e}")

    # 构建 llm_infos：每项为 (merged_text, loc, merged_lesions)
    llm_infos = []
    for loc, v in loc_agg.items():
        # 若合并数量 >1 且 LLM 可用，则调用 llm_summarize 进行简化
        if v.get("num", 0) > 1 and tokenizer is not None and llm_model is not None:
            merged_text = llm_summarize(tokenizer, llm_model, v["texts"])
        else:
            merged_text = "；".join(v["texts"]) if v["texts"] else ""
        merged_lesions = "；".join(sorted(v["lesions"])) if v["lesions"] else ""
        llm_infos.append((merged_text, loc, merged_lesions))

    report = generate_report(llm_infos, tokenizer=tokenizer, model=llm_model)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已保存到 {args.out}")


if __name__ == "__main__":
    main()
