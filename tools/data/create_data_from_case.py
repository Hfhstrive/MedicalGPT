import argparse
import json
import glob
import os
import random
import re
from typing import List
from pathlib import Path
from ipdb import set_trace


def load_asr_model(checkpoint: str, device: str = "cuda:0", hotwords_path: str = None):
    from funasr import AutoModel

    hotwords = None
    if hotwords_path and os.path.exists(hotwords_path):
        with open(hotwords_path, "r", encoding="utf-8") as f:
            hotwords = [line.strip() for line in f if line.strip()]

    if hotwords is None:
        model = AutoModel(
            model=checkpoint,
            # vad_model="fsmn-vad",
            device=device,
            disable_update=True,
            dtype="fp16",
        )
    else:
        model = AutoModel(
            model=checkpoint,
            # vad_model="fsmn-vad",
            hotwords=hotwords,
            device=device,
            disable_update=True,
            dtype="fp16",
        )
    return model, hotwords


def asr_transcribe(model, wav_path: str) -> str:
    res = model.generate(input=[wav_path], cache={}, batch_size_s=0)
    return res[0].get("text", "")


def main():
    parser = argparse.ArgumentParser(description="LLM data ")
    # -------------------------------- 方式1：口语化文本 --------------------------------------------
    parser.add_argument("--oral_path", default="/media/inno/ASR/base_data/oral/case/", help="口语化表达的文件夹路径, asr未开启使用")
    # -------------------------------- 方式2：口语化音频 --------------------------------------------
    parser.add_argument('--asr', action='store_true', help='识别音频文件以获得口语化表达')
    parser.add_argument('--asr_checkpoint', default="/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_gi_v3/", help='ASR模型路径')
    parser.add_argument("--wav_path", default="/media/inno/ASR/audio/train/real/case/", help="口语化表达的文件夹路径, asr开启使用")
    parser.add_argument("--hotwords", default="/media/inno/ASR/gi_hotwords_v3.txt", help="ASR模型热词路径，可为空")
    # --------------------------------------------------------------------------------------------
    parser.add_argument("--standard_path", default="/media/inno/ASR/base_data/standard/case/", help="规范表达的文件夹路径")
    parser.add_argument("--save_dir", default="/media/inno/LLM/report/V5/", help="训练数据集保存路径")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.asr:
        print("加载 ASR 模型...")
        asr_model, hotwords = load_asr_model(args.asr_checkpoint, hotwords_path=args.hotwords)

    for lesion in os.listdir(args.standard_path):
        train_nums, val_nums = 0, 0
        standard_files = glob.glob(f'{os.path.join(args.standard_path, lesion)}/**.txt')
        for standard_file in standard_files:
            # 判断当前病例为哪个集合
            if val_nums >= len(standard_files) * 0.1:
                mode = 'train'
                train_nums += 1
            elif val_nums == 0 or random.random() <= 0.1:
                mode = 'test'
                val_nums += 1
            else:
                mode = 'train'
                train_nums += 1
            # 口语化描述句子
            if args.asr:
                wav_oral_file = standard_file.replace(args.standard_path, args.wav_path)
                if os.path.exists(wav_oral_file):
                    with open(wav_oral_file, 'r') as f:
                        content = f.readlines()
                    assert len(content) == 1
                    oral_info = content[0].strip('\n')
                else:
                    wav_file = wav_oral_file.replace('.txt', '.mp3')
                    assert os.path.exists(wav_file)
                    oral_info = asr_transcribe(asr_model, wav_file)
                    with open(wav_oral_file, 'w') as f:
                        f.writelines(oral_info + '\n')
            else:
                oral_file = standard_file.replace(args.standard_path, args.oral_path)
                assert os.path.exists(oral_file)
                with open(oral_file, 'r') as f:
                    oral_content = f.readlines()
                assert len(oral_content) == 1
                oral_info = oral_content[0].strip('\n')
            # 标准化描述句子
            with open(standard_file, 'r') as f:
                standard_content = f.readlines()
            assert len(standard_content) >= 3, f'{standard_file}'
            standard_info = standard_content[0].strip('\n')
            diagnosis = re.split(f'[、；;]', standard_content[1].strip('\n'))
            feature_infos = []
            for line in standard_content[2:]:
                describe = line.strip('\n').strip(' ').split(' ')
                assert len(describe) >= 3, f'{standard_file}'
                feature = ' '.join(describe[1:-1])
                feature_infos.append([describe[0], feature, describe[-1]])
            message = {
                "messages": [
                    {
                        "role": "user",
                        # "content": f"规范表达以下语句：{oral_info}？"
                        "content": f"提取有效信息，规范表达以下语句：{oral_info}？"
                    },
                    {
                        "role": "assistant",
                        "content": f"{standard_info}"
                    },
                    {
                        "role": "user",
                        "content": "提取语句中所涉及的结论性疾病？"
                    },
                    {
                        "role": "assistant",
                        "content": f"{diagnosis}"
                    },
                    {
                        "role": "user",
                        # "content": "根据[部位, 特征描述, 病变名称]格式拆分语句："
                        "content": "根据[部位, 特征描述, 病变名称]格式拆分语句，并专业化特征描述："
                    },
                    {
                        "role": "assistant",
                        "content": f"{feature_infos}"
                    }
                ]
            }
            mode_path = os.path.join(args.save_dir, mode + '.jsonl')
            with open(mode_path, 'a+', encoding='utf-8') as f:
                f.write(json.dumps(message, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    random.seed(20260528)
    main()