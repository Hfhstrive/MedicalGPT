import argparse
import json
import glob
import os
import random
import re
import shutil
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


def process_gastroscope_text(standard_file: str, oral_info: str) -> dict:
    """处理胃镜文本，解析并生成微调所需的规范提示和结论等"""
    with open(standard_file, 'r', encoding='utf-8') as f:
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
                "content": "根据[部位, 特征描述, 病变名称]格式拆分语句，并专业化特征描述："
            },
            {
                "role": "assistant",
                "content": f"{feature_infos}"
            }
        ]
    }
    return message


def process_colonscope_text(standard_file: str, oral_info: str) -> dict:
    """处理肠镜文本的逻辑接口 (可在下面编写您的肠镜解析与结构转换逻辑)"""
    message = {
        "messages": [
            {
                "role": "user",
                "content": f"肠镜口语输入：{oral_info}"
            },
            {
                "role": "assistant",
                "content": "肠镜标准表达占位符"
            }
        ]
    }
    return message


def main():
    parser = argparse.ArgumentParser(description="LLM data ")
    # -------------------------------- ASR 配置 ----------------------------------------------------
    parser.add_argument('--asr_checkpoint', default="/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_gi_v3/", help='ASR模型路径')
    parser.add_argument("--hotwords", default="/media/inno/ASR/胃镜/gi_hotwords_v3.txt", help="ASR模型热词路径，可为空")
    # --------------------------------------------------------------------------------------------
    parser.add_argument("--save_dir", default="/media/inno/LLM/胃镜/report/V5_test/", help="训练数据集保存路径")
    parser.add_argument("--case_mode", default=None, help="JSON路径，包含train_cases和val_cases，用于指定病例划分集合")
    parser.add_argument("--gi_type", choices=["gastro", "colon"], default="gastro", help="内镜类型 (gastro: 胃镜, colon: 肠镜)")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # -------------------------------- 数据源配置区域 ------------------------------------------
    # ----------------(口语化文本/音频文件夹路径, 标准表达文件夹路径, 是否为口语化音频)----------------
    data_sources = [
        # ("/media/inno/ASR/胃镜/base_data/oral/case/", "/media/inno/ASR/胃镜/base_data/standard/case/", False),   # 胃镜V4
        ("/media/inno/ASR/胃镜/audio/train/real/case/", "/media/inno/ASR/胃镜/base_data/standard/case/", True),  # 胃镜V5
    ]
    # ------------------------------------------------------------------------------------------

    assert len(data_sources) > 0, "请在代码的 data_sources 列表中至少配置一个数据源！"

    # 加载指定的病例划分
    train_cases = set()
    val_cases = set()
    if args.case_mode and os.path.exists(args.case_mode):
        try:
            with open(args.case_mode, "r", encoding="utf-8") as f:
                case_data = json.load(f)
                train_cases = set(case_data.get("train_cases", []))
                val_cases = set(case_data.get("val_cases", []))
            print(f"成功加载病例划分: train_cases {len(train_cases)} 个, val_cases {len(val_cases)} 个")
        except Exception as e:
            print(f"加载 case_mode 失败: {e}")


    # 判断是否需要加载 ASR 模型
    any_asr = any(src[2] for src in data_sources)
    if any_asr:
        print("加载 ASR 模型...")
        asr_model, hotwords = load_asr_model(args.asr_checkpoint, hotwords_path=args.hotwords)

    # 清理已存在的 train.jsonl 和 val.jsonl 以免重复追加
    for mode in ['train', 'val']:
        mode_path = os.path.join(args.save_dir, mode + '.jsonl')
        if os.path.exists(mode_path):
            os.remove(mode_path)

    # 用于保存病例号与口语化表达的对应字典 (Requirement 3)
    train_ann = {}
    val_ann = {}

    # 用于保存病例的文件名 (case_name)，供 status.json 统计使用
    train_case_names = []
    val_case_names = []

    # 用于动态控制 val 比例不超过 10% 的全局计数器 (Requirement 2)
    global_train_count = 0
    global_val_count = 0

    for oral_path, standard_path, asr in data_sources:
        print(f"处理数据源: oral_path={oral_path}, standard_path={standard_path}, asr={asr}")
        if not os.path.exists(standard_path):
            print(f"警告: 标准文件夹 {standard_path} 不存在，跳过该数据源")
            continue

        for lesion in os.listdir(standard_path):
            work_lesion_path = os.path.join(standard_path, lesion)
            if not os.path.isdir(work_lesion_path):
                continue

            standard_files = glob.glob(f'{work_lesion_path}/**.txt')
            for standard_file in standard_files:
                case_name = os.path.splitext(os.path.basename(standard_file))[0]
                # 判断当前病例为哪个集合 (优先用 case_mode 划分，其次采用动态限制保证 val 占比不超过 10%)
                if case_name in train_cases:
                    mode = 'train'
                elif case_name in val_cases:
                    mode = 'val'
                else:
                    # 否则根据目前 train 和 val 的数量，保证 val 不超过 10%
                    if (global_val_count + 1) / (global_train_count + global_val_count + 1) > 0.1:
                        mode = 'train'
                    else:
                        # 保证在满足上限的前提下，如果 val_count 仍为 0，强制设定一个 val，否则以 10% 概率划分
                        if global_val_count == 0 or random.random() <= 0.1:
                            mode = 'val'
                        else:
                            mode = 'train'

                # 更新全局计数器
                if mode == 'train':
                    global_train_count += 1
                else:
                    global_val_count += 1

                # 口语化描述句子读取与生成
                if asr:
                    # 确定对应 wav 音频路径，并断言其必须存在
                    wav_oral_file = standard_file.replace(standard_path, oral_path)
                    wav_file = wav_oral_file.replace('.txt', '.wav')
                    assert os.path.exists(wav_file), f"wav音频文件不存在: {wav_file}"

                    if os.path.exists(wav_oral_file):
                        with open(wav_oral_file, 'r', encoding='utf-8') as f:
                            content = f.readlines()
                        assert len(content) == 1
                        oral_info = content[0].strip('\n')
                    else:
                        oral_info = asr_transcribe(asr_model, wav_file)
                        os.makedirs(os.path.dirname(wav_oral_file), exist_ok=True)
                        with open(wav_oral_file, 'w', encoding='utf-8') as f:
                            f.writelines(oral_info + '\n')

                    # 将口语化音频文件拷贝到 save_dir/audio/ 下的 train 或 val 子文件夹中
                    dest_audio_dir = os.path.join(args.save_dir, 'audio', mode, lesion)
                    os.makedirs(dest_audio_dir, exist_ok=True)
                    shutil.copy2(wav_file, os.path.join(dest_audio_dir, os.path.basename(wav_file)))
                else:
                    oral_file = standard_file.replace(standard_path, oral_path)
                    assert os.path.exists(oral_file), f"口语化文本文件不存在: {oral_file}"
                    with open(oral_file, 'r', encoding='utf-8') as f:
                        oral_content = f.readlines()
                    assert len(oral_content) == 1
                    oral_info = oral_content[0].strip('\n')

                # 保存病例号与 oral_info 的映射关系 (Requirement 3)
                if mode == 'train':
                    train_ann[case_name] = oral_info
                    train_case_names.append(case_name)
                else:
                    val_ann[case_name] = oral_info
                    val_case_names.append(case_name)

                # 根据内镜类型解析并生成微调消息结构
                if args.gi_type == "gastro":
                    message = process_gastroscope_text(standard_file, oral_info)
                elif args.gi_type == "colon":
                    message = process_colonscope_text(standard_file, oral_info)
                else:
                    raise ValueError(f"不支持的内镜类型: {args.gi_type}")

                mode_path = os.path.join(args.save_dir, mode + '.jsonl')
                with open(mode_path, 'a+', encoding='utf-8') as f:
                    f.write(json.dumps(message, ensure_ascii=False) + '\n')

    # 将不同数据集的病例号跟 oral_info 对应保存为 train.json 和 val.json (Requirement 3)
    ann_dir = os.path.join(args.save_dir, 'ann')
    os.makedirs(ann_dir, exist_ok=True)
    with open(os.path.join(ann_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_ann, f, ensure_ascii=False, indent=4)
    with open(os.path.join(ann_dir, 'val.json'), 'w', encoding='utf-8') as f:
        json.dump(val_ann, f, ensure_ascii=False, indent=4)
    
    # 生成 status.json (记录 train_cases 和 val_cases 的具体值以及数量)
    status_data = {
        "train_cases": sorted(list(set(train_case_names))),
        "val_cases": sorted(list(set(val_case_names))),
        "train_count": len(train_case_names),
        "val_count": len(val_case_names)
    }
    status_path = os.path.join(args.save_dir, 'status.json')
    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump(status_data, f, ensure_ascii=False, indent=4)
    
    print(f"数据处理完毕。训练集样本数: {global_train_count}, 验证集样本数: {global_val_count}")
    print(f"标注文件已保存至: {ann_dir}")
    print(f"病例划分状态汇总已保存至: {status_path}")


if __name__ == '__main__':
    random.seed(20260528)
    main()