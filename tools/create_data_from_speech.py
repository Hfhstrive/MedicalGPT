import argparse
import json
import glob
import os
from typing import List
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


def find_audio_files(wav_dir: str) -> List[str]:
    patterns = ["wav", "mp3"]
    files = []
    for type in patterns:
        files.extend(glob.glob(wav_dir + '/**/**.' + type, recursive=True))
    return sorted(files)


def asr_transcribe(model, wav_path: str) -> str:
    res = model.generate(input=[wav_path], cache={}, batch_size_s=0)
    return res[0].get("text", "")


def find_standard_file(standard_path: str) -> dict:
    gi_loc = ['食管', '贲门', '胃体', '胃窦', '幽门', '胃角', '胃底', '十二指肠']
    lesion_map = {
        'eca': '食管癌',
        'eso_barrett': 'Barrett食管',
        'eso_down': '食管黏膜隆起',
        'eso_fungus': '霉菌性食管炎',
        'eso_reflux': '反流性食管炎',
        'eso_varices': '食管静脉曲张',
        'gm_ectopia': '食管胃黏膜异位',
        'gca': '胃癌',
        'gm_down': '胃黏膜下肿瘤',
        'gm_up': '胃息肉',
        'g_ulcer': '胃溃疡',
        'atrophy': '萎缩性胃炎',
        'im': '肠化',
    }

    if os.path.isdir(standard_path):
        standard_paths = glob.glob(f'{standard_path}/**.txt')
    elif os.path.isfile(standard_path):
        standard_paths = [standard_path]

    standard_js = {}
    for s_p in standard_paths:
        with open(s_p, 'r') as f:
            lines = f.readlines()
        file_name = s_p.split('/')[-1].split('.')[0]
        for idx, line in enumerate(lines):
            mode = 'train' if idx <= 0.8 * len(lines) else 'test'
            name = file_name + f'_line{idx}'
            loc = next((word for word in gi_loc if word in line), None)
            if len(line.split(' ')) == 1:
                norm_text, lesion = line.strip('\n'), lesion_map[file_name]
            elif len(line.split(' ')) == 3:
                norm_text, lesion = line.split(' ')[2].strip('\n'), line.split(' ')[0]
            standard_js.update({
                name: {
                    'norm_text': norm_text,
                    'loc': loc,
                    'lesion': lesion,
                    'mode': mode,
                }
            })
    return standard_js


def main():
    parser = argparse.ArgumentParser(description="Batch ASR -> LLM data ")
    parser.add_argument("--wav_dir", default='/media/inno/ASR/audio/train/real/', help="音频文件路径")
    parser.add_argument("--asr_checkpoint", default="/home/inno/code/ASR/FunASR/examples/industrial_data_pretraining/fun_asr_nano/outputs/fun_asr_nano_2512_gi_v2/", help="ASR模型热词路径")
    parser.add_argument("--hotwords", default="", help="ASR模型热词路径，可为空")
    parser.add_argument("--standard_path", default="/media/inno/ASR/base_data/standard/multi_lesion.txt", help="规范表达的文件/文件夹路径")
    parser.add_argument("--asr_result", default="/media/inno/ASR/base_data/ASR_oral/", help="ASR识别结果保存路径")
    parser.add_argument("--save_dir", default="/media/inno/LLM/retrieval/V2/", help="训练数据集保存路径")
    args = parser.parse_args()

    # wav_dir = args.wav_dir
    os.makedirs(args.save_dir, exist_ok=True)
    if not os.path.isdir(args.wav_dir):
        raise SystemExit(f"wav_dir not found: {args.wav_dir}")

    print("加载 ASR 模型...")
    asr_model, hotwords = load_asr_model(args.asr_checkpoint, hotwords_path=args.hotwords)

    audio_files = find_audio_files(args.wav_dir)
    if not audio_files:
        print("未在文件夹中找到音频文件。")
        return

    standard_js = find_standard_file(args.standard_path)

    for p in audio_files:
        name = p.split('/')[-1].split('.')[0]
        oral_text = asr_transcribe(asr_model, p)
        asr_result_path = os.path.join(args.asr_result, name + '.txt')
        with open(asr_result_path, 'w') as f:
            f.writelines(oral_text + '\n')
        mode = standard_js[name]['mode']
        message = {
            "messages": [
                {
                    "role": "user",
                    "content": f"规范表达以下语句：{oral_text}"
                },
                {
                    "role": "assistant",
                    "content": f"{standard_js[name]['norm_text']}"
                },
                {
                    "role": "user",
                    "content": "描述的是哪个部位（'食管', '贲门', '胃体', '胃窦', '幽门', '胃角', '胃底', '十二指肠'）？"
                },
                {
                    "role": "assistant",
                    "content": f"{standard_js[name]['loc']}"
                },
                {
                    "role": "user",
                    "content": "描述涉及哪些疾病？"
                },
                {
                    "role": "assistant",
                    "content": f"{standard_js[name]['lesion']}"
                }
            ]
        }
        mode_path = os.path.join(args.save_dir, mode + '.jsonl')
        with open(mode_path, 'a+', encoding='utf-8') as f:
            f.write(json.dumps(message, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()