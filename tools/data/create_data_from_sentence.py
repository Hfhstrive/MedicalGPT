import argparse
import json
import glob
import os
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
        'eso_speckle': '斑驳食管',
        'hiatal_hernia': '食管裂孔疝',
    }
    loc_map = {
        'eso_speckle': '食管',
        'hiatal_hernia': '贲门',
    }

    if os.path.isdir(standard_path):
        standard_paths = glob.glob(f'{standard_path}/**/**.txt')
    elif os.path.isfile(standard_path):
        standard_paths = [standard_path]

    standard_js = {}
    for s_p in standard_paths:
        base_name = '.'.join(s_p.split('/')[-1].split('.')[:-1])
        with open(s_p, 'r') as f:
            lines = f.readlines()
        if os.path.isdir(standard_path):
            dir_file_name = s_p.replace(standard_path, '').lstrip('/')
            file_name = '.'.join(dir_file_name.replace('/', '_').split('.')[:-1])
        elif os.path.isfile(standard_path):
            file_name = base_name
        for idx, line in enumerate(lines):
            mode = 'train' if idx <= 0.8 * len(lines) else 'test'
            name = file_name + f'_line{idx}'
            loc = next((word for word in gi_loc if word in line), None)
            if loc is None:
                loc = loc_map[base_name]
            if len(line.split(' ')) == 1:
                norm_text, lesion = line.strip('\n'), lesion_map[base_name]
            elif len(line.split(' ')) == 2:
                norm_text, lesion = line.split(' ')[1].strip('\n'), line.split(' ')[0]
            elif len(line.split(' ')) == 3:
                norm_text, loc, lesion = line.split(' ')[2].strip('\n'), line.split(' ')[1].strip('\n'), line.split(' ')[0]
            describes, locations, lesions = re.split('[；;]', norm_text), re.split(f'[、；;]', loc), re.split(f'[、；;]', lesion)
            if len(locations) == 1 and (len(describes) > 1 or len(lesions) > 1):
                locations = locations * max(len(describes), len(lesions))
            assert len(describes) == len(locations) == len(lesions)
            standard_js.update({
                name: {
                    'describes': describes,
                    'locations': locations,
                    'lesions': lesions,
                    'mode': mode,
                }
            })
    return standard_js


def main():
    parser = argparse.ArgumentParser(description="Batch ASR -> LLM data ")
    # wav_dir、standard_path、asr_result下子文件夹名称一一对应
    parser.add_argument("--wav_dir", default='/media/inno/ASR/audio/train/real/sentence/', help="音频文件路径")
    parser.add_argument("--asr_checkpoint", default="/home/inno/code/ASR/FunASR/examples/industrial_data_pretraining/fun_asr_nano/outputs/fun_asr_nano_2512_gi_v2/", help="ASR模型路径")
    parser.add_argument("--hotwords", default="", help="ASR模型热词路径，可为空")
    parser.add_argument("--standard_path", default="/media/inno/ASR/base_data/standard/sentence/", help="规范表达的文件/文件夹路径")
    parser.add_argument("--asr_result", default="/media/inno/ASR/base_data/ASR_oral/sentence/", help="ASR识别结果保存路径")
    parser.add_argument("--save_dir", default="/media/inno/LLM/report/V3/", help="训练数据集保存路径")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.asr_result, exist_ok=True)
    if not os.path.isdir(args.wav_dir):
        raise SystemExit(f"wav_dir not found: {args.wav_dir}")

    print("加载 ASR 模型...")
    asr_model, hotwords = load_asr_model(args.asr_checkpoint, hotwords_path=args.hotwords)

    audio_files = find_audio_files(args.wav_dir)
    if not audio_files:
        print("未在文件夹中找到音频文件。")
        return

    standard_js = find_standard_file(args.standard_path)

    for audio_file in audio_files:
        # 相差几级目录
        relative = Path(audio_file).relative_to(Path(args.wav_dir))
        depth = len(relative.parents)
        prefix = audio_file.replace(args.wav_dir, '').split('/')
        name = '.'.join(prefix[-1].split('.')[:-1])
        asr_result_path = os.path.join(args.asr_result, '/'.join(prefix[:-(depth - 1)]), name + '.txt')
        standard_name = '_'.join(prefix[:-(depth - 1)]) + '_' + name.replace('_oral', '')

        if os.path.exists(asr_result_path):
            with open(asr_result_path, 'r') as f:
                content = f.readlines()
            assert len(content) == 1
            oral_text = content[0].strip('\n')
        else:
            oral_text = asr_transcribe(asr_model, audio_file)
            with open(asr_result_path, 'w') as f:
                f.writelines(oral_text + '\n')
        mode = standard_js[standard_name]['mode']
        message = {
            "messages": [
                # V2
                # {
                #     "role": "user",
                #     "content": f"规范表达以下语句：{oral_text}"
                # },
                # {
                #     "role": "assistant",
                #     "content": f"{standard_js[standard_name]['describes']}"
                # },
                # {
                #     "role": "user",
                #     "content": "描述的是哪个部位（'食管', '贲门', '胃体', '胃窦', '幽门', '胃角', '胃底', '十二指肠'）？"
                # },
                # {
                #     "role": "assistant",
                #     "content": f"{standard_js[standard_name]['locations']}"
                # },
                # {
                #     "role": "user",
                #     "content": "描述涉及哪些疾病？"
                # },
                # {
                #     "role": "assistant",
                #     "content": f"{standard_js[standard_name]['lesions']}"
                # }

                # V3
                {
                    "role": "user",
                    "content": f"以下描述涉及哪些疾病：{oral_text}？"
                },
                {
                    "role": "assistant",
                    "content": f"{standard_js[standard_name]['lesions']}"
                },
                {
                    "role": "user",
                    "content": "这些疾病分别在哪个部位（'食管', '贲门', '胃体', '胃窦', '幽门', '胃角', '胃底', '十二指肠'）？"
                },
                {
                    "role": "assistant",
                    "content": f"{standard_js[standard_name]['locations']}"
                },
                {
                    "role": "user",
                    "content": "对这些病变分别进行规范描述："
                },
                {
                    "role": "assistant",
                    "content": f"{standard_js[standard_name]['describes']}"
                }
            ]
        }
        mode_path = os.path.join(args.save_dir, mode + '.jsonl')
        with open(mode_path, 'a+', encoding='utf-8') as f:
            f.write(json.dumps(message, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()