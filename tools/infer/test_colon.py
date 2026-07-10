import os
import json
import time
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from ipdb import set_trace

# 1. 指定模型名称
# base_model_name = "/home/inno/.cache/modelscope/hub/models/Qwen/Qwen3.5-4B/"
# lora_adapter_path = "/media/inno/work_dirs/LLM/MedicalGPT/colon/outputs-sft-qwen3.5-4b-v1/checkpoint-66/"
base_model_name = "Qwen/Qwen3-4B-Instruct-2507"
lora_adapter_path = "/media/inno/work_dirs/LLM/MedicalGPT/colon/outputs-sft-qwen3-4b-instruct-v1/checkpoint-99/"

# 2. 加载分词器和模型
print("正在加载模型，首次运行会从 Hugging Face 下载...")
start_init_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
# model = base_model

end_init_time = time.time()
init_duration = end_init_time - start_init_time
print(f"模型加载完成，初始化耗时: {init_duration:.2f} 秒。")

if torch.cuda.is_available():
    allocated_vram = torch.cuda.memory_allocated() / 1024 ** 2  # MB
    max_allocated_vram = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB
    print(f"模型加载后 GPU 显存占用: {allocated_vram:.2f} MB (峰值: {max_allocated_vram:.2f} MB)")

# 3. 收集输入数据 (支持手动列表与 val.json 读取)
inputs_to_run = {}

# 3.1 手动输入列表 (默认 key 格式为 asr_texts_idx)
# 如果需要临时手动测试，可将下方 asr_texts 注释解除或修改
asr_texts = [
    # '这个肠道准备不错，波士顿有8分，盲升结肠、横结肠、降结肠分别为2分、3分、3分。退行过程中，在降结肠看到2枚息肉，大小约0.3到1厘米，较大的那一枚为平坦病变，靛胭脂染色后边界清晰，在病变局部有结节样的隆起，NBI黏膜呈棕色，NICE分型显示Type2。还有在肝曲和降结肠可以看到2处大片状黏膜隆，充血糜烂。',
    # '这个患者肠道有泡沫样液体，BBPS分别为2、3、2分。在末端回肠有点状糜烂，这个回盲瓣部有一枚约1cm的溃疡，中央烧脓起，有新生露牙，取活检1块，还有结肠炎。在横结肠、降结肠、乙状结肠及直肠黏膜粗糙肿胀、血管网模糊、局部有黏液浓苔，降结肠已取活检1块。同时，横结肠中段有2处黏膜烧脓起，色红、染色后边界清晰，考虑炎性息肉可能，已做圈套器冷切除。',
]

try:
    if 'asr_texts' in globals() and asr_texts:
        for idx, text in enumerate(asr_texts):
            inputs_to_run[f"asr_texts_{idx}"] = text
except NameError:
    pass

# 3.2 从 val.json 读取输入
asr_path = '/media/inno/LLM/肠镜/report/V1/ann'
val_json_path = os.path.join(asr_path, 'val.json')
if os.path.exists(val_json_path):
    print(f"正在从 {val_json_path} 读取评估数据...")
    with open(val_json_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
        for key, text in val_data.items():
            inputs_to_run[key] = text

print(f"总计收集到待测样例数: {len(inputs_to_run)}")

output_json_path = '/media/inno/output/LLM/colon/qwen3-4b-instruct-v1/val.json'
results = {}
total_generated_tokens = 0

# 4. 循环进行推理
print("\n开始推理评估样例...")
start_inference_time = time.time()

for key, asr_text in inputs_to_run.items():
    questions = [
        f"提取有效信息,生成标准肠镜报告：{asr_text}",
    ]
    messages = []
    
    sample_start_time = time.time()
    for i, q in enumerate(questions):
        messages.append({"role": "user", "content": q})
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        # 推理
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
        )

        # 解码
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)
        total_generated_tokens += len(output_ids)

        print(f"\n--- Key: {key} (问题 {i + 1}) ---")
        print(f"问：{q}")
        print(f"答：{response}")
        messages.append({"role": "assistant", "content": response})
        
        # 将最终结果存入字典
        results[key] = response.strip()

    sample_end_time = time.time()
    sample_duration = sample_end_time - sample_start_time
    print(f"[耗时统计] 样本 {key} 推理耗时: {sample_duration:.2f} 秒")
    print('----------------------------------------------------------------------------------')

end_inference_time = time.time()
total_inference_duration = end_inference_time - start_inference_time

# 5. 获取 CPU 使用占比和内存指标
cpu_percent = psutil.cpu_percent(interval=0.5)
process = psutil.Process(os.getpid())
process_cpu_percent = process.cpu_percent(interval=None)

# 打印运行性能报告
print("\n================== 性能指标汇总 ==================")
print(f"1. 模型加载/初始化耗时: {init_duration:.2f} 秒")
if torch.cuda.is_available():
    allocated_vram = torch.cuda.memory_allocated() / 1024 ** 2  # MB
    max_allocated_vram = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB
    print(f"2. GPU 显存占用量: {allocated_vram:.2f} MB (峰值: {max_allocated_vram:.2f} MB)")
print(f"3. 统 CPU 占用比例: {cpu_percent:.1f}%")
print(f"   当前进程 CPU 比例: {process_cpu_percent:.1f}%")

num_samples = len(inputs_to_run)
if num_samples > 0:
    avg_sample_time = total_inference_duration / num_samples
    avg_tokens_per_sec = total_generated_tokens / total_inference_duration if total_inference_duration > 0 else 0.0
    print(f"4. 总推理耗时: {total_inference_duration:.2f} 秒")
    print(f"5. 总生成 Token 数量: {total_generated_tokens}")
    print(f"6. 整体推理吞吐速度: {avg_tokens_per_sec:.2f} tokens/s")
    print(f"7. 平均单个 asr_text 的推理耗时: {avg_sample_time:.2f} 秒/样本")
print("=================================================\n")

# 6. 保存结果到单 JSON 文件中
output_dir = os.path.dirname(output_json_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
print(f"所有推理结果已成功保存至: {output_json_path}")