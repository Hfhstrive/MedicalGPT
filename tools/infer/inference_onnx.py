# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: ONNX Model Inference script using Optimum (Multi-turn dialogue)
"""
import argparse
import json
import os
import time
import psutil
from threading import Thread

import torch
from transformers import AutoTokenizer, TextIteratorStreamer
from optimum.onnxruntime import ORTModelForCausalLM


def llm_process_onnx(
        model,
        tokenizer,
        info: str,
        device,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.0,
        stop_str="</s>",
        use_cache=False,
):
    """根据 report_asr-llm_case.py 中的多轮对话进行推理"""
    questions = [
        f"提取有效信息，规范表达以下语句：{info}",
        "提取语句中所涉及的结论性疾病？",
        "根据[部位, 特征描述, 病变名称]格式拆分语句，并专业化特征描述：",
    ]

    messages = []
    responses = []

    for i, q in enumerate(questions):
        messages.append({"role": "user", "content": q})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0.0 else False,
            repetition_penalty=repetition_penalty,
            use_cache=use_cache,
        )
        
        generated_ids = model.generate(**generation_kwargs)
        output_ids = generated_ids[0][len(input_ids[0]):]
        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        messages.append({"role": "assistant", "content": response})
        responses.append(response)

    return responses


def stream_llm_process_onnx(
        model,
        tokenizer,
        info: str,
        device,
        max_new_tokens=512,
        temperature=0.7,
        repetition_penalty=1.0,
        stop_str="</s>",
        use_cache=False,
):
    """流式打印的多轮对话推理"""
    questions = [
        f"提取有效信息，规范表达以下语句：{info}",
        "提取语句中所涉及的结论性疾病？",
        "根据[部位, 特征描述, 病变名称]格式拆分语句，并专业化特征描述：",
    ]

    messages = []
    responses = []

    for i, q in enumerate(questions):
        print(f"\n--- 轮次 {i+1}: {q} ---")
        messages.append({"role": "user", "content": q})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0.0 else False,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
            use_cache=use_cache,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        for new_text in streamer:
            stop = False
            pos = new_text.find(stop_str)
            if pos != -1:
                new_text = new_text[:pos]
                stop = True
            generated_text += new_text
            print(new_text, end="", flush=True)
            if stop:
                break
        print()
        responses.append(generated_text.strip())
        messages.append({"role": "assistant", "content": generated_text.strip()})

    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_model_dir', default='/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v5-rank16/onnx/', type=str, 
                        help="导出的 ONNX 模型所在的目录")
    parser.add_argument('--file_name', default="model_int4.onnx", type=str,
                        help="导出的 ONNX 模型主文件名（如 model.onnx 或 decoder_model.onnx）")
    parser.add_argument('--device', default="cuda", type=str, choices=["cuda", "cpu"],
                        help="运行推理的设备，支持 cuda 或 cpu")
    parser.add_argument('--system_prompt', default="", type=str, help="系统 Prompt")
    parser.add_argument('--stop_str', default="", type=str, help="停止词，默认为 </s>")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument('--data_file', default=None, type=str,
                        help="输入测试数据文件（每行一个指令）")
    parser.add_argument('--interactive', action='store_true', help="是否开启命令行交互模式（默认多轮对话）")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--output_file', default='./predictions_onnx_result.jsonl', type=str)
    parser.add_argument('--use_cache', action='store_true', help="是否启用 KV Cache。对于单文件（如 model.onnx）建议保持默认不启用（False）")
    args = parser.parse_args()
    print("解析后的参数：", args)

    # 记录加载前的资源状态
    t_start = time.time()
    cpu_mem_start = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    gpu_start_used = 0
    gpu_total = 0
    if args.device == "cuda" and torch.cuda.is_available():
        free_gpu, total_gpu = torch.cuda.mem_get_info()
        gpu_total = total_gpu / 1024 / 1024  # MB
        gpu_start_used = (total_gpu - free_gpu) / 1024 / 1024  # MB

    # 1. 载入分词器
    tokenizer = AutoTokenizer.from_pretrained(args.onnx_model_dir, trust_remote_code=True, padding_side='left')
    
    # 2. 载入 Optimum 导出的 ONNX 模型
    provider = "CUDAExecutionProvider" if args.device == "cuda" else "CPUExecutionProvider"
    print(f"正在载入 ONNX 模型，推理设备: {args.device}，执行器: {provider} ...")
    
    # 针对 CUDAExecutionProvider 的显存分配优化，避免 ONNX Runtime 抢占所有显存导致 PyTorch 报错
    if args.device == "cuda":
        os.environ["ORT_CUDA_PROVIDER_OPTIONS"] = "arena_extend_strategy=kSameAsRequested"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

    model = ORTModelForCausalLM.from_pretrained(
        args.onnx_model_dir,
        file_name=args.file_name,
        provider=provider,
        use_cache=args.use_cache
    )
    
    # 记录并输出加载后的资源状态
    t_end = time.time()
    cpu_mem_end = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    load_time = t_end - t_start
    
    gpu_end_used = 0
    gpu_delta = 0
    gpu_ratio = 0
    if args.device == "cuda" and torch.cuda.is_available():
        free_gpu, total_gpu = torch.cuda.mem_get_info()
        gpu_end_used = (total_gpu - free_gpu) / 1024 / 1024  # MB
        gpu_delta = gpu_end_used - gpu_start_used
        gpu_ratio = (gpu_end_used / gpu_total) * 100 if gpu_total > 0 else 0
        
    print("\n================ [ONNX 模型初始化资源统计] ================")
    print(f"ONNX 模型初始化时间: {load_time:.2f} 秒")
    print(f"初始化前后 CPU 内存变化: +{cpu_mem_end - cpu_mem_start:.2f} MB (当前总内存: {cpu_mem_end:.2f} MB)")
    if args.device == "cuda" and torch.cuda.is_available():
        print(f"初始化前后 GPU 显存变化: +{gpu_delta:.2f} MB (当前总显存: {gpu_end_used:.2f} MB / {gpu_total:.2f} MB, 占比 {gpu_ratio:.2f}%)")
    print("===========================================================\n")
    
    # 3. 测试数据
    if args.data_file is None:
        examples = [
            "食管20cm至贲门口有四条蓝色的曲张静脉蛇形迂曲，红色征阳性。食管30cm处有一片黏膜粗糙，NBI下呈茶褐色，不规则、碘染不上色，警惕高级别上皮内瘤变。胃体还有马赛克样改变，散在红斑和糜烂；胃窦还有红疹样变和散在痘疹。。",
        ]
    else:
        with open(args.data_file, 'r', encoding='utf-8') as f:
            examples = [l.strip() for l in f.readlines()]
        print("前 10 个测试用例：")
        for example in examples[:10]:
            print(example)

    stop_str = args.stop_str or "</s>"

    if args.interactive:
        print("欢迎使用命令行多轮对话推理程序！输入医学描述文本，系统将自动依次进行 3 轮抽取。使用 `exit` 退出程序。")
        while True:
            try:
                query = input(f"\nUser 输入医学描述: ")
            except UnicodeDecodeError:
                print("检测到输入解码错误，请重新输入。")
                continue
            except Exception:
                raise
            if query == "":
                print("输入不能为空，请重新输入。")
                continue
            if query.strip() == "exit":
                print("正在退出...")
                break

            dev = model.device if hasattr(model, "device") else torch.device(args.device)
            stream_llm_process_onnx(
                model,
                tokenizer,
                query,
                dev,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                stop_str=stop_str,
                use_cache=getattr(model.config, "use_cache", False),
            )
    else:
        print("开始批量多轮对话推理...")
        counts = 0
        if os.path.exists(args.output_file):
            os.remove(args.output_file)
        
        dev = model.device if hasattr(model, "device") else torch.device(args.device)
        for example in examples:
            print(f"\n==================================================")
            print(f"输入语句: {example}")
            
            # 记录推理前的系统状态
            t_infer_start = time.time()
            psutil.cpu_percent(interval=None)  # 初始化 CPU 百分比计算
            
            responses = stream_llm_process_onnx(
                model,
                tokenizer,
                example,
                dev,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                stop_str=stop_str,
                use_cache=getattr(model.config, "use_cache", False),
            )
            
            # 记录推理后的资源状态
            t_infer_end = time.time()
            infer_time = t_infer_end - t_infer_start
            cpu_usage = psutil.cpu_percent(interval=None)
            cpu_mem_curr = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            print(f"\n---------------- [本次推理性能指标] ----------------")
            print(f"多轮推理总耗时: {infer_time:.2f} 秒")
            print(f"系统 CPU 核心平均占比: {cpu_usage:.1f}%")
            print(f"当前进程 CPU 内存占用: {cpu_mem_curr:.2f} MB")
            
            if args.device == "cuda" and torch.cuda.is_available():
                free_gpu, total_gpu = torch.cuda.mem_get_info()
                gpu_used = (total_gpu - free_gpu) / 1024 / 1024
                gpu_ratio = (gpu_used / (total_gpu / 1024 / 1024)) * 100
                print(f"当前进程 GPU 显存占用: {gpu_used:.2f} MB (显卡总占比: {gpu_ratio:.2f}%)")
            print(f"--------------------------------------------------\n")
            
            # 保存到结果文件
            result = {
                "Input": example,
                "Standard": responses[0] if len(responses) > 0 else "",
                "Conclusion": responses[1] if len(responses) > 1 else "",
                "Structured": responses[2] if len(responses) > 2 else ""
            }
            with open(args.output_file, 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
            counts += 1
        print(f'已保存结果至 {args.output_file}，总数量: {counts}')


if __name__ == '__main__':
    main()
