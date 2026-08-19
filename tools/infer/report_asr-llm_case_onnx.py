# -*- coding: utf-8 -*-
import argparse
import copy
import glob
import json
import os
import sys
import re
import ast
import time
import psutil
import gc
import ctypes
import numpy as np
import torch
import torchaudio
import onnxruntime as ort
from transformers import AutoTokenizer
from funasr.frontends.wav_frontend import WavFrontend
from optimum.onnxruntime import ORTModelForCausalLM

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


def print_system_status(stage_name: str):
    """打印系统当前的内存和显存状态"""
    mem = psutil.virtual_memory()
    total_mem = mem.total / (1024 ** 3)
    used_mem = mem.used / (1024 ** 3)
    process_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    
    gpu_info_str = "N/A"
    try:
        import subprocess
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        total_gpu, used_gpu = map(float, res.strip().split(","))
        gpu_percent = (used_gpu / total_gpu) * 100
        
        process_gpu_mem = 0.0
        try:
            apps_res = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            current_pid = os.getpid()
            for line in apps_res.strip().split("\n"):
                if line.strip():
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        pid = int(parts[0].strip())
                        used_mem_val = float(parts[1].strip())
                        if pid == current_pid:
                            process_gpu_mem = used_mem_val
                            break
        except Exception:
            pass
        gpu_info_str = f"显存占比: {gpu_percent:.1f}% ({used_gpu:.0f}/{total_gpu:.0f} MB), 本进程显存: {process_gpu_mem:.0f} MB"
    except Exception:
        pass
        
    print(f"[{stage_name}] 内存占比: {mem.percent:.1f}% ({used_mem:.2f}/{total_mem:.2f} GB), 本进程内存: {process_mem:.1f} MB | {gpu_info_str}")


def trim_memory():
    """强制回收内存和显存碎片，修剪 C 堆内存分配"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


class ASROnnxRecognizer:
    """ASR ONNX 推理器 (声学模型运行在 CPU，自回归 LLM 运行在 GPU 并使用 I/O Binding)"""
    def __init__(self, model_dir: str, tokenizer_dir: str, hotwords_path: str = None, llm_dtype: str = "FP16"):
        self.model_dir = model_dir
        self.tokenizer_dir = tokenizer_dir
        self.hotwords_path = hotwords_path
        self.llm_dtype = llm_dtype.upper()
        
        # 1. 载入 Tokenizer
        print(f"载入 ASR Tokenizer (从 {self.tokenizer_dir}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_dir)
        
        # 2. 载入词嵌入权重
        embed_path = os.path.join(self.model_dir, "embed_tokens.npy")
        print(f"载入 ASR LLM 词嵌入权重: {embed_path} ...")
        self.embed_tokens = np.load(embed_path)
        if str(self.embed_tokens.dtype).startswith('void') or self.embed_tokens.dtype.kind == 'V':
            import ml_dtypes
            self.embed_tokens = self.embed_tokens.view('bfloat16')
            
        if self.llm_dtype == "BF16":
            import ml_dtypes
            self.infer_dtype = 'bfloat16'
            if self.embed_tokens.dtype != ml_dtypes.bfloat16:
                self.embed_tokens = self.embed_tokens.astype('bfloat16')
            self.io_dtype = np.float32
        elif self.llm_dtype == "FP16":
            self.infer_dtype = np.float16
            self.embed_tokens = self.embed_tokens.astype(np.float16)
            self.io_dtype = np.float16
        else:
            self.infer_dtype = np.float32
            self.embed_tokens = self.embed_tokens.astype(np.float32)
            self.io_dtype = np.float32
            
        # 3. 载入热词列表
        self.hotwords = []
        if self.hotwords_path and os.path.exists(self.hotwords_path):
            try:
                with open(self.hotwords_path, "r", encoding="utf-8") as f:
                    self.hotwords = [line.strip() for line in f if line.strip()]
                print(f"载入热词成功，共 {len(self.hotwords)} 个热词。")
            except Exception as e:
                print(f"警告：读取热词文件失败: {e}")
                
        # 4. 初始化 WavFrontend
        config_path = os.path.join(self.model_dir, "config.yaml")
        print(f"载入 WavFrontend 配置: {config_path} ...")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml_safe_load(f.read())
        frontend_conf = config["frontend_conf"]
        self.frontend = WavFrontend(**frontend_conf)
        
        # 5. 初始化 ONNX 推理会话
        # 声学端强制在 CPU 上运行
        speech_onnx_path = os.path.join(self.model_dir, "model_speech_int8.onnx")
        if not os.path.exists(speech_onnx_path):
            speech_onnx_path = os.path.join(self.model_dir, "model_speech.onnx")
        print(f"初始化声学端 ONNX 会话 (CPU): {speech_onnx_path} ...")
        self.speech_session = ort.InferenceSession(speech_onnx_path, providers=["CPUExecutionProvider"])
        
        # 大模型在 GPU 上运行，配置 I/O Binding 选项
        llm_onnx_path = os.path.join(self.model_dir, "model_llm.onnx")
        print(f"初始化 ASR 大模型 ONNX 会话 (GPU): {llm_onnx_path} ...")
        cuda_provider_opts = {
            "device_id": "0",
            "arena_extend_strategy": "kSameAsRequested",
            "do_copy_in_default_stream": "True"
        }
        providers = [
            ("CUDAExecutionProvider", cuda_provider_opts),
            "CPUExecutionProvider"
        ]
        llm_opts = ort.SessionOptions()
        llm_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
        if self.llm_dtype == "BF16":
            llm_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            
        self.llm_session = ort.InferenceSession(llm_onnx_path, sess_options=llm_opts, providers=providers)
        trim_memory()

    def transcribe(self, wav_path: str) -> str:
        """从音频提取特征并运行 ONNX 自回归解码推理"""
        # 1. 载入音频并进行 16000Hz 单声道重采样
        waveform, sample_rate = torchaudio.load(wav_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        speech, speech_lengths = self.frontend(waveform, [waveform.shape[1]])
        speech = speech.numpy()
        speech_lengths = np.array(speech_lengths, dtype=np.int32)
        
        # 2. 运行声学端推理，获取语音表征
        encoder_out, encoder_out_lens = self.speech_session.run(
            ["encoder_out", "encoder_out_lens"], 
            {"speech": speech, "speech_lengths": speech_lengths}
        )
        
        # 3. 拼接 Prompt
        hotwords_str = ", ".join(self.hotwords)
        prompt = f"请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n热词列表：[{hotwords_str}]\n语音转写："
        
        prefix = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}"
        suffix = f"<|im_end|>\n<|im_start|>assistant\n"
        
        prefix_ids = self.tokenizer.encode(prefix)
        suffix_ids = self.tokenizer.encode(suffix)
        
        prefix_embeds = self.embed_tokens[prefix_ids]
        suffix_embeds = self.embed_tokens[suffix_ids]
        
        speech_len = int(encoder_out_lens[0])
        speech_token = np.array(encoder_out[0, :speech_len, :], dtype=self.infer_dtype)
        
        inputs_embeds = np.concatenate([prefix_embeds, speech_token, suffix_embeds], axis=0)
        inputs_embeds = np.expand_dims(inputs_embeds, axis=0).astype(self.io_dtype)
        
        attention_mask = np.ones((1, inputs_embeds.shape[1]), dtype=np.int32)
        
        # 4. 自回归解码循环 (ONNX I/O Binding)
        num_layers = 28
        output_names = ["logits"]
        for i in range(num_layers):
            output_names.append(f"present_key_{i}")
            output_names.append(f"present_value_{i}")
            
        io_binding = self.llm_session.io_binding()
        
        inputs_embeds_val = ort.OrtValue.ortvalue_from_numpy(inputs_embeds, 'cuda', 0)
        attention_mask_val = ort.OrtValue.ortvalue_from_numpy(attention_mask, 'cuda', 0)
        position_ids = np.arange(inputs_embeds.shape[1], dtype=np.int64).reshape(1, -1)
        position_ids_val = ort.OrtValue.ortvalue_from_numpy(position_ids, 'cuda', 0)
        
        io_binding.bind_ortvalue_input("inputs_embeds", inputs_embeds_val)
        io_binding.bind_ortvalue_input("attention_mask", attention_mask_val)
        io_binding.bind_ortvalue_input("position_ids", position_ids_val)
        
        for i in range(num_layers):
            empty_kv = np.zeros((1, 8, 0, 128), dtype=self.io_dtype)
            k_val = ort.OrtValue.ortvalue_from_numpy(empty_kv, 'cuda', 0)
            v_val = ort.OrtValue.ortvalue_from_numpy(empty_kv, 'cuda', 0)
            
            io_binding.bind_ortvalue_input(f"past_key_{i}", k_val)
            io_binding.bind_ortvalue_input(f"past_value_{i}", v_val)
            
        for name in output_names:
            io_binding.bind_output(name, device_type='cuda', device_id=0)
            
        self.llm_session.run_with_iobinding(io_binding)
        outputs_val = io_binding.get_outputs()
        
        logits_numpy = outputs_val[0].numpy()
        next_token_id = int(np.argmax(logits_numpy[0, -1, :]))
        
        total_seq_len = inputs_embeds.shape[1] + 1
        attention_mask = np.ones((1, total_seq_len), dtype=np.int32)
        
        generated_ids = []
        max_new_tokens = 512
        
        if next_token_id == 151645:
            pass
        else:
            generated_ids.append(next_token_id)
            
            for step in range(1, max_new_tokens):
                next_embed = np.expand_dims(self.embed_tokens[next_token_id].astype(self.io_dtype), axis=(0, 1))
                next_embed_val = ort.OrtValue.ortvalue_from_numpy(next_embed, 'cuda', 0)
                
                step_position_ids = np.array([[total_seq_len - 1]], dtype=np.int64)
                step_position_ids_val = ort.OrtValue.ortvalue_from_numpy(step_position_ids, 'cuda', 0)
                attention_mask_val = ort.OrtValue.ortvalue_from_numpy(attention_mask, 'cuda', 0)
                
                step_io_binding = self.llm_session.io_binding()
                step_io_binding.bind_ortvalue_input("inputs_embeds", next_embed_val)
                step_io_binding.bind_ortvalue_input("attention_mask", attention_mask_val)
                step_io_binding.bind_ortvalue_input("position_ids", step_position_ids_val)
                
                for i in range(num_layers):
                    step_io_binding.bind_ortvalue_input(f"past_key_{i}", outputs_val[1 + 2 * i])
                    step_io_binding.bind_ortvalue_input(f"past_value_{i}", outputs_val[1 + 2 * i + 1])
                    
                for name in output_names:
                    step_io_binding.bind_output(name, device_type='cuda', device_id=0)
                    
                self.llm_session.run_with_iobinding(step_io_binding)
                outputs_val = step_io_binding.get_outputs()
                
                logits_numpy = outputs_val[0].numpy()
                next_token_id = int(np.argmax(logits_numpy[0, -1, :]))
                
                if next_token_id == 151645:
                    break
                    
                generated_ids.append(next_token_id)
                total_seq_len += 1
                attention_mask = np.ones((1, total_seq_len), dtype=np.int32)
                
        decoded_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        trim_memory()
        return decoded_text.strip()


def yaml_safe_load(content: str):
    """一个极其简易的 YAML 解析，规避 PyYAML 依赖问题 (或者直接用 pyyaml)"""
    import yaml
    return yaml.safe_load(content)


class LLMOnnxEngine:
    """LLM ONNX 推理器 (基于 Optimum.onnxruntime 和 CUDA 显存优化参数)"""
    def __init__(self, onnx_model_dir: str, device: str = "cuda", gpu_mem_limit: float = 0.0):
        self.onnx_model_dir = onnx_model_dir
        self.device = device
        self.gpu_mem_limit = gpu_mem_limit
        
        # 1. 载入 Tokenizer
        print(f"载入 LLM Tokenizer (从 {self.onnx_model_dir}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.onnx_model_dir, trust_remote_code=True, padding_side='left')
        
        # 2. 载入 Optimum ONNX 模型
        provider = "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"
        print(f"载入 LLM ONNX 模型，推理设备: {self.device}，执行器: {provider} ...")
        
        session_options = ort.SessionOptions()
        # 强制 Initializers (权重) 直接使用 GPU 设备分配器，消除重复拷贝与 VRAM 碎片
        session_options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        provider_options = {}
        if self.device == "cuda":
            os.environ["ORT_CUDA_PROVIDER_OPTIONS"] = "arena_extend_strategy=kSameAsRequested"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
            
            provider_options["arena_extend_strategy"] = "kSameAsRequested"
            provider_options["do_copy_in_default_stream"] = "True"
            if self.gpu_mem_limit > 0.0:
                provider_options["gpu_mem_limit"] = str(int(self.gpu_mem_limit * 1024 * 1024 * 1024))
                
        self.model = ORTModelForCausalLM.from_pretrained(
            self.onnx_model_dir,
            file_name="model.onnx",
            provider=provider,
            use_cache=True,
            use_io_binding=True,
            provider_options=provider_options if self.device == "cuda" else None,
            session_options=session_options
        )
        self.run_device = self.model.device if hasattr(self.model, "device") else torch.device(self.device)
        trim_memory()

    def process(self, info: str) -> list:
        """执行多轮对话推理"""
        questions = [
            f"提取有效信息，规范表达以下语句：{info}",
            "提取语句中所涉及的结论性疾病？",
            "根据[部位, 特征描述, 病变名称]格式拆分语句，并专业化特征描述：",
        ]
        
        messages = []
        responses = []
        
        for i, q in enumerate(questions):
            messages.append({"role": "user", "content": q})
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            inputs = self.tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.run_device)
            attention_mask = inputs["attention_mask"].to(self.run_device)
            
            generation_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=False,  # Greedy search
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
            )
            
            generated_ids = self.model.generate(**generation_kwargs)
            output_ids = generated_ids[0][len(input_ids[0]):]
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            messages.append({"role": "assistant", "content": response})
            responses.append(response)
            
        trim_memory()
        return responses

    def summarize(self, texts: list) -> str:
        """一句话概括并简化多条文本内容"""
        if not texts:
            return ""
            
        joined = "，".join([t for t in texts if t])
        prompt = f"一句话概括以下内容，使其更连贯：\n{joined}"
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.run_device)
        attention_mask = inputs["attention_mask"].to(self.run_device)
        
        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=True,
        )
        
        generated_ids = self.model.generate(**generation_kwargs)
        output_ids = generated_ids[0][len(input_ids[0]):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        trim_memory()
        return response

def find_audio_files(wav_dir: str) -> list:
    patterns = ["*.wav", "*.mp3"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(wav_dir, p)))
    return sorted(files)


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


def deduplicate_concl(conclusion: list) -> tuple:
    """去重结论性疾病 (沿用 report_asr-llm_case.py 的去重逻辑)"""
    conclusion = [normalize_disease_name(item) for item in conclusion]
    word_to_tier = {}
    for group_idx, group in enumerate(dup_list):
        for tier_idx, tier in enumerate(group):
            for item in tier:
                word_to_tier[item] = (group_idx, tier_idx)
                
    group_highest_triggered = {}
    for item in conclusion:
        if item in word_to_tier:
            group_idx, tier_idx = word_to_tier[item]
            if group_idx not in group_highest_triggered:
                group_highest_triggered[group_idx] = tier_idx
            else:
                group_highest_triggered[group_idx] = min(group_highest_triggered[group_idx], tier_idx)
                
    result = []
    for item in conclusion:
        if item in word_to_tier:
            group_idx, tier_idx = word_to_tier[item]
            if tier_idx == group_highest_triggered[group_idx]:
                result.append(item)
        else:
            result.append(item)
            
    return list(dict.fromkeys(result)), group_highest_triggered


def generate_report(llm_infos: list, conclusion: list, llm_engine: LLMOnnxEngine) -> dict:
    """根据提取的信息生成最终的胃镜诊断报告"""
    report_concl = ["慢性浅表性胃炎"]
    report_concl.extend(conclusion)
    dedup_report_concl, group_highest_triggered = deduplicate_concl(report_concl)
    
    # 只输出包含特定字眼的诊断结论
    allowed_keywords = ['食管', '胃', '静脉', '癌', '隆起', '息肉', 'SMT', '瘤', '黏膜', '病变', '溃疡', '肠', '萎缩', '感染']
    dedup_report_concl = [item for item in dedup_report_concl if any(kw in item for kw in allowed_keywords)]
    
    # 动态获取基础胃炎类型 (慢性浅表性胃炎, 萎缩性胃炎等)
    base_gastritis = dup_list[0][group_highest_triggered[0]][0]
    # 如果基础类型不存在于 template_process 中，则默认使用 慢性浅表性胃炎
    if base_gastritis not in template_process:
        base_gastritis = '慢性浅表性胃炎'
        
    base_report = {
        "检查过程": copy.deepcopy(template_process[base_gastritis]),
        "检查结果": '；'.join(dedup_report_concl),
    }
    
    for loc, norm_text, lesion in llm_infos:
        if loc in ['胃窦', '胃角', '胃体', '贲门']:
            has_atrophy = any('胃炎' in item for item in lesion)
            if not has_atrophy:
                # 若无胃炎，用 LLM 概括默认段落和当前特征描述
                default_text = base_report["检查过程"].get(loc, "")
                norm_text = llm_engine.summarize([default_text, norm_text])
        if loc in base_report["检查过程"]:
            base_report["检查过程"][loc] = norm_text.replace(f'{loc}', '')
            
    return base_report


def analysis_result(responses: list, llm_engine: LLMOnnxEngine, out_path: str) -> tuple:
    """解析三轮对话结果，并生成报告写入文件"""
    standard = responses[0]
    
    # 对 LLM 回答的列表格式字符串进行解析
    try:
        conclusion = ast.literal_eval(responses[1])
        if not isinstance(conclusion, list):
            conclusion = [responses[1]]
    except Exception:
        conclusion = [responses[1]]
        
    try:
        split_sentences = ast.literal_eval(responses[2])
    except Exception:
        split_sentences = []
        
    loc_agg = {}
    for split_sentence in split_sentences:
        if len(split_sentence) >= 3:
            loc, describe, disease = split_sentence[:3]
        elif len(split_sentence) == 2:
            loc, describe = split_sentence
            disease = ""
        else:
            continue
            
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
                
    llm_infos = []
    for loc, v in loc_agg.items():
        if v.get("num", 0) > 1:
            merged_text = llm_engine.summarize(v["texts"])
        else:
            merged_text = "；".join(v["texts"]) if v["texts"] else ""
        merged_lesions = list(v["lesions"])
        conclusion.extend(merged_lesions)
        llm_infos.append((loc, merged_text, merged_lesions))
        
    report = generate_report(llm_infos, conclusion, llm_engine)
    
    # 创建父目录
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已成功保存到 {out_path}")
    
    return llm_infos, conclusion


def main():
    parser = argparse.ArgumentParser(description="ASR + LLM 双模型 ONNX 推理融合脚本")
    # todo: 音频文件最好是.wav格式，否则在ASR的LLM部分，其onnx模型虽然默认do_sample=False，但还是会多次结果不一致。采用.mp3格式改情况更明显。
    parser.add_argument("--wav_dir", type=str, default="/media/inno/LLM/胃镜/report/V5/test/audio/", help="音频文件夹路径")
    parser.add_argument("--asr_model_dir", type=str, default="/media/inno/work_dirs/ASR/FunASR/outputs/fun_asr_nano_2512_gi_v3/", help="ASR ONNX模型目录")
    parser.add_argument("--asr_tokenizer", type=str, default="/home/inno/code_inno/innoreport/ASR/Qwen3-0.6B", help="ASR Tokenizer目录")
    parser.add_argument("--hotwords", type=str, default="/media/inno/ASR/胃镜/gi_hotwords_v3.txt", help="ASR热词文件，可为空")
    parser.add_argument("--asr_llm_dtype", type=str, default="FP16", choices=["FP16", "BF16", "FP32"], help="ASR内部大模型的推理精度")

    parser.add_argument("--llm_model_dir", type=str, default="/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v5-rank16/onnx/", help="LLM ONNX模型目录")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="LLM推理设备")
    parser.add_argument("--gpu_mem_limit", type=float, default=0.0, help="LLM运行时的最大GPU显存限制(GB)")
    parser.add_argument("--out", type=str, default="/media/inno/output/report/胃镜/test_onnx/", help="输出文件夹路径")
    args = parser.parse_args()
    
    print("解析输入参数：", args)
    
    # 记录资源及总耗时
    t_start = time.time()
    print_system_status("初始化前")
    
    # 1. 载入 ASR ONNX 模型
    t_asr_load_start = time.time()
    asr_recognizer = ASROnnxRecognizer(
        model_dir=args.asr_model_dir,
        tokenizer_dir=args.asr_tokenizer,
        hotwords_path=args.hotwords,
        llm_dtype=args.asr_llm_dtype
    )
    t_asr_load_end = time.time()
    print(f"ASR ONNX 模型加载完成，耗时: {t_asr_load_end - t_asr_load_start:.3f} 秒")
    print_system_status("加载 ASR 后")
    
    # 2. 载入 LLM ONNX 模型
    t_llm_load_start = time.time()
    llm_engine = LLMOnnxEngine(
        onnx_model_dir=args.llm_model_dir,
        device=args.device,
        gpu_mem_limit=args.gpu_mem_limit
    )
    t_llm_load_end = time.time()
    print(f"LLM ONNX 模型加载完成，耗时: {t_llm_load_end - t_llm_load_start:.3f} 秒")
    print_system_status("加载 LLM 后")
    
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
                # 3. 运行 ASR 推理
                t_asr_infer_start = time.time()
                asr_text = asr_recognizer.transcribe(wav_path)
                t_asr_infer_end = time.time()
                print("=" * 60)
                print(f"【ASR 识别结果】:\n{asr_text}")
                print("=" * 60)
                print(f"ASR 推理耗时: {t_asr_infer_end - t_asr_infer_start:.3f} 秒")
                print_system_status("ASR 推理后")
                
                if not asr_text:
                    print("警告：ASR 转写文本为空，跳过此文件流程！")
                    continue
                    
                # 4. 运行 LLM 问答抽取
                t_llm_infer_start = time.time()
                llm_responses = llm_engine.process(asr_text)
                t_llm_infer_end = time.time()
                print("=" * 60)
                print("【LLM 提取结果】:")
                for idx, resp in enumerate(llm_responses):
                    print(f"  第 {idx+1} 轮:\n{resp}")
                print("=" * 60)
                print(f"LLM 抽取耗时: {t_llm_infer_end - t_llm_infer_start:.3f} 秒")
                print_system_status("LLM 推理后")
                
                # 5. 后处理与报告生成
                t_post_start = time.time()
                
                base_name = os.path.splitext(os.path.basename(wav_path))[0]
                out_path = os.path.join(args.out, f"{base_name}.json")
                
                # 保存 ASR 文本和 LLM 原始响应
                asr_llm_path = os.path.join(args.out, f"{base_name}_asr_llm.json")
                with open(asr_llm_path, "w", encoding="utf-8") as f:
                    json.dump({"asr_text": asr_text, "responses": llm_responses}, f, ensure_ascii=False, indent=2)
                print(f"ASR和LLM结果已保存到 {asr_llm_path}")
                
                analysis_result(llm_responses, llm_engine, out_path)
                t_post_end = time.time()
                
                print(f"处理完成，耗时: ASR推理={t_asr_infer_end - t_asr_infer_start:.2f}s, LLM推理={t_llm_infer_end - t_llm_infer_start:.2f}s, 后处理={t_post_end - t_post_start:.2f}s")
            except Exception as e:
                print(f"处理文件失败 {wav_path}: {e}")
                import traceback
                traceback.print_exc()
                
    t_end = time.time()
    
    print(f"\nASR 加载耗时: {t_asr_load_end - t_asr_load_start:.2f}s")
    print(f"LLM 加载耗时: {t_llm_load_end - t_llm_load_start:.2f}s")
    print(f"总处理耗时: {t_end - t_start:.2f}s")
    print_system_status("程序结束时")


if __name__ == "__main__":
    main()
