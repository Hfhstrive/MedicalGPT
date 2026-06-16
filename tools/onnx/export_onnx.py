# -*- coding: utf-8 -*-
import os
import multiprocessing

def merge_and_save_pytorch_model(base_model_path, lora_output_path, temp_merged_dir):
    # 将 PyTorch 相关的依赖只在子进程内部导入，以彻底杜绝主进程对 CUDA/GPU 的显存占用与干扰
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("子进程启动：正在加载 Tokenizer 和基底模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # 1. 加载基底模型（float16 精度以节省显存）
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("子进程：正在加载 LoRA 权重并进行合并 (Merge)...")
    # 2. 将 LoRA 权重加载到基底模型上并合并
    model = PeftModel.from_pretrained(base_model, lora_output_path)
    merged_model = model.merge_and_unload()
    merged_model.eval()

    # 3. 保存合并后的完整 PyTorch 模型（供后续 optimum-cli 导出使用）
    print(f"子进程：正在将合并后的完整模型临时保存至: {temp_merged_dir} ...")
    merged_model.save_pretrained(temp_merged_dir)
    tokenizer.save_pretrained(temp_merged_dir)
    print("子进程：临时合并模型保存完成。正在退出子进程并完全释放 GPU 显存...")


def merge_and_export_onnx(base_model_path, lora_output_path, onnx_export_path, device="cuda", dtype="fp16"):
    # 设置环境变量以优化显存分配，防止 ONNX Runtime 的 BFCArena 申请大块连续显存失败
    os.environ["ORT_CUDA_PROVIDER_OPTIONS"] = "arena_extend_strategy=kSameAsRequested"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

    temp_merged_dir = "./temp_qwen3_merged"

    # 1. 采用 multiprocessing 的 spawn 模式拉起子进程进行模型合并与保存
    # 这样在子进程完成退出后，PyTorch 占用的所有显存会被操作系统强制、100% 回收
    print("正在创建子进程来执行 PyTorch 合并操作...")
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(
        target=merge_and_save_pytorch_model,
        args=(base_model_path, lora_output_path, temp_merged_dir)
    )
    p.start()
    p.join()  # 等待合并子进程完全退出

    if p.exitcode != 0:
        print(f"❌ 错误：合并子进程异常退出（退出码：{p.exitcode}），已中断 ONNX 导出流程。")
        return

    print("🎉 合并子进程已成功安全退出！当前 GPU 显存完全空闲。")
    print("开始调用 Optimum 将合并后的完整模型转换为 ONNX 格式...")

    os.makedirs(onnx_export_path, exist_ok=True)

    # 2. 构建 Optimum 导出命令
    # 使用与当前运行 Python 解释器相同的 bin 目录下的 optimum-cli 绝对路径，确保在非交互式 shell 中命令可用
    import sys
    python_dir = os.path.dirname(sys.executable)
    optimum_cli_path = os.path.join(python_dir, "optimum-cli")

    export_cmd = (
        f"{optimum_cli_path} export onnx "
        f"--model {temp_merged_dir} "
        f"--task causal-lm "
        f"--device {device} "
        f"--dtype {dtype} "
        f"--no-post-process "
        f"--trust-remote-code "
        f"{onnx_export_path}"
    )

    exit_code = os.system(export_cmd)

    if exit_code == 0:
        print(f"🎉 成功！ONNX 模型已成功导出至目录: {onnx_export_path}")
    else:
        print("❌ 错误：ONNX 导出过程中出现问题，请检查上方日志。")

    # 3. 清理临时文件夹
    import shutil
    if os.path.exists(temp_merged_dir):
        shutil.rmtree(temp_merged_dir)
        print("临时合并文件夹已清理。")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LoRA 合并并导出为 ONNX 格式")
    parser.add_argument('--base_model_path', default="/home/inno/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554/", type=str,
                        help="你的基底模型路径")
    parser.add_argument('--lora_output_path', default="/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v5-rank16", type=str,
                        help="你的 LoRA 微调输出路径")
    parser.add_argument('--onnx_export_path', default="/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v5-rank16/onnx/", type=str,
                        help="期望的 ONNX 输出目录")
    parser.add_argument('--device', default="cuda", type=str, choices=["cuda", "cpu"],
                        help="Optimum 导出运行设备")
    parser.add_argument('--dtype', default="fp16", type=str, choices=["fp32", "fp16", "bf16"],
                        help="ONNX 导出权重数据精度类型")
    args = parser.parse_args()
    print("解析后的参数：", args)

    merge_and_export_onnx(
        base_model_path=args.base_model_path,
        lora_output_path=args.lora_output_path,
        onnx_export_path=args.onnx_export_path,
        device=args.device,
        dtype=args.dtype
    )


if __name__ == "__main__":
    main()