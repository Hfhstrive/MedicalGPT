# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Lightweight ONNX Model Quantization script to prevent OOM
"""
import argparse
import os
import gc

def quantize_weight_only(input_model, output_model, bits=8):
    # int8量化需onnxruntime版本==1.23.2
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
    print(f"正在将 {input_model} 仅权重块量化为 INT{bits}: {output_model} ...")
    is_symmetric = True if bits == 4 else False
    algo_config = DefaultWeightOnlyQuantConfig(
        bits=bits,
        block_size=128,
        is_symmetric=is_symmetric
    )
    quantizer = MatMulNBitsQuantizer(
        model=input_model,
        algo_config=algo_config
    )
    quantizer.process()
    quantizer.model.save_model_to_file(output_model, use_external_data_format=True)

def main():
    parser = argparse.ArgumentParser(description="轻量级 ONNX 模型量化工具")
    parser.add_argument('--input_model', type=str, default="/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v5-rank16/onnx/model.onnx", help="输入的待量化 ONNX 模型文件路径")
    parser.add_argument('--output_model', type=str, default="/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v5-rank16/onnx/model_int4.onnx", help="量化后的输出 ONNX 模型文件路径")
    parser.add_argument('--type', type=str, default="int4", choices=["int8", "int4"], help="量化类型：int8 或 int4")
    args = parser.parse_args()

    try:
        if args.type == "int8":
            quantize_weight_only(args.input_model, args.output_model, bits=8)
        elif args.type == "int4":
            quantize_weight_only(args.input_model, args.output_model, bits=4)
        gc.collect()
        print("🎉 量化完成！", flush=True)
    except Exception as e:
        print(f"❌ 量化抛出异常: {e}", flush=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
