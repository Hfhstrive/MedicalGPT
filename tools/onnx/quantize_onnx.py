# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Lightweight ONNX Model Quantization script to prevent OOM
"""
import argparse
import os
import gc

def quantize_int8(input_model, output_model):
    import onnxruntime.quantization as ort_quant
    print("⚠️ 警告: 动态 INT8 量化要求输入的 ONNX 模型为 FP32 精度。")
    print("⚠️ 如果输入的 ONNX 是以 FP16 精度导出的，此操作会导致 DequantizeLinear 算子的 scale 生成为 float16 类型，")
    print("⚠️ 进而在 ONNX Runtime 加载时触发 InvalidGraph (DequantizeLinear scale type float16 is invalid) 错误。")
    print(f"正在尝试将 {input_model} 动态量化为 INT8: {output_model} ...")
    ort_quant.quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=ort_quant.QuantType.QInt8,
        use_external_data_format=True
    )

def quantize_int4(input_model, output_model):
    from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
    print(f"正在将 {input_model} 仅权重块量化为 INT4: {output_model} ...")
    quantizer = MatMul4BitsQuantizer(
        model=input_model,
        block_size=128,
        is_symmetric=True,
        accuracy_level=1
    )
    quantizer.process()
    quantizer.model.save_model_to_file(output_model, True)

def main():
    parser = argparse.ArgumentParser(description="轻量级 ONNX 模型量化工具")
    parser.add_argument('--input_model', type=str, default="/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v5-rank16/onnx/model.onnx", help="输入的待量化 ONNX 模型文件路径")
    parser.add_argument('--output_model', type=str, default="/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v5-rank16/onnx/model_int4.onnx", help="量化后的输出 ONNX 模型文件路径")
    parser.add_argument('--type', type=str, default="int4", choices=["int8", "int4"], help="量化类型：int8 或 int4")
    args = parser.parse_args()

    if not os.path.exists(args.input_model):
        raise FileNotFoundError(f"未找到输入模型：{args.input_model}")

    if args.type == "int8":
        quantize_int8(args.input_model, args.output_model)
    elif args.type == "int4":
        quantize_int4(args.input_model, args.output_model)
    
    gc.collect()
    print("🎉 量化完成！")

if __name__ == "__main__":
    main()
