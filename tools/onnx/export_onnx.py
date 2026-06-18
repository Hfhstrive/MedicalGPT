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


def merge_and_export_onnx(base_model_path, lora_output_path, onnx_export_path, device="cuda", dtype="fp16", no_post_process=False):
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

    if os.path.exists(onnx_export_path):
        print(f"Cleaning existing export directory: {onnx_export_path} ...")
        import shutil
        try:
            # 如果是目录，用 rmtree 删除整个目录
            if os.path.isdir(onnx_export_path):
                shutil.rmtree(onnx_export_path)
            else:
                os.remove(onnx_export_path)
        except Exception as e:
            print(f"⚠️ Warning: Failed to clean directory {onnx_export_path}: {e}")
    os.makedirs(onnx_export_path, exist_ok=True)

    # 2. 构建 Optimum 导出命令
    # 使用与当前运行 Python 解释器相同的 bin 目录下的 optimum-cli 绝对路径，确保在非交互式 shell 中命令可用
    import sys
    python_dir = os.path.dirname(sys.executable)
    optimum_cli_path = os.path.join(python_dir, "optimum-cli")

    print("Monkey patching onnxruntime.InferenceSession to force CPU execution...")
    import onnxruntime
    original_init = onnxruntime.InferenceSession.__init__

    def patched_init(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
        print(f"[MonkeyPatch] Forcing CPUExecutionProvider for InferenceSession of {path_or_bytes}")
        return original_init(self, path_or_bytes, sess_options=sess_options, providers=["CPUExecutionProvider"], provider_options=None, **kwargs)

    onnxruntime.InferenceSession.__init__ = patched_init

    from optimum.exporters.onnx import main_export
    try:
        main_export(
            model_name_or_path=temp_merged_dir,
            output=onnx_export_path,
            task="causal-lm-with-past",
            device=device,
            dtype=dtype,
            no_post_process=no_post_process,
            trust_remote_code=True,
            do_validation=False
        )
        exit_code = 0
    except Exception as e:
        print(f"❌ 导出过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    if exit_code == 0:
        print(f"🎉 成功！ONNX 模型已成功导出至目录: {onnx_export_path}")
        # 自动进行 Tied Weights 去重合并
        model_onnx_path = os.path.join(onnx_export_path, "model.onnx")
        if os.path.exists(model_onnx_path):
            print("\n================ [Tied Weights 自动去重物理合并] ================")
            try:
                merge_and_shrink_tied_weights(model_onnx_path, model_onnx_path)
            except Exception as e:
                print(f"❌ Tied Weights 合并优化失败: {e}")
                import traceback
                traceback.print_exc()
            print("=================================================================\n")
    else:
        print("❌ 错误：ONNX 导出过程中出现问题，请检查上方日志。")

    # 3. 清理临时文件夹
    import shutil
    if os.path.exists(temp_merged_dir):
        shutil.rmtree(temp_merged_dir)
        print("临时合并文件夹已清理。")


def find_all_tensors(graph):
    import onnx
    tensors = []
    # 1. 收集全局 Initializer
    for init in graph.initializer:
        tensors.append(init)
        
    # 2. 遍历所有节点属性中的 Tensor (例如 Constant 节点的 value)
    for node in graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                if attr.t:
                    tensors.append(attr.t)
            elif attr.type == onnx.AttributeProto.TENSORS:
                for t in attr.tensors:
                    tensors.append(t)
            elif attr.type == onnx.AttributeProto.GRAPH:
                if attr.g:
                    tensors.extend(find_all_tensors(attr.g))
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for g in attr.graphs:
                    tensors.extend(find_all_tensors(g))
    return tensors

def merge_and_shrink_tied_weights(model_path, output_path):
    import onnx
    print(f"Loading ONNX model structure from {model_path}...")
    model = onnx.load(model_path, load_external_data=False)
    
    # 1. 查找 Embedding 和 LM Head 冗余权重并重定向图节点
    embedding_name = "model.embed_tokens.weight"
    embedding_init = None
    for init in model.graph.initializer:
        if init.name == embedding_name:
            embedding_init = init
            break
            
    if embedding_init is None:
        print(f"❌ Error: Embedding initializer '{embedding_name}' not found!")
        return
        
    embedding_dims = list(embedding_init.dims)
    print(f"Found embedding initializer: {embedding_name}, Shape: {embedding_dims}")
    
    redundant_weight_name = None
    need_transpose = False
    for init in model.graph.initializer:
        if init.name != embedding_name:
            dims = list(init.dims)
            if len(dims) == 2:
                if dims[0] == embedding_dims[0] and dims[1] == embedding_dims[1]:
                    redundant_weight_name = init.name
                    redundant_dims = dims
                    need_transpose = False
                    break
                elif dims[0] == embedding_dims[1] and dims[1] == embedding_dims[0]:
                    redundant_weight_name = init.name
                    redundant_dims = dims
                    need_transpose = True
                    break
                    
    if redundant_weight_name is None:
        print("ℹ️ No redundant weight found. Already merged?")
        return
        
    print(f"Found redundant weight to merge: '{redundant_weight_name}' ({redundant_dims})")
    
    target_weight_name = embedding_name
    if need_transpose:
        transposed_name = embedding_name + "_transposed"
        print(f"Inserting Transpose node: '{embedding_name}' -> '{transposed_name}' (perm=[1, 0])")
        transpose_node = onnx.helper.make_node(
            'Transpose',
            inputs=[embedding_name],
            outputs=[transposed_name],
            name="model.embed_tokens.weight_transpose"
        )
        model.graph.node.insert(0, transpose_node)
        target_weight_name = transposed_name
        
    # 重定向节点输入
    node_count = 0
    for node in model.graph.node:
        for idx, input_name in enumerate(node.input):
            if input_name == redundant_weight_name:
                node.input[idx] = target_weight_name
                node_count += 1
                print(f"Redirected input {idx} of node '{node.name}' ({node.op_type}) to '{target_weight_name}'")
                
    # 从 initializer 中删除冗余
    for init in list(model.graph.initializer):
        if init.name == redundant_weight_name:
            model.graph.initializer.remove(init)
            print(f"Removed initializer '{redundant_weight_name}' from graph definition.")
            
    # 获取图中所有外部 Tensor（包含全局 Initializers 和 Constant 等节点的内置外部数据属性）
    all_tensors = find_all_tensors(model.graph)
    print(f"Collected total of {len(all_tensors)} tensors from model graph and attributes.")

    # 2. 物理重组外部数据文件，消除冗余权重字节片断
    model_dir = os.path.dirname(model_path)
    old_data_filename = "model.onnx_data"
    old_data_path = os.path.join(model_dir, old_data_filename)
    new_data_filename = "model.onnx_data.tmp"
    new_data_path = os.path.join(model_dir, new_data_filename)
    
    if not os.path.exists(old_data_path):
        print(f"❌ Error: External data file {old_data_path} not found!")
        return
        
    print(f"Streaming and shrinking data from {old_data_path} to {new_data_path} ...")
    
    new_offset = 0
    with open(old_data_path, "rb") as f_in, open(new_data_path, "wb") as f_out:
        for init in all_tensors:
            if init.data_location == onnx.TensorProto.EXTERNAL:
                ext_info = {}
                for entry in init.external_data:
                    ext_info[entry.key] = entry.value
                    
                offset = int(ext_info.get("offset", 0))
                length = int(ext_info.get("length", 0))
                
                # 寻道并复制数据
                f_in.seek(offset)
                data = f_in.read(length)
                f_out.write(data)
                
                # 重新配置外部属性
                del init.external_data[:]
                e1 = init.external_data.add()
                e1.key = "location"
                e1.value = str(old_data_filename)
                e2 = init.external_data.add()
                e2.key = "offset"
                e2.value = str(new_offset)
                e3 = init.external_data.add()
                e3.key = "length"
                e3.value = str(length)
                new_offset += length
                    
    # 3. 覆盖写回与临时文件清理
    print(f"Saving modified graph to {output_path}...")
    onnx.save(model, output_path)
    
    target_data_path = os.path.join(os.path.dirname(output_path), old_data_filename)
    if os.path.exists(target_data_path) and os.path.abspath(target_data_path) != os.path.abspath(new_data_path):
        os.remove(target_data_path)
    os.rename(new_data_path, target_data_path)
    
    print("🎉 Successfully merged tied weights and shrunked model.onnx_data!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LoRA 合并并导出为 ONNX 格式")
    parser.add_argument('--base_model_path', default="/home/inno/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554/", type=str,
                        help="你的基底模型路径")
    parser.add_argument('--lora_output_path', default="/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v5-rank16", type=str,
                        help="你的 LoRA 微调输出路径")
    parser.add_argument('--onnx_export_path', default="/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v5-rank16/onnx/", type=str,
                        help="期望的 ONNX 输出目录")
    parser.add_argument('--device', default="cpu", type=str, choices=["cuda", "cpu"],
                        help="Optimum 导出运行设备")
    parser.add_argument('--dtype', default="fp16", type=str, choices=["fp32", "fp16", "bf16"],
                        help="ONNX 导出权重数据精度类型")
    parser.add_argument('--no_post_process', action='store_true',
                        help="是否禁用后处理图合并。默认不禁用（False），执行完整图融合")
    args = parser.parse_args()
    print("解析后的参数：", args)

    merge_and_export_onnx(
        base_model_path=args.base_model_path,
        lora_output_path=args.lora_output_path,
        onnx_export_path=args.onnx_export_path,
        device=args.device,
        dtype=args.dtype,
        no_post_process=args.no_post_process
    )


if __name__ == "__main__":
    # eg: python export_onnx.py --device cuda --dtype fp16 --no_post_process
    main()