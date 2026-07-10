import os
import json
import re
import ast

# 1. 配置路径
js_path_1 = '/media/inno/output/LLM/colon/qwen3-4b-instruct-v1/val.json'
js_path_2 = '/media/inno/output/LLM/colon/qwen3.5-4b-v1/val.json'
gt_dir = '/media/inno/ASR/肠镜/base_data/standard_1/case/'

# 2. 字符级相似度计算 (F1 Score) 用于评测“镜检所见”
def compute_char_f1(pred, target):
    if not pred and not target:
        return 1.0
    if not pred or not target:
        return 0.0
    
    # 统计预测和目标的字符频率
    pred_chars = list(pred.replace("\n", "").replace(" ", ""))
    target_chars = list(target.replace("\n", "").replace(" ", ""))
    
    pred_dict = {}
    for c in pred_chars:
        pred_dict[c] = pred_dict.get(c, 0) + 1
        
    common_count = 0
    for c in target_chars:
        if c in pred_dict and pred_dict[c] > 0:
            common_count += 1
            pred_dict[c] -= 1
            
    precision = common_count / len(pred_chars) if pred_chars else 0.0
    recall = common_count / len(target_chars) if target_chars else 0.0
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# 3. 诊断集合相似度计算 (Precision, Recall, F1) 用于评测“诊断结论”
def compute_set_metrics(pred_list, target_list):
    if not pred_list and not target_list:
        return 1.0, 1.0, 1.0
    if not pred_list or not target_list:
        return 0.0, 0.0, 0.0
    
    # 归一化去重
    preds = set([p.strip().replace(" ", "") for p in pred_list if p.strip()])
    targets = set([t.strip().replace(" ", "") for t in target_list if t.strip()])
    
    intersection = preds.intersection(targets)
    precision = len(intersection) / len(preds) if preds else 0.0
    recall = len(intersection) / len(targets) if targets else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

# 4. 解析 GT 文本文件 (前三行为镜检所见，第四行为诊断结论)
def parse_gt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        # 过滤空白行，保证非空行解析
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    if len(lines) < 4:
        # 兜底：如果行数不足4行
        findings = "\n".join(lines[:-1]) if len(lines) > 1 else (lines[0] if lines else "")
        diagnosis_str = lines[-1] if lines else ""
    else:
        findings = "\n".join(lines[:3])
        diagnosis_str = lines[3]
        
    # 诊断结论按中文分号或英文分号切分
    diagnoses = re.split(r'[；;]', diagnosis_str)
    diagnoses = [d.strip() for d in diagnoses if d.strip()]
    return {
        '镜检所见': findings,
        '诊断结论': diagnoses
    }

# 5. 解析模型的输出字符串 (支持 ast.literal_eval、json.loads 及正则兜底)
def parse_model_output(output_str):
    if not output_str:
        return {'镜检所见': '', '诊断结论': []}
    
    output_str = output_str.strip()
    
    # 尝试使用 ast.literal_eval
    try:
        data = ast.literal_eval(output_str)
        if isinstance(data, dict):
            findings = data.get('镜检所见', '')
            diagnoses = data.get('诊断结论', [])
            if isinstance(diagnoses, str):
                diagnoses = re.split(r'[；;]', diagnoses)
            diagnoses = [d.strip() for d in diagnoses if d.strip()]
            return {'镜检所见': findings, '诊断结论': diagnoses}
    except Exception:
        pass
        
    # 尝试使用 json.loads
    try:
        data = json.loads(output_str)
        if isinstance(data, dict):
            findings = data.get('镜检所见', '')
            diagnoses = data.get('诊断结论', [])
            if isinstance(diagnoses, str):
                diagnoses = re.split(r'[；;]', diagnoses)
            diagnoses = [d.strip() for d in diagnoses if d.strip()]
            return {'镜检所见': findings, '诊断结论': diagnoses}
    except Exception:
        pass
        
    # 正则兜底解析
    findings = ""
    diagnoses = []
    
    findings_match = re.search(r"['\"]镜检所见['\"]\s*:\s*['\"](.*?)['\"]", output_str, re.S)
    if findings_match:
        findings = findings_match.group(1)
        
    diagnoses_match = re.search(r"['\"]诊断结论['\"]\s*:\s*\[(.*?)\]", output_str, re.S)
    if diagnoses_match:
        items = re.findall(r"['\"](.*?)['\"]", diagnoses_match.group(1))
        diagnoses = [item.strip() for item in items if item.strip()]
        
    return {'镜检所见': findings, '诊断结论': diagnoses}

def main():
    # 6. 递归读取 gt_dir 所有子目录中的 txt 文本，建立 {case_name: file_path} 映射
    print("正在扫描 GT 文件...")
    gt_map = {}
    for root, _, files in os.walk(gt_dir):
        for file in files:
            if file.endswith('.txt'):
                case_name = os.path.splitext(file)[0]
                gt_map[case_name] = os.path.join(root, file)
    print(f"扫描完毕，共发现 {len(gt_map)} 个 GT 报告。")

    # 7. 加载两个模型的推理结果 JSON
    if not os.path.exists(js_path_1):
        print(f"错误: 结果文件 1 不存在: {js_path_1}")
        return
    if not os.path.exists(js_path_2):
        print(f"错误: 结果文件 2 不存在: {js_path_2}")
        return
        
    with open(js_path_1, 'r', encoding='utf-8') as f:
        res_1 = json.load(f)
    with open(js_path_2, 'r', encoding='utf-8') as f:
        res_2 = json.load(f)
        
    # 8. 进行数据比对与评估
    print(f"\n进行数据比对，当前比较范围：")
    print(f"  模型结果 1: {js_path_1} (样本数: {len(res_1)})")
    print(f"  模型结果 2: {js_path_2} (样本数: {len(res_2)})")
    
    all_cases = set(res_1.keys()).union(set(res_2.keys()))
    
    comparisons = []
    m1_findings_f1_sum, m1_diag_f1_sum = 0.0, 0.0
    m2_findings_f1_sum, m2_diag_f1_sum = 0.0, 0.0
    valid_case_count = 0
    
    for case in sorted(all_cases):
        if case not in gt_map:
            # 过滤掉无法在 GT 目录中找到的 raw asr_texts 样本
            continue
            
        # 读取 GT 标注
        gt_data = parse_gt_file(gt_map[case])
        
        # 解析模型 1 和模型 2 的输出结果
        m1_raw = res_1.get(case, "")
        m2_raw = res_2.get(case, "")
        m1_data = parse_model_output(m1_raw)
        m2_data = parse_model_output(m2_raw)
        
        # 评测 模型 1
        m1_find_f1 = compute_char_f1(m1_data['镜检所见'], gt_data['镜检所见'])
        _, _, m1_diag_f1 = compute_set_metrics(m1_data['诊断结论'], gt_data['诊断结论'])
        
        # 评测 模型 2
        m2_find_f1 = compute_char_f1(m2_data['镜检所见'], gt_data['镜检所见'])
        _, _, m2_diag_f1 = compute_set_metrics(m2_data['诊断结论'], gt_data['诊断结论'])
        
        # 累计得分
        m1_findings_f1_sum += m1_find_f1
        m1_diag_f1_sum += m1_diag_f1
        m2_findings_f1_sum += m2_find_f1
        m2_diag_f1_sum += m2_diag_f1
        valid_case_count += 1
        
        comparisons.append({
            'case': case,
            'm1_find_f1': m1_find_f1,
            'm1_diag_f1': m1_diag_f1,
            'm2_find_f1': m2_find_f1,
            'm2_diag_f1': m2_diag_f1
        })
        
    if valid_case_count == 0:
        print("未找到任何同时存在于模型输出和 GT 目录中的测试用例。")
        return
        
    # 9. 打印评估对比汇总
    print("\n" + "="*50)
    print("                模型效果对比汇总")
    print("="*50)
    print(f"评估病例总数: {valid_case_count}")
    print("-"*50)
    print("【模型 1】(qwen3-4b-instruct):")
    print(f"  - 镜检所见 (字级平均 F1): {m1_findings_f1_sum / valid_case_count:.4f}")
    print(f"  - 诊断结论 (集合平均 F1): {m1_diag_f1_sum / valid_case_count:.4f}")
    print("-"*50)
    print("【模型 2】(qwen3.5-4b):")
    print(f"  - 镜检所见 (字级平均 F1): {m2_findings_f1_sum / valid_case_count:.4f}")
    print(f"  - 诊断结论 (集合平均 F1): {m2_diag_f1_sum / valid_case_count:.4f}")
    print("="*50)
    
    # 10. 详细样本得分输出 (前 10 个样本做可视化，便于调试)
    print("\n部分病例打分明细预览 (前10例):")
    print(f"{'病例编号':<12} | {'模1镜检F1':<8} | {'模1诊断F1':<8} | {'模2镜检F1':<8} | {'模2诊断F1':<8}")
    print("-"*60)
    for comp in comparisons[:10]:
        print(f"{comp['case']:<12} | {comp['m1_find_f1']:<9.4f} | {comp['m1_diag_f1']:<9.4f} | {comp['m2_find_f1']:<9.4f} | {comp['m2_diag_f1']:<9.4f}")
    print("-"*60)

    # 11. 生成 Markdown 详细对比报告文件 (包含 GT、模型1、模型2的所有文本及打分比对)
    output_markdown_path = '/media/inno/output/LLM/colon/compare_report.md'
    os.makedirs(os.path.dirname(output_markdown_path), exist_ok=True)
    
    md_lines = []
    md_lines.append("# 肠镜微调模型对比评测详细报告\n")
    md_lines.append("## 1. 整体指标效果汇总\n")
    md_lines.append("| 评测维度 | 模型 1 (qwen3-4b-instruct) | 模型 2 (qwen3.5-4b) | 优胜者 |")
    md_lines.append("| --- | --- | --- | --- |")
    
    m1_find_avg = m1_findings_f1_sum / valid_case_count
    m2_find_avg = m2_findings_f1_sum / valid_case_count
    find_winner = "模型 1" if m1_find_avg > m2_find_avg else ("模型 2" if m2_find_avg > m1_find_avg else "平手")
    md_lines.append(f"| **镜检所见 (字级平均 F1)** | {m1_find_avg:.4f} | {m2_find_avg:.4f} | **{find_winner}** |")
    
    m1_diag_avg = m1_diag_f1_sum / valid_case_count
    m2_diag_avg = m2_diag_f1_sum / valid_case_count
    diag_winner = "模型 1" if m1_diag_avg > m2_diag_avg else ("模型 2" if m2_diag_avg > m1_diag_avg else "平手")
    md_lines.append(f"| **诊断结论 (集合平均 F1)** | {m1_diag_avg:.4f} | {m2_diag_avg:.4f} | **{diag_winner}** |")
    
    md_lines.append("\n## 2. 逐案（Case-by-Case）详细文本比对明细\n")
    
    for comp in comparisons:
        case = comp['case']
        gt_data = parse_gt_file(gt_map[case])
        m1_data = parse_model_output(res_1.get(case, ""))
        m2_data = parse_model_output(res_2.get(case, ""))
        
        md_lines.append(f"### 病例编号: `{case}`\n")
        
        # 镜检所见对比表格
        md_lines.append("#### 【镜检所见】文本及相似度比对")
        md_lines.append("| 来源类别 | 镜检所见文本内容 | 字符 F1 得分 |")
        md_lines.append("| --- | --- | --- |")
        md_lines.append(f"| **标准答案 (GT)** | {gt_data['镜检所见'].replace(chr(10), '<br>')} | - |")
        md_lines.append(f"| **模型 1 (qwen3-4b)** | {m1_data['镜检所见'].replace(chr(10), '<br>')} | {comp['m1_find_f1']:.4f} |")
        md_lines.append(f"| **模型 2 (qwen3.5)** | {m2_data['镜检所见'].replace(chr(10), '<br>')} | {comp['m2_find_f1']:.4f} |")
        md_lines.append("")
        
        # 诊断结论对比表格
        gt_diag_str = "、".join(gt_data['诊断结论']) if gt_data['诊断结论'] else "无"
        m1_diag_str = "、".join(m1_data['诊断结论']) if m1_data['诊断结论'] else "无"
        m2_diag_str = "、".join(m2_data['诊断结论']) if m2_data['诊断结论'] else "无"
        
        md_lines.append("#### 【诊断结论】提取比对")
        md_lines.append("| 来源类别 | 提取到的诊断结论列表 | 集合 F1 得分 |")
        md_lines.append("| --- | --- | --- |")
        md_lines.append(f"| **标准答案 (GT)** | {gt_diag_str} | - |")
        md_lines.append(f"| **模型 1 (qwen3-4b)** | {m1_diag_str} | {comp['m1_diag_f1']:.4f} |")
        md_lines.append(f"| **模型 2 (qwen3.5)** | {m2_diag_str} | {comp['m2_diag_f1']:.4f} |")
        md_lines.append("\n" + "-"*80 + "\n")
        
    with open(output_markdown_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))
    print(f"详细对比报告已成功保存至 Markdown 报告文件: {output_markdown_path}")

if __name__ == '__main__':
    main()
