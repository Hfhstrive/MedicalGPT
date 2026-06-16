from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from ipdb import set_trace

# 1. 指定模型名称
base_model_name = "/home/inno/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554/"
# lora_adapter_path = "/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v3/"  # 多语句特征描述
# lora_adapter_path = "/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v4-rank16/checkpoint-195/" # 单病例特征描述
lora_adapter_path = "/media/inno/work_dirs/LLM/MedicalGPT/outputs-sft-qwen3-4b-report-v5-rank16/"

# 2. 加载分词器和模型
print("正在加载模型，首次运行会从 Hugging Face 下载...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype="auto",
    device_map="auto"
)
print("模型加载完成。")
model = PeftModel.from_pretrained(base_model, lora_adapter_path)
# model = base_model

# 3. 准备对话消息
# V1 & V2
# questions = [
#     # "规范表达以下语句：大弯皱襞走向规则，黏膜光滑，小弯黏膜地图状发红，色调逆转，延及近贲门",
#     "规范表达以下语句：胃角黏膜红白相间，以白为主，后壁见IIb病变，NBI放大呈茶褐色，可见清晰边界，微血管和微结构不规则，醋酸染色见腺管结构紊乱",
#     # "规范表达以下语句：在胃食管结合部看到齿状线上方有线性糜烂，长度小于5mm，且在齿状线下方胃黏膜上发现有一息肉样隆起",
#     # "规范表达以下语句：胃角黏膜粗糙，见地图状发红，后壁见一息肉样隆起",
#     "描述的是哪个部位（'食管', '贲门', '胃体', '胃窦', '幽门', '胃角', '胃底', '十二指肠'）？",
#     "描述涉及哪些疾病？"
#     # '一句话概况以下内容：食管上中段看到多发的白黄色结节样菌斑，直径约0.2到0.3厘米，部分融合，长的很牢，周围黏膜颜色正常、没有充血，也没有糜烂溃疡，管腔不窄，蠕动正常。距门齿15厘米处，食管后壁看到一块橘红色的凹陷，大小约1.0乘1.2厘米，是圆形的，边界清晰，表面有细颗粒感和绒毛感，周围是灰白色黏膜，分界明显。'
# ]

# # V3
# questions = [
#     "以下描述涉及哪些疾病：在胃食管结合部看到齿状线上方有线性糜烂，长度小于5mm，且在齿状线下方胃黏膜上发现有一息肉样隆起。",
#     "这些疾病分别在哪个部位（'食管', '贲门', '胃体', '胃窦', '幽门', '胃角', '胃底', '十二指肠'）？",
#     "对这些病变分别进行规范描述："
# ]

# V4
asr_texts = [
    '这个考虑是C2型萎缩性胃炎，现用脑梗的现症很远，危及瘤化，现在有那个黏膜弥漫性发红，黏膜有肿胀，这是下期的。',
    '这个考虑是新二型萎缩性胃炎，现幽门螺旋杆菌性质检有，VT68检呢有粘膜弥漫性发红，粘膜肿胀，RACAS消失。',
    '这个考虑是C2型萎缩性胃炎，幽门螺旋杆菌现症感染，胃体的话，有那个黏膜弥漫性发红，黏膜有肿胀，RAC消失了.',
    '食管靠近齿状线的地方黏膜充血水肿，还有散在的纵行糜烂，考虑反流性食管炎B级。胃体前壁有一个0.5厘米的黏膜下隆起。胃体大弯侧有两颗0.3厘米的扁平息肉，采用活检钳咬除。胃体小弯下段黏膜偏白，胃角黏膜变薄、血管能看见，胃窦也是红白相间以白为主，考虑萎缩性胃炎C2。',
    '这个在食管下段有两条静脉，略迂曲，直径最大大概0.3cm，表面没有红色征，散在白色瘢痕；胃窦后壁靠近胃角的地方，有一个大小约2cm×1.5cm的隆起型黏膜病变，表面粗糙，呈褪色调，充气吸引后病变柔软；在NBI放大下病变呈红茶色，DL阳性，MV和MS潜规则，呈网格状血管。',
]
for asr_text in asr_texts:
    questions = [
        # f"规范表达以下语句：{asr_text}",
        # "提取语句中所涉及的结论性疾病？",
        # "根据[部位, 特征描述, 病变名称]格式拆分语句：",

        f"提取有效信息，规范表达以下语句：{asr_text}",
        "提取语句中所涉及的结论性疾病？",
        "根据[部位, 特征描述, 病变名称]格式拆分语句，并专业化特征描述：",
    ]
    messages = [] # 对话流更新
    # 循环进行多轮推理
    for i, q in enumerate(questions):
        # 将当前问题加入对话历史
        messages.append({"role": "user", "content": q})

        # 应用模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 推理
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            # top_p=1.0,
            # temperature=0.1,  # 医疗建议通常调低温度，增加确定性
            do_sample=False,
        )

        # 解码
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        response = tokenizer.decode(output_ids, skip_special_tokens=True)

        print(f"\n--- 问题 {i + 1} ---")
        print(f"问：{q}")
        print(f"答：{response}")

        # 重要：将模型的回答加入对话历史，这样下一轮才能实现“多问题推理”
        messages.append({"role": "assistant", "content": response})
    print('----------------------------------------------------------------------------------')