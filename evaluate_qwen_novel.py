import csv
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

NOVEL_NAME = "龙战士传说"
# 读取 a.csv 文件并加载数据
questions = []
with open('a.csv', 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        questions.append({
            "question": row['问题'],
            "options": [row['选项A'], row['选项B'], row['选项C'], row['选项D']],
            "correct_answer": row['答案']
        })

# 统计原始答案分布
original_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
for q in questions:
    original_counts[q['correct_answer']] += 1
print("原始答案分布:", original_counts)

# 调整答案分布为均匀（每种约 25 条）
target_count = 25
adjusted_questions = questions.copy()

# 计算需要调整的数量
current_counts = original_counts.copy()
to_adjust = {'A': target_count - current_counts['A'],
             'B': target_count - current_counts['B'],
             'C': target_count - current_counts['C'],
             'D': target_count - current_counts['D']}

# 随机调整答案
random.shuffle(adjusted_questions)
for answer in ['A', 'B', 'C', 'D']:
    needed = to_adjust[answer]
    if needed > 0:
        # 从其他答案中随机选择问题并替换
        candidates = [q for q in adjusted_questions if q['correct_answer'] != answer and current_counts[q['correct_answer']] > target_count]
        for i in range(min(needed, len(candidates))):
            q = candidates[i]
            current_counts[q['correct_answer']] -= 1
            q['correct_answer'] = answer
            current_counts[answer] += 1

# 检查调整后的分布
adjusted_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
for q in adjusted_questions:
    adjusted_counts[q['correct_answer']] += 1
print("调整后答案分布:", adjusted_counts)

# 加载 Qwen2.5-1.5B-Instruct 模型
model_path = "./qwen_novel_lora_optimized"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 定义推理函数
def get_model_answer(question, options):
    prompt = f"\n{question}\nA. {options[0]}\nB. {options[1]}\nC. {options[2]}\nD. {options[3]}\n 答案："
    messages = [
        {"role": "system", "content": "You are a helpful assistant with knowledge of 《{NOVEL_NAME}》. 请根据小说内容回答问题。例如：问题：《{NOVEL_NAME}》的主角是谁？答案：A. 张三\nB. 李四\nC. 王五\nD. 赵六\n答案：A"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=5)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return response

# 测试调整后的数据
correct_count = 0
total_count = 0

for q in adjusted_questions:
    question = q['question']
    options = q['options']
    correct_answer = q['correct_answer']
    
    # 获取模型答案
    model_answer = get_model_answer(question, options)
    
    # 判断是否正确
    is_correct = "✅" if model_answer == correct_answer else "❌"
    print(f"{question}")
    print(f"A. {options[0]}")
    print(f"B. {options[1]}")
    print(f"C. {options[2]}")
    print(f"D. {options[3]}")
    print(f"输出的答案: {model_answer} | 正确答案: {correct_answer} | {is_correct}")
    print("-" * 100)
    
    total_count += 1
    if model_answer == correct_answer:
        correct_count += 1

# 计算并输出结果
correct_rate = correct_count / total_count
print(f"正确个数: {correct_count}")
print(f"错误个数: {total_count - correct_count}")
print(f"正确率: {correct_rate:.2%}")