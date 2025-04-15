import torch
import json
import os
import numpy as np
from modelscope.hub.snapshot_download import snapshot_download
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, DataCollatorForLanguageModeling
from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset, load_dataset, concatenate_datasets
import re
import jieba
from concurrent.futures import ThreadPoolExecutor
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from tqdm import tqdm
import time

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-5
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
OUTPUT_DIR = "./qwen_novel_lora_optimized"
SAVE_STEPS = 100
EVAL_STEPS = 100
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION = 4

# 加载基础模型
def load_base_model(model_name):
    print("加载基础模型...")
    model_dir = snapshot_download(model_id=model_name, cache_dir="./model_cache")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        torch_dtype=torch.float16, 
        device_map="auto",
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = 'left'  # 显式设置为左侧填充
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 测试tokenizer性能
    test_text = "测试tokenizer性能"
    start_time = time.time()
    test_tokens = tokenizer(test_text, return_tensors="pt")
    print(f"Tokenizer测试用时: {time.time() - start_time:.4f}秒")
    return model, tokenizer

# 单样本推理
def get_model_answer(model, tokenizer, prompt, max_length=512):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    input_length = model_inputs.input_ids.shape[1]
    generated_ids = generated_ids[:, input_length:]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text.strip()

# 批量推理
def get_model_answer_batch(model, tokenizer, prompts, max_length=512):
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    input_lengths = model_inputs.input_ids.shape[1]
    generated_ids = generated_ids[:, input_lengths:]
    return [text.strip() for text in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]

# 数据准备 - 自回归任务
def prepare_autoregressive_data(file_path, novel_title, output_jsonl="novel_auto.jsonl", chunk_size=512, overlap=128):
    print("准备自回归数据...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 清洗文本
    content = re.sub(r'\r\n', '\n', content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？\n\s"\'—]', '', content)
    content = re.sub(r'\s+', ' ', content)
    
    # 解析章节
    chapters = []
    part_pattern = r'(前传|后传)\s*第[一二三四五六七八九十]+部\s*[\u4e00-\u9fa5]+'
    chapter_pattern = r'(第[一二三四五六七八九十百千万零]+章)\s*([^\n]{0,30}?)|(引言|引子)\s*([^\n]{0,30}?)'
    
    parts = []
    for match in re.finditer(part_pattern, content):
        parts.append({"name": match.group(0), "start": match.start()})
    
    if not parts:
        parts.append({"name": "前传 第一部 神龙物语", "start": 0})
    
    # 提取章节内容
    for i, part in enumerate(parts):
        part_start = part["start"]
        part_end = parts[i+1]["start"] if i+1 < len(parts) else len(content)
        part_content = content[part_start:part_end]
        
        for match in re.finditer(chapter_pattern, part_content):
            if match.group(1):
                chapter_num = match.group(1)
                chapter_title = match.group(2).strip() if match.group(2) else ""
                full_title = f"{chapter_num}{chapter_title}"
                start_offset = match.start()
            elif match.group(3):
                chapter_num = match.group(3)
                chapter_title = match.group(4).strip() if match.group(4) else ""
                full_title = f"{chapter_num}{chapter_title}"
                start_offset = match.start()
            
            chapters.append({
                "part": part["name"],
                "title": full_title,
                "start": part_start + start_offset
            })
    
    foreword_pattern = r'前言\s*([^\n]{0,30})'
    foreword_match = re.search(foreword_pattern, content)
    if foreword_match:
        chapters.insert(0, {
            "part": "",
            "title": f"前言 {foreword_match.group(1).strip()}",
            "start": foreword_match.start()
        })
    
    chapters.sort(key=lambda x: x["start"])
    
    # 处理章节内容
    for i in range(len(chapters)):
        start = chapters[i]["start"] + len(chapters[i]["title"])
        end = chapters[i+1]["start"] if i+1 < len(chapters) else len(content)
        chapters[i]["content"] = content[start:end].strip()
        prefix = f"{chapters[i]['part']} {chapters[i]['title']}".strip()
        chapters[i]["prefix"] = prefix
    
    if not chapters:
        chapters.append({
            "part": "",
            "title": "全文",
            "prefix": "全文",
            "content": content.strip(),
            "start": 0
        })
    
    print(f"识别到 {len(chapters)} 个章节")
    
    # 分段处理
    def improved_chunking(text, prefix, chunk_size, overlap):
        sentences = list(jieba.cut(text, cut_all=False))
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= chunk_size:
                current_chunk += sentence
                current_length += sentence_length
            else:
                if current_length > 0:
                    chunks.append(prefix + "\n\n" + current_chunk)
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_chunk = overlap_text + sentence
                current_length = len(current_chunk)
                while current_length > chunk_size:
                    chunks.append(prefix + "\n\n" + current_chunk[:chunk_size])
                    current_chunk = current_chunk[chunk_size-overlap:]
                    current_length = len(current_chunk)
        
        if current_length > 0:
            chunks.append(prefix + "\n\n" + current_chunk)
        
        return chunks
    
    examples = []
    for chapter in chapters:
        chunks = improved_chunking(chapter["content"], chapter["prefix"], chunk_size, overlap)
        for chunk in chunks:
            examples.append({"text": chunk, "type": "autoregressive"})
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"生成 {len(examples)} 个自回归训练样本")
    return examples

# 准备QA数据
def prepare_qa_data(file_path, novel_title, base_model, base_tokenizer, output_jsonl="novel_qa.jsonl"):
    print("准备QA数据...")
    
    # 先准备自回归数据以获取章节结构
    auto_examples = prepare_autoregressive_data(file_path, novel_title, "temp_auto.jsonl")
    
    # 提取章节信息
    chapters = []
    chapter_texts = {}
    
    for example in auto_examples:
        text = example["text"]
        lines = text.split("\n\n", 1)
        if len(lines) == 2:
            chapter_header = lines[0].strip()
            if chapter_header not in chapter_texts:
                chapter_texts[chapter_header] = []
            chapter_texts[chapter_header].append(lines[1])
    
    for header, content_parts in chapter_texts.items():
        content = " ".join(content_parts)
        part_title = header.split(" 第")[0] if " 第" in header else ""
        chapter_title = header[len(part_title):].strip() if part_title else header
        
        chapters.append({
            "part": part_title,
            "title": chapter_title,
            "content": content,
            "prefix": header
        })
    
    # 生成QA对
    def generate_qa_for_chapter(chapter):
        qa_questions = [
            f"根据《{novel_title}》{chapter['prefix']}，主要讲了什么内容？",
            f"《{novel_title}》{chapter['prefix']}中的主要人物是谁？",
            f"《{novel_title}》{chapter['prefix']}的关键事件是什么？",
        ]
        
        prompts = []
        content = chapter['content']
        for i in range(0, len(content), 1500):
            chunk = content[i:i+1500]
            for q in qa_questions:
                prompts.append(f"以下是《{novel_title}》{chapter['prefix']}的内容：\n\n{chunk}\n\n{q}")
        
        answers = get_model_answer_batch(base_model, base_tokenizer, prompts)
        
        qa_examples = []
        for i, (prompt, answer) in enumerate(zip(prompts, answers)):
            if len(answer.strip()) > 10:  # 只保留有实质内容的回答
                qa_examples.append({
                    "text": prompt,
                    "answer": answer,
                    "type": "qa"
                })
        
        return qa_examples
    
    # 并行生成QA
    all_qa_examples = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(generate_qa_for_chapter, chapters), total=len(chapters)))
        for result in results:
            all_qa_examples.extend(result)
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for example in all_qa_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"生成 {len(all_qa_examples)} 个QA训练样本")
    return all_qa_examples

# 数据加载和预处理 - 自回归任务（优化版）
def preprocess_autoregressive_data(dataset, tokenizer):
    print(f"开始预处理数据，共 {len(dataset)} 条...")
    
    def preprocess_function(examples):
        # 处理文本格式
        texts = [f"<novel>{text}</novel>" for text in examples["text"]]
        
        # 对文本进行tokenize
        try:
            inputs = tokenizer(
                texts, 
                truncation=True, 
                max_length=MAX_LENGTH, 
                padding="max_length", 
                return_tensors="pt"
            )
            
            # 创建标签 (用于自回归训练)
            labels = inputs["input_ids"].clone()
            
            # 将padding token的标签设置为-100 (忽略)
            padding_mask = inputs["attention_mask"].eq(0)
            labels.masked_fill_(padding_mask, -100)
            
            inputs["labels"] = labels
            return inputs
        except Exception as e:
            print(f"数据处理错误: {e}")
            print(f"错误样本长度: {[len(t) for t in texts]}")
            raise
    
    # 使用分批处理
    batch_size = 100  # 每批处理的样本数
    tokenized_batches = []
    
    for i in range(0, len(dataset), batch_size):
        start_time = time.time()
        end = min(i + batch_size, len(dataset))
        print(f"处理批次 {i//batch_size + 1}/{(len(dataset)-1)//batch_size + 1} (样本 {i}-{end-1})")
        
        batch = dataset.select(range(i, end))
        processed_batch = batch.map(
            preprocess_function, 
            batched=True,
            remove_columns=["text", "type"]
        )
        
        tokenized_batches.append(processed_batch)
        print(f"批次处理完成，用时 {time.time() - start_time:.2f} 秒")
    
    # 合并所有批次
    print("合并处理后的批次...")
    if len(tokenized_batches) > 1:
        tokenized_dataset = concatenate_datasets(tokenized_batches)
    else:
        tokenized_dataset = tokenized_batches[0]
    
    tokenized_dataset.set_format("torch")
    print(f"数据预处理完成，共 {len(tokenized_dataset)} 条")
    return tokenized_dataset

# 数据加载和预处理 - QA任务
def preprocess_qa_data(dataset, tokenizer):
    def preprocess_function(examples):
        # 构建输入和目标
        inputs = []
        targets = []
        
        for text, answer in zip(examples["text"], examples["answer"]):
            inputs.append(f"<question>{text}</question>")
            targets.append(f"<answer>{answer}</answer>")
        
        # 对输入文本进行tokenize
        model_inputs = tokenizer(
            inputs, 
            truncation=True, 
            max_length=MAX_LENGTH, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        # 对目标文本进行tokenize
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, 
                truncation=True, 
                max_length=MAX_LENGTH//2,  # 答案长度限制更小
                padding="max_length", 
                return_tensors="pt"
            ).input_ids
        
        # 将padding token的标签设置为-100
        padding_mask = labels.eq(tokenizer.pad_token_id)
        labels.masked_fill_(padding_mask, -100)
        
        model_inputs["labels"] = labels
        return model_inputs
    
    # 应用预处理
    tokenized_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        num_proc=4,
        remove_columns=["text", "answer", "type"]
    )
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

# 配置 LoRA
def setup_lora_config(model):
    # 优化的LoRA配置
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 准备训练
    model.train()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    return model

# 训练模型 - 第一阶段：自回归
def train_autoregressive(model, dataset, output_dir=OUTPUT_DIR + "/auto"):
    print("开始自回归训练...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 分割训练和验证集
    train_val = dataset.train_test_split(test_size=0.1, seed=42)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # 优化器设置
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        
        # 混合精度训练
        fp16=True,  # 使用FP16
        fp16_opt_level="O1",
        
        # 性能优化
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_val["train"],
        eval_dataset=train_val["test"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 开始训练
    trainer.train()
    
    # 保存中间模型
    trainer.save_model(output_dir)
    
    return model

# 训练模型 - 第二阶段：问答
def train_qa(model, dataset, output_dir=OUTPUT_DIR + "/qa"):
    print("开始QA训练...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 分割训练和验证集
    train_val = dataset.train_test_split(test_size=0.1, seed=42)
    
    # 训练参数 - 更小的学习率
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE/5,  # 使用更小的学习率进行微调
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # 优化器设置
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        
        # 混合精度训练
        fp16=True,
        fp16_opt_level="O1",
        
        # 性能优化
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_val["train"],
        eval_dataset=train_val["test"],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型
    trainer.save_model(output_dir)
    
    return model

# 保存最终模型
def save_merged_model(model, tokenizer, output_dir=OUTPUT_DIR + "/final"):
    print("保存最终模型...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并LoRA权重
    model = model.merge_and_unload()
    
    # 保存模型和tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"模型已保存至 {output_dir}")

# 定性评估
def evaluate_model(model, tokenizer, prompts, model_name="模型"):
    print(f"\n=== {model_name}评估 ===")
    
    model.eval()
    results = []
    
    for prompt in prompts:
        generated_text = get_model_answer(model, tokenizer, prompt)
        print(f"提示：{prompt}\n生成：{generated_text}\n")
        results.append({"prompt": prompt, "response": generated_text})
    
    return results

# 主函数
def main():
    novel_name = "龙战士传说"
    input_file = "novel.txt"
    auto_jsonl = "novel_auto.jsonl"
    qa_jsonl = "novel_qa.jsonl"
    
    # 加载模型和tokenizer
    base_model, base_tokenizer = load_base_model(MODEL_NAME)
    
    # 检查文件是否存在，不存在则生成
    if not os.path.exists(auto_jsonl):
        prepare_autoregressive_data(input_file, novel_name, auto_jsonl)
    
    if not os.path.exists(qa_jsonl):
        prepare_qa_data(input_file, novel_name, base_model, base_tokenizer, qa_jsonl)
    
    # 测试提示
    test_prompts = [
        f"根据《{novel_name}》前传 第一部 神龙物语 第一章 新人类，描述主角是谁及其背景。",
        f"《{novel_name}》前传 第一部 神龙物语 第二章 暗黑龙卡鲁兹的关键事件是什么？",
        f"《{novel_name}》前传 第二部 双星传奇的结局是什么？",
        f"《{novel_name}》中秀耐达的最终命运如何？",
    ]
    
    # 微调前评估
    base_eval = evaluate_model(base_model, base_tokenizer, test_prompts, "微调前")
    
    # 阶段1：自回归训练
    print("加载自回归数据...")
    auto_dataset = Dataset.from_json(auto_jsonl)
    auto_tokenized = preprocess_autoregressive_data(auto_dataset, base_tokenizer)
    
    # 配置LoRA
    print("设置LoRA...")
    lora_model = setup_lora_config(base_model)
    
    # 阶段1训练
    trained_auto = train_autoregressive(lora_model, auto_tokenized)
    
    # 阶段1评估
    auto_eval = evaluate_model(trained_auto, base_tokenizer, test_prompts, "自回归训练后")
    
    # 阶段2：QA训练
    print("加载QA数据...")
    qa_dataset = Dataset.from_json(qa_jsonl)
    qa_tokenized = preprocess_qa_data(qa_dataset, base_tokenizer)
    
    # 阶段2训练
    trained_qa = train_qa(trained_auto, qa_tokenized)
    
    # 最终评估
    final_eval = evaluate_model(trained_qa, base_tokenizer, test_prompts, "微调后")
    
    # 保存最终模型
    save_merged_model(trained_qa, base_tokenizer)
    
    # 保存评估结果
    with open(f"{OUTPUT_DIR}/eval_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            "base_eval": base_eval,
            "auto_eval": auto_eval,
            "final_eval": final_eval
        }, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()