import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList, 
    StoppingCriteria
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from rouge import Rouge
import pandas as pd
import seaborn as sns
from collections import defaultdict
import glob
import inspect

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="大语言模型量化评估")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="要评估的模型路径",
    )
    # 为不同任务提供不同数据集的参数
    parser.add_argument(
        "--perplexity_dataset",
        type=str,
        default=None,
        help="用于困惑度评估的数据集，支持HuggingFace数据集名称或本地文件路径，多个用逗号分隔",
    )
    parser.add_argument(
        "--generation_prompts",
        type=str,
        default=None,
        help="用于生成任务的提示文件路径，多个用逗号分隔",
    )
    parser.add_argument(
        "--qa_dataset",
        type=str,
        default=None,
        help="用于问答评估的数据集，支持HuggingFace数据集名称或本地文件路径，多个用逗号分隔",
    )
    parser.add_argument(
        "--classification_dataset",
        type=str,
        default=None,
        help="用于分类评估的数据集，支持HuggingFace数据集名称或本地文件路径，多个用逗号分隔",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="validation",
        help="使用的HuggingFace数据集分割，如'train', 'validation', 'test'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="保存评估结果的目录",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="生成文本的最大长度",
    )
    parser.add_argument(
        "--temperature",
        type=int,
        default=0.7,
        help="生成的温度",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="每个任务要评估的样本数量，-1表示全部",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="评估批次大小",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="是否使用wandb记录结果",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="llm-evaluation",
        help="Wandb项目名称",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Wandb运行名称",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="perplexity,generation,qa,classification",
        help="要执行的评估任务，用逗号分隔",
    )
    parser.add_argument(
        "--no_cuda", 
        action="store_true", 
        help="不使用CUDA即使可用"
    )
    parser.add_argument(
        "--fp16", 
        action="store_true", 
        help="使用半精度FP16"
    )
    parser.add_argument(
        "--save_generations", 
        action="store_true", 
        help="保存模型生成的文本到文件"
    )
    
    args = parser.parse_args()
    return args

# 提取公共方法：使用Qwen模式生成文本
def generate_with_qwen_format(model, tokenizer, prompt, system_prompt, max_new_tokens, temperature=0.7, do_sample=True, top_p=0.7):
    """使用Qwen对话格式生成文本
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 用户提示
        system_prompt: 系统提示
        max_new_tokens: 最大生成token数
        temperature: 温度
        do_sample: 是否使用采样
        top_p: top-p采样参数
        
    Returns:
        生成的文本
    """
    # 构建Qwen格式输入
    full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    full_prompt += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # 处理输入
    inputs = tokenizer(full_prompt, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(model.device)
    
    # 设置生成参数
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    # 添加停止标准
    generate_signature = inspect.signature(model.generate)
    if "stopping_criteria" in generate_signature.parameters:
        # 自定义停止标准
        class StopOnTokens(StoppingCriteria):
            def __init__(self, stop_token_ids):
                self.stop_token_ids = stop_token_ids
            
            def __call__(self, input_ids, scores, **kwargs):
                for stop_id in self.stop_token_ids:
                    if input_ids[0][-1] == stop_id:
                        return True
                return False
        
        # 获取终止词的token IDs
        stop_words_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_words_ids)])
        gen_kwargs["stopping_criteria"] = stopping_criteria
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    # 解码输出
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取助手的回复部分
    assistant_start = "<|im_start|>assistant\n"
    assistant_end = "<|im_end|>"
    
    # 查找最后一个助手部分
    last_assistant_start = full_response.rfind(assistant_start)
    if last_assistant_start != -1:
        response_start = last_assistant_start + len(assistant_start)
        response_end = full_response.find(assistant_end, response_start)
        if response_end != -1:
            generated_text = full_response[response_start:response_end].strip()
        else:
            # 如果没有找到结束标记，就取所有剩余文本
            generated_text = full_response[response_start:].strip()
    else:
        # 回退到简单解码
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    return generated_text

# 加载多个数据集并合并
def load_multiple_datasets(dataset_paths_or_names, dataset_split, task_type):
    """
    加载多个数据集并合并为一个
    
    Args:
        dataset_paths_or_names: 逗号分隔的数据集路径或名称
        dataset_split: 对HuggingFace数据集使用的分割
        task_type: 任务类型，用于确定数据格式
        
    Returns:
        合并后的数据集
    """
    if not dataset_paths_or_names:
        return None
        
    # 分割路径/名称列表
    paths_or_names = [p.strip() for p in dataset_paths_or_names.split(',')]
    combined_dataset = None
    
    for path_or_name in paths_or_names:
        try:
            current_dataset = None
            
            # 检查是否是本地文件路径
            if os.path.exists(path_or_name) or "*" in path_or_name:
                # 处理通配符
                if "*" in path_or_name:
                    files = glob.glob(path_or_name)
                    if not files:
                        logger.warning(f"未找到匹配 {path_or_name} 的文件")
                        continue
                    
                    # 合并所有匹配的文件
                    file_datasets = []
                    for file_path in files:
                        file_dataset = load_file_dataset(file_path, task_type)
                        if file_dataset:
                            file_datasets.append(file_dataset)
                    
                    if file_datasets:
                        # 尝试合并所有文件数据集
                        current_dataset = file_datasets[0]
                        for ds in file_datasets[1:]:
                            current_dataset.extend(ds)
                else:
                    # 加载单个文件
                    current_dataset = load_file_dataset(path_or_name, task_type)
            else:
                # 尝试作为HuggingFace数据集名称加载
                try:
                    logger.info(f"尝试从HuggingFace加载数据集 {path_or_name}")
                    current_dataset = load_dataset(path_or_name, split=dataset_split)
                    
                    # 验证数据集格式是否符合任务要求
                    if task_type == "perplexity":
                        if not any(col in current_dataset.column_names for col in ["text", "content", "sentence"]):
                            logger.warning(f"数据集 {path_or_name} 缺少用于困惑度评估的文本列")
                            current_dataset = None
                    elif task_type == "qa":
                        if not ("question" in current_dataset.column_names and 
                               ("answer" in current_dataset.column_names or "answers" in current_dataset.column_names)):
                            logger.warning(f"数据集 {path_or_name} 不是标准问答格式")
                            current_dataset = None
                    elif task_type == "classification":
                        if not (("text" in current_dataset.column_names or "sentence" in current_dataset.column_names) 
                               and "label" in current_dataset.column_names):
                            logger.warning(f"数据集 {path_or_name} 不是标准分类格式")
                            current_dataset = None
                            
                except Exception as e:
                    logger.warning(f"加载HuggingFace数据集 {path_or_name} 失败: {e}")
                    current_dataset = None
            
            # 合并数据集
            if current_dataset:
                if combined_dataset is None:
                    combined_dataset = current_dataset
                else:
                    # 对于列表格式的数据集
                    if isinstance(combined_dataset, list):
                        if isinstance(current_dataset, list):
                            combined_dataset.extend(current_dataset)
                        else:
                            # 将Dataset转换为列表格式
                            combined_dataset.extend(current_dataset.to_list())
                    # 对于HuggingFace Dataset格式
                    elif isinstance(combined_dataset, Dataset):
                        if isinstance(current_dataset, Dataset):
                            # 尝试合并Dataset
                            try:
                                combined_dataset = concatenate_datasets([combined_dataset, current_dataset])
                            except Exception as e:
                                logger.warning(f"合并Dataset失败，转为列表格式: {e}")
                                combined_dataset = combined_dataset.to_list() + current_dataset.to_list()
                        else:
                            # 将Dataset转为列表并合并
                            combined_dataset = combined_dataset.to_list() + current_dataset
                
                logger.info(f"成功加载数据集 {path_or_name} 用于 {task_type} 任务")
        
        except Exception as e:
            logger.error(f"处理数据集 {path_or_name} 时出错: {e}")
    
    if combined_dataset:
        # 记录合并后的数据集大小
        dataset_size = len(combined_dataset) if hasattr(combined_dataset, "__len__") else "未知"
        logger.info(f"{task_type} 任务最终合并了 {len(paths_or_names)} 个数据源，共 {dataset_size} 个样本")
    else:
        logger.warning(f"所有 {task_type} 数据集加载失败")
    
    return combined_dataset

# 加载单个文件数据集
def load_file_dataset(file_path, task_type):
    """加载单个文件作为数据集"""
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查数据格式
            if isinstance(data, list):
                # 根据任务类型验证和转换数据
                if task_type == "perplexity":
                    # 如果是文本列表，直接使用
                    if all(isinstance(item, str) for item in data):
                        return data
                    # 如果是对象列表，尝试提取文本字段
                    elif all(isinstance(item, dict) for item in data):
                        if all("text" in item for item in data):
                            return [item["text"] for item in data]
                        elif all("content" in item for item in data):
                            return [item["content"] for item in data]
                
                # 问答和分类数据格式验证
                elif task_type == "qa" or task_type == "classification":
                    if all(isinstance(item, dict) for item in data):
                        # 验证关键字段
                        if task_type == "qa" and all("question" in item and "answer" in item for item in data):
                            return data
                        elif task_type == "classification" and all("text" in item and "label" in item for item in data):
                            return data
            
            elif task_type == "generation" and isinstance(data, list) and all(isinstance(p, str) for p in data):
                # 对于生成任务，接受字符串提示列表
                return data
            
            logger.warning(f"文件 {file_path} 格式不符合 {task_type} 任务要求")
            return None
                
        elif file_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(file_path)
            
            # 转换为列表格式
            data = df.to_dict('records')
            
            # 根据任务类型验证和转换数据
            if task_type == "perplexity":
                if "text" in df.columns:
                    return df["text"].tolist()
                elif "content" in df.columns:
                    return df["content"].tolist()
            
            elif task_type == "qa":
                if "question" in df.columns and "answer" in df.columns:
                    return data
            
            elif task_type == "classification":
                if "text" in df.columns and "label" in df.columns:
                    return data
            
            elif task_type == "generation":
                if "prompt" in df.columns:
                    return df["prompt"].tolist()
            
            logger.warning(f"CSV文件 {file_path} 不包含 {task_type} 任务所需的列")
            return None
            
        elif file_path.endswith('.txt'):
            # 对于txt文件，按行读取
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            if task_type == "perplexity":
                # 对于困惑度评估，每行作为一个文本样本
                return lines
            elif task_type == "generation":
                # 对于生成任务，每行作为一个提示
                return lines
            else:
                logger.warning(f"无法将txt文件 {file_path} 用于 {task_type} 任务")
                return None
        else:
            logger.warning(f"不支持的文件格式: {file_path}")
            return None
    
    except Exception as e:
        logger.error(f"加载文件 {file_path} 时出错: {e}")
        return None

# 加载评估数据集
def load_evaluation_datasets(args):
    """
    为每个任务加载对应的数据集
    """
    datasets = {}
    
    # 任务列表
    tasks = [task.strip() for task in args.tasks.split(",")]
    
    # 为每个任务加载数据集
    if "perplexity" in tasks and args.perplexity_dataset:
        datasets["perplexity"] = load_multiple_datasets(
            args.perplexity_dataset, args.dataset_split, "perplexity"
        )
    
    if "qa" in tasks and args.qa_dataset:
        datasets["qa"] = load_multiple_datasets(
            args.qa_dataset, args.dataset_split, "qa"
        )
    
    if "classification" in tasks and args.classification_dataset:
        datasets["classification"] = load_multiple_datasets(
            args.classification_dataset, args.dataset_split, "classification"
        )
    
    if "generation" in tasks and args.generation_prompts:
        # 加载生成提示
        prompts = []
        for prompts_file in args.generation_prompts.split(','):
            prompts_file = prompts_file.strip()
            if os.path.exists(prompts_file):
                try:
                    if prompts_file.endswith('.json'):
                        with open(prompts_file, 'r', encoding='utf-8') as f:
                            file_prompts = json.load(f)
                            if isinstance(file_prompts, list):
                                prompts.extend(file_prompts)
                    elif prompts_file.endswith('.txt'):
                        with open(prompts_file, 'r', encoding='utf-8') as f:
                            file_prompts = [line.strip() for line in f if line.strip()]
                            prompts.extend(file_prompts)
                    logger.info(f"从 {prompts_file} 加载了 {len(file_prompts)} 个提示")
                except Exception as e:
                    logger.error(f"加载提示文件 {prompts_file} 时出错: {e}")
            else:
                logger.warning(f"提示文件 {prompts_file} 不存在")
        
        if prompts:
            datasets["generation"] = prompts
        else:
            # 默认提示
            default_prompts = [
                "请写一首关于人工智能的诗歌。",
                "解释量子计算的基本原理。",
                "描述一下你心目中理想的度假胜地。",
                "如果你可以穿越时空去任何地方，你会去哪里？为什么？",
                "请分析当今社会中科技发展对教育的影响。"
            ]
            datasets["generation"] = default_prompts
            logger.info(f"使用 {len(default_prompts)} 个默认提示")
    
    # 如果未提供特定数据集，尝试使用其他任务的数据集
    for task in tasks:
        if task not in datasets or not datasets[task]:
            if task == "perplexity" and "classification" in datasets:
                # 从分类数据集创建困惑度数据集
                texts = []
                for item in datasets["classification"]:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                if texts:
                    datasets["perplexity"] = texts
                    logger.info(f"从分类数据集创建了困惑度评估数据集，包含 {len(texts)} 个样本")
            
            elif task == "generation" and not args.generation_prompts:
                # 默认提示
                default_prompts = [
                    "请写一首关于人工智能的诗歌。",
                    "解释量子计算的基本原理。",
                    "描述一下你心目中理想的度假胜地。",
                    "如果你可以穿越时空去任何地方，你会去哪里？为什么？",
                    "请分析当今社会中科技发展对教育的影响。"
                ]
                datasets["generation"] = default_prompts
                logger.info(f"使用 {len(default_prompts)} 个默认提示")
    
    return datasets

# 计算困惑度
def calculate_perplexity(model, tokenizer, eval_dataset, device, args):
    logger.info("计算困惑度...")
    model.eval()
    total_loss = 0
    total_tokens = 0
    perplexities = []
    
    # 确保eval_dataset是正确的格式
    texts = []
    if isinstance(eval_dataset, Dataset):
        if "text" in eval_dataset.column_names:
            texts = eval_dataset["text"]
        elif "content" in eval_dataset.column_names:
            texts = eval_dataset["content"]
        elif "sentence" in eval_dataset.column_names:
            texts = eval_dataset["sentence"]
        else:
            # 尝试获取第一个列作为文本
            first_col = eval_dataset.column_names[0]
            texts = eval_dataset[first_col]
            logger.warning(f"未找到标准文本列，使用{first_col}列作为文本")
    else:
        # 如果数据集格式不是Dataset，尝试转换
        texts = eval_dataset
    
    if args.num_samples > 0 and args.num_samples < len(texts):
        texts = texts[:args.num_samples]
    
    # 将数据分成批次处理
    batch_size = args.batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="计算困惑度"):
            batch_texts = texts[i:i+batch_size]
            
            encodings = tokenizer(batch_texts, padding=True, truncation=True, 
                                 max_length=args.max_length, return_tensors="pt")
            input_ids = encodings.input_ids.to(device)
            attention_mask = encodings.attention_mask.to(device)
            
            # 创建标签，与输入相同但偏移一位
            labels = input_ids.clone()
            
            # 计算损失
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # 计算每个样本的困惑度
            for j in range(len(batch_texts)):
                sample_input_ids = input_ids[j]
                sample_loss = outputs.logits[j][:-1]
                sample_target = labels[j][1:]
                sample_length = attention_mask[j].sum().item()
                
                # 提取有效的logits和targets
                valid_indices = sample_target != tokenizer.pad_token_id
                valid_logits = sample_loss[valid_indices]
                valid_targets = sample_target[valid_indices]
                
                # 使用交叉熵损失计算样本级损失
                if len(valid_targets) > 0:
                    crit = torch.nn.CrossEntropyLoss(reduction='sum')
                    sample_tokens = len(valid_targets)
                    sample_loss_val = crit(valid_logits, valid_targets).item()
                    sample_perplexity = math.exp(sample_loss_val / sample_tokens)
                    perplexities.append(sample_perplexity)
                    
                    total_loss += sample_loss_val
                    total_tokens += sample_tokens
    
    # 计算整体困惑度
    if total_tokens > 0:
        avg_perplexity = math.exp(total_loss / total_tokens)
    else:
        avg_perplexity = float('inf')
        
    # 计算困惑度统计数据
    perplexity_stats = {
        "avg_perplexity": avg_perplexity,
        "median_perplexity": np.median(perplexities) if perplexities else float('inf'),
        "min_perplexity": min(perplexities) if perplexities else float('inf'),
        "max_perplexity": max(perplexities) if perplexities else float('inf'),
        "perplexity_std": np.std(perplexities) if perplexities else 0,
        "per_sample_perplexities": perplexities
    }
    
    return perplexity_stats

# 重构评估生成能力函数
def evaluate_generation(model, tokenizer, eval_prompts, device, args, use_accelerate=False):
    """评估生成能力"""
    logger.info("评估生成能力...")
    results = []
    
    for prompt in tqdm(eval_prompts, desc="生成文本"):
        try:
            # 使用公共生成方法
            system_prompt = "你是一个专业的文本生成助手，请根据提供的上下文生成符合要求的文本。"
            generated_text = generate_with_qwen_format(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=args.max_length,
                temperature=args.temperature
            )
                
            results.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "length": len(generated_text),
                "unique_words": len(set(generated_text.split()))
            })
        except Exception as e:
            logger.error(f"生成文本时出错: {str(e)}")
            results.append({
                "prompt": prompt,
                "generated_text": "生成错误",
                "length": 0,
                "unique_words": 0,
                "error": str(e)
            })
    
    # 计算生成文本的统计信息
    lengths = [r["length"] for r in results]
    unique_words = [r["unique_words"] for r in results]
    
    generation_stats = {
        "avg_length": np.mean(lengths),
        "median_length": np.median(lengths),
        "max_length": max(lengths),
        "min_length": min(lengths),
        "avg_unique_words": np.mean(unique_words),
        "lexical_diversity": np.mean([r["unique_words"]/r["length"] if r["length"] > 0 else 0 for r in results]),
        "generation_samples": results
    }
    
    return generation_stats

# 重构评估问答能力函数
def evaluate_qa(model, tokenizer, qa_dataset, device, args, use_accelerate=False):
    """评估问答能力"""
    logger.info("评估问答能力...")
    model.eval()
    
    # 提取问题和答案
    questions, answers = extract_qa_data(qa_dataset, args.num_samples)
    
    if not questions:
        logger.error("无法从数据集中提取问答对")
        return {"error": "无法从数据集中提取问答对"}
    
    # 评估问答
    rouge = Rouge()
    exact_matches = 0
    rouges = []
    results = []
    
    for question, expected_answer in tqdm(zip(questions, answers), desc="问答评估", total=len(questions)):
        try:
            # 使用公共生成方法
            system_prompt = "你是小说的阅读专家，请根据小说内容进行简要回答,无需回复与提问无关的内容和解释。"
            generated_answer = generate_with_qwen_format(
                model=model,
                tokenizer=tokenizer,
                prompt=question,
                system_prompt=system_prompt,
                max_new_tokens=args.max_length,
                temperature=0.3  # 问答使用低温度
            )
            
            # 计算指标
            exact_match = 1 if generated_answer.strip() == expected_answer.strip() else 0
            exact_matches += exact_match
            
            # 计算ROUGE分数
            try:
                rouge_scores = rouge.get_scores(generated_answer, expected_answer)[0]
                rouges.append(rouge_scores)
            except Exception as e:
                logger.warning(f"计算ROUGE时出错: {str(e)}")
                rouges.append({
                    "rouge-1": {"f": 0, "p": 0, "r": 0},
                    "rouge-2": {"f": 0, "p": 0, "r": 0},
                    "rouge-l": {"f": 0, "p": 0, "r": 0}
                })
            
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "exact_match": exact_match,
                "rouge_scores": rouge_scores if 'rouge_scores' in locals() else {}
            })
        except Exception as e:
            logger.error(f"处理问题时出错: {str(e)}")
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": "生成错误",
                "exact_match": 0,
                "rouge_scores": {},
                "error": str(e)
            })
    
    # 计算和聚合指标
    qa_stats = aggregate_qa_metrics(results, exact_matches, rouges, len(questions))
    return qa_stats

# 提取问答数据的辅助函数
def extract_qa_data(qa_dataset, num_samples):
    """从不同格式的数据集中提取问题和答案对"""
    questions = []
    answers = []
    
    if isinstance(qa_dataset, Dataset):
        if "question" in qa_dataset.column_names and "answer" in qa_dataset.column_names:
            questions = qa_dataset["question"]
            answers = qa_dataset["answer"]
        elif "question" in qa_dataset.column_names and "answers" in qa_dataset.column_names:
            questions = qa_dataset["question"]
            answers = [a[0] if isinstance(a, list) else a for a in qa_dataset["answers"]]
        elif "context" in qa_dataset.column_names and "question" in qa_dataset.column_names:
            questions = [f"上下文: {c}\n问题: {q}" for c, q in zip(qa_dataset["context"], qa_dataset["question"])]
            if "answers" in qa_dataset.column_names:
                answers = [a["text"][0] if isinstance(a, dict) and "text" in a else a for a in qa_dataset["answers"]]
            elif "answer" in qa_dataset.column_names:
                answers = qa_dataset["answer"]
    elif isinstance(qa_dataset, list) and len(qa_dataset) > 0:
        if "question" in qa_dataset[0] and "answer" in qa_dataset[0]:
            questions = [item["question"] for item in qa_dataset]
            answers = [item["answer"] for item in qa_dataset]
    
    # 限制样本数量
    if num_samples > 0 and num_samples < len(questions):
        questions = questions[:num_samples]
        answers = answers[:num_samples]
        
    return questions, answers

# 聚合问答指标的辅助函数
def aggregate_qa_metrics(results, exact_matches, rouges, total_questions):
    """聚合问答评估指标"""
    # 计算平均值
    exact_match_acc = exact_matches / total_questions if total_questions else 0
    
    # 聚合ROUGE分数
    rouge_1_f = np.mean([r["rouge-1"]["f"] for r in rouges]) if rouges else 0
    rouge_2_f = np.mean([r["rouge-2"]["f"] for r in rouges]) if rouges else 0
    rouge_l_f = np.mean([r["rouge-l"]["f"] for r in rouges]) if rouges else 0
    
    qa_stats = {
        "exact_match": exact_match_acc,
        "rouge-1-f": rouge_1_f,
        "rouge-2-f": rouge_2_f,
        "rouge-l-f": rouge_l_f,
        "samples": results
    }
    
    return qa_stats

# 重构评估分类能力函数
def evaluate_classification(model, tokenizer, classification_dataset, device, args, use_accelerate=False):
    """评估分类能力"""
    logger.info("评估分类能力...")
    model.eval()
    
    # 提取文本和标签
    texts, labels = extract_classification_data(classification_dataset, args.num_samples)
    
    if not texts:
        logger.error("无法从数据集中提取分类数据")
        return {"error": "无法从数据集中提取分类数据"}
    
    # 获取所有可能的标签
    unique_labels = list(set(labels))
    
    # 评估分类
    predictions = []
    results = []
    
    for text, true_label in tqdm(zip(texts, labels), desc="分类评估", total=len(texts)):
        try:
            # 构建分类提示
            prompt = f"请对以下文本进行分类，可选的类别有: {', '.join(unique_labels)}\n\n文本: {text}\n\n请直接回答类别名称，无需其他解释。"
            system_prompt = "你是一个专业的文本分类助手，请根据提供的选项做出准确分类。"
            
            # 使用公共生成方法
            generated_label = generate_with_qwen_format(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=min(50, args.max_length),
                temperature=0.1,  # 分类使用极低温度
                do_sample=False   # 分类不使用采样
            )
            
            # 清理生成的标签
            if generated_label not in unique_labels:
                generated_label = find_best_matching_label(generated_label, unique_labels)
            
            correct = 1 if str(generated_label).strip() == str(true_label).strip() else 0
            predictions.append(generated_label)
            
            results.append({
                "text": text,
                "true_label": true_label,
                "predicted_label": generated_label,
                "correct": correct
            })
        except Exception as e:
            logger.error(f"生成分类预测时出错: {str(e)}")
            predictions.append("")
            results.append({
                "text": text,
                "true_label": true_label,
                "predicted_label": "预测错误",
                "correct": 0,
                "error": str(e)
            })
    
    # 计算分类指标
    classification_stats = calculate_classification_metrics(results, predictions, labels, unique_labels)
    return classification_stats

# 提取分类数据的辅助函数
def extract_classification_data(classification_dataset, num_samples):
    """从不同格式的数据集中提取文本和标签对"""
    texts = []
    labels = []
    
    if isinstance(classification_dataset, Dataset):
        if "text" in classification_dataset.column_names and "label" in classification_dataset.column_names:
            texts = classification_dataset["text"]
            labels = classification_dataset["label"]
        elif "sentence" in classification_dataset.column_names and "label" in classification_dataset.column_names:
            texts = classification_dataset["sentence"]
            labels = classification_dataset["label"]
    elif isinstance(classification_dataset, list) and len(classification_dataset) > 0:
        if "text" in classification_dataset[0] and "label" in classification_dataset[0]:
            texts = [item["text"] for item in classification_dataset]
            labels = [item["label"] for item in classification_dataset]
    
    # 限制样本数量
    if num_samples > 0 and num_samples < len(texts):
        texts = texts[:num_samples]
        labels = labels[:num_samples]
        
    return texts, labels

# 找到最匹配的标签
def find_best_matching_label(generated_label, unique_labels):
    """找到生成标签最匹配的标准标签"""
    # 检查是否包含完整标签
    for label in unique_labels:
        if label in generated_label:
            return label
    
    # 如果没有完全匹配，取第一个单词或整个短语
    return generated_label.split()[0] if generated_label else ""

# 计算分类指标的辅助函数
def calculate_classification_metrics(results, predictions, labels, unique_labels):
    """计算分类评估指标"""
    # 计算基本指标
    accuracy = accuracy_score(labels, predictions) if predictions else 0
    
    try:
        precision = precision_score(labels, predictions, average="weighted", zero_division=0)
        recall = recall_score(labels, predictions, average="weighted", zero_division=0)
        f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    except Exception as e:
        logger.warning(f"计算分类指标时出错: {str(e)}")
        precision = recall = f1 = 0
    
    # 创建混淆矩阵
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(labels, predictions):
        confusion_matrix[true][pred] += 1
    
    # 转换为字典
    confusion_dict = {str(k): {str(k2): v2 for k2, v2 in v.items()} for k, v in confusion_matrix.items()}
    
    classification_stats = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_dict,
        "samples": results
    }
    
    return classification_stats

# 创建简化版可视化图表
def create_visualizations(evaluation_results, args, datasets):
    logger.info("创建极简版可视化图表...")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    visualizations = {}
    
    # 为所有任务创建单个核心指标图表
    core_metrics = {}
    dataset_info = {}
    
    # 提取每个任务的核心指标
    if "perplexity" in evaluation_results:
        core_metrics["困惑度(PPL)"] = evaluation_results["perplexity"]["avg_perplexity"]
        dataset_size = len(evaluation_results["perplexity"]["per_sample_perplexities"])
        dataset_info["困惑度(PPL)"] = f"样本数: {dataset_size}"
    
    if "qa" in evaluation_results:
        # 仅保留ROUGE-L指标
        core_metrics["问答(ROUGE-L)"] = evaluation_results["qa"]["rouge-l-f"]
        dataset_size = len(evaluation_results["qa"]["samples"])
        dataset_info["问答(ROUGE-L)"] = f"样本数: {dataset_size}"
    
    if "classification" in evaluation_results:
        # 仅保留准确率
        core_metrics["分类(准确率)"] = evaluation_results["classification"]["accuracy"]
        dataset_size = len(evaluation_results["classification"]["samples"])
        dataset_info["分类(准确率)"] = f"样本数: {dataset_size}"
        
    if "generation" in evaluation_results:
        # 生成任务使用词汇多样性指标
        core_metrics["生成(词汇多样性)"] = evaluation_results["generation"]["lexical_diversity"]
        dataset_size = len(evaluation_results["generation"]["generation_samples"])
        dataset_info["生成(词汇多样性)"] = f"样本数: {dataset_size}"
    
    if core_metrics:
        # 创建统一的核心指标图
        plt.figure(figsize=(10, 6))
        
        # 分离困惑度和其他指标（因为困惑度通常值较大）
        ppl_metrics = {k: v for k, v in core_metrics.items() if "困惑度" in k}
        other_metrics = {k: v for k, v in core_metrics.items() if "困惑度" not in k}
        
        # 绘制非困惑度指标的条形图
        idx = np.arange(len(other_metrics))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(other_metrics)]
        bars = plt.bar(idx, list(other_metrics.values()), color=colors)
        plt.xticks(idx, list(other_metrics.keys()), rotation=15)
        plt.ylim(0, min(1.1, max(list(other_metrics.values())) * 1.2))  # 调整Y轴范围
        plt.title("模型评估核心指标")
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数值和数据集信息标签
        for i, bar in enumerate(bars):
            metric_name = list(other_metrics.keys())[i]
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 在底部添加数据集信息
            if metric_name in dataset_info:
                plt.text(bar.get_x() + bar.get_width()/2., -0.05,
                        dataset_info[metric_name], ha='center', va='top', 
                        fontsize=8, rotation=15, color='darkblue')
        
        # 如果有困惑度，添加单独的文本框
        if ppl_metrics:
            ppl_key = list(ppl_metrics.keys())[0]
            ppl_value = ppl_metrics[ppl_key]
            ppl_dataset = dataset_info.get(ppl_key, "")
            
            # 在图表底部添加困惑度文本框
            bbox_props = dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="orange", alpha=0.7)
            plt.figtext(0.5, 0.01, 
                      f"{ppl_key}: {ppl_value:.2f}   {ppl_dataset}", 
                      ha="center", fontsize=10, bbox=bbox_props)
        
        summary_plot_path = os.path.join(args.output_dir, "core_metrics.png")
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(bottom=0.15)  # 为底部文本提供空间
        plt.savefig(summary_plot_path)
        plt.close()
        visualizations["core_metrics"] = summary_plot_path
    
    # 保存可视化信息
    visualizations_json_path = os.path.join(args.output_dir, "visualizations.json")
    with open(visualizations_json_path, "w") as f:
        json.dump(visualizations, f, indent=2)
    
    return visualizations

# 优化wandb上报函数，重点显示采样数据
def upload_to_wandb(evaluation_results, args, datasets):
    logger.info("正在将模型评估采样结果上传到wandb...")
    
    # 初始化wandb
    run_name = args.wandb_name or f"eval-{args.model_path.split('/')[-1]}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args)
    )
    
    # 记录核心评估指标
    metrics = {}
    
    # 基本信息
    model_info = {
        "model_path": args.model_path,
        "tasks": args.tasks
    }
    wandb.config.update(model_info)
    
    # 上报问答任务结果
    if "qa" in evaluation_results:
        metrics["qa/rouge-l"] = evaluation_results["qa"]["rouge-l-f"]
        metrics["qa/exact_match"] = evaluation_results["qa"]["exact_match"]
        
        # 上报多个有代表性的问答样本
        samples = evaluation_results["qa"]["samples"]
        if samples:
            # 选择几个典型样本：最佳、中等和最差的样本
            try:
                # 按rouge-l分数排序
                sorted_samples = sorted(
                    samples, 
                    key=lambda x: x.get("rouge_scores", {}).get("rouge-l", {}).get("f", 0) 
                    if isinstance(x.get("rouge_scores", {}), dict) else 0
                )
                
                # 最多选择5个样本展示
                max_samples = min(5, len(sorted_samples))
                step_size = max(1, len(sorted_samples) // max_samples)
                
                qa_examples = []
                # 选择分布均匀的样本
                for i in range(0, len(sorted_samples), step_size):
                    if len(qa_examples) >= max_samples:
                        break
                    sample = sorted_samples[i]
                    
                    # 获取rouge-l得分
                    rouge_l = sample.get("rouge_scores", {}).get("rouge-l", {}).get("f", 0)
                    if isinstance(rouge_l, float):
                        rouge_l = f"{rouge_l:.3f}"
                        
                    qa_examples.append([
                        sample["question"],
                        sample["expected_answer"],
                        sample["generated_answer"],
                        sample["exact_match"],
                        rouge_l
                    ])
            except Exception as e:
                # 如果排序失败，直接选择前几个样本
                logger.warning(f"样本排序失败: {e}")
                qa_examples = []
                for i, sample in enumerate(samples[:max_samples]):
                    qa_examples.append([
                        sample["question"],
                        sample["expected_answer"],
                        sample["generated_answer"],
                        sample["exact_match"],
                        "N/A"
                    ])
            
            metrics["qa/samples"] = wandb.Table(
                columns=["问题", "标准答案", "模型生成答案", "精确匹配", "ROUGE-L"],
                data=qa_examples
            )
    
    # 上报分类任务结果
    if "classification" in evaluation_results:
        metrics["classification/accuracy"] = evaluation_results["classification"]["accuracy"]
        metrics["classification/f1"] = evaluation_results["classification"]["f1"]
        
        # 上报分类样本
        samples = evaluation_results["classification"]["samples"]
        if samples:
            # 选择一些典型样本：正确和错误的样本
            classification_examples = []
            correct_samples = [s for s in samples if s["correct"] == 1]
            wrong_samples = [s for s in samples if s["correct"] == 0]
            
            # 最多选择6个样本：3个正确，3个错误
            for i in range(min(3, len(correct_samples))):
                sample = correct_samples[i]
                classification_examples.append([
                    sample["text"][:100] + "..." if len(sample["text"]) > 100 else sample["text"],
                    sample["true_label"],
                    sample["predicted_label"],
                    "✓" if sample["correct"] else "✗"
                ])
            
            for i in range(min(3, len(wrong_samples))):
                sample = wrong_samples[i]
                classification_examples.append([
                    sample["text"][:100] + "..." if len(sample["text"]) > 100 else sample["text"],
                    sample["true_label"],
                    sample["predicted_label"],
                    "✓" if sample["correct"] else "✗"
                ])
            
            metrics["classification/samples"] = wandb.Table(
                columns=["文本", "真实标签", "预测标签", "是否正确"],
                data=classification_examples
            )
            
            # 尝试上报混淆矩阵
            if "confusion_matrix" in evaluation_results["classification"]:
                confusion_data = []
                for true_label, predictions in evaluation_results["classification"]["confusion_matrix"].items():
                    for pred_label, count in predictions.items():
                        confusion_data.append({"真实标签": true_label, "预测标签": pred_label, "数量": count})
                
                if confusion_data:
                    confusion_df = pd.DataFrame(confusion_data)
                    metrics["classification/confusion_matrix"] = wandb.Table(
                        dataframe=confusion_df
                    )
    
    # 上报困惑度评估结果
    if "perplexity" in evaluation_results:
        metrics["perplexity/avg"] = evaluation_results["perplexity"]["avg_perplexity"]
        
        # 上报困惑度分布直方图
        perplexities = evaluation_results["perplexity"]["per_sample_perplexities"]
        if perplexities:
            # 过滤掉无穷值
            valid_perplexities = [p for p in perplexities if not math.isinf(p)]
            if valid_perplexities:
                # 上报困惑度直方图
                metrics["perplexity/distribution"] = wandb.Histogram(valid_perplexities)
    
    # 上报生成能力评估结果
    if "generation" in evaluation_results:
        metrics["generation/lexical_diversity"] = evaluation_results["generation"]["lexical_diversity"]
        metrics["generation/avg_length"] = evaluation_results["generation"]["avg_length"]
        
        # 上报生成样本
        samples = evaluation_results["generation"]["generation_samples"]
        if samples:
            # 选择几个典型样本：最长、中等长度和最短的样本
            try:
                # 按长度排序
                sorted_samples = sorted(samples, key=lambda x: x["length"])
                
                # 最多选择5个样本
                max_samples = min(5, len(sorted_samples))
                step_size = max(1, len(sorted_samples) // max_samples)
                
                generation_examples = []
                # 选择分布均匀的样本
                for i in range(0, len(sorted_samples), step_size):
                    if len(generation_examples) >= max_samples:
                        break
                    sample = sorted_samples[i]
                    generation_examples.append([
                        sample["prompt"],
                        sample["generated_text"],
                        sample["length"],
                        sample["unique_words"] / sample["length"] if sample["length"] > 0 else 0
                    ])
            except Exception as e:
                # 如果排序失败，直接选择前几个样本
                logger.warning(f"样本排序失败: {e}")
                generation_examples = []
                for i, sample in enumerate(samples[:max_samples]):
                    generation_examples.append([
                        sample["prompt"],
                        sample["generated_text"],
                        sample["length"],
                        sample["unique_words"] / sample["length"] if sample["length"] > 0 else 0
                    ])
            
            metrics["generation/samples"] = wandb.Table(
                columns=["提示", "生成文本", "长度", "词汇多样性"],
                data=generation_examples
            )
    
    # 创建核心指标摘要
    task_counts = {}
    for task in ["qa", "classification", "perplexity", "generation"]:
        if task in evaluation_results:
            if task == "qa":
                task_counts[task] = len(evaluation_results[task]["samples"])
            elif task == "classification":
                task_counts[task] = len(evaluation_results[task]["samples"])
            elif task == "perplexity":
                task_counts[task] = len(evaluation_results[task]["per_sample_perplexities"])
            elif task == "generation":
                task_counts[task] = len(evaluation_results[task]["generation_samples"])
    
    summary_text = f"<h3>模型评估结果: {args.model_path}</h3><br>"
    summary_text += "<table style='width:100%; border:1px solid #ddd;'>"
    summary_text += "<tr style='background-color:#f8f8f8;'><th>任务</th><th>核心指标</th><th>样本数</th></tr>"
    
    if "perplexity" in evaluation_results:
        ppl = evaluation_results["perplexity"]["avg_perplexity"]
        summary_text += f"<tr><td>困惑度</td><td>{ppl:.3f}</td><td>{task_counts.get('perplexity', 'N/A')}</td></tr>"
    
    if "qa" in evaluation_results:
        rouge_l = evaluation_results["qa"]["rouge-l-f"]
        exact_match = evaluation_results["qa"]["exact_match"]
        summary_text += f"<tr><td>问答</td><td>ROUGE-L: {rouge_l:.3f}<br>精确匹配: {exact_match:.3f}</td><td>{task_counts.get('qa', 'N/A')}</td></tr>"
    
    if "classification" in evaluation_results:
        acc = evaluation_results["classification"]["accuracy"]
        f1 = evaluation_results["classification"]["f1"]
        summary_text += f"<tr><td>分类</td><td>准确率: {acc:.3f}<br>F1分数: {f1:.3f}</td><td>{task_counts.get('classification', 'N/A')}</td></tr>"
    
    if "generation" in evaluation_results:
        lex_div = evaluation_results["generation"]["lexical_diversity"]
        avg_len = evaluation_results["generation"]["avg_length"]
        summary_text += f"<tr><td>生成</td><td>词汇多样性: {lex_div:.3f}<br>平均长度: {avg_len:.1f}</td><td>{task_counts.get('generation', 'N/A')}</td></tr>"
    
    summary_text += "</table>"
    
    # 添加摘要 HTML
    metrics["summary"] = wandb.Html(summary_text)
    
    # 记录所有指标
    wandb.log(metrics)
    
    # 上传原始结果文件
    results_json = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_json, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    wandb.save(results_json)
    
    # 结束wandb运行
    wandb.finish()

# 修复后的保存评估结果函数
def save_evaluation_results(evaluation_results, args, datasets):
    """仅保存评估结果JSON，不创建可视化"""
    # 保存评估结果
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        # 过滤掉不可序列化的对象
        filtered_results = {}
        for task, result in evaluation_results.items():
            if task == "perplexity":
                # 过滤掉可能的无穷值
                result["per_sample_perplexities"] = [p for p in result["per_sample_perplexities"] if not math.isinf(p)]
            filtered_results[task] = result
        
        json.dump(filtered_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估结果已保存到 {results_file}")
    
    # 如果启用wandb，直接上传结果
    if args.use_wandb:
        upload_to_wandb(evaluation_results, args, datasets)
        logger.info("结果已上传到wandb")
    
    # 保存生成文本示例（如果需要）
    if args.save_generations and "generation" in evaluation_results:
        save_generated_texts(evaluation_results["generation"]["generation_samples"], args.output_dir)

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备和加载模型
    device, model, tokenizer, use_accelerate = load_model_and_tokenizer(args)
    if model is None:
        return
    
    # 加载数据集
    datasets = load_evaluation_datasets(args)
    
    # 执行评估任务
    evaluation_results = run_evaluations(model, tokenizer, datasets, device, args, use_accelerate)
    
    # 仅保存结果，不创建本地可视化
    save_evaluation_results(evaluation_results, args, datasets)
    
    logger.info("评估完成!")

# 加载模型和分词器
def load_model_and_tokenizer(args):
    """加载模型和分词器"""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model_kwargs = {}
        use_accelerate = False
        
        if args.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        
        # 根据设备类型决定如何加载模型
        if device.type == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                device_map="auto",
                **model_kwargs
            )
            use_accelerate = True
            logger.info("使用accelerate自动设备映射加载模型")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                **model_kwargs
            )
            model = model.to(device)
            logger.info(f"模型已加载到 {device}")
        
        return device, model, tokenizer, use_accelerate
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        return device, None, None, False

# 运行所有评估任务
def run_evaluations(model, tokenizer, datasets, device, args, use_accelerate):
    """运行所有选定的评估任务"""
    evaluation_results = {}
    tasks = [task.strip() for task in args.tasks.split(",")]
    
    task_evaluators = {
        "perplexity": lambda: calculate_perplexity(model, tokenizer, datasets["perplexity"], device, args),
        "generation": lambda: evaluate_generation(model, tokenizer, datasets["generation"], device, args, use_accelerate),
        "qa": lambda: evaluate_qa(model, tokenizer, datasets["qa"], device, args, use_accelerate),
        "classification": lambda: evaluate_classification(model, tokenizer, datasets["classification"], device, args, use_accelerate)
    }
    
    for task in tasks:
        if task in datasets and datasets[task]:
            logger.info(f"开始{task}评估...")
            if task in task_evaluators:
                evaluation_results[task] = task_evaluators[task]()
        else:
            logger.warning(f"未找到{task}任务的数据集，跳过评估")
    
    return evaluation_results

# 保存和可视化结果 - 更新调用
def save_and_visualize_results(evaluation_results, args, datasets):
    """保存评估结果并创建可视化"""
    # 保存评估结果
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        # 过滤掉不可序列化的对象
        filtered_results = {}
        for task, result in evaluation_results.items():
            if task == "perplexity":
                # 过滤掉可能的无穷值
                result["per_sample_perplexities"] = [p for p in result["per_sample_perplexities"] if not math.isinf(p)]
            filtered_results[task] = result
        
        json.dump(filtered_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估结果已保存到 {results_file}")
    
    # 创建可视化
    visualizations = create_visualizations(evaluation_results, args, datasets)
    
    # 如果启用wandb，上传结果
    if args.use_wandb:
        upload_to_wandb(evaluation_results, args, datasets)
        logger.info("结果已上传到wandb")
    
    # 保存生成文本示例
    if args.save_generations and "generation" in evaluation_results:
        save_generated_texts(evaluation_results["generation"]["generation_samples"], args.output_dir)

# 保存生成文本示例
def save_generated_texts(samples, output_dir):
    """保存生成的文本示例到文件"""
    generations_file = os.path.join(output_dir, "generated_texts.txt")
    with open(generations_file, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(samples):
            f.write(f"提示 {i+1}: {sample['prompt']}\n")
            f.write(f"生成: {sample['generated_text']}\n")
            f.write("-" * 50 + "\n")
    
    logger.info(f"生成文本已保存到 {generations_file}")

if __name__ == "__main__":
    main() 