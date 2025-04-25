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
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from rouge import Rouge
import pandas as pd
import seaborn as sns
from collections import defaultdict
import glob

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

# 评估生成能力
def evaluate_generation(model, tokenizer, eval_prompts, device, args, use_accelerate=False):
    logger.info("评估生成能力...")
    
    # 根据是否使用accelerate决定如何创建pipeline
    if use_accelerate:
        pipeline = TextGenerationPipeline(
            model=model, 
            tokenizer=tokenizer,
        )
    else:
        pipeline = TextGenerationPipeline(
            model=model, 
            tokenizer=tokenizer,
        )
    
    results = []
    for prompt in tqdm(eval_prompts, desc="生成文本"):
        try:
            outputs = pipeline(
                prompt,
                max_length=args.max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=args.temperature
            )
            
            generated_text = outputs[0]["generated_text"]
            # 去除提示部分，只保留生成的内容
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
                
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

# 评估问答能力
def evaluate_qa(model, tokenizer, qa_dataset, device, args, use_accelerate):
    logger.info("评估问答能力...")
    model.eval()
    results = []
    
    # 修改这部分代码，不再指定device参数
    pipeline = TextGenerationPipeline(
        model=model, 
        tokenizer=tokenizer,
    )
    
    # 确保qa_dataset有正确的格式
    questions = []
    answers = []
    
    if isinstance(qa_dataset, Dataset):
        # 尝试不同的列名格式
        if "question" in qa_dataset.column_names and "answer" in qa_dataset.column_names:
            questions = qa_dataset["question"]
            answers = qa_dataset["answer"]
        elif "question" in qa_dataset.column_names and "answers" in qa_dataset.column_names:
            questions = qa_dataset["question"]
            # 如果answers是列表，取第一个答案
            answers = [a[0] if isinstance(a, list) else a for a in qa_dataset["answers"]]
        else:
            logger.warning("未找到标准问答列名，尝试使用上下文进行问答评估")
            if "context" in qa_dataset.column_names and "question" in qa_dataset.column_names:
                questions = [f"上下文: {c}\n问题: {q}" for c, q in zip(qa_dataset["context"], qa_dataset["question"])]
                # 尝试不同格式的答案
                if "answers" in qa_dataset.column_names:
                    answers = [a["text"][0] if isinstance(a, dict) and "text" in a else a for a in qa_dataset["answers"]]
                elif "answer" in qa_dataset.column_names:
                    answers = qa_dataset["answer"]
                else:
                    logger.error("未找到答案列")
                    return {"error": "未找到答案列"}
            else:
                logger.error("数据集格式不兼容问答任务")
                return {"error": "数据集格式不兼容问答任务"}
    else:
        # 如果不是Dataset对象，则假设是包含问题和答案的字典列表
        if isinstance(qa_dataset, list) and len(qa_dataset) > 0:
            if "question" in qa_dataset[0] and "answer" in qa_dataset[0]:
                questions = [item["question"] for item in qa_dataset]
                answers = [item["answer"] for item in qa_dataset]
            else:
                logger.error("数据集格式不兼容问答任务")
                return {"error": "数据集格式不兼容问答任务"}
        else:
            logger.error("无效的问答数据集")
            return {"error": "无效的问答数据集"}
    
    # 限制样本数量
    if args.num_samples > 0 and args.num_samples < len(questions):
        questions = questions[:args.num_samples]
        answers = answers[:args.num_samples]
    
    # 评估问答
    rouge = Rouge()
    exact_matches = 0
    rouges = []
    
    for question, expected_answer in tqdm(zip(questions, answers), desc="问答评估", total=len(questions)):
        try:
            # 添加提示以获得更好的QA结果
            prompt = f"问题: {question}\n回答: "
            
            outputs = pipeline(
                prompt,
                max_length=args.max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.3  # 降低温度以获得更确定的答案
            )
            
            generated_text = outputs[0]["generated_text"]
            # 提取生成的答案部分
            if "回答: " in generated_text:
                generated_answer = generated_text.split("回答: ", 1)[1].strip()
            else:
                generated_answer = generated_text[len(prompt):].strip()
            
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
    
    # 计算平均值
    exact_match_acc = exact_matches / len(questions) if questions else 0
    
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

# 评估分类能力
def evaluate_classification(model, tokenizer, classification_dataset, device, args, use_accelerate):
    logger.info("评估分类能力...")
    model.eval()
    pipeline = TextGenerationPipeline(
        model=model, 
        tokenizer=tokenizer,
    )
    
    # 提取数据集
    texts = []
    labels = []
    
    if isinstance(classification_dataset, Dataset):
        if "text" in classification_dataset.column_names and "label" in classification_dataset.column_names:
            texts = classification_dataset["text"]
            labels = classification_dataset["label"]
        elif "sentence" in classification_dataset.column_names and "label" in classification_dataset.column_names:
            texts = classification_dataset["sentence"]
            labels = classification_dataset["label"]
        else:
            logger.error("分类数据集格式不兼容")
            return {"error": "分类数据集格式不兼容"}
    elif isinstance(classification_dataset, list) and len(classification_dataset) > 0:
        if "text" in classification_dataset[0] and "label" in classification_dataset[0]:
            texts = [item["text"] for item in classification_dataset]
            labels = [item["label"] for item in classification_dataset]
        else:
            logger.error("分类数据集格式不兼容")
            return {"error": "分类数据集格式不兼容"}
    else:
        logger.error("无效的分类数据集")
        return {"error": "无效的分类数据集"}
    
    # 限制样本数量
    if args.num_samples > 0 and args.num_samples < len(texts):
        texts = texts[:args.num_samples]
        labels = labels[:args.num_samples]
    
    # 获取所有可能的标签
    unique_labels = list(set(labels))
    
    # 评估分类
    predictions = []
    results = []
    
    for text, true_label in tqdm(zip(texts, labels), desc="分类评估", total=len(texts)):
        try:
            # 构建分类提示
            prompt = f"文本: {text}\n分类任务，请选择最合适的标签: {', '.join(unique_labels)}\n答案: "
            
            outputs = pipeline(
                prompt,
                max_length=args.max_length,
                num_return_sequences=1,
                do_sample=False  # 分类任务不使用采样
            )
            
            generated_text = outputs[0]["generated_text"]
            # 提取生成的答案部分
            if "答案: " in generated_text:
                generated_label = generated_text.split("答案: ", 1)[1].strip()
            else:
                generated_label = generated_text[len(prompt):].strip()
            
            # 清理生成的标签
            # 如果生成的标签不在预定义标签列表中，找出最相似的
            if generated_label not in unique_labels:
                # 找出最相似的标签
                for label in unique_labels:
                    if label in generated_label:
                        generated_label = label
                        break
                else:
                    # 如果仍未找到匹配项，取第一个单词或整个短语
                    generated_label = generated_label.split()[0] if generated_label else ""
            
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
    accuracy = accuracy_score(labels, predictions) if predictions else 0
    
    # 尝试计算其他指标（如果数据是离散的）
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

# 创建量化评估结果的可视化
def create_visualizations(evaluation_results, args):
    logger.info("创建可视化图表...")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    visualizations = {}
    
    # 困惑度分布图
    if "perplexity" in evaluation_results:
        perplexities = evaluation_results["perplexity"]["per_sample_perplexities"]
        plt.figure(figsize=(10, 6))
        plt.hist(perplexities, bins=30, alpha=0.7)
        plt.axvline(evaluation_results["perplexity"]["avg_perplexity"], color='r', linestyle='dashed', linewidth=1)
        plt.title("Perplexity Distribution")
        plt.xlabel("Perplexity")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        
        # 添加平均值标签
        plt.text(evaluation_results["perplexity"]["avg_perplexity"] * 1.1, 
                plt.ylim()[1] * 0.9, 
                f'Avg: {evaluation_results["perplexity"]["avg_perplexity"]:.2f}', 
                color='red')
        
        perplexity_plot_path = os.path.join(args.output_dir, "perplexity_distribution.png")
        plt.savefig(perplexity_plot_path)
        plt.close()
        visualizations["perplexity_distribution"] = perplexity_plot_path
    
    # 生成文本长度分布
    if "generation" in evaluation_results:
        lengths = [sample["length"] for sample in evaluation_results["generation"]["generation_samples"]]
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=20, alpha=0.7)
        plt.axvline(evaluation_results["generation"]["avg_length"], color='r', linestyle='dashed', linewidth=1)
        plt.title("Generated Text Length Distribution")
        plt.xlabel("Length")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        
        length_plot_path = os.path.join(args.output_dir, "generation_length_distribution.png")
        plt.savefig(length_plot_path)
        plt.close()
        visualizations["generation_length_distribution"] = length_plot_path
    
    # 问答任务的ROUGE分数
    if "qa" in evaluation_results:
        rouge_scores = {
            "ROUGE-1": evaluation_results["qa"]["rouge-1-f"],
            "ROUGE-2": evaluation_results["qa"]["rouge-2-f"],
            "ROUGE-L": evaluation_results["qa"]["rouge-l-f"],
            "Exact Match": evaluation_results["qa"]["exact_match"]
        }
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(rouge_scores.keys(), rouge_scores.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title("Question Answering Metrics")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        qa_plot_path = os.path.join(args.output_dir, "qa_metrics.png")
        plt.savefig(qa_plot_path)
        plt.close()
        visualizations["qa_metrics"] = qa_plot_path
    
    # 分类指标
    if "classification" in evaluation_results:
        classification_metrics = {
            "Accuracy": evaluation_results["classification"]["accuracy"],
            "Precision": evaluation_results["classification"]["precision"],
            "Recall": evaluation_results["classification"]["recall"],
            "F1 Score": evaluation_results["classification"]["f1"]
        }
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(classification_metrics.keys(), classification_metrics.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.title("Classification Metrics")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        classification_plot_path = os.path.join(args.output_dir, "classification_metrics.png")
        plt.savefig(classification_plot_path)
        plt.close()
        visualizations["classification_metrics"] = classification_plot_path
        
        # 混淆矩阵热图
        if evaluation_results["classification"]["confusion_matrix"]:
            # 转换为pandas DataFrame
            confusion_data = []
            for true_label, predictions in evaluation_results["classification"]["confusion_matrix"].items():
                for pred_label, count in predictions.items():
                    confusion_data.append({"True": true_label, "Predicted": pred_label, "Count": count})
            
            if confusion_data:
                confusion_df = pd.DataFrame(confusion_data)
                confusion_pivot = confusion_df.pivot(index="True", columns="Predicted", values="Count")
                confusion_pivot = confusion_pivot.fillna(0)
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(confusion_pivot, annot=True, fmt=".0f", cmap="Blues")
                plt.title("Confusion Matrix")
                plt.tight_layout()
                
                confusion_matrix_path = os.path.join(args.output_dir, "confusion_matrix.png")
                plt.savefig(confusion_matrix_path)
                plt.close()
                visualizations["confusion_matrix"] = confusion_matrix_path
    
    # 保存可视化信息
    visualizations_json_path = os.path.join(args.output_dir, "visualizations.json")
    with open(visualizations_json_path, "w") as f:
        json.dump(visualizations, f, indent=2)
    
    return visualizations

# 将结果上传到wandb
def upload_to_wandb(evaluation_results, visualizations, args):
    logger.info("正在将结果上传到wandb...")
    
    # 初始化wandb
    run_name = args.wandb_name or f"eval-{args.model_path.split('/')[-1]}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args)
    )
    
    # 上传结果指标
    metrics = {}
    
    # 困惑度指标
    if "perplexity" in evaluation_results:
        metrics["perplexity/avg"] = evaluation_results["perplexity"]["avg_perplexity"]
        metrics["perplexity/median"] = evaluation_results["perplexity"]["median_perplexity"]
        metrics["perplexity/min"] = evaluation_results["perplexity"]["min_perplexity"]
        metrics["perplexity/max"] = evaluation_results["perplexity"]["max_perplexity"]
        metrics["perplexity/std"] = evaluation_results["perplexity"]["perplexity_std"]
        
        # 上传困惑度分布图
        if "perplexity_distribution" in visualizations:
            metrics["perplexity/distribution"] = wandb.Image(visualizations["perplexity_distribution"])

    # 生成能力指标
    if "generation" in evaluation_results:
        metrics["generation/avg_length"] = evaluation_results["generation"]["avg_length"]
        metrics["generation/median_length"] = evaluation_results["generation"]["median_length"]
        metrics["generation/lexical_diversity"] = evaluation_results["generation"]["lexical_diversity"]
        
        # 上传生成文本长度分布图
        if "generation_length_distribution" in visualizations:
            metrics["generation/length_distribution"] = wandb.Image(visualizations["generation_length_distribution"])
        
        # 上传生成示例
        generation_examples = []
        for i, sample in enumerate(evaluation_results["generation"]["generation_samples"][:10]):  # 取前10个示例
            generation_examples.append([
                sample["prompt"],
                sample["generated_text"]
            ])
        
        metrics["generation/examples"] = wandb.Table(
            columns=["Prompt", "Generated Text"],
            data=generation_examples
        )

    # 问答能力指标
    if "qa" in evaluation_results:
        metrics["qa/exact_match"] = evaluation_results["qa"]["exact_match"]
        metrics["qa/rouge-1-f"] = evaluation_results["qa"]["rouge-1-f"]
        metrics["qa/rouge-2-f"] = evaluation_results["qa"]["rouge-2-f"]
        metrics["qa/rouge-l-f"] = evaluation_results["qa"]["rouge-l-f"]
        
        # 上传问答指标图
        if "qa_metrics" in visualizations:
            metrics["qa/metrics_chart"] = wandb.Image(visualizations["qa_metrics"])
        
        # 上传问答示例
        qa_examples = []
        for i, sample in enumerate(evaluation_results["qa"]["samples"][:10]):  # 取前10个示例
            qa_examples.append([
                sample["question"],
                sample["expected_answer"],
                sample["generated_answer"],
                sample["exact_match"]
            ])
        
        metrics["qa/examples"] = wandb.Table(
            columns=["Question", "Expected Answer", "Generated Answer", "Exact Match"],
            data=qa_examples
        )

    # 分类能力指标
    if "classification" in evaluation_results:
        metrics["classification/accuracy"] = evaluation_results["classification"]["accuracy"]
        metrics["classification/precision"] = evaluation_results["classification"]["precision"]
        metrics["classification/recall"] = evaluation_results["classification"]["recall"]
        metrics["classification/f1"] = evaluation_results["classification"]["f1"]
        
        # 上传分类指标图
        if "classification_metrics" in visualizations:
            metrics["classification/metrics_chart"] = wandb.Image(visualizations["classification_metrics"])
        
        # 上传混淆矩阵
        if "confusion_matrix" in visualizations:
            metrics["classification/confusion_matrix"] = wandb.Image(visualizations["confusion_matrix"])
        
        # 上传分类示例
        classification_examples = []
        for i, sample in enumerate(evaluation_results["classification"]["samples"][:10]):  # 取前10个示例
            classification_examples.append([
                sample["text"],
                sample["true_label"],
                sample["predicted_label"],
                sample["correct"]
            ])
        
        metrics["classification/examples"] = wandb.Table(
            columns=["Text", "True Label", "Predicted Label", "Correct"],
            data=classification_examples
        )

    # 记录所有指标
    wandb.log(metrics)

    # 上传结果文件
    results_json = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_json, 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    wandb.save(results_json)

    # 结束wandb运行
    wandb.finish()

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型和分词器
    logger.info(f"加载模型: {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model_kwargs = {}
        use_accelerate = False  # 添加标志跟踪是否使用accelerate
        
        if args.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        
        # 根据设备类型决定如何加载模型
        if device.type == "cuda":
            # 在CUDA环境中，使用自动设备映射
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                device_map="auto",  # 使用accelerate自动管理设备
                **model_kwargs
            )
            use_accelerate = True
            logger.info("使用accelerate自动设备映射加载模型")
        else:
            # 在CPU环境中，直接加载模型到CPU
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                **model_kwargs
            )
            model = model.to(device)
            logger.info(f"模型已加载到 {device}")
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        return
    
    # 加载数据集
    datasets = load_evaluation_datasets(args)
    
    # 执行评估任务
    evaluation_results = {}
    tasks = [task.strip() for task in args.tasks.split(",")]
    
    for task in tasks:
        if task in datasets and datasets[task]:
            if task == "perplexity":
                logger.info("开始困惑度评估...")
                evaluation_results["perplexity"] = calculate_perplexity(
                    model, tokenizer, datasets["perplexity"], device, args
                )
            
            elif task == "generation":
                logger.info("开始生成能力评估...")
                # 传递use_accelerate标志
                evaluation_results["generation"] = evaluate_generation(
                    model, tokenizer, datasets["generation"], device, args, use_accelerate
                )
            
            elif task == "qa":
                logger.info("开始问答能力评估...")
                # 传递use_accelerate标志
                evaluation_results["qa"] = evaluate_qa(
                    model, tokenizer, datasets["qa"], device, args, use_accelerate
                )
            
            elif task == "classification":
                logger.info("开始分类能力评估...")
                # 传递use_accelerate标志
                evaluation_results["classification"] = evaluate_classification(
                    model, tokenizer, datasets["classification"], device, args, use_accelerate
                )
        else:
            logger.warning(f"未找到{task}任务的数据集，跳过评估")
    
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
    visualizations = create_visualizations(evaluation_results, args)
    
    # 如果启用wandb，上传结果
    if args.use_wandb:
        upload_to_wandb(evaluation_results, visualizations, args)
        logger.info("结果已上传到wandb")
    
    # 保存生成文本示例
    if args.save_generations and "generation" in evaluation_results:
        generations_file = os.path.join(args.output_dir, "generated_texts.txt")
        with open(generations_file, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(evaluation_results["generation"]["generation_samples"]):
                f.write(f"提示 {i+1}: {sample['prompt']}\n")
                f.write(f"生成: {sample['generated_text']}\n")
                f.write("-" * 50 + "\n")
        
        logger.info(f"生成文本已保存到 {generations_file}")
    
    logger.info("评估完成!")

if __name__ == "__main__":
    main() 