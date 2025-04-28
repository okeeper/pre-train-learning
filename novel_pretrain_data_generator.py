#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用本地千问大语言模型对小说章节进行数据扩充，生成C4格式预训练数据
优化版：处理章节分片并提高长文本理解能力
"""

import json
import os
import time
import logging
import random
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import jsonlines

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("novel_pretrain_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NovelPretrainGenerator:
    """使用大语言模型扩充小说数据，生成预训练数据集"""
    
    def __init__(
        self, 
        input_file: str,
        output_dir: str,
        model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
        batch_size: int = 1,
        max_length: int = 4096,
        temperature: float = 0.7,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_tokens_per_chapter: int = 3000,  # 每个章节的最大标记数
    ):
        """初始化生成器
        
        Args:
            input_file: 输入的小说章节JSON文件路径
            output_dir: 输出预训练数据的目录
            model_name_or_path: 使用的模型名称或路径
            batch_size: 批处理大小
            max_length: 生成文本的最大长度
            temperature: 生成温度
            device: 使用的设备
            max_tokens_per_chapter: 每个章节的最大标记数
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.temperature = temperature
        self.device = device
        self.max_tokens_per_chapter = max_tokens_per_chapter
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载模型和分词器
        logger.info(f"加载模型: {model_name_or_path}")
        self._load_model()
        
        # 加载小说数据并合并章节分片
        logger.info("加载小说数据并处理章节分片...")
        self.novel_data = self._load_novel_data()
        self.merged_chapters = self._merge_chapter_chunks()
        
        # 定义扩充任务列表
        self.augmentation_tasks = [
            self._generate_chapter_summary,
            self._generate_character_analysis,
            self._generate_chapter_qa,
            self._generate_chapter_events,
            self._generate_next_chapter_prediction,
            self._generate_character_dialogue,
            self._generate_emotion_analysis,
            self._generate_location_description,
            self._generate_alternate_pov
        ]
    
    def _load_novel_data(self) -> List[Dict[str, Any]]:
        """加载小说数据"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功加载小说数据: {self.input_file}, 共{len(data)}个分片")
            return data
        except Exception as e:
            logger.error(f"加载小说数据失败: {e}")
            raise
    
    def _merge_chapter_chunks(self) -> List[Dict[str, Any]]:
        """合并属于同一章节的分片"""
        chapter_map = {}
        pattern = re.compile(r'(.*?)(\(\d+\))$')  # 匹配章节标题后的(1)、(2)等
        
        for chunk in self.novel_data:
            if not isinstance(chunk, dict):
                continue
                
            title = chunk.get("chapter_title", "")
            content = chunk.get("content", "")
            
            # 提取基础章节标题（去除(1)、(2)等后缀）
            match = pattern.match(title)
            base_title = match.group(1).strip() if match else title
            
            if base_title not in chapter_map:
                chapter_map[base_title] = {
                    "chapter_title": base_title,
                    "content": content,
                    "chunks": [chunk],
                    "chunk_titles": [title]
                }
            else:
                chapter_map[base_title]["content"] += "\n" + content
                chapter_map[base_title]["chunks"].append(chunk)
                chapter_map[base_title]["chunk_titles"].append(title)
        
        merged_chapters = list(chapter_map.values())
        logger.info(f"合并后的章节数量: {len(merged_chapters)}")
        return merged_chapters
    
    def _load_model(self) -> None:
        """加载大语言模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, 
                trust_remote_code=True
            )
            
            # 检查是否需要低精度加载
            if "cuda" in self.device:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16  # 使用半精度加载
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    device_map=self.device,
                    trust_remote_code=True
                )
                
            # 创建生成管道
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto" if "cuda" in self.device else self.device
            )
            
            logger.info(f"模型加载成功，使用设备: {self.device}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def _tokenize(self, text: str) -> List[str]:
        """将文本转换为标记"""
        return self.tokenizer.encode(text)
    
    def _smart_truncate(self, text: str, max_tokens: int, preserve_end: bool = False) -> str:
        """智能截断文本，优先保留开头和结尾的内容"""
        tokens = self._tokenize(text)
        
        if len(tokens) <= max_tokens:
            return text
            
        if preserve_end:
            # 保留开头和结尾
            start_tokens = tokens[:max_tokens//2]
            end_tokens = tokens[-(max_tokens//2):]
            truncated_tokens = start_tokens + end_tokens
            # 由于直接解码截断的标记可能导致错误，这里简单返回
            return self.tokenizer.decode(start_tokens) + "...\n[中间内容已省略]...\n" + self.tokenizer.decode(end_tokens)
        else:
            # 只保留开头
            return self.tokenizer.decode(tokens[:max_tokens]) + "..."
    
    def generate_data(self) -> None:
        """生成预训练数据"""
        logger.info("开始生成预训练数据...")
        
        # 创建C4格式输出文件
        output_file = os.path.join(self.output_dir, "novel_pretrain_data.jsonl")
        
        generated_count = 0
        
        # 对每个合并后的章节进行处理
        with jsonlines.open(output_file, mode='w') as writer:
            # 首先保存原始章节内容 (C4格式)
            for chapter in tqdm(self.merged_chapters, desc="处理原始章节"):
                # 保存原始内容
                writer.write({
                    "text": chapter["content"],
                    "meta": {
                        "url": f"novel_chapter_{generated_count}",
                        "source": "original_text",
                        "title": chapter["chapter_title"]
                    }
                })
                generated_count += 1
            
            # 然后进行数据扩充
            for i, chapter in enumerate(tqdm(self.merged_chapters, desc="数据扩充")):
                title = chapter["chapter_title"]
                full_content = chapter["content"]
                
                # 获取上下文信息
                prev_chapter = self.merged_chapters[i-1] if i > 0 else None
                next_chapter = self.merged_chapters[i+1] if i < len(self.merged_chapters)-1 else None
                
                prev_text = prev_chapter["content"] if prev_chapter else ""
                next_text = next_chapter["content"] if next_chapter else ""
                
                # 为不同类型的任务准备不同的上下文
                # 为了适应模型的上下文窗口，我们需要智能地截断文本
                
                # 获取chapter_content的标记数
                chapter_tokens_count = len(self._tokenize(full_content))
                
                # 如果章节内容太长，需要进行智能截断
                if chapter_tokens_count > self.max_tokens_per_chapter:
                    # 对于不同任务，采用不同的截断策略
                    truncated_content = self._smart_truncate(full_content, self.max_tokens_per_chapter)
                    # 对于特定任务，我们可能需要保留章节结尾
                    truncated_content_with_end = self._smart_truncate(full_content, self.max_tokens_per_chapter, preserve_end=True)
                else:
                    truncated_content = full_content
                    truncated_content_with_end = full_content
                
                # 准备章节上下文信息
                chapter_contexts = {
                    "basic": truncated_content,  # 基本上下文
                    "with_end": truncated_content_with_end,  # 保留结尾的上下文
                    "with_prev": self._prepare_context_with_prev(truncated_content, prev_text),  # 包含前一章节的上下文
                    "with_next": self._prepare_context_with_next(truncated_content, next_text),  # 包含后一章节的上下文
                }
                
                # 对当前章节应用随机3-5个扩充任务
                num_tasks = random.randint(3, 5)
                selected_tasks = random.sample(self.augmentation_tasks, num_tasks)
                
                # 执行每个扩充任务
                for task_func in selected_tasks:
                    try:
                        # 根据任务类型选择适当的上下文
                        context_type = self._select_context_for_task(task_func.__name__)
                        context = chapter_contexts[context_type]
                        
                        # 使用选定的上下文执行任务
                        augmented_data = task_func(context, title, prev_text, next_text)
                        if augmented_data:
                            writer.write(augmented_data)
                            generated_count += 1
                    except Exception as e:
                        logger.error(f"执行任务 {task_func.__name__} 失败: {e}")
                        continue
                    
                    # 短暂休息，避免模型过载
                    time.sleep(0.5)
        
        logger.info(f"预训练数据生成完成，共生成 {generated_count} 条数据，保存至: {output_file}")
    
    def _prepare_context_with_prev(self, current_text: str, prev_text: str) -> str:
        """准备包含前一章节摘要的上下文"""
        if not prev_text:
            return current_text
            
        # 对前一章节进行摘要
        prev_summary = self._generate_quick_summary(prev_text)
        context = f"前一章节摘要: {prev_summary}\n\n当前章节内容:\n{current_text}"
        
        # 确保上下文不超过最大标记数
        if len(self._tokenize(context)) > self.max_tokens_per_chapter:
            return self._smart_truncate(context, self.max_tokens_per_chapter)
        return context
    
    def _prepare_context_with_next(self, current_text: str, next_text: str) -> str:
        """准备包含后一章节摘要的上下文"""
        if not next_text:
            return current_text
            
        # 对后一章节进行摘要
        next_summary = self._generate_quick_summary(next_text)
        context = f"{current_text}\n\n后一章节摘要: {next_summary}"
        
        # 确保上下文不超过最大标记数
        if len(self._tokenize(context)) > self.max_tokens_per_chapter:
            return self._smart_truncate(context, self.max_tokens_per_chapter)
        return context
    
    def _generate_quick_summary(self, text: str) -> str:
        """快速生成文本摘要，无需调用模型"""
        # 这是一个简单的规则方法，提取前几句话作为摘要
        sentences = re.split(r'([。！？])', text)
        temp_sentences = []
        
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                temp_sentences.append(sentences[i] + sentences[i + 1])
                
        # 取前3句或20%的句子，以较小者为准
        summary_count = min(3, max(1, len(temp_sentences) // 5))
        return "".join(temp_sentences[:summary_count])
    
    def _select_context_for_task(self, task_name: str) -> str:
        """根据任务选择适当的上下文类型"""
        if task_name == "_generate_next_chapter_prediction":
            return "with_next"
        elif task_name == "_generate_chapter_summary":
            return "basic"
        elif task_name == "_generate_character_analysis":
            return "basic"
        elif task_name == "_generate_alternate_pov":
            return "with_end"  # 可能需要了解章节结尾
        elif task_name == "_generate_emotion_analysis":
            return "with_end"  # 情感可能会随着章节进展而变化
        else:
            return "basic"
    
    def _extract_title(self, text: str) -> str:
        """从文本中提取章节标题"""
        # 简单提取第一行作为标题
        lines = text.strip().split('\n')
        return lines[0] if lines else "无标题章节"
    
    def _generate_text(self, prompt: str) -> str:
        """使用模型生成文本"""
        try:
            # 计算prompt标记数
            prompt_tokens = len(self._tokenize(prompt))
            
            # 调整max_new_tokens，确保不超过模型限制
            max_new_tokens = min(2048, self.max_length - prompt_tokens)
            
            if max_new_tokens <= 0:
                logger.warning(f"提示过长: {prompt_tokens} 标记，超过模型限制")
                # 截断提示
                prompt = self._smart_truncate(prompt, self.max_length // 2)
                max_new_tokens = 1024  # 设置一个安全值
            
            result = self.generator(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
                num_return_sequences=1
            )
            
            generated_text = result[0]['generated_text']
            
            # 提取生成的部分（去除原始提示）
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
                
            return generated_text
        except Exception as e:
            logger.error(f"生成文本失败: {e}")
            return ""
    
    def _generate_chapter_summary(self, text: str, title: str, prev_text: str, next_text: str) -> Dict[str, Any]:
        """生成章节摘要"""
        prompt = f"""请对以下小说章节进行详细总结，包括主要情节、人物行动和关键事件。

章节标题: {title}

章节内容:
{text[:1500]}...  # 增加字符数，提供更多上下文

请提供一个全面的总结:"""

        summary = self._generate_text(prompt)
        
        return {
            "text": f"《{title}》章节总结: {summary}",
            "meta": {
                "url": f"novel_summary_{title}",
                "source": "chapter_summary",
                "title": title
            }
        }
    
    def _generate_character_analysis(self, text: str, title: str, prev_text: str, next_text: str) -> Dict[str, Any]:
        """生成角色分析"""
        prompt = f"""请分析以下小说章节中出现的主要人物，包括他们的特点、动机、行为和关系。

章节标题: {title}

章节内容:
{text[:1500]}...

请提供详细的角色分析:"""

        analysis = self._generate_text(prompt)
        
        return {
            "text": f"《{title}》角色分析: {analysis}",
            "meta": {
                "url": f"novel_character_analysis_{title}",
                "source": "character_analysis",
                "title": title
            }
        }
    
    def _generate_chapter_qa(self, text: str, title: str, prev_text: str, next_text: str) -> Dict[str, Any]:
        """生成章节问答对"""
        prompt = f"""请根据以下小说章节内容，创建5个问答对，问题应该涵盖章节中的关键信息，例如人物、事件、地点和情节发展。
每个问题应该是对章节理解的测试，答案应该详细且准确。

章节标题: {title}

章节内容:
{text[:1500]}...

请提供5个高质量的问答对，每个问答对包含一个问题和对应的详细答案:"""

        qa_pairs = self._generate_text(prompt)
        
        return {
            "text": f"《{title}》问答: {qa_pairs}",
            "meta": {
                "url": f"novel_qa_{title}",
                "source": "chapter_qa",
                "title": title
            }
        }
    
    def _generate_chapter_events(self, text: str, title: str, prev_text: str, next_text: str) -> Dict[str, Any]:
        """生成章节事件时间线"""
        prompt = f"""请根据以下小说章节内容，按时间顺序列出发生的主要事件，形成事件时间线。
对于每个事件，请简要描述事件内容、相关人物和对情节的影响。

章节标题: {title}

章节内容:
{text[:1500]}...

请提供详细的事件时间线:"""

        events = self._generate_text(prompt)
        
        return {
            "text": f"《{title}》事件时间线: {events}",
            "meta": {
                "url": f"novel_events_{title}",
                "source": "chapter_events",
                "title": title
            }
        }
    
    def _generate_next_chapter_prediction(self, text: str, title: str, prev_text: str, next_text: str) -> Dict[str, Any]:
        """生成下一章节预测"""
        prompt = f"""请根据以下小说章节内容，预测下一章可能发生的情节和事件。
考虑当前章节结尾的情况、角色面临的处境、未解决的问题和可能的情节发展方向。

章节标题: {title}

章节内容:
{text[:1500]}...

请基于上述内容，详细预测下一章的情节发展:"""

        prediction = self._generate_text(prompt)
        
        return {
            "text": f"《{title}》后续预测: {prediction}",
            "meta": {
                "url": f"novel_prediction_{title}",
                "source": "next_chapter_prediction",
                "title": title
            }
        }
    
    def _generate_character_dialogue(self, text: str, title: str, prev_text: str, next_text: str) -> Dict[str, Any]:
        """生成角色对话扩展"""
        prompt = f"""请根据以下小说章节内容，选择章节中的两个或多个主要角色，编写一段扩展对话。
对话应符合角色性格、背景和当前的情节状况，展现角色之间的关系和冲突。

章节标题: {title}

章节内容:
{text[:1500]}...

请创作一段生动、符合角色特点的对话场景:"""

        dialogue = self._generate_text(prompt)
        
        return {
            "text": f"《{title}》角色对话: {dialogue}",
            "meta": {
                "url": f"novel_dialogue_{title}",
                "source": "character_dialogue",
                "title": title
            }
        }
    
    def _generate_emotion_analysis(self, text: str, title: str, prev_text: str, next_text: str) -> Dict[str, Any]:
        """生成情感分析"""
        prompt = f"""请分析以下小说章节的情感基调和氛围，包括主要角色的情感变化、情绪转折点和整体氛围塑造。
分析章节中的情感描写技巧和情感如何推动情节发展。

章节标题: {title}

章节内容:
{text[:1500]}...

请提供详细的情感分析:"""

        emotion = self._generate_text(prompt)
        
        return {
            "text": f"《{title}》情感分析: {emotion}",
            "meta": {
                "url": f"novel_emotion_{title}",
                "source": "emotion_analysis",
                "title": title
            }
        }
    
    def _generate_location_description(self, text: str, title: str, prev_text: str, next_text: str) -> Dict[str, Any]:
        """生成场景详细描述"""
        prompt = f"""请根据以下小说章节内容，详细描述章节中出现的主要场景和地点，包括环境、氛围、物理特征和文化背景。
分析场景设置如何影响人物行动和情节发展，以及作者如何通过场景描写表达主题。

章节标题: {title}

章节内容:
{text[:1500]}...

请提供详细的场景分析与描述:"""

        location = self._generate_text(prompt)
        
        return {
            "text": f"《{title}》场景描述: {location}",
            "meta": {
                "url": f"novel_location_{title}",
                "source": "location_description",
                "title": title
            }
        }
    
    def _generate_alternate_pov(self, text: str, title: str, prev_text: str, next_text: str) -> Dict[str, Any]:
        """生成不同视角的叙述"""
        prompt = f"""请选择以下小说章节中的一个配角或次要角色，用该角色的第一人称视角重新讲述章节中的事件。
叙述应反映该角色的性格、价值观和对主要事件的独特理解，可以揭示原文中未表现的背景故事或想法。

章节标题: {title}

章节内容:
{text[:1500]}...

请从配角视角重新讲述这个章节的故事:"""

        pov = self._generate_text(prompt)
        
        return {
            "text": f"《{title}》配角视角: {pov}",
            "meta": {
                "url": f"novel_alternate_pov_{title}",
                "source": "alternate_pov",
                "title": title
            }
        }


def main():
    parser = argparse.ArgumentParser(description="使用本地千问大语言模型生成小说预训练数据")
    parser.add_argument("--input", type=str, default="data/xd_chunks_4096.json", help="输入的小说章节JSON文件")
    parser.add_argument("--output", type=str, default="data/pretrain_output", help="输出的预训练数据目录")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="使用的模型名称或路径")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--batch-size", type=int, default=1, help="批处理大小")
    parser.add_argument("--max-length", type=int, default=4096, help="生成文本的最大长度")
    parser.add_argument("--max-tokens-per-chapter", type=int, default=3000, help="每个章节的最大标记数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="使用的设备")
    
    args = parser.parse_args()
    
    generator = NovelPretrainGenerator(
        input_file=args.input,
        output_dir=args.output,
        model_name_or_path=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        device=args.device,
        max_tokens_per_chapter=args.max_tokens_per_chapter
    )
    
    generator.generate_data()

if __name__ == "__main__":
    main()