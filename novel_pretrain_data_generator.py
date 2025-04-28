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
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, before_log, after_log
import tiktoken
import concurrent.futures
from threading import Lock
import queue

# 增强日志配置，添加请求ID
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s',
    handlers=[
        logging.FileHandler("logs/novel_pretrain_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 添加请求ID过滤器
class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'request_id'):
            record.request_id = 'N/A'
        return True

logger.addFilter(RequestIdFilter())

class NovelPretrainGenerator:
    """使用大语言模型扩充小说数据，生成预训练数据集"""
    
    def __init__(
        self, 
        input_file: str,
        output_dir: str,
        openai_api_key: str,
        openai_model: str = "gpt-4",
        openai_base_url: str = "https://api.openai.com/v1",
        batch_size: int = 1,
        max_length: int = 8000,
        temperature: float = 0.7,
        log_level: str = "INFO",
        max_workers: int = 5,  # 新增参数：最大工作线程数
        resume_from: str = None,  # 添加新参数：从指定章节标题恢复
    ):
        """初始化生成器
        
        Args:
            input_file: 输入的小说章节JSON文件路径
            output_dir: 输出预训练数据的目录
            openai_api_key: OpenAI API密钥
            openai_model: 使用的OpenAI模型名称
            openai_base_url: OpenAI API的基础URL
            batch_size: 批处理大小
            max_length: 生成文本的最大长度(包括输入和输出)
            temperature: 生成温度
            log_level: 日志级别
            max_workers: 最大并行工作线程数
            resume_from: 从指定章节标题恢复处理
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.openai_model = openai_model
        self.openai_base_url = openai_base_url
        self.batch_size = batch_size
        self.max_length = max_length
        self.temperature = temperature
        
        # 自动计算输出token的最大数量（约30%）和输入token的最大数量（约70%）
        # 预留大约500个token用于提示词和系统消息
        reserved_tokens = 500
        self.max_output_tokens = int((max_length - reserved_tokens) * 0.3)
        self.max_input_tokens = int((max_length - reserved_tokens) * 0.7)
        
        logger.info(f"自动计算的参数: 最大输入tokens={self.max_input_tokens}, 最大输出tokens={self.max_output_tokens}")
        
        # 设置OpenAI API配置
        openai.api_key = openai_api_key
        openai.api_base = openai_base_url  # 设置API基础URL
        
        # 初始化tiktoken编码器
        self.tiktoken_model = self._get_tiktoken_model(openai_model)
        self.encoder = tiktoken.encoding_for_model(self.tiktoken_model)
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 创建详细日志目录
        self.log_dir = "logs"
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # 请求计数器，用于生成唯一的请求ID
        self.request_counter = 0
        
        # 加载小说数据但不合并章节分片
        logger.info("加载小说数据...")
        self.chunks = self._load_novel_data()
        
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
        
        # 设置日志级别
        numeric_level = getattr(logging, log_level.upper(), None)
        if isinstance(numeric_level, int):
            logger.setLevel(numeric_level)
        
        self.max_workers = max_workers
        logger.info(f"设置最大并行线程数: {max_workers}")
        
        # 添加线程安全的计数器和锁
        self.generated_count = 0
        self.lock = Lock()
        
        # 用于存储结果的队列
        self.result_queue = queue.Queue()
        
        self.resume_from = resume_from
        if resume_from:
            logger.info(f"将从章节标题 '{resume_from}' 恢复处理")
    
    def _load_novel_data(self) -> List[Dict[str, Any]]:
        """加载小说数据"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 过滤掉非字典对象
            chunks = [chunk for chunk in data if isinstance(chunk, dict)]
            logger.info(f"成功加载小说数据: {self.input_file}, 共{len(chunks)}个有效分片")
            return chunks
        except Exception as e:
            logger.error(f"加载小说数据失败: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=3),
        before=before_log(logger, logging.INFO),
        after=after_log(logger, logging.INFO)
    )
    def _generate_text(self, prompt: str) -> str:
        """使用OpenAI API生成文本，并记录输入输出"""
        # 生成唯一的请求ID
        self.request_counter += 1
        request_id = f"req_{self.request_counter}_{int(time.time())}"
        
        # 创建日志上下文
        log_extra = {'request_id': request_id}
        
        # 记录输入内容
        logger.info(f"开始生成文本，提示词长度: {len(prompt)} 字符, {len(self._tokenize(prompt))} tokens", extra=log_extra)
        
        # 直接打印提示词到控制台
        print("\n" + "="*80)
        print(f"【提示词】请求ID: {request_id}")
        print("-"*80)
        print(prompt)
        print("="*80 + "\n")
        
        start_time = time.time()
        
        try:
            # 尝试使用ChatCompletion API
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                top_p=0.9,
            )
            
            output_text = response.choices[0].message.content.strip()
            
            # 计算生成时间和token数量
            generation_time = time.time() - start_time
            output_tokens = len(self._tokenize(output_text))
            
            # 记录输出内容
            logger.info(f"文本生成完成，耗时: {generation_time:.2f}秒，输出长度: {len(output_text)} 字符, {output_tokens} tokens", extra=log_extra)
            
            # 直接打印生成内容到控制台
            print("\n" + "="*80)
            print(f"【生成内容】请求ID: {request_id}，耗时: {generation_time:.2f}秒")
            print("-"*80)
            print(output_text)
            print("="*80 + "\n")
            
            return output_text
            
        except (AttributeError, TypeError) as e:
            logger.error(f"ChatCompletion API调用失败: {e}", extra=log_extra)
            # 打印异常信息到控制台
            print(f"\n【错误】请求ID: {request_id}")
            print(f"ChatCompletion API调用失败: {e}")
            raise
        except Exception as e:
            logger.error(f"调用OpenAI API失败: {e}", extra=log_extra)
            # 打印异常信息到控制台
            print(f"\n【错误】请求ID: {request_id}")
            print(f"调用OpenAI API失败: {e}")
            raise
    
    def _get_tiktoken_model(self, openai_model: str) -> str:
        """获取与OpenAI模型对应的tiktoken模型名称"""
        # OpenAI模型到tiktoken模型的映射
        model_map = {
            "gpt-4o": "gpt-4",
            "gpt-4-turbo": "gpt-4",
            "gpt-4": "gpt-4",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            # 添加其他可能的模型映射
        }
        
        # 返回对应的tiktoken模型名称
        if openai_model in model_map:
            return model_map[openai_model]
        else:
            # 默认使用通用编码器
            logger.warning(f"未知模型: {openai_model}，使用cl100k_base编码器")
            return "cl100k_base"
    
    def _tokenize(self, text: str) -> List[int]:
        """使用tiktoken将文本转换为标记ID列表"""
        try:
            return self.encoder.encode(text)
        except Exception as e:
            logger.error(f"Tiktoken编码失败: {e}")
            # 失败时使用字符数作为备用估计
            return list(range(len(text)))
    
    def _smart_truncate(self, text: str, max_tokens: int, preserve_end: bool = False) -> str:
        """智能截断文本，使用tiktoken确保不超过token限制"""
        tokens = self._tokenize(text)
        
        if len(tokens) <= max_tokens:
            return text
            
        if preserve_end:
            # 保留开头和结尾
            start_length = max_tokens // 2
            end_length = max_tokens - start_length
            start_tokens = tokens[:start_length]
            end_tokens = tokens[-end_length:]
            
            # 将token ID转回文本
            start_text = self.encoder.decode(start_tokens)
            end_text = self.encoder.decode(end_tokens)
            
            return f"{start_text}\n...\n[中间内容已省略]\n...\n{end_text}"
        else:
            # 只保留开头
            truncated_tokens = tokens[:max_tokens]
            return self.encoder.decode(truncated_tokens)
    
    def generate_data(self) -> None:
        """生成预训练数据，使用多线程并行处理每个chunk，支持从指定章节恢复"""
        logger.info(f"开始生成预训练数据... 最大上下文长度: {self.max_length}, 最大输入token: {self.max_input_tokens}, 最大输出token: {self.max_output_tokens}")
        logger.info(f"使用 {self.max_workers} 个线程并行处理")
        
        # 创建C4格式输出文件
        output_file = os.path.join(self.output_dir, "novel_pretrain_data.jsonl")
        
        # 重置生成计数器
        self.generated_count = 0
        
        # 如果需要恢复处理，找到起始索引
        start_index = 0
        if self.resume_from:
            for i, chunk in enumerate(self.chunks):
                if chunk.get("chapter_title", "") == self.resume_from:
                    start_index = i
                    logger.info(f"找到恢复点: 章节标题 '{self.resume_from}' 在索引 {start_index}")
                    break
            else:
                logger.warning(f"未找到指定的章节标题 '{self.resume_from}'，将从头开始处理")
        
        # 确定处理模式
        mode = 'a' if start_index > 0 and os.path.exists(output_file) else 'w'
        if mode == 'a':
            logger.info(f"将在现有输出文件 {output_file} 中追加数据")
        else:
            logger.info(f"将创建新的输出文件 {output_file}")
            
        # 创建线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 只提交从start_index开始的分片处理任务
            future_to_chunk = {
                executor.submit(self._process_chunk, i, chunk): (i, chunk) 
                for i, chunk in enumerate(self.chunks[start_index:], start=start_index)
            }
            
            # 创建进度条
            with tqdm(total=len(future_to_chunk), desc="处理分片进度") as pbar:
                # 创建输出文件写入器
                with jsonlines.open(output_file, mode=mode) as writer:
                    # 处理完成的任务
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        i, chunk = future_to_chunk[future]
                        title = chunk.get("chapter_title", f"无标题分片_{i}")
                        
                        try:
                            # 获取该分片的所有结果
                            results = future.result()
                            
                            # 写入结果到文件
                            for result in results:
                                writer.write(result)
                            
                            # 更新进度条
                            pbar.update(1)
                            pbar.set_description(f"已处理: {title}")
                            
                        except Exception as e:
                            logger.error(f"处理分片 {title} 时发生错误: {e}")
                            pbar.update(1)
        
        logger.info(f"预训练数据生成完成，共生成 {self.generated_count} 条数据，保存至: {output_file}")
    
    def _prepare_context_with_prev_simple(self, current_text: str, prev_text: str) -> str:
        """准备包含前一分片摘要的简单上下文"""
        if not prev_text:
            return current_text
        
        # 对前一分片进行摘要
        prev_summary = self._generate_quick_summary(prev_text)
        return f"前一分片摘要: {prev_summary}\n\n当前内容:\n{current_text}"
    
    def _prepare_context_with_next_simple(self, current_text: str, next_text: str) -> str:
        """准备包含后一分片摘要的简单上下文"""
        if not next_text:
            return current_text
        
        # 对后一分片进行摘要
        next_summary = self._generate_quick_summary(next_text)
        return f"{current_text}\n\n后一分片摘要: {next_summary}"
    
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
    
    def _generate_chapter_summary(self, text: str, title: str, prev_text: str, next_text: str) -> Dict[str, Any]:
        """生成分片摘要"""
        task_name = "分片摘要"
        logger.info(f"开始执行任务: {task_name}, 分片: {title}")
        
        # 计算提示词的预估token长度
        prompt_template = f"""请对以下小说分片进行详细总结，包括主要情节、人物行动和关键事件。

分片标题: {title}

分片内容:
{{text}}

请提供一个全面的总结:"""
        
        # 计算提示词模板的token数（不包括分片内容）
        prompt_tokens = len(self._tokenize(prompt_template.format(text="")))
        
        # 计算可用于分片内容的token数
        available_tokens = self.max_input_tokens - prompt_tokens
        
        # 对分片内容进行截断
        #truncated_text = self._smart_truncate(text, available_tokens)
        logger.info(f"任务: {task_name}, 分片: {title}, 原始内容长度: {len(self._tokenize(text))} tokens, 截断后: {len(self._tokenize(truncated_text))} tokens")
        
        # 组装完整提示词
        prompt = prompt_template.format(text=text)

        summary = self._generate_text(prompt)
        logger.info(f"任务: {task_name}, 分片: {title}, 生成完成, 输出长度: {len(self._tokenize(summary))} tokens")
        
        return {
            "text": f"《{title}》分片摘要: {summary}",
            "meta": {
                "url": f"novel_summary_{title}",
                "source": "chunk_summary",
                "title": title
            }
        }
    
    def _generate_character_analysis(self, text: str, title: str, prev_text: str, next_text: str) -> Dict[str, Any]:
        """生成角色分析"""
        task_name = "角色分析"
        logger.info(f"开始执行任务: {task_name}, 分片: {title}")
        
        # 计算提示词模板
        prompt_template = f"""请分析以下小说分片中出现的主要人物，包括他们的特点、动机、行为和关系。

分片标题: {title}

分片内容:
{{text}}

请提供详细的角色分析:"""
        
        # 计算提示词模板的token数（不包括分片内容）
        prompt_tokens = len(self._tokenize(prompt_template.format(text="")))
        
        # 计算可用于分片内容的token数
        available_tokens = self.max_input_tokens - prompt_tokens
        
        # 对分片内容进行截断
        #truncated_text = self._smart_truncate(text, available_tokens)
        
        # 组装完整提示词
        prompt = prompt_template.format(text=text)

        analysis = self._generate_text(prompt)
        
        logger.info(f"任务: {task_name}, 分片: {title}, 生成完成, 输出长度: {len(self._tokenize(analysis))} tokens")
        
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
        task_name = "章节问答对"
        logger.info(f"开始执行任务: {task_name}, 章节: {title}")
        
        prompt = f"""请根据以下小说章节内容，创建5个问答对，问题应该涵盖章节中的关键信息，例如人物、事件、地点和情节发展。
每个问题应该是对章节理解的测试，答案应该详细且准确。

章节标题: {title}

章节内容:
{text}

请提供5个高质量的问答对，每个问答对包含一个问题和对应的详细答案:"""

        qa_pairs = self._generate_text(prompt)
        
        logger.info(f"任务: {task_name}, 章节: {title}, 生成完成, 输出长度: {len(self._tokenize(qa_pairs))} tokens")
        
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
        task_name = "事件时间线"
        logger.info(f"开始执行任务: {task_name}, 章节: {title}")
        
        prompt = f"""请根据以下小说章节内容，按时间顺序列出发生的主要事件，形成事件时间线。
对于每个事件，请简要描述事件内容、相关人物和对情节的影响。

章节标题: {title}

章节内容:
{text}...

请提供详细的事件时间线:"""

        events = self._generate_text(prompt)
        
        logger.info(f"任务: {task_name}, 章节: {title}, 生成完成, 输出长度: {len(self._tokenize(events))} tokens")
        
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
        task_name = "后续预测"
        logger.info(f"开始执行任务: {task_name}, 章节: {title}")
        
        prompt = f"""请根据以下小说章节内容，预测下一章可能发生的情节和事件。
考虑当前章节结尾的情况、角色面临的处境、未解决的问题和可能的情节发展方向。

章节标题: {title}

章节内容:
{text}...

请基于上述内容，详细预测下一章的情节发展:"""

        prediction = self._generate_text(prompt)
        
        logger.info(f"任务: {task_name}, 章节: {title}, 生成完成, 输出长度: {len(self._tokenize(prediction))} tokens")
        
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
        task_name = "角色对话扩展"
        logger.info(f"开始执行任务: {task_name}, 章节: {title}")
        
        prompt = f"""请根据以下小说章节内容，选择章节中的两个或多个主要角色，编写一段扩展对话。
对话应符合角色性格、背景和当前的情节状况，展现角色之间的关系和冲突。

章节标题: {title}

章节内容:
{text}...

请创作一段生动、符合角色特点的对话场景:"""

        dialogue = self._generate_text(prompt)
        
        logger.info(f"任务: {task_name}, 章节: {title}, 生成完成, 输出长度: {len(self._tokenize(dialogue))} tokens")
        
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
        task_name = "情感分析"
        logger.info(f"开始执行任务: {task_name}, 章节: {title}")
        
        prompt = f"""请分析以下小说章节的情感基调和氛围，包括主要角色的情感变化、情绪转折点和整体氛围塑造。
分析章节中的情感描写技巧和情感如何推动情节发展。

章节标题: {title}

章节内容:
{text}...

请提供详细的情感分析:"""

        emotion = self._generate_text(prompt)
        
        logger.info(f"任务: {task_name}, 章节: {title}, 生成完成, 输出长度: {len(self._tokenize(emotion))} tokens")
        
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
        task_name = "场景分析与描述"
        logger.info(f"开始执行任务: {task_name}, 章节: {title}")
        
        prompt = f"""请根据以下小说章节内容，详细描述章节中出现的主要场景和地点，包括环境、氛围、物理特征和文化背景。
分析场景设置如何影响人物行动和情节发展，以及作者如何通过场景描写表达主题。

章节标题: {title}

章节内容:
{text}...

请提供详细的场景分析与描述:"""

        location = self._generate_text(prompt)
        
        logger.info(f"任务: {task_name}, 章节: {title}, 生成完成, 输出长度: {len(self._tokenize(location))} tokens")
        
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
        task_name = "配角视角"
        logger.info(f"开始执行任务: {task_name}, 章节: {title}")
        
        prompt = f"""请选择以下小说章节中的一个配角或次要角色，用该角色的第一人称视角重新讲述章节中的事件。
叙述应反映该角色的性格、价值观和对主要事件的独特理解，可以揭示原文中未表现的背景故事或想法。

章节标题: {title}

章节内容:
{text}...

请从配角视角重新讲述这个章节的故事:"""

        pov = self._generate_text(prompt)
        
        logger.info(f"任务: {task_name}, 章节: {title}, 生成完成, 输出长度: {len(self._tokenize(pov))} tokens")
        
        return {
            "text": f"《{title}》配角视角: {pov}",
            "meta": {
                "url": f"novel_alternate_pov_{title}",
                "source": "alternate_pov",
                "title": title
            }
        }

    def _process_chunk(self, chunk_index: int, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理单个分片，生成扩充数据
        
        Args:
            chunk_index: 分片索引
            chunk: 分片数据
            
        Returns:
            list: 生成的数据列表
        """
        results = []
        
        title = chunk.get("chapter_title", f"无标题分片_{chunk_index}")
        content = chunk.get("content", "")
        
        # 如果内容为空，跳过
        if not content.strip():
            logger.warning(f"跳过空分片: {title}")
            return results
        
        # 记录开始处理此分片
        logger.info(f"开始处理分片 {chunk_index+1}/{len(self.chunks)}: {title}")
        
        # 首先添加原始内容
        with self.lock:
            self.generated_count += 1
            original_count = self.generated_count
            
        results.append({
            "text": content,
            "meta": {
                "url": f"novel_chunk_{original_count}",
                "source": "original_text",
                "title": title
            }
        })
        
        # 获取上下文信息（前一个和后一个分片）
        prev_chunk = self.chunks[chunk_index-1] if chunk_index > 0 else None
        next_chunk = self.chunks[chunk_index+1] if chunk_index < len(self.chunks)-1 else None
        
        prev_text = prev_chunk.get("content", "") if prev_chunk else ""
        next_text = next_chunk.get("content", "") if next_chunk else ""
        
        # 准备上下文信息
        chunk_contexts = {
            "basic": content,
            "with_end": content,
            "with_prev": self._prepare_context_with_prev_simple(content, prev_text),
            "with_next": self._prepare_context_with_next_simple(content, next_text),
        }
        
        # 对当前分片应用随机3-5个扩充任务
        num_tasks = random.randint(3, 5)
        selected_tasks = random.sample(self.augmentation_tasks, num_tasks)
        
        # 执行每个扩充任务
        for task_func in selected_tasks:
            task_name = task_func.__name__.replace("_generate_", "")
            try:
                logger.info(f"分片 '{title}': 开始执行任务 {task_name}")
                start_time = time.time()
                
                # 根据任务类型选择适当的上下文
                context_type = self._select_context_for_task(task_func.__name__)
                context = chunk_contexts[context_type]
                
                # 使用选定的上下文执行任务
                augmented_data = task_func(context, title, prev_text, next_text)
                if augmented_data:
                    results.append(augmented_data)
                    with self.lock:
                        self.generated_count += 1
                
                # 记录任务完成信息
                end_time = time.time()
                logger.info(f"分片 '{title}': 任务 {task_name} 完成，耗时: {end_time - start_time:.2f}秒")
            except Exception as e:
                logger.error(f"分片 '{title}': 执行任务 {task_name} 失败: {e}", exc_info=True)
                continue
            
            # 短暂休息，避免模型过载（每个线程单独休息）
            time.sleep(0.5)
        
        logger.info(f"分片 '{title}' 处理完成，生成了 {len(results)} 条数据")
        return results


def main():
    parser = argparse.ArgumentParser(description="使用OpenAI GPT模型生成小说预训练数据")
    parser.add_argument("--input", type=str, default="data/xd_chunks_tokens_4096.json", help="输入的小说章节JSON文件")
    parser.add_argument("--output", type=str, default="data/pretrain_output", help="输出的预训练数据目录")
    parser.add_argument("--openai-api-key", type=str, default="", help="OpenAI API密钥")
    parser.add_argument("--openai-model", type=str, default="gpt-4o", help="使用的OpenAI模型名称")
    parser.add_argument("--openai-base-url", type=str, default="http://kugpt-openapi.akulaku.com/v1", help="OpenAI API基础URL")
    parser.add_argument("--temperature", type=float, default=0.5, help="生成温度")
    parser.add_argument("--batch-size", type=int, default=1, help="批处理大小")
    parser.add_argument("--max-length", type=int, default=8192, help="生成文本的最大上下文长度(包括输入和输出)")
    parser.add_argument("--max-workers", type=int, default=5, help="最大并行工作线程数")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                     default="INFO", help="日志级别")
    parser.add_argument("--resume-from", type=str, default=None, help="从指定章节标题恢复处理")
    
    args = parser.parse_args()
    
    try:
        import tiktoken
    except ImportError:
        print("请先安装tiktoken: pip install tiktoken")
        return
        
    generator = NovelPretrainGenerator(
        input_file=args.input,
        output_dir=args.output,
        openai_api_key=args.openai_api_key,
        openai_model=args.openai_model,
        openai_base_url=args.openai_base_url,
        batch_size=args.batch_size,
        max_length=args.max_length,
        temperature=args.temperature,
        log_level=args.log_level,
        max_workers=args.max_workers,
        resume_from=args.resume_from,
    )
    
    generator.generate_data()

if __name__ == "__main__":
    main()