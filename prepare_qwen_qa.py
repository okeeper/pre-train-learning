import torch
import json
import os
import re
import time
import random
from tqdm import tqdm
from modelscope.hub.snapshot_download import snapshot_download
from transformers import Trainer, TrainingArguments
from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from openai import OpenAI
from openai import RateLimitError
import backoff  # 用于重试机制

# 清空显存
torch.cuda.empty_cache()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 1024
BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 5e-5
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
OUTPUT_DIR = "./qwen_novel_lora"
QUESTION_TYPES = ["人物", "情节", "背景", "主题", "场景", "对话", "情感", "描写"]
QA_PAIRS_PER_CHUNK = 5  # 每个文本块生成的问答对数量

# OpenAI API配置
def setup_openai_client(api_key=None, base_url=None):
    """设置OpenAI客户端"""
    # 优先使用传入的参数，否则使用环境变量
    api_key = api_key or os.getenv('OPENAI_API_KEY')
    base_url = base_url or os.getenv('OPENAI_API_BASE')
    
    # 如果没有API密钥，提示用户输入
    if not api_key:
        api_key = input("\n请输入你的OpenAI API密钥: ").strip()
        with open(os.path.expanduser("~/.bashrc"), "a") as f:
            f.write(f'\nexport OPENAI_API_KEY="{api_key}"\n')
            print("API密钥已添加到 ~/.bashrc，请重新加载配置文件使其生效")
    
    if not api_key:
        raise ValueError("未提供API密钥！")
    
    # 如果没有设置base_url，询问是否需要设置
    if not base_url:
        use_base_url = input("\n是否需要设置自定义API地址？(y/n): ").strip().lower()
        if use_base_url == 'y':
            base_url = input("请输入API地址（例如：http://kugpt-openapi.akulaku.com/v1）: ").strip()
            with open(os.path.expanduser("~/.bashrc"), "a") as f:
                f.write(f'\nexport OPENAI_API_BASE="{base_url}"\n')
                print("API地址已添加到 ~/.bashrc，请重新加载配置文件使其生效")
    
    client_params = {
        "api_key": api_key,
        "base_url": base_url
    }
        
    return OpenAI(**client_params)

def extract_novel_title(file_path):
    """从文件第一行提取小说标题，标题通常用<>括起"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # 尝试匹配<>中的标题
            title_match = re.search(r'<([^>]+)>', first_line)
            if title_match:
                return title_match.group(1)
            # 如果没有<>标记，则返回第一行作为标题
            return first_line or os.path.splitext(os.path.basename(file_path))[0]
    except Exception as e:
        print(f"警告：无法从文件中提取标题：{str(e)}")
        # 如果出错，使用文件名作为标题
        return os.path.splitext(os.path.basename(file_path))[0]

def split_novel_into_chapters_and_chunks(file_path, chunk_size=1024, overlap=256):
    """将小说分割为章节和较小的文本块"""
    print("分割小说文本...")
    
    # 读取小说文本
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 清理文本，规范化空白
    content = re.sub(r'\r\n', '\n', content)  # 统一换行符
    content = re.sub(r'\n{3,}', '\n\n', content)  # 减少过多空行
    
    # 分割成行
    lines = content.split('\n')
    chapters = []
    current_chapter = {'title': '', 'content': []}
    
    for line in lines:
        original_line = line
        line = line.rstrip()
        if not line:  # 跳过空行
            if current_chapter['title']:
                current_chapter['content'].append('')
            continue
            
        # 计算行首空格数
        leading_spaces = len(original_line) - len(original_line.lstrip())
        
        # 行首无空格或只有一个空格的为章节标题
        if leading_spaces <= 1:
            # 保存前一个章节
            if current_chapter['title']:
                current_chapter['content'] = '\n'.join(current_chapter['content'])
                chapters.append(current_chapter.copy())
            # 开始新章节
            current_chapter = {'title': line.strip(), 'content': []}
        else:  # 其他情况为正文内容
            if current_chapter['title']:  # 如果已有章节标题，添加内容
                current_chapter['content'].append(line)
    
    # 保存最后一个章节
    if current_chapter['title']:
        current_chapter['content'] = '\n'.join(current_chapter['content'])
        chapters.append(current_chapter)
    
    # 如果没有找到任何章节，将整个内容作为一个章节
    if not chapters:
        chapters.append({
            'title': '第1章',
            'content': content
        })

    print(f"成功分割出 {len(chapters)} 个章节")
    
    # 将章节内容分割成重叠的块
    chunks = []
    
    for chapter in chapters:
        chapter_content = chapter["content"]
        chapter_title = chapter["title"]
        
        # 如果章节内容少于最小块大小，则整章作为一块
        if len(chapter_content) < chunk_size:
            chunks.append({
                "title": chapter_title,
                "chapter": chapter_title,
                "content": chapter_content
            })
            continue
        
        # 否则，滑动窗口分块
        start = 0
        while start < len(chapter_content):
            end = min(start + chunk_size, len(chapter_content))
            
            # 尝试在句子结束处截断
            if end < len(chapter_content):
                while end > start and chapter_content[end] not in '.。!！?？':
                    end -= 1
                if end > start:
                    end += 1  # 包含句号
                else:
                    end = min(start + chunk_size, len(chapter_content))  # 回退到原来的位置
            
            chunk_text = chapter_content[start:end]
            chunks.append({
                "title": f"{chapter_title} (片段)",
                "chapter": chapter_title,
                "content": chunk_text
            })
            
            # 滑动窗口，考虑重叠
            start = end - overlap if end < len(chapter_content) else len(chapter_content)
    
    print(f"将小说分割为 {len(chunks)} 个文本块")
    return chunks

def prepare_prompts_for_qa_generation(chunk, novel_title):
    """为每个文本块准备生成问答对的提示"""
    context = chunk["content"]
    chapter = chunk["chapter"]
    
    # 准备几种不同类型的提示
    prompt_templates = [
        f"""你是一位文学分析专家。请根据下面的小说《{novel_title}》中的片段，生成5个高质量的问答对，这些问答对应该涵盖人物、情节、背景、文学手法等方面。
问题应该多样化，包括事实性问题和分析性问题。回答应该基于文本内容，详细且准确。

小说片段：
{context}

请以以下JSON格式输出问答对：
[
  {{"question": "问题1", "answer": "回答1"}},
  {{"question": "问题2", "answer": "回答2"}},
  ...
]""",
        
        f"""作为一个文学教育助手，你需要从以下《{novel_title}》的章节文本中创建教学问答对。
这些问答应该帮助读者理解小说中的重要元素，如人物发展、关键情节、场景描写、情感表达等。

章节：{chapter}
文本：
{context}

请生成5个深入的问答对，并以JSON格式返回：
[
  {{"question": "问题1", "answer": "详细回答1"}},
  {{"question": "问题2", "answer": "详细回答2"}},
  ...
]""",
        
        f"""请你作为小说分析专家，根据下面《{novel_title}》的文本片段，创建5个问答对。
着重关注以下几个方面：情节发展、人物关系、环境描写、主题探讨、写作技巧。

文本片段：
{context}

输出格式为JSON：
[
  {{"question": "问题1", "answer": "回答1"}},
  {{"question": "问题2", "answer": "回答2"}},
  ...
]"""
    ]
    
    # 随机选择一个提示模板
    return random.choice(prompt_templates)

@backoff.on_exception(backoff.expo, RateLimitError, max_tries=8)
def generate_qa_pairs_with_openai(chunks, novel_title, output_file):
    """使用OpenAI API生成问答对"""
    print("使用OpenAI生成问答对...")
    qa_pairs = []
    # 清空output_file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.truncate(0)
    
    for chunk in tqdm(chunks, desc="处理文本块"):
        prompt = f"""你是一位文学分析专家。请根据下面的小说《{novel_title}》中的片段，生成5个高质量的问答对。
这些问答对应该涵盖人物、情节、背景、文学手法等方面。问题应该多样化，包括事实性问题和分析性问题。
回答应该基于文本内容，详细且准确。

小说片段：
{chunk['content']}

请以JSON数组格式输出5个问答对，格式如下：
[
  {{"question": "问题1", "answer": "回答1"}},
  {{"question": "问题2", "answer": "回答2"}},
  ...
]"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # 或使用 gpt-4
                messages=[
                    {"role": "system", "content": "你是一个专业的文学分析助手，善于生成高质量的文学作品相关问答对。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                n=1,
            )
            
            # 解析响应
            try:
                response_text = response.choices[0].message.content
                # 查找JSON部分
                json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
                if json_match:
                    pairs = json.loads(json_match.group(0))
                    
                    # 验证并清理每个问答对
                    for pair in pairs:
                        if "question" in pair and "answer" in pair:
                            # 确保问题以问号结尾
                            if not pair["question"].strip().endswith('?') and not pair["question"].strip().endswith('？'):
                                pair["question"] = pair["question"].strip() + '？'
                            
                            # 添加章节信息
                            pair["chapter"] = chunk["title"]
                            pair["novel_title"] = novel_title
                            
                            qa_pairs.append(pair)
                            # append to output_file
                            with open(output_file, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(pair, ensure_ascii=False) + '\n')

                else:
                    print(f"警告：无法从响应中解析JSON：{response_text[:100]}...")
            except json.JSONDecodeError:
                print(f"警告：JSON解析错误，响应：{response_text[:100]}...")
                
        except Exception as e:
            print(f"生成问答对时出错：{str(e)}")
        
        # 添加短暂延迟，避免触发速率限制
        time.sleep(0.5)
    
    print(f"成功生成 {len(qa_pairs)} 个问答对")
    return qa_pairs

def format_qa_pairs_for_training(qa_pairs, novel_title, output_file):
    """将问答对格式化为训练数据集"""
    formatted_data = []
    
    for pair in qa_pairs:
        # 创建指令格式的训练样本
        sample = {
            "messages": [
                {"role": "system", "content": f"你是一个关于小说《{novel_title}》的文学助手，你可以回答有关小说内容、人物、情节和主题的问题。"},
                {"role": "user", "content": pair["question"]},
                {"role": "assistant", "content": pair["answer"]}
            ]
        }
        formatted_data.append(sample)
    
    print(f"格式化了 {len(formatted_data)} 个训练样本")
    save_json_data(formatted_data, output_file)
    return formatted_data

def save_json_data(data, output_file):
    """保存训练数据到JSONL文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"训练数据已保存到 {output_file}")

def generate_novel_qa_dataset(novel_file, output_dir="data", api_key=None, base_url=None):
    """主函数：从小说文件生成问答对数据集"""
    # 设置OpenAI客户端
    global client
    client = setup_openai_client(api_key, base_url)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取小说标题
    novel_title = extract_novel_title(novel_file)

    # 分割小说文本
    chunks = split_novel_into_chapters_and_chunks(novel_file)

    # 保存chunks
    with open(os.path.join(output_dir, "novel_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    # 使用OpenAI生成问答对
    qa_pairs_output_file = os.path.join(output_dir, "novel_qa_pairs.json")
    qa_pairs = generate_qa_pairs_with_openai(chunks, novel_title, qa_pairs_output_file)
    
    # 格式化为训练数据
    training_data_output_file = os.path.join(output_dir, "novel_qa_data.jsonl")
    training_data = format_qa_pairs_for_training(qa_pairs, novel_title)
    
    return novel_title

if __name__ == "__main__":
    novel_file = "./data/novel.txt"
    output_dir = "data"
    
    try:
        # 生成问答对数据集
        novel_title = generate_novel_qa_dataset(novel_file, output_dir)
        print(f"\n成功为《{novel_title}》创建问答对训练数据集！")
        print("接下来，您可以使用这个数据集来微调 Qwen 模型")
    except ValueError as e:
        print(f"\n错误：{str(e)}")
    except Exception as e:
        print(f"\n发生未知错误：{str(e)}")