import json
import os
import re
import tiktoken  # 添加tiktoken库用于计算token

def count_tokens(text, encoding_name="cl100k_base"):
    """计算文本的token数量"""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

def split_novel_into_chapters_and_chunks(file_path, output_file_name, novel_name, chapter_is_split_by_newline=True, max_chunk_tokens=1024, min_chunk_tokens=768, overlap_tokens=256):
    """
    将小说分割为章节和较小的文本块,使用换行将文本拆分为句子
    然后按递归分组，直至符合最大块大小和最小块大小，要求不要尽可能不要截断句子的完整性
    file_path: 小说文件路径
    novel_name: 小说名称
    chapter_is_split_by_newline: 是否使用首行空格判断章节标题，True则使用首行空格判断，False则使用正则识别章节标题
    max_chunk_tokens: 最大块token数
    min_chunk_tokens: 最小块token数
    overlap_tokens: 重叠token数

    返回的chunks格式为：
    {
        "novel_name": "小说名称",
        "chapter_title": "章节标题(1)",
        "content": "章节内容"
    }

    如果章节内容少于最小块大小，则整章作为一块
    如果章节内容大于最大块大小，则按最大块大小分割

    chapter_title章节标题如果被拆分为多个，则每个章节标题后面加(1/2/3/...)
    """

    print(f"分割小说文本...{file_path} {novel_name}")
    
    # 读取小说文本
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 清理文本，规范化空白
    content = re.sub(r'\r\n', '\n', content)  # 统一换行符
    
    # 删除分隔线，如"------------"
    content = re.sub(r'-{3,}', '', content)
    
    # 删除多个（多于1个）换行符，替换为单个换行符
    #content = re.sub(r'\n{2,}', '\n', content)
    
    # 分割成行
    lines = content.split('\n')
    chapters = []
    current_chapter = {'title': '', 'content': []}
    current_volume = None  # 当前卷名
    
    # 编译正则表达式，用于识别章节标题模式
    # 匹配 "第x章 xxx" 或 "第x节 xxx" 或 "第x回 xxx" "七日之书卷　第七日　审判 （四）"等格式
    chapter_pattern = re.compile(r'^\S+.*第*[0-9一二三四五六七八九十百千万零初余]+[章节回卷集部篇日]\s*.*$')
    
    # 匹配序言、前言、楔子、引子等特殊章节格式
    special_chapter_pattern = re.compile(r'^(序章|前言|楔子|引子|序幕|引言|后记|尾声|终章)\s*.*$')
    
    # 匹配"章x xxx"、"章一 末路 全" 这样的格式
    complex_chapter_pattern = re.compile(r'^([章][0-9一二三四五六七八九十百千万零]+.*)|(\S+.*[章][0-9一二三四五六七八九十百千万零]+.*)$')
    
    # 匹配卷名格式："第x卷 xxx"、"第四卷　问情"
    volume_pattern1 = re.compile(r'^(第[0-9一二三四五六七八九十百千万零]+卷).*$')
    # 匹配卷名格式："在地狱中仰望天堂　章十八　天明　上"、"盛世年华卷　章四　中线　下"、"在地狱中仰望天堂　章四　回归　上"
    volume_pattern2 = re.compile(r'^(\S+.*)[章][0-9一二三四五六七八九十百千万零]+.*$')
    # 匹配卷名格式："黄昏之十二乐章卷　余章　绕梁　下"
    volume_pattern3= re.compile(r'^(\S+卷).*[0-9一二三四五六七八九十百千万零初余]+[章节回卷集部篇日]\s*.*$')
    
    # 匹配特殊卷名格式："xxx之卷" 
    special_volume_pattern = re.compile(r'^(\S*之卷).*')


    if chapter_is_split_by_newline:
        # 使用首行空格来判断章节标题
        for line in lines:
            original_line = line
            line = line.rstrip()
            if not line:  # 跳过空行
                continue
                
            # 计算行首空格数
            leading_spaces = len(original_line) - len(original_line.lstrip())
            
            # 行首无空格或只有一个空格的为章节标题或卷标题
            if leading_spaces <= 1:
                # 检查是否是卷标题,并抽取卷名
                if volume_pattern1.match(line):
                    current_volume = volume_pattern1.match(line).group(1)
                elif volume_pattern2.match(line):
                    current_volume = volume_pattern2.match(line).group(1)
                if special_volume_pattern.match(line):
                    current_volume = special_volume_pattern.match(line).group(1)
                
                # 保存前一个章节
                if current_chapter['title']:
                    current_chapter['content'] = '\n'.join(current_chapter['content'])
                    chapters.append(current_chapter.copy())
                
                # 开始新章节，添加卷名前缀
                chapter_title = line.strip()
                if current_volume:
                    # 检查特殊情况：如果章节标题已经包含卷名，不重复添加
                    if not current_volume in chapter_title:
                        chapter_title = f"{current_volume} - {chapter_title}"
                
                current_chapter = {'title': chapter_title, 'content': []}
            else:  # 其他情况为正文内容
                if current_chapter['title']:  # 如果已有章节标题，添加内容
                    current_chapter['content'].append(line)
    else:
        # 使用正则表达式识别章节标题
        for line in lines:

            # if not line.strip():  # 跳过空行
            #     continue    

            # 先检查是否是卷标题
            if volume_pattern1.match(line):
                current_volume = volume_pattern1.match(line).group(1)
            elif volume_pattern2.match(line):
                current_volume = volume_pattern2.match(line).group(1)
            elif volume_pattern3.match(line):
                current_volume = volume_pattern3.match(line).group(1)
            if special_volume_pattern.match(line):
                current_volume = special_volume_pattern.match(line).group(1)
                      
            # 判断是否是章节标题（包括常规章节、特殊章节和复杂格式章节）
            if (chapter_pattern.match(line) or 
                special_chapter_pattern.match(line) or 
                complex_chapter_pattern.match(line)):
                # 保存前一个章节
                if current_chapter['title']:
                    current_chapter['content'] = '\n'.join(current_chapter['content'])
                    chapters.append(current_chapter.copy())
                
                # 开始新章节，添加卷名前缀
                chapter_title = line
                
                # 检查特殊情况：如果章节标题本身就包含卷名（如"罪与罚　章一　晨昏　上"），则不添加前缀
                if current_volume and not current_volume in chapter_title:
                    chapter_title = f"{current_volume} - {chapter_title}"
                
                current_chapter = {'title': chapter_title, 'content': []}
            else:  # 其他情况为正文内容
                if current_chapter['title']:  # 如果已有章节标题，添加内容
                    current_chapter['content'].append(line)
                else:
                    # 如果尚未找到章节标题，将内容添加到临时集合
                    if not chapters and 'tmp_content' not in current_chapter:
                        current_chapter['tmp_content'] = []
                    if 'tmp_content' in current_chapter:
                        current_chapter['tmp_content'].append(line)
    
    # 保存最后一个章节
    if current_chapter['title']:
        current_chapter['content'] = '\n'.join(current_chapter['content'])
        chapters.append(current_chapter)
    

    print(f"成功分割出 {len(chapters)} 个章节")
    
    # 将章节内容分割成重叠的块
    chunks = []
    
    for chapter in chapters:
        chapter_content = chapter["content"]
        chapter_title = chapter["title"]

        # 如果章节内容为空，则跳过
        if not chapter_content:
            continue
        
        # 计算章节内容的token数
        chapter_tokens = count_tokens(chapter_content)
        
        # 如果章节内容少于最小块token数，则整章作为一块
        if chapter_tokens < min_chunk_tokens:
            full_content = novel_name + " " + chapter_title + "\n\n" + chapter_content
            chunks.append({
                "chapter_title": novel_name + " " + chapter_title,
                "content": full_content,
                "length": len(chapter_content),
                "tokens": count_tokens(full_content)
            })
            continue
        
        # 使用句号等标点符号将章节内容拆分为句子
        sentences = []
        # 首先按照标点符号分割
        temp_parts = re.split(r'([。！？\.\!\?\n])', chapter_content)
        
        # 重新组合句子（包含标点符号）并清理格式
        i = 0
        while i < len(temp_parts) - 1:
            if i + 1 < len(temp_parts):
                # 组合句子和标点
                sentence = temp_parts[i] + temp_parts[i+1]
                i += 2
            else:
                # 处理最后一个可能没有标点的句子
                sentence = temp_parts[i]
                i += 1
            
            # 清理句子格式
            # 1. 去除前后空白
            #sentence = sentence.strip()
            # 2. 将句子内部的多个空白字符替换为单个空格
            #sentence = re.sub(r'\s+', ' ', sentence)
            # 3. 如果句子不为空，添加到结果列表
            sentences.append(sentence)
        
        # 使用迭代方法而非递归来分组句子
        def split_sentences_into_chunks(sentences_list):
            result_chunks = []
            
            # 当前块的起始位置
            start_idx = 0
            
            while start_idx < len(sentences_list):
                # 找到合适的结束位置
                end_idx = start_idx
                current_chunk = []
                current_tokens = 0
                
                # 尝试扩展当前块直到达到最大token数
                while end_idx < len(sentences_list):
                    sentence = sentences_list[end_idx]
                    sentence_tokens = count_tokens(sentence)
                    
                    # 如果添加这个句子会超过最大块token数，则停止添加
                    if current_tokens + sentence_tokens > max_chunk_tokens:
                        break
                        
                    # 否则添加这个句子
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
                    end_idx += 1
                
                # 如果没有添加任何句子（可能是一个非常长的句子），至少包含一个句子
                if len(current_chunk) == 0:
                    current_chunk.append(sentences_list[start_idx])
                    end_idx = start_idx + 1
                    current_tokens = count_tokens(sentences_list[start_idx])
                
                # 确保当前块满足最小token数要求（如果有指定）
                if min_chunk_tokens > 0 and current_tokens < min_chunk_tokens and end_idx < len(sentences_list):
                    # 继续添加句子直到满足最小token数
                    while end_idx < len(sentences_list) and current_tokens < min_chunk_tokens:
                        sentence = sentences_list[end_idx]
                        sentence_tokens = count_tokens(sentence)
                        
                        # 如果添加这个句子会超过最大块token数，则停止添加
                        if current_tokens + sentence_tokens > max_chunk_tokens:
                            break
                        
                        current_chunk.append(sentence)
                        current_tokens += sentence_tokens
                        end_idx += 1
                
                # 再次计算当前块的token数
                chunk_text = ''.join(current_chunk)
                actual_tokens = count_tokens(chunk_text)
                
                # 添加当前块
                result_chunks.append({
                    "text": current_chunk, 
                    "length": len(chunk_text),
                    "tokens": actual_tokens
                })
                
                # 移动到下一个块的起始位置，考虑重叠
                if overlap_tokens > 0 and end_idx < len(sentences_list):
                    # 找到重叠部分的起始位置
                    overlap_sentences = []
                    overlap_token_count = 0
                    i = end_idx - 1
                    
                    # 从末尾开始累积句子，直到达到指定的重叠token数
                    while i >= start_idx and overlap_token_count < overlap_tokens:
                        sentence_tokens = count_tokens(sentences_list[i])
                        overlap_sentences.insert(0, sentences_list[i])
                        overlap_token_count += sentence_tokens
                        i -= 1
                    
                    # 设置下一个块的起始位置
                    if overlap_sentences:
                        start_idx = end_idx - len(overlap_sentences)
                    else:
                        start_idx = end_idx
                else:
                    start_idx = end_idx
            
            return result_chunks
        
        # 使用迭代方法分割句子
        sentence_groups = split_sentences_into_chunks(sentences)
        
        # 将分组后的句子转换为文本块
        for i, group in enumerate(sentence_groups):
            chunk_text = ''.join(group["text"])
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            
            # 如果章节被分成多个块，则在标题后添加编号
            chunk_title = chapter_title
            if len(sentence_groups) > 1:
                chunk_title = f"{chapter_title}({i+1})"
            
            full_content = novel_name + " " + chunk_title + "\n\n" + chunk_text
            
            chunks.append({
                "chapter_title": novel_name + " " + chunk_title,
                "content": full_content,
                "length": len(chunk_text),
                "tokens": count_tokens(full_content)
            })
    
    print(f"将小说分割为 {len(chunks)} 个文本块")
    # 保存chunks
    with open(os.path.join(output_file_name), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return chunks

if __name__ == "__main__":
    # 使用不同的token大小生成chunks
    for chunk_tokens in [32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 4096, 10240, 32768]:
        chunks = split_novel_into_chapters_and_chunks("data/亵渎.txt", 
                                                  f"data/xd_chunks_tokens_{chunk_tokens}.json",
                                                  "亵渎", 
                                                  chapter_is_split_by_newline=False, 
                                                  max_chunk_tokens=chunk_tokens, 
                                                  min_chunk_tokens=0, 
                                                  overlap_tokens=0)

   