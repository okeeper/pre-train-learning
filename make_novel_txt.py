import os
import json

def convert_novel_to_json(input_file="data/novel.txt", output_file="data/novel_txt.json"):
    """
    将整个小说文本文件转换为指定JSON格式
    
    参数:
    input_file -- 输入的小说文本文件路径
    output_file -- 输出的JSON文件路径
    
    格式:
    [
      {"content": "完整小说内容"}
    ]
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 读取小说全文
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            novel_content = f.read()
        
        # 创建JSON数据结构
        json_data = [{"content": novel_content}]
        
        # 写入JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        print(f"小说已成功转换为JSON格式并保存到: {output_file}")
        return True
    
    except Exception as e:
        print(f"转换过程中出错: {str(e)}")
        return False

if __name__ == "__main__":
    convert_novel_to_json()