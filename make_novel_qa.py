from loguru import logger
import torch
import json
from tqdm import tqdm
import time
import os
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置loguru输出到文件
logger.remove()  # 移除默认的控制台输出
logger.add("logs/app_{time:YYYY-MM-DD}.log", level="TRACE", rotation="00:00", retention="10 days", compression="zip")

# 加载 Qwen/Qwen2.5-1.5B-Instruct 模型
model_path = "/data/hf-models/Qwen2.5-14B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)



def get_model_answer(model, tokenizer, messages, max_length=512):
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
    # 打印输入输出
    print(f"输入：{text}")
    print(f"输出：{generated_text}")
    return generated_text.strip()


def build_dataset_control(novel_name, novel_chunks, output_file):
    instruction_prompt = f"你是一个熟读各类小说的专家，正在写标题为{novel_name}的小说，请你根据要求写一段800字左右的小说片段。"
    dataset = []
    dataset_error = []
    for chunk in tqdm(novel_chunks, desc=f"Processing {novel_name}", total=len(novel_chunks)):
        try:
            messages=[
            {
                'role': 'system', 
                'content': f'你是一个熟读各类小说的专家，正在阅读小说《{novel_name}》，请你用一句话总结这段小说的情节。字数控制在800字左右，仅回答总结，不需要添加其他内容'
            },
            # {
            #     'role': 'user', 
            #     'content': "他承认班纳特小姐是漂亮的，可惜她笑得太多。赫斯脱太太姐妹同意他这种看法……可是她们仍然羡慕她，喜欢她，说她是个甜姐儿，她们并不反对跟她这样的一位小姐做个深交。班纳特小姐就这样成为一个甜姐儿了，她们的兄弟听到了这番赞美，便觉得今后可以爱怎么样想她就怎么样想她了。\n距离浪博恩不远的地方，住着一家人家，这就是威廉·卢卡斯爵士府上。班纳特府上跟他们特别知已。爵士从前是在麦里屯做生意起家发迹的，曾在当市长的任内上书皇上，获得了一个爵士头衔；这个显要的身份使他觉得太荣幸，从此他就讨厌做生意，讨厌住在一个小镇上，于是歇了生意，告别小镇，带着家属迁到那离开麦里屯大约一英里路的一幢房子里去住，从那时候起就把那地方叫做卢家庄。他可以在这儿自得其乐，以显要自居，而且，既然摆脱了生意的纠缠，他大可以一心一意地从事社交活动。他尽管以自己的地位欣然自得，却并不因此而目空一切，反而对什么人都应酬得非常周到。他生来不肯得罪人，待人接物总是和蔼可亲，殷勤体贴，而且自从皇上觐见以来，更加彬彬有礼。卢卡斯太太是个很善良的女人，真是班纳特太太一位宝贵的邻居。卢府上有好几个孩子。大女儿是个明理懂事的年轻小姐，年纪大约二十六七岁，她是伊丽莎白的要好朋友。且说卢府上几位小姐跟班府上几位小姐这回非要见见面，谈谈这次跳舞会上的事业不可。于是在开完了跳舞会的第二天上午，卢府上的小姐们到浪博恩来跟班府上的小姐交换意见。\n班纳特太太一看见卢卡斯小姐，便客客气气，从容不迫地说："那天晚上全靠你开场开得好，你做了彬格莱先生的第一个意中人。""是呀；可是他喜欢的倒是第二个意中人。""哦，我想你是说吉英吧，因为他跟她跳了两次。看起来，他是真的爱上她呢……我的确相信他是真的……我听到了一些话……可是我弄不清究竟……我听到了一些有关鲁宾逊先生的话。""说不定你指的是我喻听到他和鲁宾逊先生的谈话吧；我不是跟你说过了吗？鲁宾逊先生问他喜欢不喜欢我们麦里屯的跳舞会，问他是否觉得到场的女宾们中间有许多人很美，问他认为哪一个最美？"
            # },
            # {
            #     'role': 'assistant', 
            #     'content': "浪博恩的班纳特家与卢卡斯家交好，班纳特家的二小姐伊丽莎白在舞会上因笑容过多而未得到达西的好感，但得到了彬格莱的青睐，卢卡斯家的大女儿夏洛特则是伊丽莎白的好友，两家人在舞会后讨论着舞会上的趣事和可能的姻缘。"
            # },
            {
                'role': 'user', 
                'content': chunk["chapter"] + "\n" + chunk["title"] + "\n" + chunk["content"]
                }
            ]
            summary = get_model_answer(model, tokenizer, messages)
            dataset.append({
                "instruction": instruction_prompt,
                "input": summary,
                "output": chunk["content"]
            })
        except Exception as e:
            dataset_error.append(chunk)
            logger.error(f"Failed to process text: {chunk}. Error: {e}")
            
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(dataset, ensure_ascii=False, indent=4))

    with open(f"{output_file}_error.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(dataset_error, ensure_ascii=False, indent=4))
    return dataset

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
    # with open("data/novel_chunks.json", "r", encoding="utf-8")as f:
    #     novel_chunks = json.load(f)
    # novel_name = "龙战士传说"
    # output_file = "data/novel_qa_control.json"
    # build_dataset_control(novel_name, novel_chunks, output_file)
    convert_novel_to_json()