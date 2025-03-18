#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import soundfile as sf
import numpy as np
import time
from ebook_reader import EbookReader
from client_http import prepare_request
import requests
import concurrent.futures
import re

def get_args():
    parser = argparse.ArgumentParser(
        description='将电子书转换为语音',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--ebook-path",
        type=str,
        required=True,
        help="电子书文件路径",
    )
    
    parser.add_argument(
        "--server-url",
        type=str,
        default="localhost:8000",
        help="TTS服务器地址",
    )
    
    parser.add_argument(
        "--reference-audio",
        type=str,
        default="../../example/prompt_audio.wav",
        help="参考音频文件路径",
    )
    
    parser.add_argument(
        "--reference-text",
        type=str,
        default="吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
        help="参考文本",
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="spark_tts",
        choices=["f5_tts", "spark_tts"],
        help="TTS模型名称",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output_audio",
        help="输出音频目录",
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="最大并发工作线程数",
    )
    
    parser.add_argument(
        "--chapter-range",
        type=str,
        default=None,
        help="要处理的章节范围，格式为'start-end'，例如'1-5'",
    )
    
    return parser.parse_args()

def clean_text(text):
    """清理文本，移除不必要的字符"""
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff,.，。!?！？:：;；()（）《》""\']+', '', text)
    return text.strip()

def split_long_text(text, max_length=1000):
    """将长文本分割成较短的段落"""
    if len(text) <= max_length:
        return [text]
    
    # 按句子分割
    sentences = re.split(r'([。！？.!?])', text)
    
    # 重新组合句子（保留分隔符）
    sentences = [''.join(sentences[i:i+2]) for i in range(0, len(sentences), 2) if i+1 < len(sentences)]
    
    # 合并短句子
    paragraphs = []
    current_paragraph = ""
    
    for sentence in sentences:
        if len(current_paragraph) + len(sentence) <= max_length:
            current_paragraph += sentence
        else:
            if current_paragraph:
                paragraphs.append(current_paragraph)
            current_paragraph = sentence
    
    if current_paragraph:
        paragraphs.append(current_paragraph)
    
    return paragraphs

def text_to_speech(text, server_url, reference_audio, reference_text, model_name, output_path):
    """将文本转换为语音"""
    # 清理文本
    text = clean_text(text)
    
    if not text:
        print(f"警告: 空文本，跳过生成 {output_path}")
        return False
    
    # 分割长文本
    paragraphs = split_long_text(text)
    all_audio = []
    
    # 读取参考音频
    waveform, sr = sf.read(reference_audio)
    assert sr == 16000, "sample rate hardcoded in server"
    samples = np.array(waveform, dtype=np.float32)
    
    # 确保服务器URL格式正确
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"
    
    url = f"{server_url}/v2/models/{model_name}/infer"
    
    # 处理每个段落
    for i, paragraph in enumerate(paragraphs):
        try:
            # 准备请求
            data = prepare_request(samples, reference_text, paragraph)
            
            # 发送请求
            rsp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=data,
                verify=False,
                params={"request_id": f'{i}'}
            )
            
            # 处理响应
            result = rsp.json()
            if "outputs" in result and len(result["outputs"]) > 0:
                audio = result["outputs"][0]["data"]
                audio = np.array(audio, dtype=np.float32)
                all_audio.append(audio)
                print(f"已生成段落 {i+1}/{len(paragraphs)} 的音频 ({len(audio)} 样本)")
            else:
                print(f"警告: 段落 {i+1}/{len(paragraphs)} 未返回音频数据")
                
        except Exception as e:
            print(f"错误: 处理段落 {i+1}/{len(paragraphs)} 时出错: {e}")
    
    # 合并所有音频
    if all_audio:
        combined_audio = np.concatenate(all_audio)
        
        # 保存音频
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, combined_audio, 16000, "PCM_16")
        print(f"已保存音频到: {output_path}")
        return True
    else:
        print(f"错误: 未能生成任何音频")
        return False

def process_chapter(args, chapter_id, chapter, reader):
    """处理单个章节"""
    chapter_title = chapter['title']
    print(f"\n处理章节 {chapter_id}: {chapter_title}")
    
    # 创建章节目录
    chapter_dir = os.path.join(args.output_dir, f"Chapter_{chapter_id:03d}")
    os.makedirs(chapter_dir, exist_ok=True)
    
    if chapter['sections']:
        # 有小节的情况
        for section_id, section in chapter['sections'].items():
            section_title = section['title']
            print(f"  处理小节 {chapter_id}.{section_id}: {section_title}")
            
            # 获取小节内容
            content = reader.get_section_content(chapter_id, section_id)
            
            # 生成音频文件名
            output_path = os.path.join(chapter_dir, f"Section_{section_id:03d}.wav")
            
            # 转换为语音
            success = text_to_speech(
                content, 
                args.server_url, 
                args.reference_audio, 
                args.reference_text, 
                args.model_name, 
                output_path
            )
            
            if success:
                print(f"  已完成小节 {chapter_id}.{section_id}")
            else:
                print(f"  小节 {chapter_id}.{section_id} 处理失败")
    else:
        # 没有小节的情况
        content = reader.get_chapter_content(chapter_id)
        
        # 生成音频文件名
        output_path = os.path.join(chapter_dir, "Chapter.wav")
        
        # 转换为语音
        success = text_to_speech(
            content, 
            args.server_url, 
            args.reference_audio, 
            args.reference_text, 
            args.model_name, 
            output_path
        )
        
        if success:
            print(f"已完成章节 {chapter_id}")
        else:
            print(f"章节 {chapter_id} 处理失败")
    
    return chapter_id

def main():
    args = get_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析章节范围
    chapter_range = None
    if args.chapter_range:
        try:
            start, end = map(int, args.chapter_range.split('-'))
            chapter_range = (start, end)
        except:
            print(f"警告: 无效的章节范围格式: {args.chapter_range}，将处理所有章节")
    
    try:
        # 创建电子书解析器
        reader = EbookReader(args.ebook_path)
        
        # 提取章节结构
        chapters = reader.extract_chapters()
        
        # 读取内容
        reader.read_contents(max_workers=args.max_workers)
        
        # 获取书籍信息
        book_info = reader.get_book_info()
        print(f"\n开始处理电子书: {book_info.get('title', '未知标题')}")
        
        # 筛选要处理的章节
        if chapter_range:
            start, end = chapter_range
            chapters_to_process = {
                chapter_id: chapter 
                for chapter_id, chapter in chapters.items() 
                if start <= chapter_id <= end
            }
            print(f"将处理章节 {start} 到 {end}，共 {len(chapters_to_process)} 个章节")
        else:
            chapters_to_process = chapters
            print(f"将处理所有章节，共 {len(chapters_to_process)} 个章节")
        
        # 并发处理章节
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            
            for chapter_id, chapter in chapters_to_process.items():
                future = executor.submit(process_chapter, args, chapter_id, chapter, reader)
                futures.append(future)
            
            # 等待所有任务完
            for future in concurrent.futures.as_completed(futures):
                try:
                    chapter_id = future.result()
                    print(f"章节 {chapter_id} 处理完成")
                except Exception as e:
                    print(f"处理章节时出错: {e}")
        
        print("\n所有章节处理完成!")
        
        # 创建一个简单的HTML索引文件
        index_path = os.path.join(args.output_dir, "index.html")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{book_info.get('title', '电子书')} - 有声版</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; margin-top: 30px; }}
        h3 {{ color: #777; margin-left: 20px; }}
        .chapter {{ margin-bottom: 20px; }}
        .section {{ margin-left: 40px; margin-bottom: 10px; }}
        audio {{ width: 100%; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>{book_info.get('title', '电子书')} - 有声版</h1>
    <p>作者: {', '.join(book_info.get('authors', ['未知']))}</p>
""")
            
            # 添加章节和小节
            for chapter_id, chapter in sorted(chapters_to_process.items()):
                chapter_dir = f"Chapter_{chapter_id:03d}"
                f.write(f'    <div class="chapter">\n')
                f.write(f'        <h2>第{chapter_id}章: {chapter["title"]}</h2>\n')
                
                if chapter['sections']:
                    # 有小节的情况
                    for section_id, section in sorted(chapter['sections'].items()):
                        section_file = f"{chapter_dir}/Section_{section_id:03d}.wav"
                        f.write(f'        <div class="section">\n')
                        f.write(f'            <h3>第{chapter_id}.{section_id}节: {section["title"]}</h3>\n')
                        f.write(f'            <audio controls src="{section_file}"></audio>\n')
                        f.write(f'        </div>\n')
                else:
                    # 没有小节的情况
                    chapter_file = f"{chapter_dir}/Chapter.wav"
                    f.write(f'        <audio controls src="{chapter_file}"></audio>\n')
                
                f.write(f'    </div>\n')
            
            f.write("""</body>
</html>""")
        
        print(f"已创建索引页面: {index_path}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())