#!/usr/bin/env python3

import argparse
import os
import time
import math
import shutil
import tempfile
import subprocess
import threading  # 用于创建线程（实现异步读取 ffmpeg 的输出）
import queue  # 用于线程间的安全通信（存储 ffmpeg 的输出）
import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--split-method",
        type=str,
        choices=["basic", "spacy", "transformers", "hybrid", "nature"],
        default="basic",
        help="文本分割方法: basic (基于标点), spacy (基于NLP), transformers (基于BERT), hybrid (混合方法), nature (自然语句分割)",
    )

    parser.add_argument(
        "--server-addr",
        type=str,
        default="localhost",
        help="Address of the server",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=8001,
        help="Grpc port of the triton server, default is 8001",
    )

    parser.add_argument(
        "--reference-audio",
        type=str,
        default="../../example/prompt_audio.wav",
        help="Path to a single audio file",
    )

    parser.add_argument(
        "--reference-text",
        type=str,
        default="吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
        help="Reference text for the prompt audio",
    )

    parser.add_argument(
        "--target-text",
        type=str,
        default="这是一个测试句子，用于检测文本长度限制。",
        help="Text to synthesize",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="./output_grpc.wav",
        help="Path to save output audio file",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="spark_tts",
        help="Model name in Triton server",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=40,  # 进一步减小默认块大小
        help="Maximum characters per chunk for long text",
    )
    
    parser.add_argument(
        "--use-chunking",
        action="store_true",
        help="Enable text chunking for long texts",
    )
    
    parser.add_argument(
        "--add-pause",
        action="store_true",
        help="Add short pause between chunks",
    )
    
    parser.add_argument(
        "--pause-duration",
        type=float,
        default=0.3,
        help="Duration of pause between chunks in seconds",
    )
    
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=5,  # 将默认值改为5，因为测试表明这是较好的值
        help="Number of characters to overlap between chunks for smoother transitions",
    )
    
    parser.add_argument(
        "--save-chunks",
        action="store_true",
        help="Save individual audio chunks for debugging",
    )
    
    parser.add_argument(
        "--trim-overlap",
        action="store_true",
        help="Trim overlapped audio to avoid repetition",
    )
    
    parser.add_argument(
        "--trim-factor",
        type=float,
        default=1.0,  # 降低默认修剪因子，避免过度裁剪
        help="Factor to adjust overlap trimming (higher values trim more)",
    )
    
    # 添加新参数，用于选择最佳处理模式
    parser.add_argument(
        "--processing-mode",
        type=str,
        choices=["balanced", "complete", "no-overlap"],
        default="balanced",
        help="Processing mode: balanced (best quality), complete (minimal trimming), no-overlap (no repetition)",
    )
    
    
    # 添加音频合并方法选择
    parser.add_argument(
        "--merge-method",
        type=str,
        choices=["basic", "pydub", "ffmpeg"],
        default="basic",
        help="Audio merging method: basic (numpy), pydub (crossfade), ffmpeg (professional)",
    )
    
    # 添加交叉淡入淡出参数
    parser.add_argument(
        "--crossfade-duration",
        type=float,
        default=0.05,
        help="Crossfade duration in seconds for audio merging (pydub/ffmpeg only)",
    )
    return parser.parse_args()

def split_text_with_nature(text, max_chunk_size=70, overlap_chars=0):
    """
    按照自然语句分割文本，避免文字重叠和过度分割
    
    参数:
        text: 要分割的文本
        max_chunk_size: 每个块的最大字符数
        overlap_chars: 重叠字符数（此参数仅为兼容接口，实际不使用）
    
    返回:
        分割后的文本块列表
    """
    # 步骤1: 预处理文本，去除空白和空行
    text = text.replace('\n', ' ')
    orig_text_trimmed = "\n".join([line.strip() for line in text.strip().split("\n") if line.strip()])
    
    # 如果处理后的文本长度小于最大块大小，直接返回
    if len(orig_text_trimmed) <= max_chunk_size:
        return [orig_text_trimmed]
    
    # 定义句子结束的标点符号（主要分割点）
    primary_breaks = ['。', '！', '？', '；', '.', '!', '?', ';', '…', '"', "'", '）', '】', '》', '」', '』', '〕', '〉', '〗', '〞', '〟', '—']
    # 定义次要分割点
    secondary_breaks = ['，', ',', '：', ':', '、', '-', '–', '~', '～', '·']
    # 所有可能的分割点
    all_breaks = primary_breaks + secondary_breaks
    
    # 步骤2: 按照自然语句分割文本
    chunks = []
    start_idx = 0
    current_idx = 0
    last_break_idx = -1
    
    while current_idx < len(orig_text_trimmed):
        char = orig_text_trimmed[current_idx]
        
        # 记录最后一个分割点的位置
        if char in all_breaks:
            last_break_idx = current_idx
        
        # 当达到最大块大小或接近最大块大小时，尝试在合适的位置分割
        if current_idx - start_idx >= max_chunk_size - 1:
            # 检查当前位置是否为分割点
            if char in all_breaks:
                chunks.append(orig_text_trimmed[start_idx:current_idx + 1])
                start_idx = current_idx + 1
            # 如果不是分割点，回退到上一个分割点
            elif last_break_idx > start_idx:
                chunks.append(orig_text_trimmed[start_idx:last_break_idx + 1])
                start_idx = last_break_idx + 1
            # 如果没有找到合适的分割点，强制分割
            else:
                chunks.append(orig_text_trimmed[start_idx:current_idx])
                start_idx = current_idx
        
        current_idx += 1
    
    # 添加最后一个块
    if start_idx < len(orig_text_trimmed):
        chunks.append(orig_text_trimmed[start_idx:])
    
    # 步骤3: 检查并优化分割结果
    optimized_chunks = []
    
    for i, chunk in enumerate(chunks):
        # 确保每个块不超过最大长度
        if len(chunk) > max_chunk_size:
            # 尝试在分割点处再次分割
            sub_chunks = []
            sub_start = 0
            
            for j in range(len(chunk)):
                if j - sub_start >= max_chunk_size - 1:
                    # 查找合适的分割点
                    sub_break_idx = -1
                    for k in range(j, sub_start, -1):
                        if chunk[k] in all_breaks:
                            sub_break_idx = k
                            break
                    
                    if sub_break_idx > sub_start:
                        sub_chunks.append(chunk[sub_start:sub_break_idx + 1])
                        sub_start = sub_break_idx + 1
                    else:
                        # 强制分割
                        sub_chunks.append(chunk[sub_start:j])
                        sub_start = j
            
            # 添加最后一个子块
            if sub_start < len(chunk):
                sub_chunks.append(chunk[sub_start:])
            
            optimized_chunks.extend(sub_chunks)
        else:
            optimized_chunks.append(chunk)
    
    # 步骤4: 检查相邻块是否可以合并
    final_chunks = []
    i = 0
    
    while i < len(optimized_chunks):
        current = optimized_chunks[i]
        
        # 如果不是最后一个块，且当前块和下一个块合并后不超过最大长度
        if i < len(optimized_chunks) - 1:
            next_chunk = optimized_chunks[i + 1]
            combined_length = len(current) + len(next_chunk)
            
            if combined_length <= max_chunk_size:
                final_chunks.append(current + next_chunk)
                i += 2  # 跳过下一个块
                continue
        
        # 如果不能合并，直接添加当前块
        final_chunks.append(current)
        i += 1
    
    # 步骤5: 检查并去除重叠内容
    result_chunks = []
    processed_text = ""
    
    for chunk in final_chunks:
        # 检查当前块是否与已处理文本有重叠
        if processed_text and chunk in processed_text:
            # 完全重叠，跳过
            continue
        elif processed_text:
            # 查找部分重叠
            overlap_start = -1
            for i in range(1, min(len(processed_text), len(chunk)) + 1):
                if processed_text[-i:] == chunk[:i]:
                    overlap_start = i
            
            if overlap_start > 0:
                # 去除重叠部分
                chunk = chunk[overlap_start:]
        
        # 确保块不为空且不超过最大长度
        if chunk and len(chunk) <= max_chunk_size:
            result_chunks.append(chunk)
            processed_text += chunk
    
    return result_chunks

def split_text_with_spacy(text, max_chunk_size=60, overlap_chars=0):
    """使用spaCy进行更智能的文本分割"""
    try:
        # 加载中文模型
        import spacy
        nlp = spacy.load("zh_core_web_sm")
    except (ImportError, OSError):
        print("警告: spaCy或中文模型未安装，尝试安装...")
        import subprocess
        try:
            subprocess.run(["pip", "install", "spacy"], check=True)
            subprocess.run(["python", "-m", "spacy", "download", "zh_core_web_sm"], check=True)
            nlp = spacy.load("zh_core_web_sm")
        except Exception as e:
            print(f"安装spaCy失败: {e}")
            print("回退到基本分割方法")
            return split_text_with_overlap(text, max_chunk_size, overlap_chars)
    
    # 使用spaCy处理文本
    doc = nlp(text)
    
    # 按句子分割
    sentences = list(doc.sents)
    
    # 组合句子成块，确保不超过最大长度
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
            
        # 如果当前句子加上当前块会超过最大长度，则保存当前块并开始新块
        if len(current_chunk) + len(sent_text) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = sent_text
        else:
            current_chunk += sent_text
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
    
    # 如果需要重叠，添加重叠部分
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i in range(len(chunks)):
            if i == 0:
                overlapped_chunks.append(chunks[i])
            else:
                prev_end = chunks[i-1][-overlap_chars:] if len(chunks[i-1]) >= overlap_chars else chunks[i-1]
                overlapped_chunks.append(prev_end + chunks[i])
        return overlapped_chunks
    
    return chunks
def split_text_with_transformers(text, max_chunk_size=60, overlap_chars=0):
    """使用Transformers进行文本分割，同时保留原始文本不被替换为[UNK]"""
    try:
        from transformers import AutoTokenizer
        import re
        # 加载预训练的中文分词模型
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    except ImportError:
        print("警告: transformers库未安装，尝试安装...")
        import subprocess
        try:
            subprocess.run(["pip", "install", "transformers"], check=True)
            from transformers import AutoTokenizer
            import re
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        except Exception as e:
            print(f"安装transformers失败: {e}")
            print("回退到基本分割方法")
            return split_text_with_overlap(text, max_chunk_size, overlap_chars)
    
    # 预处理：找出所有可能被识别为[UNK]的特殊术语
    # 使用正则表达式找出英文单词、数字和特殊符号，包括连续的单词
    special_terms_pattern = r'[a-zA-Z0-9_\-+.]+(?:\s+[a-zA-Z0-9_\-+.]+)*'
    special_terms = re.findall(special_terms_pattern, text)
    
    # 创建一个映射，记录每个特殊术语在文本中的位置
    term_positions = {}
    for term in special_terms:
        start = 0
        while True:
            pos = text.find(term, start)
            if pos == -1:
                break
            # 记录术语的起始位置和结束位置
            term_positions[(pos, pos + len(term))] = term
            start = pos + 1
    
    # 使用transformer的tokenizer进行分词
    encoded = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(encoded)
    
    # 重建文本并按最大长度分块
    chunks = []
    current_chunk_tokens = []
    current_chunk_text = ""
    token_positions = []  # 记录每个token在原始文本中的位置
    
    # 跟踪当前处理的文本位置
    current_pos = 0
    for token in tokens:
        # 如果是[UNK]，尝试从原始文本中恢复
        if token == "[UNK]":
            # 查找当前位置附近的特殊术语
            found_term = False
            for (start, end), term in term_positions.items():
                if start <= current_pos < end:
                    # 找到匹配的术语
                    token_text = term
                    token_len = len(term)
                    found_term = True
                    break
            
            if not found_term:
                # 如果没找到匹配的术语，保留[UNK]
                token_text = "[UNK]"
                token_len = 1
        else:
            # 正常token，去除##前缀
            token_text = token.replace("##", "")
            token_len = len(token_text)
        
        # 记录token位置
        token_positions.append((current_pos, current_pos + token_len))
        current_pos += token_len
        
        # 如果当前块加上新token会超过最大长度，保存当前块并开始新块
        if len(current_chunk_tokens) + 1 > max_chunk_size and current_chunk_tokens:
            # 查找合适的断句点
            break_points = ['.', '。', '!', '！', '?', '？', ';', '；', ',', '，']
            found_break = False
            
            # 从后向前查找断句点
            for i in range(len(current_chunk_text)-1, max(0, len(current_chunk_text)-10), -1):
                if current_chunk_text[i] in break_points:
                    # 找到断句点，分割文本
                    chunk_end_pos = token_positions[len(current_chunk_tokens)-1][1]
                    chunk_text = text[:chunk_end_pos]
                    chunks.append(chunk_text)
                    
                    # 更新文本和位置信息
                    text = text[chunk_end_pos:]
                    
                    # 更新term_positions
                    new_term_positions = {}
                    for (start, end), term in term_positions.items():
                        if start >= chunk_end_pos:
                            new_term_positions[(start - chunk_end_pos, end - chunk_end_pos)] = term
                    term_positions = new_term_positions
                    
                    # 重置当前块
                    current_chunk_tokens = []
                    current_chunk_text = ""
                    token_positions = []
                    current_pos = 0
                    
                    # 重新进行分词
                    encoded = tokenizer.encode(text, add_special_tokens=False)
                    tokens = tokenizer.convert_ids_to_tokens(encoded)
                    
                    found_break = True
                    break
            
            # 如果没找到合适的断句点，直接截断
            if not found_break:
                chunk_end_pos = token_positions[len(current_chunk_tokens)-1][1]
                chunk_text = text[:chunk_end_pos]
                chunks.append(chunk_text)
                
                # 更新文本和位置信息
                text = text[chunk_end_pos:]
                
                # 更新term_positions
                new_term_positions = {}
                for (start, end), term in term_positions.items():
                    if start >= chunk_end_pos:
                        new_term_positions[(start - chunk_end_pos, end - chunk_end_pos)] = term
                term_positions = new_term_positions
                
                # 重置当前块
                current_chunk_tokens = []
                current_chunk_text = ""
                token_positions = []
                current_pos = 0
                
                # 重新进行分词
                encoded = tokenizer.encode(text, add_special_tokens=False)
                tokens = tokenizer.convert_ids_to_tokens(encoded)
                
                # 如果文本已经处理完，跳出循环
                if not text:
                    break
        
        # 添加当前token到chunk
        current_chunk_tokens.append(token)
        current_chunk_text += token_text
    
    # 添加最后一个块
    if text:
        chunks.append(text)
    
    # 处理重叠
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i in range(len(chunks)):
            if i == 0:
                overlapped_chunks.append(chunks[i])
            else:
                prev_end = chunks[i-1][-overlap_chars:] if len(chunks[i-1]) >= overlap_chars else chunks[i-1]
                overlapped_chunks.append(prev_end + chunks[i])
        return overlapped_chunks
    
    return chunks
def split_text_with_transformers_backup(text, max_chunk_size=60, overlap_chars=0):
    """使用Transformers进行文本分割，同时保留原始文本不被替换为[UNK]"""
    try:
        from transformers import AutoTokenizer
        import re
        # 加载预训练的中文分词模型
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    except ImportError:
        print("警告: transformers库未安装，尝试安装...")
        import subprocess
        try:
            subprocess.run(["pip", "install", "transformers"], check=True)
            from transformers import AutoTokenizer
            import re
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        except Exception as e:
            print(f"安装transformers失败: {e}")
            print("回退到基本分割方法")
            return split_text_with_overlap(text, max_chunk_size, overlap_chars)
    
    # 预处理：标记可能被识别为[UNK]的英文单词和专业术语
    # 使用正则表达式找出英文单词和数字
    special_terms = re.findall(r'[a-zA-Z0-9]+', text)
    term_positions = {}
    
    # 记录每个特殊术语在文本中的位置
    for term in special_terms:
        start = 0
        while True:
            pos = text.find(term, start)
            if pos == -1:
                break
            # 修改这里：将键值对改为 pos -> (end_pos, term)
            term_positions[pos] = (pos + len(term), term)
            start = pos + 1
    
    # 使用基本分割方法先获取初步的块
    basic_chunks = split_text_by_sentence(text, max_chunk_size)
    
    # 处理每个块，确保英文单词和数字不被替换为[UNK]
    processed_chunks = []
    for chunk in basic_chunks:
        # 检查这个块中是否有特殊术语
        chunk_terms = {}
        # 修改这里：正确处理term_positions的键值对
        for pos, (end_pos, term) in term_positions.items():
            # 找出在当前块范围内的术语
            chunk_start = text.find(chunk)
            chunk_end = chunk_start + len(chunk)
            if chunk_start <= pos < chunk_end:
                relative_start = pos - chunk_start
                chunk_terms[relative_start] = term
        
        # 使用tokenizer处理，但保留特殊术语
        tokens = tokenizer.tokenize(chunk)
        reconstructed = ""
        pos = 0
        
        for token in tokens:
            if token == "[UNK]" and pos in chunk_terms:
                # 使用原始术语替换[UNK]
                reconstructed += chunk_terms[pos]
                pos += len(chunk_terms[pos])
            else:
                # 正常添加token
                token_text = token.replace("##", "")
                reconstructed += token_text
                pos += len(token_text)
        
        processed_chunks.append(reconstructed)
    
    # 处理重叠
    if overlap_chars > 0 and len(processed_chunks) > 1:
        overlapped_chunks = []
        for i in range(len(processed_chunks)):
            if i == 0:
                overlapped_chunks.append(processed_chunks[i])
            else:
                prev_end = processed_chunks[i-1][-overlap_chars:] if len(processed_chunks[i-1]) >= overlap_chars else processed_chunks[i-1]
                overlapped_chunks.append(prev_end + processed_chunks[i])
        return overlapped_chunks
    
    return processed_chunks

def split_text_with_transformers_orig(text, max_chunk_size=60, overlap_chars=0):
    """使用Transformers进行文本分割"""
    try:
        from transformers import AutoTokenizer
        import re
        # 加载预训练的中文分词模型
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    except ImportError:
        print("警告: transformers库未安装，尝试安装...")
        import subprocess
        try:
            subprocess.run(["pip", "install", "transformers"], check=True)
            from transformers import AutoTokenizer
            import re
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        except Exception as e:
            print(f"安装transformers失败: {e}")
            print("回退到基本分割方法")
            return split_text_with_overlap(text, max_chunk_size, overlap_chars)
    
    # 预处理：标记可能被识别为[UNK]的英文单词和专业术语
    # 使用正则表达式找出英文单词和数字
    special_terms = re.findall(r'[a-zA-Z0-9]+', text)
    term_positions = {}
    
    # 记录每个特殊术语在文本中的位置
    for term in special_terms:
        start = 0
        while True:
            pos = text.find(term, start)
            if pos == -1:
                break
            term_positions[pos] = (pos + len(term), term)
            start = pos + 1
    
    # 使用tokenizer分词，但保留原始文本信息
    tokens = tokenizer.tokenize(text)
    token_text_map = []  # 存储token与原始文本的映射
    
    
    # 重建文本并按最大长度分块
    chunks = []
    current_chunk = ""
    current_tokens = []
    
    for token in tokens:
        if len(current_tokens) + 1 > max_chunk_size:
            # 查找合适的断句点
            break_points = ['.', '。', '!', '！', '?', '？', ';', '；', ',', '，']
            found_break = False
            
            # 从后向前查找断句点
            for i in range(len(current_chunk)-1, max(0, len(current_chunk)-10), -1):
                if current_chunk[i] in break_points:
                    chunks.append(current_chunk[:i+1])
                    current_chunk = current_chunk[i+1:]
                    current_tokens = tokenizer.tokenize(current_chunk)
                    found_break = True
                    break
            
            # 如果没找到合适的断句点，直接截断
            if not found_break:
                chunks.append(current_chunk)
                current_chunk = ""
                current_tokens = []
        
        # 添加当前token到chunk，处理[UNK]标记
        if token == "[UNK]":
            # 尝试从原始文本中恢复
            pos = len(current_chunk)
            if pos in term_positions:
                _, term = term_positions[pos]
                current_chunk += term
            else:
                # 如果无法恢复，保留[UNK]
                current_chunk += token
        else:
            current_chunk += token.replace("##", "")
        
        current_tokens.append(token)
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
    
    # 处理重叠
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i in range(len(chunks)):
            if i == 0:
                overlapped_chunks.append(chunks[i])
            else:
                prev_end = chunks[i-1][-overlap_chars:] if len(chunks[i-1]) >= overlap_chars else chunks[i-1]
                overlapped_chunks.append(prev_end + chunks[i])
        return overlapped_chunks
    
    return chunks

def split_text_hybrid(text, max_chunk_size=60, overlap_chars=0):
    """混合分割方法：使用spaCy进行句子分割，同时保留原始文本"""
    try:
        import spacy
        import re
        # 加载中文模型
        nlp = spacy.load("zh_core_web_sm")
    except (ImportError, OSError):
        print("警告: spaCy或中文模型未安装，回退到基本分割方法")
        return split_text_by_sentence(text, max_chunk_size, overlap_chars)
    
    # 预处理：保护英文单词和数字，避免被错误分割
    # 使用正则表达式找出英文单词和数字
    special_terms = re.findall(r'[a-zA-Z0-9]+', text)
    term_positions = {}
    
    # 记录每个特殊术语在文本中的位置
    for term in special_terms:
        start = 0
        while True:
            pos = text.find(term, start)
            if pos == -1:
                break
            term_positions[(pos, pos + len(term))] = term
            start = pos + 1
    
    # 使用spaCy进行句子分割
    doc = nlp(text)
    sentences = list(doc.sents)
    
    # 组合句子成块，确保不超过最大长度
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
            
        # 如果当前句子加上当前块会超过最大长度，则保存当前块并开始新块
        if len(current_chunk) + len(sent_text) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = sent_text
        else:
            current_chunk += sent_text
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
    
    # 如果需要重叠，添加重叠部分
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i in range(len(chunks)):
            if i == 0:
                overlapped_chunks.append(chunks[i])
            else:
                prev_end = chunks[i-1][-overlap_chars:] if len(chunks[i-1]) >= overlap_chars else chunks[i-1]
                overlapped_chunks.append(prev_end + chunks[i])
        return overlapped_chunks
    
    return chunks
    
def split_text_hybrid_orgi(text, max_chunk_size=60, overlap_chars=0):
    """混合分割方法：使用spaCy进行句子分割，同时保留原始文本"""
    try:
        import spacy
        import re
        # 加载中文模型
        nlp = spacy.load("zh_core_web_sm")
    except (ImportError, OSError):
        print("警告: spaCy或中文模型未安装，回退到transformers方法")
        return split_text_with_transformers(text, max_chunk_size, overlap_chars)
    
    # 使用spaCy进行句子分割
    doc = nlp(text)
    sentences = list(doc.sents)
    
    # 组合句子成块，确保不超过最大长度
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
            
        # 如果当前句子加上当前块会超过最大长度，则保存当前块并开始新块
        if len(current_chunk) + len(sent_text) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = sent_text
        else:
            current_chunk += sent_text
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
    
    # 如果需要重叠，添加重叠部分
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i in range(len(chunks)):
            if i == 0:
                overlapped_chunks.append(chunks[i])
            else:
                prev_end = chunks[i-1][-overlap_chars:] if len(chunks[i-1]) >= overlap_chars else chunks[i-1]
                overlapped_chunks.append(prev_end + chunks[i])
        return overlapped_chunks
    
    return chunks

def load_audio(wav_path, target_sample_rate=16000):
    assert target_sample_rate == 16000, "hard coding in server"
    waveform, sample_rate = sf.read(wav_path)
    if sample_rate != target_sample_rate:
        from scipy.signal import resample
        num_samples = int(len(waveform) * (target_sample_rate / sample_rate))
        waveform = resample(waveform, num_samples)
    return waveform, target_sample_rate

def process_text_chunk(triton_client, args, waveform, sample_rate, text_chunk):
    """处理单个文本块并返回音频"""
    lengths = np.array([[len(waveform)]], dtype=np.int32)
    samples = waveform.reshape(1, -1).astype(np.float32)
    
    # 准备请求输入
    inputs = [
        grpcclient.InferInput(
            "reference_wav", samples.shape, np_to_triton_dtype(samples.dtype)
        ),
        grpcclient.InferInput(
            "reference_wav_len", lengths.shape, np_to_triton_dtype(lengths.dtype)
        ),
        grpcclient.InferInput("reference_text", [1, 1], "BYTES"),
        grpcclient.InferInput("target_text", [1, 1], "BYTES")
    ]
    
    inputs[0].set_data_from_numpy(samples)
    inputs[1].set_data_from_numpy(lengths)
    
    reference_text_data = np.array([args.reference_text], dtype=object).reshape((1, 1))
    inputs[2].set_data_from_numpy(reference_text_data)
    
    target_text_data = np.array([text_chunk], dtype=object).reshape((1, 1))
    inputs[3].set_data_from_numpy(target_text_data)
    
    # 准备输出
    outputs = [grpcclient.InferRequestedOutput("waveform")]
    
    # 发送请求
    response = triton_client.infer(
        args.model_name, inputs, request_id="1", outputs=outputs
    )
    
    # 处理响应
    audio = response.as_numpy("waveform").reshape(-1)
    return audio

def split_text_by_sentence(text, max_chunk_size=60):
    """按句子边界分割文本，确保每个块不超过最大长度"""
    # 定义句子结束的标点符号
    sentence_endings = ['。', '！', '？', '；', '.', '!', '?', ';']
    # 定义次要分割点
    secondary_breaks = ['，', ',', '：', ':', '、']
    
    # 如果文本长度小于最大块大小，直接返回
    if len(text) <= max_chunk_size:
        return [text]
    
    # 首先尝试按句号等主要标点分割
    chunks = []
    start_idx = 0
    
    # 遍历文本寻找句子结束点
    for i in range(len(text)):
        # 如果找到句子结束符，并且当前块长度适中
        if text[i] in sentence_endings and i - start_idx + 1 >= max_chunk_size * 0.5:
            chunks.append(text[start_idx:i+1])
            start_idx = i + 1
        # 如果当前块已经太长，需要在次要分割点处分割
        elif i - start_idx + 1 >= max_chunk_size:
            # 向后查找最近的次要分割点
            found_break = False
            for j in range(i, max(start_idx, i - 20), -1):
                if text[j] in secondary_breaks:
                    chunks.append(text[start_idx:j+1])
                    start_idx = j + 1
                    found_break = True
                    break
            
            # 如果没找到合适的分割点，强制分割
            if not found_break:
                chunks.append(text[start_idx:i+1])
                start_idx = i + 1
    
    # 处理剩余文本
    if start_idx < len(text):
        # 如果剩余文本很短，尝试合并到上一个块
        if chunks and len(text) - start_idx < max_chunk_size * 0.3:
            if len(chunks[-1]) + (len(text) - start_idx) <= max_chunk_size:
                chunks[-1] += text[start_idx:]
            else:
                chunks.append(text[start_idx:])
        else:
            chunks.append(text[start_idx:])
    
    # 最后检查一遍，确保没有块超过最大长度
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= max_chunk_size:
            final_chunks.append(chunk)
        else:
            # 尝试在次要分割点处再次分割
            temp = ""
            sub_chunks = []
            for char in chunk:
                temp += char
                if len(temp) >= max_chunk_size * 0.7 and char in secondary_breaks + sentence_endings:
                    sub_chunks.append(temp)
                    temp = ""
            
            if temp:
                sub_chunks.append(temp)
            
            # 如果仍然没有找到合适的分割点，强制按长度分割
            if not sub_chunks:
                for i in range(0, len(chunk), max_chunk_size - 5):  # 留5个字符的余量
                    sub_chunks.append(chunk[i:min(i + max_chunk_size - 5, len(chunk))])
            
            final_chunks.extend(sub_chunks)
    
    return final_chunks

def split_text_with_overlap(text, max_chunk_size=60, overlap_chars=0):
    """Split text into chunks with optional overlap between chunks"""
    # First get the basic chunks
    basic_chunks = split_text_by_sentence(text, max_chunk_size)
    
    # If no overlap needed or only one chunk, return as is
    if overlap_chars <= 0 or len(basic_chunks) <= 1:
        return basic_chunks
    
    # Create chunks with overlap
    overlapped_chunks = []
    for i in range(len(basic_chunks)):
        if i == 0:
            # First chunk remains unchanged
            overlapped_chunks.append(basic_chunks[i])
        else:
            # Get the end of the previous chunk to add as overlap
            prev_end = basic_chunks[i-1][-overlap_chars:] if len(basic_chunks[i-1]) >= overlap_chars else basic_chunks[i-1]
            # Add this overlap to the beginning of the current chunk
            overlapped_chunks.append(prev_end + basic_chunks[i])
    
    return overlapped_chunks

def process_chunks_with_overlap_trimming(triton_client, args, waveform, sample_rate, chunks):
    """处理文本块并处理重叠部分"""
    audio_segments = []
    chunk_durations = []
    
    # 根据处理模式调整参数
    if args.processing_mode == "balanced":
        # 平衡模式 - 适度修剪，保持音频质量
        char_duration_estimate = 0.22  # 稍微降低估计值
        adaptive_factor = 1.0  # 较低的修剪因子
    elif args.processing_mode == "complete":
        # 完整模式 - 最小修剪，确保内容完整
        char_duration_estimate = 0.18  # 更低的估计值
        adaptive_factor = 0.8  # 更低的修剪因子
    elif args.processing_mode == "no-overlap":
        # 无重叠模式 - 更激进的修剪，消除所有重叠
        char_duration_estimate = 0.25  # 较高的估计值
        adaptive_factor = 1.5  # 较高的修剪因子
    else:
        # 使用用户指定的值
        char_duration_estimate = 0.25
        adaptive_factor = args.trim_factor
    
    # 记录处理模式
    if args.trim_overlap:
        print(f"使用 {args.processing_mode} 处理模式 (字符时长估计: {char_duration_estimate}秒, 修剪因子: {adaptive_factor})")
    
    for i, chunk in enumerate(chunks):
        print(f"处理块 {i+1}/{len(chunks)}...")
        
        # 添加重试逻辑
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                chunk_audio = process_text_chunk(triton_client, args, waveform, sample_rate, chunk)
                success = True
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"处理块 {i+1} 失败，正在重试 ({retry_count}/{max_retries})...")
                    time.sleep(1)  # 等待一秒再重试
                else:
                    print(f"处理块 {i+1} 失败，跳过此块: {e}")
                    # 使用一小段静音代替失败的块
                    chunk_audio = np.zeros(int(1.0 * sample_rate), dtype=np.float32)
        
        # 记录块的预期时长
        chunk_duration = len(chunk) / 4.5
        chunk_durations.append(chunk_duration)
        
        # 保存单独的块音频
        if args.save_chunks:
            chunk_filename = f"{os.path.splitext(args.output_file)[0]}_chunk_{i+1}.wav"
            sf.write(chunk_filename, chunk_audio, sample_rate, "PCM_16")
            print(f"块 {i+1} 音频已保存到 {chunk_filename}")
        
        # 如果需要修剪重叠部分
        if args.trim_overlap and i > 0 and args.overlap_chars > 0:
            # 计算当前块中重叠字符的实际数量
            actual_overlap_chars = min(args.overlap_chars, len(chunks[i-1]))
            
            # 根据实际重叠字符数和自适应因子调整裁剪长度
            adjusted_overlap_samples = int(actual_overlap_chars * char_duration_estimate * sample_rate * adaptive_factor)
            
            # 确保不会裁剪过多 - 降低最大裁剪比例
            max_trim = min(adjusted_overlap_samples, len(chunk_audio) // 4)  # 最多裁剪音频的1/4
            
            # 只保留当前块的非重叠部分
            if len(chunk_audio) > max_trim:
                # 创建一个新数组而不是修改原始数组
                trimmed_audio = chunk_audio[max_trim:].copy()
                
                # 使用淡入效果使过渡更自然
                fade_length = min(int(0.05 * sample_rate), len(trimmed_audio))  # 50ms的淡入
                if fade_length > 0:
                    fade_in = np.linspace(0, 1, fade_length)
                    trimmed_audio[:fade_length] = trimmed_audio[:fade_length] * fade_in
                
                chunk_audio = trimmed_audio
        
        audio_segments.append(chunk_audio)
        
        # 如果需要，在块之间添加短暂的停顿
        if args.add_pause and i < len(chunks) - 1:
            # 使用用户指定的停顿时长
            pause_length = int(args.pause_duration * sample_rate)
            pause = np.zeros(pause_length, dtype=np.float32)
            audio_segments.append(pause)
        
        print(f"块 {i+1} 处理完成，生成音频长度: {len(chunk_audio)/sample_rate:.2f} 秒")

    # 根据选择的合并方法合并所有音频段
    print(f"使用 {args.merge_method} 方法合并音频片段...")
    if args.merge_method == "pydub":
        crossfade_ms = int(args.crossfade_duration * 1000)  # 转换为毫秒
        audio = merge_audio_with_pydub(audio_segments, sample_rate, crossfade_ms)
        print(f"使用pydub合并音频，应用 {crossfade_ms}ms 交叉淡入淡出")
    elif args.merge_method == "ffmpeg":
        audio = merge_audio_with_ffmpeg_google(audio_segments, sample_rate, None, args.crossfade_duration)
        print(f"使用FFmpeg合并音频，应用 {args.crossfade_duration}秒 交叉淡入淡出")
    else:
        # 基本方法 - 直接连接
        audio = np.concatenate(audio_segments)
        print("使用基本方法合并音频（直接连接）")
    
    return audio, chunk_durations    

def analyze_audio_quality(audio, sample_rate, chunk_durations):
    """分析生成的音频质量，检测潜在问题"""
    issues = []
    
    # 检查音频是否太短
    expected_min_duration = sum(chunk_durations) * 0.8
    actual_duration = len(audio) / sample_rate
    if actual_duration < expected_min_duration:
        issues.append(f"警告: 音频总时长({actual_duration:.2f}秒)明显短于预期最小时长({expected_min_duration:.2f}秒)")
    
    # 检查音频是否有突然的静音段（可能表示截断）
    chunk_size = int(0.1 * sample_rate)  # 分析窗口大小：0.1秒
    for i in range(0, len(audio) - chunk_size, chunk_size):
        chunk = audio[i:i+chunk_size]
        rms = np.sqrt(np.mean(chunk**2))
        if rms < 0.01 and i > chunk_size and i < len(audio) - 2*chunk_size:
            # 检查前后是否都有声音
            prev_rms = np.sqrt(np.mean(audio[i-chunk_size:i]**2))
            next_rms = np.sqrt(np.mean(audio[i+chunk_size:i+2*chunk_size]**2))
            if prev_rms > 0.05 and next_rms > 0.05:
                issues.append(f"警告: 在 {i/sample_rate:.2f} 秒处检测到可能的不自然静音")
    
    return issues

def merge_audio_with_pydub(audio_segments, sample_rate=16000, crossfade_ms=50):
    """使用pydub合并音频片段，支持交叉淡入淡出"""
    try:
        from pydub import AudioSegment
        import io
        import numpy as np
        import wave
    except ImportError:
        print("警告: pydub库未安装，尝试安装...")
        import subprocess
        try:
            subprocess.run(["pip", "install", "pydub"], check=True)
            from pydub import AudioSegment
            import io
            import wave
        except Exception as e:
            print(f"安装pydub失败: {e}")
            print("回退到基本合并方法")
            return np.concatenate(audio_segments)
    
    # 将NumPy数组转换为AudioSegment对象
    pydub_segments = []
    for i, segment in enumerate(audio_segments):
        # 将float32转换为16位PCM
        segment_int16 = (segment * 32767).astype(np.int16)
        
        # 创建内存文件对象
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16位
            wf.setframerate(sample_rate)
            wf.writeframes(segment_int16.tobytes())
        
        buffer.seek(0)
        pydub_segment = AudioSegment.from_wav(buffer)
        
        # 对每个片段应用淡入淡出效果，使过渡更自然
        if len(pydub_segment) > 100:  # 确保片段足够长
            fade_duration = min(crossfade_ms, len(pydub_segment) // 4)  # 不超过片段长度的1/4
            pydub_segment = pydub_segment.fade_in(fade_duration).fade_out(fade_duration)
            print(f"片段 {i+1}: 应用了 {fade_duration}ms 的淡入淡出效果")
        
        pydub_segments.append(pydub_segment)
    
    # 如果没有片段，返回空数组
    if not pydub_segments:
        return np.zeros(0, dtype=np.float32)
    
    # 如果只有一个片段，直接返回
    if len(pydub_segments) == 1:
        return audio_segments[0]
    
    # 合并片段，应用交叉淡入淡出
    print(f"使用pydub合并 {len(pydub_segments)} 个音频片段，应用 {crossfade_ms}ms 交叉淡入淡出")
    
    # 自适应调整交叉淡入淡出时长
    result = pydub_segments[0]
    for i in range(1, len(pydub_segments)):
        # 计算当前片段和下一个片段的长度
        current_len = len(result)
        next_len = len(pydub_segments[i])
        
        # 自适应调整交叉淡入淡出时长，不超过两个片段长度的1/5
        adaptive_crossfade = min(crossfade_ms, current_len // 5, next_len // 5)
        
        if adaptive_crossfade < 10:  # 如果太短，使用简单连接
            result = result + pydub_segments[i]
            print(f"片段 {i}: 片段太短，使用简单连接")
        else:
            # 应用交叉淡入淡出
            result = result.append(pydub_segments[i], crossfade=adaptive_crossfade)
            print(f"片段 {i+1}: 应用了 {adaptive_crossfade}ms 的交叉淡入淡出")
    
    # 将结果转换回NumPy数组
    buffer = io.BytesIO()
    result.export(buffer, format="wav")
    buffer.seek(0)
    
    with wave.open(buffer, 'rb') as wf:
        sample_width = wf.getsampwidth()
        n_channels = wf.getnchannels()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        
        # 读取所有帧
        frames = wf.readframes(n_frames)
        
        # 将字节转换为NumPy数组
        if sample_width == 2:  # 16位
            dtype = np.int16
        elif sample_width == 4:  # 32位
            dtype = np.int32
        else:
            dtype = np.uint8
        
        result_array = np.frombuffer(frames, dtype=dtype)
        
        # 如果是立体声，转换为单声道
        if n_channels == 2:
            result_array = result_array.reshape(-1, 2).mean(axis=1)
        
        # 转换为float32
        result_array = result_array.astype(np.float32) / 32767.0
    
    print(f"音频合并完成，最终时长: {len(result_array)/sample_rate:.2f}秒")
    return result_array

def merge_audio_with_ffmpeg(audio_segments, sample_rate=16000, output_file=None, crossfade_duration=0.05):
    """使用FFmpeg合并音频片段，支持高级音频处理"""
    
    # 检查ffmpeg是否已安装
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("警告: ffmpeg未安装或无法运行")
        print("回退到pydub合并方法...")
        
        # 尝试使用pydub作为备选方案
        try:
            crossfade_ms = int(crossfade_duration * 1000)
            return merge_audio_with_pydub(audio_segments, sample_rate, crossfade_ms)
        except Exception as e:
            print(f"pydub合并失败: {e}")
            print("回退到基本合并方法")
            return np.concatenate(audio_segments)

    if output_file is None:
        # 创建临时输出文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
            output_file = temp_output.name
    
    # 创建临时目录存放音频片段
    with tempfile.TemporaryDirectory() as temp_dir:
        segment_files = []
        
        # 保存每个音频片段为临时文件
        for i, segment in enumerate(audio_segments):
            segment_file = os.path.join(temp_dir, f"segment_{i}.wav")
            segment_files.append(segment_file)
            
            # 保存为WAV文件
            sf.write(segment_file, segment, sample_rate, 'PCM_16')
        
        # 使用两阶段方法进行合并
        # 第一阶段：使用concat协议合并所有片段
        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, 'w') as f:
            for segment_file in segment_files:
                f.write(f"file '{segment_file}'\n")
        
        # 先创建一个简单合并的临时文件
        temp_merged = os.path.join(temp_dir, "temp_merged.wav")
        concat_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
            "-i", concat_file, "-c:a", "pcm_s16le", "-ar", str(sample_rate), temp_merged
        ]
        
        try:
            subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # 第二阶段：如果需要交叉淡入淡出，应用音频滤镜
            if crossfade_duration > 0:
                # 计算交叉淡入淡出的样本数
                crossfade_samples = int(crossfade_duration * sample_rate)
                
                # 使用单独的滤镜命令应用淡入淡出效果
                fade_cmd = [
                    "ffmpeg", "-y", "-i", temp_merged,
                    "-filter_complex", f"afade=t=in:st=0:d={crossfade_duration},afade=t=out:st={len(audio_segments) * 2 - crossfade_duration}:d={crossfade_duration}",
                    "-c:a", "pcm_s16le", "-ar", str(sample_rate), output_file
                ]
                
                subprocess.run(fade_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            else:
                # 如果不需要交叉淡入淡出，直接使用合并后的文件
                shutil.copy(temp_merged, output_file)
                
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg合并失败: {e}")
            print(f"FFmpeg错误输出: {e.stderr.decode('utf-8', errors='replace')}")
            print("尝试备用方法...")
            
            # 备用方法：使用更简单的命令
            try:
                simple_cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
                    "-i", concat_file, "-c", "copy", output_file
                ]
                subprocess.run(simple_cmd, check=True)
            except subprocess.CalledProcessError:
                print("备用方法也失败，回退到基本合并方法")
                return np.concatenate(audio_segments)
        
        # 读取合并后的音频
        merged_audio, _ = sf.read(output_file)
        
        # 如果是临时文件，删除它
        if output_file.startswith(tempfile.gettempdir()):
            os.unlink(output_file)
        
        return merged_audio

def reader_thread(pipe, queue):
    """读取管道中的数据并放入队列
       参数:
           pipe: 要读取的管道 (subprocess.Popen.stdout 或 subprocess.Popen.stderr)
           queue: 用于存储读取到的数据的队列 (queue.Queue)
    """
    try:
        with pipe:  # 确保管道在使用完毕后被关闭
            for line in iter(pipe.readline, ''):  # 逐行读取管道中的数据，直到遇到空字符串（表示管道结束）
                queue.put(line)  # 将读取到的每一行数据放入队列
    except Exception as e:  # 捕获可能出现的异常
        queue.put(e)  # 将异常对象也放入队列，以便主线程处理


def get_audio_duration(file_path):
    """获取音频文件的时长（秒）"""
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
               "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"获取音频时长失败: {e}")
        return 0

def merge_audio_with_ffmpeg_google(audio_segments, sample_rate=16000, output_file=None, crossfade_duration=0.1):
    """使用FFmpeg合并音频片段，使用改进的方法实现平滑过渡"""

    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("警告: ffmpeg未安装或无法运行")
        print("回退到基本合并方法...")
        return np.concatenate(audio_segments)

    if output_file is None:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
            output_file = temp_output.name

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"临时目录: {temp_dir}")
        segment_files = []
        normalized_files = []

        # 第一步：保存音频片段并进行音量标准化
        for i, segment in enumerate(audio_segments):
            segment_file = os.path.join(temp_dir, f"segment_{i}.wav")
            normalized_file = os.path.join(temp_dir, f"normalized_{i}.wav")
            
            # 显式转换为 int16 (如果输入是 float 类型)
            if segment.dtype.kind == 'f':
                segment = (segment * 32767).astype(np.int16)
            sf.write(segment_file, segment, sample_rate, 'PCM_16')
            segment_files.append(segment_file)
            
            # 检查生成的文件
            if not os.path.exists(segment_file) or os.path.getsize(segment_file) == 0:
                print(f"警告: 片段 {i} 文件创建失败或为空")
                continue
                
            # 对每个片段进行音量标准化和动态范围压缩
            try:
                normalize_cmd = [
                    "ffmpeg", "-y", "-i", segment_file,
                    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=summary,acompressor=threshold=-20dB:ratio=3:attack=150:release=1000",
                    "-c:a", "pcm_s16le", "-ar", str(sample_rate), normalized_file
                ]
                subprocess.run(normalize_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if os.path.exists(normalized_file) and os.path.getsize(normalized_file) > 0:
                    normalized_files.append(normalized_file)
                    print(f"片段 {i}: 音量标准化成功，时长: {get_audio_duration(normalized_file):.2f}秒")
                else:
                    print(f"警告: 片段 {i} 音量标准化失败，使用原始片段")
                    normalized_files.append(segment_file)
            except Exception as e:
                print(f"片段 {i} 音量标准化失败: {e}，使用原始片段")
                normalized_files.append(segment_file)
        
        # 如果没有有效的音频片段，返回空数组
        if not normalized_files:
            print("没有有效的音频片段，返回空音频")
            return np.zeros(0, dtype=np.float32)

        original_dir = os.getcwd()
        os.chdir(temp_dir)

        try:
            # 如果只有一个片段，直接复制
            if len(normalized_files) == 1:
                shutil.copy(normalized_files[0], output_file)
                merged_audio, _ = sf.read(output_file)
                os.chdir(original_dir)
                return merged_audio
            
            # 创建concat文件
            concat_file = os.path.join(temp_dir, "concat.txt")
            with open(concat_file, "w") as f:
                for file in normalized_files:
                    f.write(f"file '{os.path.basename(file)}'\n")
            
            # 使用concat滤镜一次性合并所有片段
            filter_complex = ""
            
            # 为每个片段添加淡入淡出效果
            for i, file in enumerate(normalized_files):
                # 获取当前片段的时长
                duration = get_audio_duration(file)
                
                if duration == 0:
                    print(f"警告: 片段 {i} 时长为0，跳过")
                    continue
                
                # 对中间片段使用更长的淡变时间
                if 1 < i < len(normalized_files) - 1:
                    fade_factor = 1.5  # 增加中间片段的淡变因子
                else:
                    fade_factor = 1.0  # 首尾片段使用标准淡变
                
                # 增加淡变时长，但不超过音频长度的1/3
                fade_in = min(crossfade_duration * fade_factor, duration/3)
                fade_out = min(crossfade_duration * fade_factor, duration/3)
                
                # 添加淡入淡出效果，使用exp曲线（FFmpeg支持的曲线）
                filter_complex += f"[{i}:a]afade=t=in:st=0:d={fade_in}:curve=exp,afade=t=out:st={duration-fade_out}:d={fade_out}:curve=exp[a{i}];"
            
            # 使用concat滤镜合并所有片段
            for i in range(len(normalized_files)):
                if get_audio_duration(normalized_files[i]) > 0:
                    filter_complex += f"[a{i}]"
            
            filter_complex += f"concat=n={len(normalized_files)}:v=0:a=1[aout]"
            
            # 构建FFmpeg命令
            concat_cmd = ["ffmpeg", "-y"]
            
            # 添加所有输入文件
            for file in normalized_files:
                concat_cmd.extend(["-i", file])
            
            # 添加滤镜复杂图和输出映射
            concat_cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[aout]",
                # 添加最终的音频处理
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,acompressor=threshold=-16dB:ratio=2:attack=200:release=1000,highpass=f=50:width_type=h:width=100",
                "-c:a", "pcm_s16le",
                "-ar", str(sample_rate),
                output_file
            ])
            
            print(f"使用一体化合并方法...")
            try:
                subprocess.run(concat_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"一体化合并成功，输出文件: {output_file}")
            except Exception as e:
                print(f"一体化合并失败: {e}，尝试备用方法")
                
                # 备用方法：使用acrossfade滤镜逐对合并
                current_output = "temp_0.wav"
                shutil.copy(normalized_files[0], current_output)
                
                # 设置基础交叉淡变时长
                base_crossfade = crossfade_duration
                
                for i in range(1, len(normalized_files)):
                    next_output = f"temp_{i}.wav"
                    
                    # 计算交叉淡变的持续时间
                    duration1 = get_audio_duration(current_output)
                    duration2 = get_audio_duration(normalized_files[i])
                    
                    if duration1 == 0 or duration2 == 0:
                        print(f"警告: 片段 {i-1} 或 {i} 时长为0，跳过合并")
                        continue
                    
                    # 对中间片段使用更长的淡变时间
                    if 1 < i < len(normalized_files) - 1:
                        adaptive_crossfade = base_crossfade * 1.5  # 中间片段使用更长的淡变
                    else:
                        adaptive_crossfade = base_crossfade
                    
                    # 确保淡变时长合理，不超过音频长度的1/4
                    actual_crossfade = min(adaptive_crossfade, duration1/4, duration2/4)
                    
                    # 使用acrossfade滤镜合并当前输出和下一个片段
                    # 使用FFmpeg支持的交叉淡变曲线(exp)
                    acrossfade_cmd = [
                        "ffmpeg",
                        "-i", current_output,
                        "-i", normalized_files[i],
                        "-filter_complex", f"[0:a][1:a]acrossfade=d={actual_crossfade}:c1=exp:c2=exp[out]",
                        "-map", "[out]",
                        "-c:a", "pcm_s16le",
                        "-ar", str(sample_rate),
                        "-y",
                        next_output
                    ]
                    
                    print(f"合并片段 {i-1} 和 {i}: 交叉淡变时长 = {actual_crossfade:.3f}秒")
                    try:
                        result = subprocess.run(acrossfade_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        
                        # 验证输出文件
                        if os.path.exists(next_output) and os.path.getsize(next_output) > 0:
                            # 更新当前输出为新生成的文件
                            current_output = next_output
                            print(f"片段 {i-1} 和 {i} 合并成功，合并后时长: {get_audio_duration(current_output):.2f}秒")
                        else:
                            print(f"警告: 片段 {i-1} 和 {i} 合并失败，输出文件无效")
                            # 保持当前输出不变
                    except subprocess.CalledProcessError as e:
                        print(f"合并片段 {i-1} 和 {i} 失败: {e}")
                        # 保持当前输出不变
                
                # 复制最终结果到输出文件
                if os.path.exists(current_output) and os.path.getsize(current_output) > 0:
                    shutil.copy(current_output, output_file)
                    print(f"备用方法合并成功，输出文件: {output_file}")
                else:
                    print(f"警告: 备用方法合并失败，尝试基本合并")
                    # 如果备用方法也失败，使用基本合并
                    basic_merged = np.concatenate([sf.read(f)[0] for f in normalized_files])
                    sf.write(output_file, basic_merged, sample_rate, 'PCM_16')
            
            # 读取合并后的音频
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                merged_audio, _ = sf.read(output_file)
                print(f"最终音频时长: {len(merged_audio)/sample_rate:.2f}秒")
            else:
                print("警告: 最终输出文件无效，返回基本合并结果")
                merged_audio = np.concatenate(audio_segments)
            
            os.chdir(original_dir)
            return merged_audio
            
        except Exception as e:
            print(f"音频合并过程中发生错误: {e}")
            os.chdir(original_dir)
            print("回退到基本合并方法...")
            return np.concatenate(audio_segments)

def main():
    args = get_args()
    url = f"{args.server_addr}:{args.server_port}"

    # 创建 gRPC 客户端
    triton_client = grpcclient.InferenceServerClient(url=url, verbose=False)
    
    # 加载参考音频
    waveform, sample_rate = load_audio(args.reference_audio, target_sample_rate=16000)
    
    # 打印目标文本长度信息
    print(f"目标文本长度: {len(args.target_text)} 字符")
    print(f"目标文本: {args.target_text}")
    
    # 估计目标音频时长
    expected_duration = len(args.target_text) / 4.5  # 中文的粗略估计
    print(f"预期音频时长: {expected_duration:.2f} 秒")
    
    # 发送请求并计时
    start_time = time.time()
    
    try:
        if args.use_chunking and len(args.target_text) > args.chunk_size:
            print(f"文本长度超过 {args.chunk_size} 字符，启用分块处理")
            
            # 根据处理模式自动设置重叠和修剪参数
            if args.processing_mode == "balanced" and args.overlap_chars == 0:
                args.overlap_chars = 5  # 平衡模式默认使用5字符重叠
                print(f"平衡模式: 自动设置重叠字符数为 {args.overlap_chars}")
            
            if args.processing_mode != "no-overlap" and args.overlap_chars > 0 and not args.trim_overlap:
                args.trim_overlap = True
                print(f"自动启用重叠修剪以提高音频质量")
            
            # Use the splitting function with overlap if specified
            # 根据选择的分割方法分割文本
            print(f"使用 {args.split_method} 方法进行文本分割")
            if args.split_method == "spacy":
                chunks = split_text_with_spacy(args.target_text, args.chunk_size, args.overlap_chars)
            elif args.split_method == "transformers":
                chunks = split_text_with_transformers(args.target_text, args.chunk_size, args.overlap_chars)
            elif args.split_method == "hybrid":
                chunks = split_text_hybrid(args.target_text, args.chunk_size, args.overlap_chars)
            elif args.split_method == "nature":
                chunks = split_text_with_nature(args.target_text, args.chunk_size, args.overlap_chars)
            else:
                if args.overlap_chars > 0:
                    chunks = split_text_with_overlap(args.target_text, args.chunk_size, args.overlap_chars)
                    print(f"使用 {args.overlap_chars} 字符的重叠进行分块")
                else:
                    chunks = split_text_by_sentence(args.target_text, args.chunk_size)
            
            print(f"将文本分为 {len(chunks)} 个块进行处理")
            for i, chunk in enumerate(chunks):
                print(f"块 {i+1}: 长度={len(chunk)}, 内容={chunk}")
            
            # 使用新的处理函数处理块
            audio, chunk_durations = process_chunks_with_overlap_trimming(triton_client, args, waveform, sample_rate, chunks)
            
        else:
            # 处理整个文本
            audio = process_text_chunk(triton_client, args, waveform, sample_rate, args.target_text)
            chunk_durations = [expected_duration]
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 保存音频输出
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        sf.write(args.output_file, audio, 16000, "PCM_16")
        
        # 计算实际音频时长
        audio_duration = len(audio) / 16000
        
        print(f"成功! 音频已保存到 {args.output_file}")
        print(f"处理时间: {processing_time:.2f} 秒")
        print(f"音频时长: {audio_duration:.2f} 秒")
        print(f"预期时长: {expected_duration:.2f} 秒")
        print(f"比例: {audio_duration/expected_duration:.2f}")
        
        # 添加重叠修剪提示
        if args.use_chunking and args.overlap_chars > 0:
            if args.trim_overlap:
                print(f"已应用重叠修剪 (修剪因子: {args.trim_factor})")
                print("如果仍有重叠问题，可尝试增加 --trim-factor 的值 (例如: --trim-factor 1.5)")
            else:
                print("检测到使用了重叠但未启用修剪，如需去除重叠，请添加 --trim-overlap 参数")
        
        # 判断是否有截断
        if not args.use_chunking and audio_duration < expected_duration * 0.8:
            print("警告: 音频时长明显短于预期，可能存在文本截断问题")
            print("建议使用 --use-chunking 参数启用文本分块处理")
        
        # 分析音频质量
        quality_issues = analyze_audio_quality(audio, 16000, chunk_durations)
        if quality_issues:
            print("\n音频质量分析:")
            for issue in quality_issues:
                print(f"- {issue}")
                
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        # 确保在出错时不会尝试访问未定义的变量
        print("处理过程中出现错误，未能生成音频。")

if __name__ == "__main__":
    main()