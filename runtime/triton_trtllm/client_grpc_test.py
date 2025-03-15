#!/usr/bin/env python3

import argparse
import os
import time
import numpy as np
import soundfile as sf
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    
    return parser.parse_args()

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
    
    # 合并所有音频段
    audio = np.concatenate(audio_segments)
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