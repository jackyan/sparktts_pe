
import os
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
        
        # 创建FFmpeg合并脚本
        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, 'w') as f:
            for segment_file in segment_files:
                f.write(f"file '{segment_file}'\n")
        
        # 使用FFmpeg合并音频
        if crossfade_duration > 0 and len(segment_files) > 1:
            # 对于大量片段，使用更简单的方法而不是复杂的filter_complex
            if len(segment_files) > 10:
                # 使用简单的concat滤镜，不尝试复杂的交叉淡入淡出
                cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
                    "-i", concat_file, "-c:a", "pcm_s16le", "-ar", str(sample_rate), output_file
                ]
                print("片段数量较多，使用简化的FFmpeg合并方法")
            else:
                # 对于少量片段，尝试使用交叉淡入淡出
                filter_complex = ""
                for i in range(len(segment_files)):
                    filter_complex += f"[{i}:0]"
                
                filter_complex += f"concat=n={len(segment_files)}:v=0:a=1"
                
                cmd = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
                    "-i", concat_file, "-filter_complex", filter_complex,
                    "-ar", str(sample_rate), output_file
                ]
        else:
            # 简单合并
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
                "-i", concat_file, "-c", "copy", output_file
            ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg合并失败: {e}")
            print("回退到基本合并方法")
            return np.concatenate(audio_segments)
        
        # 读取合并后的音频
        merged_audio, _ = sf.read(output_file)
        
        # 如果是临时文件，删除它
        if output_file.startswith(tempfile.gettempdir()):
            os.unlink(output_file)
        
        return merged_audio
        
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
if __name__ == "__main__":
    text = """随着大模型为主的生成式AI技术取得快速发展，各大PC厂商不仅在积极探索全新的AI PC形态，为推动大模型推理快速高效实现也在积极采纳和部署强劲的AI芯片。传统AI PC解决方案是在CPU中嵌入iNPU，在运行大语言模型时，通常依赖GPU进行加速，iNPU只有在特定的场景中才能被调用。然而，GPU在处理大模型时可能会面临一些性能瓶颈，如GPU的架构虽然适合并行计算，但在处理深度学习任务时，会导致资源利用率不足或延迟较高。此外，GPU在推理阶段的功耗相对较高。
而且在群雄逐鹿的通用GPU市场中，面临着英伟达、英特尔、AMD等巨头的强大竞争，国内厂商要在重重壁垒中开辟自己的天地，需要独辟蹊径，打造全生态。芯动力敏锐地观察到，高性价比是边缘计算核心要求，且性能与TOPS不直接挂钩，不同计算阶段对性能要求不同，采用探索创新型的计算机架构的GPGPU是解决通用高算力和低功耗需求的必由之路，并已成为业界共识。
"""
    result_chunks = split_text_with_nature(text)
    
    print(len(result_chunks))
    i = 0
    for result_chunk in result_chunks:
        i = i + 1
        print(f"第{i}个块:" + result_chunk)