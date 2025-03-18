import json
import os
import re
import logging
import datetime

def preprocess_text(text, min_segment_length=80, max_segment_length=120, logger=None):
    """
    预处理文本，根据文本长度决定是否分段并进行分段处理
    
    Args:
        text: 需要处理的文本
        min_segment_length: 分段的最小长度
        max_segment_length: 强制分割时的最大段落长度
        logger: 日志记录器
        
    Returns:
        分割后的文本段落列表
    """
    # 如果没有提供logger，创建一个新的
    if logger is None:
        log_file = f"./sparktts_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger = setup_logger('sparktts', log_file)
    
    # 判断是否需要分段处理
    need_split = len(text) > 130
    
    if not need_split:
        # 不需要分段，将整个文本作为一个段落处理
        logger.info("文本长度适中，不需要分段处理")
        return [text]
    
    logger.info(f"目标文本长度为{len(text)}字符，进行分段处理")
    
    # 第一步：按空行分割成段落
    # 匹配一个或多个空行（包含可能的空格和制表符）
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]  # 移除空段落
    
    logger.info(f"按空行分割成{len(paragraphs)}个段落")
    for i, para in enumerate(paragraphs):
        logger.info(f"原始段落{i+1}: {para[:50]}{'...' if len(para) > 50 else ''}")
    
    # 如果没有找到多个段落，则整个文本作为一个段落
    if len(paragraphs) <= 1:
        paragraphs = [text]
        logger.info("未检测到多个段落，将整个文本视为一个段落")
    
    # 更完备的中文和英文标点符号列表
    punctuations = [
        '，', '。', '！', '？', '；', '：', '、', '…', '"', '"', ''', ''', '【', '】', '《', '》', '（', '）',
        ',', '.', '!', '?', ';', ':', '...', '"', "'", '(', ')', '[', ']', '{', '}', '<', '>'
    ]
    
    # 优先级排序的标点符号（句号、问号、感叹号优先级高）
    priority_puncts = ['。', '！', '？', '.', '!', '?']
    secondary_puncts = ['；', '，', ';', ',']
    
    # 第二步：对每个段落进行进一步分割
    final_segments = []
    
    for para_idx, paragraph in enumerate(paragraphs):
        # 如果段落长度小于最小分段长度的2倍，则不再分割
        if len(paragraph) < min_segment_length * 2:
            logger.info(f"段落{para_idx+1}长度为{len(paragraph)}字符，小于最小分段长度的2倍，保持完整")
            final_segments.append(paragraph)
            continue
        
        # 找到段落中所有可能的分割点
        potential_splits = []
        
        for i, char in enumerate(paragraph):
            if char in punctuations and i >= min_segment_length - 1:
                # 根据标点符号类型分配优先级
                priority = 1 if char in priority_puncts else (2 if char in secondary_puncts else 3)
                potential_splits.append((i, priority))
        
        # 如果没有找到任何潜在分割点，则使用强制分割
        if not potential_splits:
            para_segments = []
            for i in range(0, len(paragraph), max_segment_length):
                para_segments.append(paragraph[i:min(i+max_segment_length, len(paragraph))])
            
            final_segments.extend(para_segments)
            
            logger.info(f"段落{para_idx+1}未找到合适的分割点，使用强制分割，共{len(para_segments)}段")
            continue
        
        # 对分割点进行优化：尽量均匀分布，同时考虑标点符号优先级
        para_segments = []
        last_split = -1
        target_length = (len(paragraph) // (len(potential_splits) // 2 + 1)) if len(potential_splits) > 1 else max_segment_length
        target_length = min(max(target_length, min_segment_length), max_segment_length)
        
        logger.info(f"段落{para_idx+1}目标分段长度: {target_length}字符")
        
        # 按位置排序分割点
        potential_splits.sort(key=lambda x: x[0])
        
        # 防止死循环，设置最大迭代次数
        max_iterations = len(potential_splits) + 10
        iteration_count = 0
        
        while last_split < len(paragraph) - 1 and iteration_count < max_iterations:
            iteration_count += 1
            
            # 找出当前位置之后的最佳分割点
            best_split = None
            best_score = float('inf')
            
            for pos, priority in potential_splits:
                if pos <= last_split:
                    continue
                    
                # 计算与目标长度的差距，并考虑标点符号优先级
                segment_length = pos - last_split
                if segment_length < min_segment_length:
                    continue
                    
                # 分数越低越好：长度接近目标长度，优先级高的标点符号
                score = abs(segment_length - target_length) * priority
                
                if score < best_score:
                    best_score = score
                    best_split = pos
            
            # 如果找不到合适的分割点，或者剩余文本不长，则将剩余文本作为最后一段
            if best_split is None:
                # 如果找不到合适的分割点，但剩余文本较长，则强制分割
                remaining_text = paragraph[last_split+1:]
                if len(remaining_text) > max_segment_length * 1.5:
                    # 强制分割剩余文本
                    for i in range(0, len(remaining_text), max_segment_length):
                        para_segments.append(remaining_text[i:min(i+max_segment_length, len(remaining_text))])
                    logger.info(f"段落{para_idx+1}剩余文本({len(remaining_text)}字符)较长，进行强制分割")
                else:
                    para_segments.append(remaining_text)
                    logger.info(f"段落{para_idx+1}剩余文本({len(remaining_text)}字符)作为最后一段")
                break
            
            # 添加分割出的段落
            para_segments.append(paragraph[last_split+1:best_split+1])
            last_split = best_split
            
            # 检查是否已经处理到文本末尾
            if last_split >= len(paragraph) - min_segment_length:
                # 如果剩余文本较短，直接添加到最后一个段落
                if last_split < len(paragraph) - 1:
                    para_segments[-1] += paragraph[last_split+1:]
                    logger.info(f"段落{para_idx+1}剩余文本较短，合并到最后一段")
                break
        
        # 如果迭代达到最大次数但仍未处理完文本，强制处理剩余部分
        if iteration_count >= max_iterations and last_split < len(paragraph) - 1:
            remaining_text = paragraph[last_split+1:]
            logger.warning(f"段落{para_idx+1}处理达到最大迭代次数，强制处理剩余文本({len(remaining_text)}字符)")
            
            # 强制分割剩余文本
            for i in range(0, len(remaining_text), max_segment_length):
                para_segments.append(remaining_text[i:min(i+max_segment_length, len(remaining_text))])
        
        # 处理最后可能的空段落
        para_segments = [seg for seg in para_segments if seg]
        
        logger.info(f"段落{para_idx+1}被分为{len(para_segments)}个子段落")
        for i, seg in enumerate(para_segments):
            logger.info(f"  子段落{i+1}: {seg[:30]}...({len(seg)}字符)")
        
        # 检查是否有过长的段落需要强制分割
        for segment in para_segments:
            if len(segment) > max_segment_length * 1.5:  # 如果段落长度超过最大长度的1.5倍
                # 强制分割过长段落
                logger.info(f"检测到过长段落({len(segment)}字符)，进行强制分割")
                for i in range(0, len(segment), max_segment_length):
                    final_segments.append(segment[i:min(i+max_segment_length, len(segment))])
            else:
                final_segments.append(segment)
    
    # 最终检查：确保所有段落都不为空且长度合适
    final_segments = [seg for seg in final_segments if seg and len(seg.strip()) > 0]
    
    # 智能合并过短的段落
    if len(final_segments) > 1:
        logger.info("开始智能合并过短段落")
        merged_segments = []
        current_segment = final_segments[0]
        
        for i in range(1, len(final_segments)):
            # 如果当前段落和下一个段落合并后长度不超过最大长度，则合并
            if len(current_segment) + len(final_segments[i]) <= max_segment_length:
                # 检查是否需要添加标点符号连接
                if not current_segment[-1] in punctuations:
                    logger.info(f"合并段落 '{current_segment[-10:]}' 和 '{final_segments[i][:10]}...'，添加逗号连接")
                    current_segment += "，" + final_segments[i]  # 使用逗号连接
                else:
                    logger.info(f"合并段落 '{current_segment[-10:]}' 和 '{final_segments[i][:10]}...'")
                    current_segment += final_segments[i]
            else:
                merged_segments.append(current_segment)
                current_segment = final_segments[i]
        
        # 添加最后一个段落
        merged_segments.append(current_segment)
        final_segments = merged_segments
        
        logger.info(f"智能合并后，段落数量从{len(final_segments)}减少到{len(merged_segments)}")
    
    # 确保至少返回一个段落
    if not final_segments:
        # 如果所有处理后没有得到有效段落，则使用简单的强制分割
        final_segments = []
        for i in range(0, len(text), max_segment_length):
            final_segments.append(text[i:min(i+max_segment_length, len(text))])
        
        logger.warning(f"分段处理未产生有效段落，使用强制分割，共{len(final_segments)}段")
    
    logger.info(f"最终文本被分为{len(final_segments)}段")
    for i, seg in enumerate(final_segments):
        logger.info(f"最终段落{i+1}({len(seg)}字符): {seg}")
    
    return final_segments
# 设置日志
def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器"""
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger
    
def preprocess_text_back(text, min_segment_length=80, max_segment_length=120, logger=None):
    """
    预处理文本，根据文本长度决定是否分段并进行分段处理
    
    Args:
        text: 需要处理的文本
        min_segment_length: 分段的最小长度
        max_segment_length: 强制分割时的最大段落长度
        logger: 日志记录器
        
    Returns:
        分割后的文本段落列表
    """
    # 创建日志文件和日志记录器
    log_file = f"./sparktts_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger("preprocess_text", log_file)
    # 判断是否需要分段处理
    need_split = len(text) > 130
    
    if not need_split:
        # 不需要分段，将整个文本作为一个段落处理
        if logger:
            logger.info("文本长度适中，不需要分段处理")
        return [text]
    
    if logger:
        logger.info(f"目标文本长度超过100，进行分段处理")
    
    # 更完备的中文和英文标点符号列表
    punctuations = [
        '，', '。', '！', '？', '；', '：', '、', '…', '"', '"', ''', ''', '【', '】', '《', '》', '（', '）',
        ',', '.', '!', '?', ';', ':', '...', '"', "'", '(', ')', '[', ']', '{', '}', '<', '>'
    ]
    
    # 优先级排序的标点符号（句号、问号、感叹号优先级高）
    priority_puncts = ['。', '！', '？', '.', '!', '?']
    secondary_puncts = ['；', '，', ';', ',']
    
    # 改进的分段算法：先找到所有可能的分割点
    potential_splits = []
    
    for i, char in enumerate(text):
        if char in punctuations and i >= min_segment_length - 1:
            # 根据标点符号类型分配优先级
            priority = 1 if char in priority_puncts else (2 if char in secondary_puncts else 3)
            potential_splits.append((i, priority))
    
    # 如果没有找到任何潜在分割点，则使用强制分割
    if not potential_splits:
        segments = []
        for i in range(0, len(text), max_segment_length):
            segments.append(text[i:min(i+max_segment_length, len(text))])
        
        if logger:
            logger.info(f"未找到合适的分割点，使用强制分割，共{len(segments)}段")
            for i, seg in enumerate(segments):
                logger.info(f"段落{i+1}: {seg}")
        
        return segments
    
    # 对分割点进行优化：尽量均匀分布，同时考虑标点符号优先级
    segments = []
    last_split = -1
    target_length = (len(text) // (len(potential_splits) // 2 + 1)) if len(potential_splits) > 1 else max_segment_length
    target_length = min(max(target_length, min_segment_length), max_segment_length)
    
    # 按位置排序分割点
    potential_splits.sort(key=lambda x: x[0])
    
    while last_split < len(text) - 1:
        # 找出当前位置之后的最佳分割点
        best_split = None
        best_score = float('inf')
        
        for pos, priority in potential_splits:
            if pos <= last_split:
                continue
                
            # 计算与目标长度的差距，并考虑标点符号优先级
            segment_length = pos - last_split
            if segment_length < min_segment_length:
                continue
                
            # 分数越低越好：长度接近目标长度，优先级高的标点符号
            score = abs(segment_length - target_length) * priority
            
            if score < best_score:
                best_score = score
                best_split = pos
        
        # 如果找不到合适的分割点，或者剩余文本不长，则将剩余文本作为最后一段
        if best_split is None or len(text) - last_split - 1 <= max_segment_length:
            segments.append(text[last_split+1:])
            break
        
        # 添加分割出的段落
        segments.append(text[last_split+1:best_split+1])
        last_split = best_split
    
    # 处理最后可能的空段落
    segments = [seg for seg in segments if seg]
    
    # 检查是否有过长的段落需要强制分割
    final_segments = []
    for segment in segments:
        if len(segment) > max_segment_length * 1.5:  # 如果段落长度超过最大长度的1.5倍
            # 强制分割过长段落
            for i in range(0, len(segment), max_segment_length):
                final_segments.append(segment[i:min(i+max_segment_length, len(segment))])
        else:
            final_segments.append(segment)
    
    if logger:
        logger.info(f"文本被分为{len(final_segments)}段")
        for i, seg in enumerate(final_segments):
            logger.info(f"段落{i+1}: {seg}")
    
    return final_segments

if __name__ == "__main__":
    preprocess_text("""作为PC界的龙头，联想给出了自己的答案。在2025年3月3日在西班牙巴塞罗那的MWC Barcelona2025盛会上，联想展示了全面升级的AI PC。新款AI PC首次采用国内珠海市芯动力科技有限公司基于可重构并行处理器RPP的AzureBlade M.2加速卡，并将其命名为dNPU，不仅显著提升了推理速度和整体性能，让系统运行更加流畅，而且还显著降低了系统整体功耗，实现了高效运行和节能降耗的双重目标和双重优化。
“dNPU代表了未来大模型在PC等本地端推理的技术方向和趋势。”上述负责人强调。端侧AI算力追求极致性价比 GPGPU站上舞台中央
随着大模型为主的生成式AI技术取得快速发展，各大PC厂商不仅在积极探索全新的AI PC形态，为推动大模型推理快速高效实现也在积极采纳和部署强劲的AI芯片。
传统AI PC解决方案是在CPU中嵌入iNPU，在运行大语言模型时，通常依赖GPU进行加速，iNPU只有在特定的场景中才能被调用。然而，GPU在处理大模型时可能会面临一些性能瓶颈，如GPU的架构虽然适合并行计算，但在处理深度学习任务时，会导致资源利用率不足或延迟较高。此外，GPU在推理阶段的功耗相对较高。
而且在群雄逐鹿的通用GPU市场中，面临着英伟达、英特尔、AMD等巨头的强大竞争，国内厂商要在重重壁垒中开辟自己的天地，需要独辟蹊径，打造全生态。芯动力敏锐地观察到，高性价比是边缘计算核心要求，且性能与TOPS不直接挂钩，不同计算阶段对性能要求不同，采用探索创新型的计算机架构的GPGPU是解决通用高算力和低功耗需求的必由之路，并已成为业界共识。
""")