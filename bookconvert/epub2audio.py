import os
import urllib.parse
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import logging
import re
import numpy as np
import soundfile as sf
#import tritonclient
#import tritonclient.grpc.aio as grpcclient
#from tritonclient.utils import np_to_triton_dtype
from logging.handlers import RotatingFileHandler

# 配置日志
# 创建日志目录
log_dir = os.path.join(".", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "epub2audio.log")

# 创建日志处理器
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
console_handler = logging.StreamHandler()

# 设置格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 配置日志记录器
logger = logging.getLogger('epub_extractor')
logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别以捕获所有日志
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 移除root logger的默认处理器，避免重复输出
logging.basicConfig(level=logging.WARNING, handlers=[])

logger.info("日志系统初始化完成，日志文件位置: %s", log_file)

def extract_epub_to_chapters(epub_path, output_dir=None):
    """
    从EPUB电子书中提取目录结构和内容，并按章节保存为TXT文件
    
    Args:
        epub_path: EPUB文件的路径
        output_dir: 输出目录，默认为书名创建的文件夹
    
    Returns:
        保存的章节文件列表
    """
    logger.info(f"开始处理EPUB文件: {epub_path}")
    
    # 打开EPUB文件
    book = epub.read_epub(epub_path)
    
    # 获取书名作为默认输出目录
    title = book.get_metadata('DC', 'title')
    book_title = title[0][0] if title else os.path.basename(epub_path).split('.')[0]
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(".", "Extracted_Books", book_title)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 获取目录结构
    toc = book.toc
    logger.info(f"获取到目录项数量: {len(toc)}")
    
    # 保存的文件列表
    saved_files = []
    
    # 收集所有章节信息
    all_chapters = {}
    
    # 处理目录项
    for item in toc:
        chapter_info = extract_chapter_info(item)
        if chapter_info:
            all_chapters.update(chapter_info)
    
    # 对章节进行排序
    sorted_chapters = sort_chapters(all_chapters)
    logger.info(f"排序后的章节数量: {len(sorted_chapters)}")
    
    # 按排序后的顺序处理章节
    for title, hrefs in sorted_chapters:
        # 提取章节内容
        content = extract_content_from_hrefs(book, hrefs)
        if content:
            # 创建合法的文件名
            # 先将全角空格替换为半角空格
            normalized_title = title.replace('\u3000', ' ')
            # 再处理连续的多个空格
            safe_title = re.sub(r'\s{2,}', ' ', normalized_title)
            # 将文件名中的中文数字替换为阿拉伯数字
            cn_num_map = {
                '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
                '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
                '十': '10', '百': '100', '千': '1000', '万': '10000'
            }
            
            # 查找第一个中文数字并替换
            pattern = r'第([一二三四五六七八九十百千万]+)[章节篇]'
            match = re.search(pattern, safe_title)
            if match:
                cn_num = match.group(1)
                # 简单处理中文数字转阿拉伯数字
                if len(cn_num) == 1 and cn_num in cn_num_map:
                    ar_num = cn_num_map[cn_num]
                elif len(cn_num) == 2 and cn_num[0] == '十':
                    ar_num = '1' + cn_num_map.get(cn_num[1], '0')
                elif len(cn_num) == 2 and cn_num[1] == '十':
                    ar_num = cn_num_map[cn_num[0]] + '0'
                elif len(cn_num) == 3 and cn_num[1] == '十':
                    ar_num = cn_num_map[cn_num[0]] + cn_num_map.get(cn_num[2], '0')
                else:
                    # 复杂情况保持原样
                    ar_num = cn_num
                # 替换文件名中的中文数字
                safe_title = safe_title.replace(f"第{cn_num}", f"第{ar_num}")
                
            filename = os.path.join(output_dir, f"{safe_title}.txt")
            
            # 保存内容到文件
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"已保存章节: {title} -> {filename}")
            saved_files.append(filename)
    
    logger.info(f"处理完成，共保存 {len(saved_files)} 个章节文件")
    return saved_files

def sort_chapters(chapters_dict):
    """
    对章节进行排序
    
    Args:
        chapters_dict: 包含章节标题和链接的字典 {标题: [链接列表]}
    
    Returns:
        排序后的章节列表 [(标题, [链接列表])]
    """
    # 提取章节标题和链接
    chapters = list(chapters_dict.items())
    
    # 定义章节排序函数
    def chapter_sort_key(chapter_item):
        title = chapter_item[0]
        # 尝试提取章节序号
        match = re.search(r'第([一二三四五六七八九十百千万零\d]+)[章节篇]', title)
        if match:
            # 如果找到了章节序号
            num_str = match.group(1)
            # 将中文数字转换为阿拉伯数字
            if re.match(r'^[一二三四五六七八九十百千万零]+$', num_str):
                # 中文数字转阿拉伯数字的映射
                cn_num = {
                    '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, 
                    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
                    '十': 10, '百': 100, '千': 1000, '万': 10000
                }
                
                # 简单处理中文数字（仅处理常见情况）
                if len(num_str) == 1 and num_str in cn_num:
                    return cn_num[num_str]
                elif len(num_str) == 2 and num_str[0] == '十':
                    return 10 + cn_num.get(num_str[1], 0)
                elif len(num_str) == 2 and num_str[1] == '十':
                    return cn_num[num_str[0]] * 10
                elif len(num_str) == 3 and num_str[0] in cn_num and num_str[1] == '十':
                    return cn_num[num_str[0]] * 10 + cn_num.get(num_str[2], 0)
                else:
                    # 复杂中文数字，使用简单估计
                    return 1000 + len(title)  # 放到后面
            else:
                # 尝试直接转换为整数
                try:
                    return int(num_str)
                except ValueError:
                    return 2000 + len(title)  # 无法解析的放到最后
        else:
            # 没有找到章节序号，使用标题长度作为次要排序依据
            # 特殊处理前言、序言等
            if '前言' in title or '序言' in title or '引言' in title:
                return -1
            elif '目录' in title:
                return -2
            elif '封面' in title or '扉页' in title:
                return -3
            elif '版权' in title:
                return -4
            else:
                return 3000 + len(title)  # 其他无法识别的章节放到最后
    
    # 对章节进行排序
    try:
        sorted_chapters = sorted(chapters, key=chapter_sort_key)
        logger.info("章节排序成功")
    except Exception as e:
        logger.error(f"章节排序出错: {str(e)}")
        # 如果排序出错，返回原始顺序
        sorted_chapters = chapters
    
    # 记录排序结果
    for i, (title, _) in enumerate(sorted_chapters):
        logger.debug(f"排序后章节 {i+1}: {title}")
    
    return sorted_chapters

def extract_chapter_info(item, level=0):
    """
    递归提取目录项的标题和链接信息
    
    Args:
        item: 目录项
        level: 当前层级
    
    Returns:
        包含标题和链接的字典 {标题: [链接列表]}
    """
    result = {}
    
    if isinstance(item, tuple) or isinstance(item, list):
        if len(item) > 0:
            # 提取标题
            title = item[0]
            if hasattr(title, 'title'):
                title = title.title
            
            # 提取链接
            hrefs = []
            if len(item) > 1:
                href = item[1]
                if isinstance(href, list):
                    for link in href:
                        if hasattr(link, 'href'):
                            hrefs.append(link.href)
                elif hasattr(href, 'href'):
                    hrefs.append(href.href)
                else:
                    hrefs.append(href)
            
            if title and hrefs:
                result[title] = hrefs
            
            # 处理子项
            if len(item) > 2 and item[2]:
                for child in item[2]:
                    child_result = extract_chapter_info(child, level + 1)
                    result.update(child_result)
    
    elif hasattr(item, 'title') and hasattr(item, 'href'):
        title = item.title
        hrefs = [item.href]
        result[title] = hrefs
        logger.debug(f"{'  ' * level}标题: {title}, 链接: {item.href}")
        
        # 处理子项
        if hasattr(item, 'children') and item.children:
            for child in item.children:
                child_result = extract_chapter_info(child, level + 1)
                result.update(child_result)
    
    return result

def extract_content_from_hrefs(book, hrefs):
    """
    从多个href链接中提取内容
    
    Args:
        book: epub.read_epub()返回的书籍对象
        hrefs: 链接列表
    
    Returns:
        合并后的文本内容
    """
    all_content = []
    
    for href in hrefs:
        content = extract_content_from_href(book, href)
        if content and content != "未找到链接对应的内容":
            # 删除内容中的空白行
            content_lines = content.split('\n')
            content_lines = [line for line in content_lines if line.strip()]
            content = '\n'.join(content_lines)
            all_content.append(content)
    
    return "\n\n".join(all_content)

def extract_content_from_href(book, href):
    """
    从单个href链接中提取内容
    
    Args:
        book: epub.read_epub()返回的书籍对象
        href: 链接
    
    Returns:
        提取的文本内容
    """
    # URL解码href
    decoded_href = urllib.parse.unquote(href)
    
    # 处理锚点
    if '#' in decoded_href:
        file_path, fragment = decoded_href.split('#', 1)
    else:
        file_path, fragment = decoded_href, None
    
    logger.debug(f"处理链接 - 原始链接: {href}")
    logger.debug(f"处理链接 - 解码后链接: {decoded_href}")
    logger.debug(f"处理链接 - 文件路径: {file_path}, 锚点: {fragment}")
    
    # 记录所有文档项的名称，用于调试
    doc_names = [item.get_name() for item in book.get_items() 
                if item.get_type() == ebooklib.ITEM_DOCUMENT]
    logger.debug(f"文档项列表: {doc_names}")
    
    # 查找对应的文档项
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            item_name = item.get_name()
            
            # 记录详细的匹配尝试
            logger.debug(f"尝试匹配 - 文档名: {item_name}")
            logger.debug(f"精确匹配: {item_name == file_path}")
            
            # 尝试多种匹配方式
            if (item_name == file_path or 
                item_name.endswith('/' + file_path) or
                file_path.endswith(item_name) or
                os.path.basename(item_name) == os.path.basename(file_path)):
                
                logger.info(f"找到匹配文档: {item_name} 对应链接: {href}")
                # 获取文档内容
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                
                # 如果有锚点，尝试找到对应的元素
                if fragment:
                    logger.debug(f"处理锚点: {fragment}")
                    # 尝试通过id查找
                    target_element = soup.find(id=fragment)
                    if target_element:
                        logger.debug(f"通过id找到锚点元素: {target_element.name}")
                        
                        # 提取锚点所在章节的内容
                        content = extract_section_content(soup, target_element)
                        logger.debug(f"提取章节内容长度: {len(content)} 字符")
                        return content
                    
                    # 如果没有找到id，尝试查找name属性
                    target_element = soup.find(attrs={"name": fragment})
                    if target_element:
                        logger.debug(f"通过name找到锚点元素: {target_element.name}")
                        
                        # 提取锚点所在章节的内容
                        content = extract_section_content(soup, target_element)
                        logger.debug(f"提取章节内容长度: {len(content)} 字符")
                        return content
                    
                    logger.warning(f"未找到锚点元素: {fragment}")
                
                # 如果没有锚点或找不到锚点元素，返回整个文档内容
                content = soup.get_text(strip=True)
                logger.debug(f"提取内容长度: {len(content)} 字符")
                return content
    
    logger.warning(f"未找到精确匹配文档，尝试部分匹配...")
    
    # 如果没有找到匹配的文档，尝试更宽松的匹配
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            item_name = item.get_name()
            
            # 记录部分匹配尝试
            contains_match = file_path in item_name
            parts_match = any(part in item_name for part in file_path.split('/'))
            logger.debug(f"部分匹配 - 文档名: {item_name}")
            logger.debug(f"包含匹配: {contains_match}")
            logger.debug(f"部分路径匹配: {parts_match}")
            
            # 尝试部分匹配
            if contains_match or parts_match:
                logger.info(f"找到部分匹配文档: {item_name} 对应链接: {href}")
                # 获取文档内容
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                content = soup.get_text(strip=True)
                logger.debug(f"提取内容长度: {len(content)} 字符")
                return content
    
    # 尝试更宽松的匹配 - 忽略大小写
    logger.warning(f"尝试忽略大小写匹配...")
    file_path_lower = file_path.lower()
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            item_name = item.get_name()
            item_name_lower = item_name.lower()
            
            if (item_name_lower == file_path_lower or 
                item_name_lower.endswith('/' + file_path_lower) or
                file_path_lower.endswith(item_name_lower) or
                os.path.basename(item_name_lower) == os.path.basename(file_path_lower)):
                
                logger.info(f"忽略大小写找到匹配文档: {item_name} 对应链接: {href}")
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                content = soup.get_text(strip=True)
                logger.debug(f"提取内容长度: {len(content)} 字符")
                return content
    
    logger.error(f"未找到链接对应的内容: {href}")
    return "未找到链接对应的内容"

def extract_section_content(soup, target_element):
    """
    提取锚点元素所在章节的内容
    
    Args:
        soup: BeautifulSoup对象
        target_element: 锚点元素
    
    Returns:
        章节内容
    """
    # 获取所有标题元素
    heading_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    
    # 确定当前元素是否为标题
    current_tag = target_element.name.lower()
    is_heading = current_tag in heading_tags
    
    # 如果锚点元素本身是标题，直接从这个标题开始提取
    if is_heading:
        heading_level = heading_tags.index(current_tag)
        logger.debug(f"锚点元素是标题: {current_tag}, 级别: {heading_level+1}")
        
        # 获取所有内容，直到下一个同级或更高级标题
        content_parts = [target_element.get_text(strip=True)]
        next_element = target_element.find_next()
        
        while next_element:
            # 检查是否遇到同级或更高级标题
            if next_element.name and next_element.name.lower() in heading_tags:
                next_level = heading_tags.index(next_element.name.lower())
                if next_level <= heading_level:
                    break
            
            # 添加文本内容
            if next_element.string and next_element.string.strip():
                content_parts.append(next_element.string.strip())
            
            next_element = next_element.find_next()
        
        return "\n".join(content_parts)
    
    # 如果锚点元素不是标题，查找其最近的父标题
    parent_heading = None
    parent = target_element.find_parent()
    
    while parent and parent.name != 'body':
        if parent.name.lower() in heading_tags:
            parent_heading = parent
            break
        parent = parent.find_parent()
    
    # 如果找到父标题，从父标题开始提取
    if parent_heading:
        logger.debug(f"找到锚点元素的父标题: {parent_heading.name}")
        return extract_section_content(soup, parent_heading)
    
    # 如果没有找到父标题，尝试找到下一个标题前的所有内容
    logger.debug("未找到父标题，提取锚点元素后的内容直到下一个标题")
    content_parts = []
    current_element = target_element
    
    # 添加锚点元素本身的内容
    if target_element.string and target_element.string.strip():
        content_parts.append(target_element.string.strip())
    
    # 添加后续元素的内容直到遇到标题
    next_element = target_element.find_next()
    while next_element:
        if next_element.name and next_element.name.lower() in heading_tags:
            break
        
        if next_element.string and next_element.string.strip():
            content_parts.append(next_element.string.strip())
        
        next_element = next_element.find_next()
    
    # 如果没有提取到内容，返回锚点元素所在的整个段落
    if not content_parts and target_element.find_parent('p'):
        logger.debug("提取锚点元素所在段落的内容")
        return target_element.find_parent('p').get_text(strip=True)
    
    # 如果仍然没有内容，返回从锚点到文档末尾的所有内容
    if not content_parts:
        logger.debug("提取从锚点到文档末尾的所有内容")
        elements = list(target_element.next_elements)
        content_parts = [el.string.strip() for el in elements if el.string and el.string.strip()]
    
    return "\n".join(content_parts)

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

def preprocess_text(text, min_segment_length=70, max_segment_length=120, logger=None):
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

def extract_epub_to_chapters_dict(epub_path, output_dir=None):
    """
    从EPUB电子书中提取目录结构和内容，返回章节内容字典
    
    Args:
        epub_path: EPUB文件的路径
        output_dir: 输出目录，默认为书名创建的文件夹
    
    Returns:
        章节文件名与内容的字典 {文件名: 内容}
    """
    logger.info(f"开始处理EPUB文件: {epub_path}")
    
    # 打开EPUB文件
    book = epub.read_epub(epub_path)
    
    # 获取书名作为默认输出目录
    title = book.get_metadata('DC', 'title')
    book_title = title[0][0] if title else os.path.basename(epub_path).split('.')[0]
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(".", "Extracted_Books", book_title)
    else:
        output_dir = os.path.join(output_dir, book_title)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 获取目录结构
    toc = book.toc
    logger.info(f"获取到目录项数量: {len(toc)}")
    
    # 章节内容字典
    chapters_dict = {}
    
    # 收集所有章节信息
    all_chapters = {}
    
    # 处理目录项
    for item in toc:
        chapter_info = extract_chapter_info(item)
        if chapter_info:
            all_chapters.update(chapter_info)
    
    # 对章节进行排序
    sorted_chapters = sort_chapters(all_chapters)
    logger.info(f"排序后的章节数量: {len(sorted_chapters)}")
    
    # 按排序后的顺序处理章节
    for title, hrefs in sorted_chapters:
        # 提取章节内容
        content = extract_content_from_hrefs(book, hrefs)
        if content:
            # 创建合法的文件名
            # 先将全角空格替换为半角空格
            normalized_title = title.replace('\u3000', ' ')
            # 再处理连续的多个空格
            safe_title = re.sub(r'\s{2,}', ' ', normalized_title)
            # 将文件名中的中文数字替换为阿拉伯数字
            cn_num_map = {
                '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
                '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
                '十': '10', '百': '100', '千': '1000', '万': '10000'
            }
            
            # 查找第一个中文数字并替换
            pattern = r'第([一二三四五六七八九十百千万]+)[章节篇]'
            match = re.search(pattern, safe_title)
            if match:
                cn_num = match.group(1)
                # 简单处理中文数字转阿拉伯数字
                if len(cn_num) == 1 and cn_num in cn_num_map:
                    ar_num = cn_num_map[cn_num]
                elif len(cn_num) == 2 and cn_num[0] == '十':
                    ar_num = '1' + cn_num_map.get(cn_num[1], '0')
                elif len(cn_num) == 2 and cn_num[1] == '十':
                    ar_num = cn_num_map[cn_num[0]] + '0'
                elif len(cn_num) == 3 and cn_num[1] == '十':
                    ar_num = cn_num_map[cn_num[0]] + cn_num_map.get(cn_num[2], '0')
                else:
                    # 复杂情况保持原样
                    ar_num = cn_num
                # 替换文件名中的中文数字
                safe_title = safe_title.replace(f"第{cn_num}", f"第{ar_num}")
                
            # 使用wav作为后缀而不是txt
            filename = os.path.join(output_dir, f"{safe_title}.wav")
            
            # 将内容添加到字典中
            chapters_dict[filename] = content
            
            logger.info(f"已处理章节: {title} -> {filename}")
    
    logger.info(f"处理完成，共处理 {len(chapters_dict)} 个章节")
    return chapters_dict

async def process_chapter_to_audio(triton_client, args, waveform, sample_rate, chapter_file, chapter_text):
    """
    处理单个章节的文本，将其转换为语音并保存
    
    Args:
        triton_client: Triton客户端
        args: 命令行参数
        waveform: 参考音频波形
        sample_rate: 采样率
        chapter_file: 章节文件路径
        chapter_text: 章节文本内容
        
    Returns:
        保存的音频文件路径
    """
    logger.info(f"开始处理章节: {os.path.basename(chapter_file)}")
    
    # 使用preprocess_text函数分段处理文本
    text_segments = preprocess_text(chapter_text, min_segment_length=70, max_segment_length=120, logger=logger)
    logger.info(f"章节文本被分为{len(text_segments)}个段落")
    
    # 处理每个文本段落并获取音频
    audio_segments = []
    for i, segment in enumerate(text_segments):
        logger.info(f"处理段落 {i+1}/{len(text_segments)}")
        try:
            # 使用process_text_chunk处理文本段落
            audio = await process_text_chunk(triton_client, args, waveform, sample_rate, segment)
            audio_segments.append(audio)
            logger.info(f"段落 {i+1} 处理完成，生成音频长度: {len(audio)}")
        except Exception as e:
            logger.error(f"处理段落 {i+1} 时出错: {str(e)}")
            # 如果处理失败，添加一段静音
            audio_segments.append(np.zeros(int(0.5 * sample_rate)))  # 0.5秒静音
    
    # 合并所有音频段落
    combined_audio = np.concatenate(audio_segments)
    logger.info(f"合并后的音频长度: {len(combined_audio)}")
    
    # 保存音频文件
    try:
        sf.write(chapter_file, combined_audio, sample_rate)
        logger.info(f"已保存音频文件: {chapter_file}")
        return chapter_file
    except Exception as e:
        logger.error(f"保存音频文件时出错: {str(e)}")
        return None

async def process_book_to_audio(triton_client, args, waveform, sample_rate, epub_path, output_dir=None):
    """
    处理整本书，将所有章节转换为音频
    
    Args:
        triton_client: Triton客户端
        args: 命令行参数
        waveform: 参考音频波形
        sample_rate: 采样率
        epub_path: EPUB文件路径
        output_dir: 输出目录
        
    Returns:
        处理的章节数量
    """
    logger.info(f"开始处理书籍: {os.path.basename(epub_path)}")
    
    # 提取章节内容
    chapters_dict = extract_epub_to_chapters_dict(epub_path, output_dir)
    
    if not chapters_dict:
        logger.warning(f"未从书籍中提取到任何章节: {epub_path}")
        return 0
    
    # 使用asyncio.Semaphore限制并发数量
    import asyncio
    semaphore = asyncio.Semaphore(4)  # 最多4个并发
    
    async def process_chapter_with_semaphore(chapter_file, chapter_text):
        async with semaphore:
            return await process_chapter_to_audio(triton_client, args, waveform, sample_rate, chapter_file, chapter_text)
    
    # 创建所有章节处理任务
    tasks = []
    for chapter_file, chapter_text in chapters_dict.items():
        task = asyncio.create_task(process_chapter_with_semaphore(chapter_file, chapter_text))
        tasks.append(task)
    
    # 等待所有任务完成
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 统计成功处理的章节数量
    success_count = sum(1 for result in results if result is not None and not isinstance(result, Exception))
    logger.info(f"书籍处理完成: {os.path.basename(epub_path)}, 成功处理 {success_count}/{len(chapters_dict)} 个章节")
    
    return success_count

async def main_async():
    """异步主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EPUB电子书转语音工具")
    parser.add_argument("input_path", help="EPUB文件或包含EPUB文件的目录路径",default="./ebooks")
    parser.add_argument("--output-dir", help="输出目录，默认为书名创建的文件夹",default="./audiobooks")
    parser.add_argument("--reference-wav", help="参考音频文件路径",default="../example/prompt_audio.wav")
    parser.add_argument("--reference-text", required=True, help="参考音频对应的文本",default="吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。")
    parser.add_argument("--url", default="localhost:8001", help="Triton服务器URL")
    parser.add_argument("--model-name", default="sparktts", help="Triton模型名称")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    
    args = parser.parse_args()
    
    # 加载参考音频
    try:
        waveform, sample_rate = load_audio(args.reference_wav)
        logger.info(f"已加载参考音频: {args.reference_wav}, 长度: {len(waveform)}, 采样率: {sample_rate}")
    except Exception as e:
        logger.error(f"加载参考音频时出错: {str(e)}")
        return
    
    # 创建Triton客户端
    try:
        import grpc
        channel = grpc.insecure_channel(
            args.url,
            options=[
                ('grpc.max_send_message_length', 10 * 1024 * 1024),  # 10MB
                ('grpc.max_receive_message_length', 10 * 1024 * 1024)  # 10MB
            ]
        )
        triton_client = grpcclient.InferenceServerClient(
            channel=channel, verbose=args.verbose
        )
        logger.info(f"已连接到Triton服务器: {args.url}")
    except Exception as e:
        logger.error(f"连接Triton服务器时出错: {str(e)}")
        return
    
    # 判断输入路径是文件还是目录
    input_path = args.input_path
    if os.path.isfile(input_path):
        # 处理单个EPUB文件
        if input_path.lower().endswith('.epub'):
            logger.info(f"处理单个EPUB文件: {input_path}")
            await process_book_to_audio(triton_client, args, waveform, sample_rate, input_path, args.output_dir)
        else:
            logger.error(f"输入文件不是EPUB格式: {input_path}")
    elif os.path.isdir(input_path):
        # 处理目录中的所有EPUB文件
        logger.info(f"处理目录中的EPUB文件: {input_path}")
        epub_files = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                     if f.lower().endswith('.epub') and os.path.isfile(os.path.join(input_path, f))]
        
        if not epub_files:
            logger.warning(f"目录中未找到EPUB文件: {input_path}")
            return
        
        logger.info(f"找到 {len(epub_files)} 个EPUB文件")
        
        # 逐个处理每本书
        for epub_file in epub_files:
            logger.info(f"开始处理: {os.path.basename(epub_file)}")
            book_output_dir = os.path.join(args.output_dir, os.path.basename(epub_file).split('.')[0]) if args.output_dir else None
            await process_book_to_audio(triton_client, args, waveform, sample_rate, epub_file, book_output_dir)
    else:
        logger.error(f"输入路径不存在: {input_path}")
def test_main(output_dir="./ebooks/txt"):
    """测试函数"""
    # 测试提取章节内容
    chapters_dict = extract_epub_to_chapters_dict("./ebooks/The Golden Bowl - Henry James.epub", output_dir="./ebooks/txt")
    print(f"提取到 {len(chapters_dict)} 个章节")

    os.makedirs(output_dir, exist_ok=True)
    # 测试处理单个章节
    for chapter_file, chapter_text in chapters_dict.items():
        print(f"开始处理章节: {chapter_file}")
        filename = chapter_file + ".txt"
            # 保存内容到文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(chapter_text)
            logger.info(f"已保存章节: {chapter_file} -> {filename}")

if __name__ == "__main__":
    import asyncio
    test_main()
    try:
        #asyncio.run(main_async())
        logger.info("脚本执行完成")
    except Exception as e:
        logger.exception("脚本执行过程中发生错误: %s", str(e))
        import sys
        sys.exit(1)