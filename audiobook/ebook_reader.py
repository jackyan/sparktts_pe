#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import argparse
import concurrent.futures
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any

# 导入calibre相关库
from calibre.ebooks.metadata.meta import get_metadata
from calibre.ebooks.conversion.plumber import Plumber
from calibre.customize.conversion import OptionRecommendation
from calibre.ebooks.oeb.reader import OEBReader
from calibre.ebooks.oeb.base import TOC
from calibre.utils.logging import Log
from bs4 import BeautifulSoup

class EbookReader:
    """电子书解析器，支持多种格式的电子书解析"""
    
    def __init__(self, file_path: str, output_dir: str = None):
        """
        初始化电子书解析器
        
        Args:
            file_path: 电子书文件路径
            output_dir: 输出目录，默认为当前目录
        """
        self.file_path = file_path
        self.output_dir = output_dir or os.path.dirname(os.path.abspath(file_path))
        self.book_format = os.path.splitext(file_path)[1].lower().replace('.', '')
        self.log = Log()
        self.chapters = OrderedDict()  # 保存章节结构 {chapter_id: {'title': title, 'sections': {section_id: title}}}
        self.contents = {}  # 保存内容 {chapter_id: content} 或 {(chapter_id, section_id): content}
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 检查文件格式是否支持
        supported_formats = ['epub', 'mobi', 'azw', 'azw3', 'txt', 'html', 'pdf']
        if self.book_format not in supported_formats:
            raise ValueError(f"不支持的文件格式: {self.book_format}，支持的格式有: {', '.join(supported_formats)}")
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 转换为EPUB格式以统一处理
        self.epub_path = self._convert_to_epub()
        
    def _convert_to_epub(self) -> str:
        """
        将电子书转换为EPUB格式以统一处理
        
        Returns:
            转换后的EPUB文件路径
        """
        if self.book_format == 'epub':
            return self.file_path
            
        print(f"正在将 {self.book_format} 格式转换为 EPUB 格式...")
        
        # 设置输出路径
        output_path = os.path.join(
            self.output_dir, 
            f"{os.path.splitext(os.path.basename(self.file_path))[0]}.epub"
        )
        
        # 如果已经存在转换后的文件，直接返回
        if os.path.exists(output_path):
            print(f"使用已存在的转换文件: {output_path}")
            return output_path
            
        # 设置转换选项
        options = [
            OptionRecommendation(name='output_profile', recommended_value='tablet'),
            OptionRecommendation(name='input_profile', recommended_value='tablet'),
            OptionRecommendation(name='prefer_metadata_cover', recommended_value=True),
        ]
        
        # 执行转换
        plumber = Plumber(self.file_path, output_path, self.log)
        plumber.merge_ui_recommendations(options)
        plumber.run()
        
        print(f"转换完成: {output_path}")
        return output_path
    
    def extract_chapters(self) -> Dict:
        """
        提取电子书的章节结构
        
        Returns:
            章节结构字典
        """
        print("正在提取章节结构...")
        
        # 获取书籍元数据
        with open(self.epub_path, 'rb') as f:
            mi = get_metadata(f, self.book_format)
            
        print(f"书名: {mi.title}")
        print(f"作者: {', '.join(mi.authors)}")
        
        # 解析目录结构
        from calibre.ebooks.epub.reader import EpubReader
        from calibre.ebooks.oeb.iterator import EbookIterator
        
        iterator = EbookIterator(self.epub_path)
        iterator.__enter__()
        
        # 获取目录
        toc = iterator.toc
        
        # 处理目录结构
        chapter_id = 0
        for entry in toc:
            if hasattr(entry, 'children') and entry.children:
                # 有子章节的情况
                chapter_id += 1
                chapter_title = entry.title or f"第{chapter_id}章"
                
                self.chapters[chapter_id] = {
                    'title': chapter_title,
                    'sections': OrderedDict(),
                    'href': entry.href
                }
                
                # 处理小节
                for section_id, section in enumerate(entry.children, 1):
                    section_title = section.title or f"第{section_id}节"
                    self.chapters[chapter_id]['sections'][section_id] = {
                        'title': section_title,
                        'href': section.href
                    }
            else:
                # 没有子章节的情况
                chapter_id += 1
                chapter_title = entry.title or f"第{chapter_id}章"
                
                self.chapters[chapter_id] = {
                    'title': chapter_title,
                    'sections': OrderedDict(),
                    'href': entry.href
                }
        
        # 如果没有提取到章节，尝试从文件内容中提取
        if not self.chapters:
            print("未从目录中提取到章节，尝试从内容中提取...")
            self._extract_chapters_from_content(iterator)
            
        iterator.__exit__(None, None, None)
        
        print(f"共提取到 {len(self.chapters)} 个章节")
        for chapter_id, chapter in self.chapters.items():
            print(f"章节 {chapter_id}: {chapter['title']} (包含 {len(chapter['sections'])} 个小节)")
            
        return self.chapters
    
    def _extract_chapters_from_content(self, iterator) -> None:
        """
        从内容中提取章节结构
        
        Args:
            iterator: 电子书迭代器
        """
        # 章节标题的正则表达式模式
        chapter_patterns = [
            r'第\s*[一二三四五六七八九十百千万零\d]+\s*章',  # 中文数字章节
            r'Chapter\s*\d+',  # 英文章节
            r'^\s*\d+\.\s+.+$',  # 数字编号章节
        ]
        
        # 小节标题的正则表达式模式
        section_patterns = [
            r'第\s*[一二三四五六七八九十百千万零\d]+\s*节',  # 中文数字小节
            r'Section\s*\d+',  # 英文小节
            r'^\s*\d+\.\d+\s+.+$',  # 数字编号小节
        ]
        
        chapter_id = 0
        current_chapter = None
        
        # 遍历所有HTML文件
        for spine_item in iterator.spine:
            soup = BeautifulSoup(spine_item.data, 'lxml')
            
            # 查找所有可能的标题元素
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'strong', 'b']):
                text = heading.get_text().strip()
                
                # 检查是否是章节标题
                is_chapter = any(re.search(pattern, text, re.IGNORECASE) for pattern in chapter_patterns)
                
                if is_chapter:
                    chapter_id += 1
                    current_chapter = chapter_id
                    self.chapters[chapter_id] = {
                        'title': text,
                        'sections': OrderedDict(),
                        'href': spine_item.href
                    }
                    continue
                
                # 检查是否是小节标题
                is_section = any(re.search(pattern, text, re.IGNORECASE) for pattern in section_patterns)
                
                if is_section and current_chapter:
                    section_id = len(self.chapters[current_chapter]['sections']) + 1
                    self.chapters[current_chapter]['sections'][section_id] = {
                        'title': text,
                        'href': spine_item.href
                    }
    
    def read_contents(self, max_workers: int = 4) -> Dict:
        """
        并发读取章节内容
        
        Args:
            max_workers: 最大并发工作线程数
            
        Returns:
            内容字典
        """
        if not self.chapters:
            self.extract_chapters()
            
        print(f"正在并发读取内容 (最大并发数: {max_workers})...")
        
        # 准备任务列表
        tasks = []
        
        for chapter_id, chapter in self.chapters.items():
            if chapter['sections']:
                # 有小节的情况，按小节读取
                for section_id, section in chapter['sections'].items():
                    tasks.append((chapter_id, section_id, section.get('href', chapter.get('href'))))
            else:
                # 没有小节的情况，按章节读取
                tasks.append((chapter_id, None, chapter.get('href')))
        
        # 并发执行任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 创建任务
            future_to_task = {
                executor.submit(self._read_content, chapter_id, section_id, href): (chapter_id, section_id)
                for chapter_id, section_id, href in tasks
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_task):
                chapter_id, section_id = future_to_task[future]
                try:
                    content = future.result()
                    if section_id is None:
                        # 按章节存储
                        self.contents[chapter_id] = content
                        print(f"已读取章节 {chapter_id}: {self.chapters[chapter_id]['title']} ({len(content)} 字符)")
                    else:
                        # 按小节存储
                        self.contents[(chapter_id, section_id)] = content
                        section_title = self.chapters[chapter_id]['sections'][section_id]['title']
                        print(f"已读取章节 {chapter_id} 的小节 {section_id}: {section_title} ({len(content)} 字符)")
                except Exception as e:
                    print(f"读取内容时出错: {e}")
        
        return self.contents
    
    def _read_content(self, chapter_id: int, section_id: Optional[int], href: str) -> str:
        """
        读取指定章节或小节的内容
        
        Args:
            chapter_id: 章节ID
            section_id: 小节ID，如果没有小节则为None
            href: 内容链接
            
        Returns:
            章节或小节的内容
        """
        from calibre.ebooks.oeb.iterator import EbookIterator
        
        iterator = EbookIterator(self.epub_path)
        iterator.__enter__()
        
        content = ""
        
        try:
            # 查找对应的内容
            for spine_item in iterator.spine:
                if spine_item.href == href:
                    # 解析HTML内容
                    soup = BeautifulSoup(spine_item.data, 'lxml')
                    
                    # 移除脚本和样式
                    for script in soup(["script", "style"]):
                        script.extract()
                    
                    # 获取文本内容
                    content = soup.get_text(separator="\n").strip()
                    break
        finally:
            iterator.__exit__(None, None, None)
            
        return content
    
    def save_to_txt(self, output_file: str = None) -> str:
        """
        将内容保存为TXT文件
        
        Args:
            output_file: 输出文件路径，默认为原文件名.txt
            
        Returns:
            输出文件路径
        """
        if not self.contents:
            self.read_contents()
            
        if output_file is None:
            output_file = os.path.join(
                self.output_dir,
                f"{os.path.splitext(os.path.basename(self.file_path))[0]}.txt"
            )
            
        print(f"正在保存内容到: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # 按章节顺序写入
            for chapter_id in sorted(self.chapters.keys()):
                chapter = self.chapters[chapter_id]
                
                # 写入章节标题
                f.write(f"\n\n{'='*40}\n")
                f.write(f"第{chapter_id}章: {chapter['title']}\n")
                f.write(f"{'='*40}\n\n")
                
                if chapter['sections']:
                    # 有小节的情况
                    for section_id in sorted(chapter['sections'].keys()):
                        section = chapter['sections'][section_id]
                        
                        # 写入小节标题
                        f.write(f"\n{'-'*30}\n")
                        f.write(f"第{chapter_id}.{section_id}节: {section['title']}\n")
                        f.write(f"{'-'*30}\n\n")
                        
                        # 写入小节内容
                        content = self.contents.get((chapter_id, section_id), "")
                        f.write(content)
                        f.write("\n\n")
                else:
                    # 没有小节的情况
                                        # 没有小节的情况
                    content = self.contents.get(chapter_id, "")
                    f.write(content)
                    f.write("\n\n")
        
        print(f"内容已保存到: {output_file}")
        return output_file
    
    def get_chapter_content(self, chapter_id: int) -> str:
        """
        获取指定章节的内容
        
        Args:
            chapter_id: 章节ID
            
        Returns:
            章节内容
        """
        if not self.contents:
            self.read_contents()
            
        if chapter_id not in self.chapters:
            raise ValueError(f"章节ID {chapter_id} 不存在")
            
        if self.chapters[chapter_id]['sections']:
            # 有小节的情况，合并所有小节内容
            content = ""
            for section_id in sorted(self.chapters[chapter_id]['sections'].keys()):
                section_content = self.contents.get((chapter_id, section_id), "")
                content += section_content + "\n\n"
            return content
        else:
            # 没有小节的情况
            return self.contents.get(chapter_id, "")
    
    def get_section_content(self, chapter_id: int, section_id: int) -> str:
        """
        获取指定小节的内容
        
        Args:
            chapter_id: 章节ID
            section_id: 小节ID
            
        Returns:
            小节内容
        """
        if not self.contents:
            self.read_contents()
            
        if chapter_id not in self.chapters:
            raise ValueError(f"章节ID {chapter_id} 不存在")
            
        if section_id not in self.chapters[chapter_id]['sections']:
            raise ValueError(f"小节ID {section_id} 在章节 {chapter_id} 中不存在")
            
        return self.contents.get((chapter_id, section_id), "")
    
    def get_book_info(self) -> Dict:
        """
        获取书籍信息
        
        Returns:
            书籍信息字典
        """
        with open(self.epub_path, 'rb') as f:
            mi = get_metadata(f, 'epub')
            
        return {
            'title': mi.title,
            'authors': mi.authors,
            'publisher': mi.publisher,
            'language': mi.language,
            'comments': mi.comments,
            'pubdate': mi.pubdate,
            'identifiers': mi.identifiers,
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='电子书解析器')
    parser.add_argument('file_path', help='电子书文件路径')
    parser.add_argument('--output-dir', '-o', help='输出目录')
    parser.add_argument('--save-txt', '-s', action='store_true', help='保存为TXT文件')
    parser.add_argument('--max-workers', '-w', type=int, default=4, help='最大并发工作线程数')
    
    args = parser.parse_args()
    
    try:
        # 创建电子书解析器
        reader = EbookReader(args.file_path, args.output_dir)
        
        # 提取章节结构
        chapters = reader.extract_chapters()
        
        # 读取内容
        contents = reader.read_contents(max_workers=args.max_workers)
        
        # 保存为TXT文件
        if args.save_txt:
            output_file = reader.save_to_txt()
            print(f"已保存为TXT文件: {output_file}")
        
        # 打印书籍信息
        book_info = reader.get_book_info()
        print("\n书籍信息:")
        for key, value in book_info.items():
            if value:
                print(f"{key}: {value}")
        
        print("\n处理完成!")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())