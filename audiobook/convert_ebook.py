#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess

def get_args():
    parser = argparse.ArgumentParser(
        description='电子书转换工具',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "ebook_path",
        type=str,
        help="电子书文件路径",
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["text", "audio", "both"],
        default="both",
        help="转换模式: text(仅文本), audio(仅音频), both(文本和音频)",
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
        "--output-dir",
        type=str,
        default="./output",
        help="输出目录",
    )
    
    parser.add_argument(
        "--chapter-range",
        type=str,
        default=None,
        help="要处理的章节范围，格式为'start-end'，例如'1-5'",
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="最大并发工作线程数",
    )
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.ebook_path):
        print(f"错误: 文件不存在: {args.ebook_path}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根据模式执行不同的操作
    if args.mode in ["text", "both"]:
        print("\n=== 提取电子书文本 ===")
        cmd = [
            sys.executable, 
            "ebook_reader.py",
            args.ebook_path,
            "--output-dir", args.output_dir,
            "--save-txt",
            "--max-workers", str(args.max_workers)
        ]
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("文本提取失败")
            if args.mode == "text":
                return 1
    
    if args.mode in ["audio", "both"]:
        print("\n=== 转换电子书为音频 ===")
        cmd = [
            sys.executable,
            "ebook_to_tts.py",
            "--ebook-path", args.ebook_path,
            "--server-url", args.server_url,
            "--reference-audio", args.reference_audio,
            "--output-dir", os.path.join(args.output_dir, "audio"),
            "--max-workers", str(args.max_workers)
        ]
        
        if args.chapter_range:
            cmd.extend(["--chapter-range", args.chapter_range])
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("音频转换失败")
            return 1
    
    print(f"\n转换完成! 输出目录: {args.output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())