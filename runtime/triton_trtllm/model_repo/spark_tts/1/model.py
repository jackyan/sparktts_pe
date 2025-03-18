# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import os
import re
import logging
import datetime
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

from sparktts.utils.token_parser import TASK_TOKEN_MAP

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

def process_prompt(
    text: str,
    prompt_text: Optional[str] = None,
    global_token_ids: torch.Tensor = None,
    semantic_token_ids: torch.Tensor = None,
) -> Tuple[str, torch.Tensor]:
    """
    Process input for voice cloning.

    Args:
        text: The text input to be converted to speech.
        prompt_text: Transcript of the prompt audio.
        global_token_ids: Global token IDs extracted from reference audio.
        semantic_token_ids: Semantic token IDs extracted from reference audio.

    Returns:
        Tuple containing the formatted input prompt and global token IDs.
    """
    # Convert global tokens to string format
    global_tokens = "".join(
        [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
    )

    
    # Prepare the input tokens for the model
    if prompt_text is not None:
        # Include semantic tokens when prompt text is provided
        semantic_tokens = "".join(
            [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
        )

        inputs = [
            TASK_TOKEN_MAP["tts"],
            "<|start_content|>",
            prompt_text,
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
            "<|start_semantic_token|>",
            semantic_tokens,
        ]
    else:
        # Without prompt text, exclude semantic tokens
        inputs = [
            TASK_TOKEN_MAP["tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
        ]

    # Join all input components into a single string
    inputs = "".join(inputs)
    return inputs, global_token_ids


class TritonPythonModel:
    """Triton Python model for Spark TTS.
    
    This model orchestrates the end-to-end TTS pipeline by coordinating
    between audio tokenizer, LLM, and vocoder components.
    """
    def initialize(self, args):
        """Initialize the model."""
        # 创建日志文件和日志记录器
        log_file = f"/workspace/sparktts_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = setup_logger('sparktts', log_file)
        self.logger.info("初始化模型...")
        
        # Parse model parameters
        parameters = json.loads(args['model_config'])['parameters']
        model_params = {k: v["string_value"] for k, v in parameters.items()}
        
        # 记录模型参数
        self.logger.info(f"模型参数: {model_params}")
        
        # Initialize tokenizer
        llm_tokenizer_dir = model_params["llm_tokenizer_dir"]
        self.tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_dir)
        self.device = torch.device("cuda")
        self.decoupled = False
        self.logger.info(f"模型初始化完成，使用设备: {self.device}")
    
    def initialize_orig(self, args):
        """Initialize the model.
        
        Args:
            args: Dictionary containing model configuration
        """
        # 创建日志文件
        import datetime
        self.log_file = f"/workspace/sparktts_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        with open(self.log_file, "w") as f:
            f.write(f"[{datetime.datetime.now()}] 初始化模型...\n")
        # Parse model parameters
        parameters = json.loads(args['model_config'])['parameters']
        model_params = {k: v["string_value"] for k, v in parameters.items()}
        
        # 记录模型参数
        with open(self.log_file, "a") as f:
            f.write(f"[{datetime.datetime.now()}] 模型参数: {model_params}\n")
        # Initialize tokenizer
        llm_tokenizer_dir = model_params["llm_tokenizer_dir"]
        self.tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_dir)
        self.device = torch.device("cuda")
        self.decoupled = False
        with open(self.log_file, "a") as f:
            f.write(f"[{datetime.datetime.now()}] 模型初始化完成，使用设备: {self.device}\n")

    def forward_llm(self, input_ids):
        """
        Prepares the response from the language model based on the provided
        inputs. Creates a `pb_utils.InferenceRequest` object with passed
        `llm_request_inputs` to send to a decoupled TensorRTLLM model.
        For each response from the language model:
            - Checks for errors and raise an exception if any are found.
            - Extracts the "output_ids" tensor from the response.
            - Determines the finish reason based on the presence of the
              end-of-sequence token or reaching the maximum length.
            - Appends the generated token IDs to `output_ids`.
            - If the finish reason is determined, decodes the output IDs to text
              and prepares the final response.

        The final response includes the generated text, finish reason,
        completion tokens, prompt tokens, and total tokens.

        Parameters
        ----------
        - llm_request_inputs (dict): A dictionary containing the inputs for the language model.

        Returns
        -------
        - pb_utils.InferenceResponse: The response object containing the generated text and additional metadata.
        """
        # convert input_ids to numpy, with shape [1, sequence_length]
        input_ids = input_ids.cpu().numpy()
        max_tokens = 5120
        input_dict = {
            "request_output_len": np.array([[max_tokens]], dtype=np.int32),
            "end_id": np.array([[self.tokenizer.eos_token_id]], dtype=np.int32),
            "pad_id": np.array([[self.tokenizer.pad_token_id]], dtype=np.int32),
            "streaming": np.array([[self.decoupled]], dtype=np.bool_),
            "runtime_top_p": np.array([[0.95]], dtype=np.float32),
            "runtime_top_k": np.array([[50]], dtype=np.int32),
            "temperature": np.array([[0.8]], dtype=np.float32),
            "input_ids": input_ids,
            "input_lengths": np.array([[input_ids.shape[1]]], dtype=np.int32),
        }
        
        # Convert inputs to Triton tensors
        input_tensor_list = [
            pb_utils.Tensor(k, v) for k, v in input_dict.items()
        ]
        
        # Create and execute inference request
        llm_request = pb_utils.InferenceRequest(
            model_name="tensorrt_llm",
            requested_output_names=["output_ids", "sequence_length"],
            inputs=input_tensor_list,
        )
        
        llm_response = llm_request.exec(decoupled=self.decoupled)
        if llm_response.has_error():
            raise pb_utils.TritonModelException(llm_response.error().message())
        
        # Extract and process output
        output_ids = pb_utils.get_output_tensor_by_name(
            llm_response, "output_ids").as_numpy()
        seq_lens = pb_utils.get_output_tensor_by_name(
            llm_response, "sequence_length").as_numpy()
        
        # Get actual output IDs up to the sequence length
        actual_output_ids = output_ids[0][0][:seq_lens[0][0]]
        
        return actual_output_ids

    def forward_audio_tokenizer(self, wav, wav_len):
        """Forward pass through the audio tokenizer component.
        
        Args:
            wav: Input waveform tensor
            wav_len: Waveform length tensor
            
        Returns:
            Tuple of global and semantic tokens
        """
        inference_request = pb_utils.InferenceRequest(
            model_name='audio_tokenizer',
            requested_output_names=['global_tokens', 'semantic_tokens'],
            inputs=[wav, wav_len]
        )
        
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        
        # Extract and convert output tensors
        global_tokens = pb_utils.get_output_tensor_by_name(inference_response, 'global_tokens')
        global_tokens = torch.utils.dlpack.from_dlpack(global_tokens.to_dlpack()).cpu()
        
        semantic_tokens = pb_utils.get_output_tensor_by_name(inference_response, 'semantic_tokens')
        semantic_tokens = torch.utils.dlpack.from_dlpack(semantic_tokens.to_dlpack()).cpu()
        
        return global_tokens, semantic_tokens

    def forward_vocoder_with_log(self, global_token_ids: torch.Tensor, pred_semantic_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the vocoder component."""
        import datetime
        
        with open(self.log_file, "a") as f:
            f.write(f"[{datetime.datetime.now()}] Vocoder输入 - global_token_ids shape: {global_token_ids.shape}\n")
            f.write(f"[{datetime.datetime.now()}] Vocoder输入 - pred_semantic_ids shape: {pred_semantic_ids.shape}\n")
            if pred_semantic_ids.shape[1] > 1000:
                f.write(f"[{datetime.datetime.now()}] 警告: 语义token数量较大，可能超出vocoder处理能力\n")
                # 记录前10个和后10个语义token，检查是否有重复模式
                f.write(f"[{datetime.datetime.now()}] 前10个语义token: {pred_semantic_ids[0, :10].tolist()}\n")
                f.write(f"[{datetime.datetime.now()}] 后10个语义token: {pred_semantic_ids[0, -10:].tolist()}\n")
        
        # 尝试分段处理长语义序列
        max_segment_length = 800  # 设置一个合理的分段长度
        if pred_semantic_ids.shape[1] > max_segment_length:
            with open(self.log_file, "a") as f:
                f.write(f"[{datetime.datetime.now()}] 语义token数量({pred_semantic_ids.shape[1]})超过{max_segment_length}，尝试分段处理\n")
            
            # 分段处理
            segments = []
            for i in range(0, pred_semantic_ids.shape[1], max_segment_length):
                end_idx = min(i + max_segment_length, pred_semantic_ids.shape[1])
                segment = pred_semantic_ids[:, i:end_idx]
                
                # 为每个段落创建输入张量
                global_token_ids_tensor = pb_utils.Tensor.from_dlpack("global_tokens", to_dlpack(global_token_ids))
                segment_tensor = pb_utils.Tensor.from_dlpack("semantic_tokens", to_dlpack(segment))
                
                # 处理当前段落
                inference_request = pb_utils.InferenceRequest(
                    model_name='vocoder',
                    requested_output_names=['waveform'],
                    inputs=[global_token_ids_tensor, segment_tensor]
                )
                
                inference_response = inference_request.exec()
                if inference_response.has_error():
                    with open(self.log_file, "a") as f:
                        f.write(f"[{datetime.datetime.now()}] 段落处理错误: {inference_response.error().message()}\n")
                    continue
                
                # 提取当前段落的波形
                segment_waveform = pb_utils.get_output_tensor_by_name(inference_response, 'waveform')
                segment_waveform = torch.utils.dlpack.from_dlpack(segment_waveform.to_dlpack()).cpu()
                segments.append(segment_waveform)
                
                with open(self.log_file, "a") as f:
                    f.write(f"[{datetime.datetime.now()}] 段落 {i//max_segment_length + 1} 处理完成，波形shape: {segment_waveform.shape}\n")
            
            # 合并所有段落
            if segments:
                waveform = torch.cat(segments, dim=1)
                with open(self.log_file, "a") as f:
                    f.write(f"[{datetime.datetime.now()}] 合并后的波形shape: {waveform.shape}\n")
                return waveform
            else:
                with open(self.log_file, "a") as f:
                    f.write(f"[{datetime.datetime.now()}] 所有段落处理失败，尝试使用原始方法\n")
        
        # 原始处理逻辑
        global_token_ids_tensor = pb_utils.Tensor.from_dlpack("global_tokens", to_dlpack(global_token_ids))
        pred_semantic_ids_tensor = pb_utils.Tensor.from_dlpack("semantic_tokens", to_dlpack(pred_semantic_ids))
        
        inference_request = pb_utils.InferenceRequest(
            model_name='vocoder',
            requested_output_names=['waveform'],
            inputs=[global_token_ids_tensor, pred_semantic_ids_tensor]
        )
        
        inference_response = inference_request.exec()
        if inference_response.has_error():
            error_msg = inference_response.error().message()
            with open(self.log_file, "a") as f:
                f.write(f"[{datetime.datetime.now()}] Vocoder错误: {error_msg}\n")
            raise pb_utils.TritonModelException(error_msg)
        
        waveform = pb_utils.get_output_tensor_by_name(inference_response, 'waveform')
        waveform = torch.utils.dlpack.from_dlpack(waveform.to_dlpack()).cpu()
        
        with open(self.log_file, "a") as f:
            f.write(f"[{datetime.datetime.now()}] Vocoder输出 - waveform shape: {waveform.shape}\n")
        
        return waveform
    
    def forward_vocoder(self, global_token_ids: torch.Tensor, pred_semantic_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the vocoder component.
        
        Args:
            global_token_ids: Global token IDs tensor
            pred_semantic_ids: Predicted semantic token IDs tensor
            
        Returns:
            Generated waveform tensor
        """
        # Convert tensors to Triton format
        global_token_ids_tensor = pb_utils.Tensor.from_dlpack("global_tokens", to_dlpack(global_token_ids))
        pred_semantic_ids_tensor = pb_utils.Tensor.from_dlpack("semantic_tokens", to_dlpack(pred_semantic_ids))
        
        # Create and execute inference request
        inference_request = pb_utils.InferenceRequest(
            model_name='vocoder',
            requested_output_names=['waveform'],
            inputs=[global_token_ids_tensor, pred_semantic_ids_tensor]
        )
        
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        
        # Extract and convert output waveform
        waveform = pb_utils.get_output_tensor_by_name(inference_response, 'waveform')
        waveform = torch.utils.dlpack.from_dlpack(waveform.to_dlpack()).cpu()
        
        return waveform

    def preprocess_text(self, text, min_segment_length=80, max_segment_length=120, logger=None):
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
            log_file = f"/workspace/sparktts_debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

    def process_text_segment(self, segment, reference_text, global_tokens, semantic_tokens, segment_index, total_segments):
        """
        处理单个文本段落，生成对应的音频
        
        Args:
            segment: 要处理的文本段落
            reference_text: 参考文本，仅在第一段使用
            global_tokens: 全局token
            semantic_tokens: 语义token
            segment_index: 当前段落索引
            total_segments: 总段落数
            
        Returns:
            生成的音频段落，处理失败时返回None
        """
        try:
            self.logger.info(f"处理段落 {segment_index+1}/{total_segments}: {segment}")
            
            # 处理当前段落
            prompt, global_token_ids = process_prompt(
                text=segment,
                prompt_text=reference_text if segment_index == 0 else None,  # 只在第一段使用参考文本
                global_token_ids=global_tokens,
                semantic_token_ids=semantic_tokens,
            )
            
            self.logger.info(f"段落{segment_index+1}拼接后的输入长度: {len(prompt)}")
            
            # Tokenize prompt for LLM
            model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            input_ids = model_inputs.input_ids.to(torch.int32)
            
            self.logger.info(f"段落{segment_index+1}分词后的输入token数量: {input_ids.shape[1]}")
            
            # Generate semantic tokens with LLM
            generated_ids = self.forward_llm(input_ids)
            
            # Decode and extract semantic token IDs from generated text
            predicted_text = self.tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0]
            
            self.logger.info(f"段落{segment_index+1}生成文本长度: {len(predicted_text)}")
            
            pred_semantic_ids = (
                torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicted_text)])
                .unsqueeze(0).to(torch.int32)
            )
            
            self.logger.info(f"段落{segment_index+1}提取的语义token数量: {pred_semantic_ids.shape[1]}")
            
            # 生成音频
            segment_audio = self.forward_vocoder(
                global_token_ids.to(self.device),
                pred_semantic_ids.to(self.device),
            )
            
            self.logger.info(f"段落{segment_index+1}生成的音频shape: {segment_audio.shape}")
            
            return segment_audio
        except Exception as e:
            # 记录异常信息
            self.logger.error(f"处理段落{segment_index+1}时发生错误: {str(e)}", exc_info=True)
            return None

    def process_text_segments(self, segments, reference_text, global_tokens, semantic_tokens):
        """
        处理所有文本段落并合并结果
        
        Args:
            segments: 文本段落列表
            reference_text: 参考文本
            global_tokens: 全局token
            semantic_tokens: 语义token
            
        Returns:
            合并后的音频，如果所有段落处理失败则返回None
        """
        # 处理所有段落并收集结果
        all_audio_segments = []
        
        for i, segment in enumerate(segments):
            segment_audio = self.process_text_segment(
                segment, 
                reference_text,  # 对于单段文本，始终使用参考文本
                global_tokens, 
                semantic_tokens, 
                i, 
                len(segments)
            )
            
            if segment_audio is not None:
                all_audio_segments.append(segment_audio)
        
        # 处理结果
        if all_audio_segments:
            # 如果有多个段落，需要合并
            if len(all_audio_segments) > 1:
                self.logger.info(f"合并{len(all_audio_segments)}个音频段落")
                audio = torch.cat(all_audio_segments, dim=1)
                self.logger.info(f"合并后的音频shape: {audio.shape}")
            else:
                # 只有一个段落，直接使用
                audio = all_audio_segments[0]
            
            return audio
        else:
            # 所有段落处理失败
            self.logger.error("所有文本段落处理失败")
            return None

    def execute(self, requests):
        """Execute inference on the batched requests."""
        responses = []
        
        self.logger.info(f"收到请求数量: {len(requests)}")
        
        for request_idx, request in enumerate(requests):
            self.logger.info(f"处理请求 {request_idx+1}/{len(requests)}")
            
            try:
                # Extract input tensors
                wav = pb_utils.get_input_tensor_by_name(request, "reference_wav")
                wav_len = pb_utils.get_input_tensor_by_name(request, "reference_wav_len")
                
                # Process reference audio through audio tokenizer
                global_tokens, semantic_tokens = self.forward_audio_tokenizer(wav, wav_len)
                
                # Extract text inputs
                reference_text = pb_utils.get_input_tensor_by_name(request, "reference_text").as_numpy()
                reference_text = reference_text[0][0].decode('utf-8')
                
                target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
                target_text = target_text[0][0].decode('utf-8')
                
                self.logger.info(f"参考文本: {reference_text}")
                self.logger.info(f"目标文本: {target_text}")
                self.logger.info(f"目标文本长度: {len(target_text)}")
                self.logger.info(f"Global token IDs shape: {global_tokens.shape}")
                self.logger.info(f"Semantic token IDs shape: {semantic_tokens.shape}")
                
                # 预处理文本，决定是否分段并进行分段
                segments = preprocess_text(target_text, min_segment_length=50, max_segment_length=80, logger=self.logger)
                
                # 处理所有段落并获取合并后的音频
                audio = self.process_text_segments(segments, reference_text, global_tokens, semantic_tokens)
                
                # 准备响应
                if audio is not None:
                    audio_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(audio))
                    inference_response = pb_utils.InferenceResponse(output_tensors=[audio_tensor])
                else:
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError("所有文本段落处理失败")
                    )
                    
                responses.append(inference_response)
                self.logger.info(f"请求 {request_idx+1} 处理完成")
            
            except Exception as e:
                # 处理整个请求的异常
                self.logger.error(f"处理请求 {request_idx+1} 时发生错误: {str(e)}", exc_info=True)
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"处理请求失败: {str(e)}")
                )
                responses.append(inference_response)
                        
        return responses

    def execute_backup(self, requests):
        """Execute inference on the batched requests."""
        responses = []
        
        self.logger.info(f"收到请求数量: {len(requests)}")
        
        for request_idx, request in enumerate(requests):
            self.logger.info(f"处理请求 {request_idx+1}/{len(requests)}")
            
            # Extract input tensors
            wav = pb_utils.get_input_tensor_by_name(request, "reference_wav")
            wav_len = pb_utils.get_input_tensor_by_name(request, "reference_wav_len")
            
            # Process reference audio through audio tokenizer
            global_tokens, semantic_tokens = self.forward_audio_tokenizer(wav, wav_len)
            
            # Extract text inputs
            reference_text = pb_utils.get_input_tensor_by_name(request, "reference_text").as_numpy()
            reference_text = reference_text[0][0].decode('utf-8')
            
            target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
            target_text = target_text[0][0].decode('utf-8')
            
            self.logger.info(f"参考文本: {reference_text}")
            self.logger.info(f"目标文本: {target_text}")
            self.logger.info(f"目标文本长度: {len(target_text)}")
            self.logger.info(f"Global token IDs shape: {global_tokens.shape}")
            self.logger.info(f"Semantic token IDs shape: {semantic_tokens.shape}")
            
            # 检查文本长度，如果超过100个字符，则分段处理
            if len(target_text) > 100:
                self.logger.info("目标文本长度超过100，尝试分段处理")
                
                # 按标点符号分段
                segments = []
                current_segment = ""
                punctuations = ['，', '。', '！', '？', '；', '：', ',', '!', '?', ';', ':']
                
                for char in target_text:
                    current_segment += char
                    if char in punctuations and len(current_segment) >= 50:
                        segments.append(current_segment)
                        current_segment = ""
                
                # 处理最后一段
                if current_segment:
                    segments.append(current_segment)
                
                # 如果没有找到合适的分割点，则强制分割
                if len(segments) <= 1 and len(target_text) > 100:
                    segments = []
                    for i in range(0, len(target_text), 80):
                        segments.append(target_text[i:min(i+80, len(target_text))])
                
                self.logger.info(f"文本被分为{len(segments)}段")
                for i, seg in enumerate(segments):
                    self.logger.info(f"段落{i+1}: {seg}")
                
                # 处理每个段落并合并结果
                all_audio_segments = []
                
                for i, segment in enumerate(segments):
                    try:  # 添加异常处理
                        self.logger.info(f"处理段落 {i+1}/{len(segments)}: {segment}")
                        
                        # 处理当前段落
                        prompt, global_token_ids = process_prompt(
                            text=segment,
                            prompt_text=reference_text if i == 0 else None,  # 只在第一段使用参考文本
                            global_token_ids=global_tokens,
                            semantic_token_ids=semantic_tokens,
                        )
                        
                        self.logger.info(f"段落{i+1}拼接后的输入长度: {len(prompt)}")
                        
                        # Tokenize prompt for LLM
                        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
                        input_ids = model_inputs.input_ids.to(torch.int32)
                        
                        self.logger.info(f"段落{i+1}分词后的输入token数量: {input_ids.shape[1]}")
                        
                        # Generate semantic tokens with LLM
                        generated_ids = self.forward_llm(input_ids)
                        
                        # Decode and extract semantic token IDs from generated text
                        predicted_text = self.tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0]
                        
                        self.logger.info(f"段落{i+1}生成文本长度: {len(predicted_text)}")
                        
                        pred_semantic_ids = (
                            torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicted_text)])
                            .unsqueeze(0).to(torch.int32)
                        )
                        
                        self.logger.info(f"段落{i+1}提取的语义token数量: {pred_semantic_ids.shape[1]}")
                        
                        # 使用正确的vocoder函数
                        segment_audio = self.forward_vocoder(
                            global_token_ids.to(self.device),
                            pred_semantic_ids.to(self.device),
                        )
                        
                        self.logger.info(f"段落{i+1}生成的音频shape: {segment_audio.shape}")
                        
                        all_audio_segments.append(segment_audio)
                    except Exception as e:
                        # 记录异常信息
                        self.logger.error(f"处理段落{i+1}时发生错误: {str(e)}", exc_info=True)
                        continue  # 继续处理下一个段落
                
                # 合并所有音频段落
                if all_audio_segments:
                    self.logger.info(f"合并{len(all_audio_segments)}个音频段落")
                    
                    # 简单拼接音频段落
                    audio = torch.cat(all_audio_segments, dim=1)
                    
                    self.logger.info(f"合并后的音频shape: {audio.shape}")
                else:
                    # 如果所有段落处理失败，记录错误并返回空响应
                    self.logger.error("所有段落处理失败，返回空响应")
                    inference_response = pb_utils.InferenceResponse(
                        output_tensors=[],
                        error=pb_utils.TritonError("所有文本段落处理失败")
                    )
                    responses.append(inference_response)
                    continue
            else:
                # 原始处理逻辑，不分段
                prompt, global_token_ids = process_prompt(
                    text=target_text,
                    prompt_text=reference_text,
                    global_token_ids=global_tokens,
                    semantic_token_ids=semantic_tokens,
                )
                
                self.logger.info(f"拼接后的输入长度: {len(prompt)}")
                self.logger.info(f"拼接后的输入前100个字符: {prompt[:100]}...")
                if len(prompt) > 200:
                    self.logger.info(f"拼接后的输入后100个字符: ...{prompt[-100:]}")
                
                # Tokenize prompt for LLM
                model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
                input_ids = model_inputs.input_ids.to(torch.int32)
                
                self.logger.info(f"分词后的输入token数量: {input_ids.shape[1]}")
                
                # Generate semantic tokens with LLM
                generated_ids = self.forward_llm(input_ids)
                
                # Decode and extract semantic token IDs from generated text
                predicted_text = self.tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0]
                
                self.logger.info(f"生成文本长度: {len(predicted_text)}")
                self.logger.info(f"生成文本前100个字符: {predicted_text[:100]}...")
                
                pred_semantic_ids = (
                    torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicted_text)])
                    .unsqueeze(0).to(torch.int32)
                )
                
                self.logger.info(f"提取的语义token数量: {pred_semantic_ids.shape[1]}")
                
                # Generate audio with vocoder
                audio = self.forward_vocoder(
                    global_token_ids.to(self.device),
                    pred_semantic_ids.to(self.device),
                )
            
            # Prepare response
            audio_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(audio))
            inference_response = pb_utils.InferenceResponse(output_tensors=[audio_tensor])
            responses.append(inference_response)
            
            self.logger.info(f"请求 {request_idx+1} 处理完成")
                         
        return responses

    def execute_with_file(self, requests):
        """Execute inference on the batched requests.
        
        Args:
            requests: List of inference requests
            
        Returns:
            List of inference responses containing generated audio
        """
        import datetime
        responses = []
        
        with open(self.log_file, "a") as f:
            f.write(f"[{datetime.datetime.now()}] 收到请求数量: {len(requests)}\n")
        
        for request_idx, request in enumerate(requests):
            with open(self.log_file, "a") as f:
                f.write(f"[{datetime.datetime.now()}] 处理请求 {request_idx+1}/{len(requests)}\n")
            
            # Extract input tensors
            wav = pb_utils.get_input_tensor_by_name(request, "reference_wav")
            wav_len = pb_utils.get_input_tensor_by_name(request, "reference_wav_len")
            
            # Process reference audio through audio tokenizer
            global_tokens, semantic_tokens = self.forward_audio_tokenizer(wav, wav_len)
            
            # Extract text inputs
            reference_text = pb_utils.get_input_tensor_by_name(request, "reference_text").as_numpy()
            reference_text = reference_text[0][0].decode('utf-8')
            
            target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
            target_text = target_text[0][0].decode('utf-8')
            
            with open(self.log_file, "a") as f:
                f.write(f"[{datetime.datetime.now()}] 参考文本: {reference_text}\n")
                f.write(f"[{datetime.datetime.now()}] 目标文本: {target_text}\n")
                f.write(f"[{datetime.datetime.now()}] 目标文本长度: {len(target_text)}\n")
                f.write(f"[{datetime.datetime.now()}] Global token IDs shape: {global_tokens.shape}\n")
                f.write(f"[{datetime.datetime.now()}] Semantic token IDs shape: {semantic_tokens.shape}\n")
            
            # 检查文本长度，如果超过100个字符，则分段处理
            if len(target_text) > 100:
                with open(self.log_file, "a") as f:
                    f.write(f"[{datetime.datetime.now()}] 目标文本长度超过100，尝试分段处理\n")
                
                # 按标点符号分段
                segments = []
                current_segment = ""
                punctuations = ['，', '。', '！', '？', '；', '：', ',', '.', '!', '?', ';', ':']
                
                for char in target_text:
                    current_segment += char
                    if char in punctuations and len(current_segment) >= 50:
                        segments.append(current_segment)
                        current_segment = ""
                
                # 处理最后一段
                if current_segment:
                    segments.append(current_segment)
                
                # 如果没有找到合适的分割点，则强制分割
                if len(segments) <= 1 and len(target_text) > 100:
                    segments = []
                    for i in range(0, len(target_text), 80):
                        segments.append(target_text[i:min(i+80, len(target_text))])
                
                with open(self.log_file, "a") as f:
                    f.write(f"[{datetime.datetime.now()}] 文本被分为{len(segments)}段\n")
                    for i, seg in enumerate(segments):
                        f.write(f"[{datetime.datetime.now()}] 段落{i+1}: {seg}\n")
                
                # 处理每个段落并合并结果
                all_audio_segments = []
                
                for i, segment in enumerate(segments):
                    with open(self.log_file, "a") as f:
                        f.write(f"[{datetime.datetime.now()}] 处理段落 {i+1}/{len(segments)}: {segment}\n")
                    
                    # 处理当前段落
                    prompt, global_token_ids = process_prompt(
                        text=segment,
                        prompt_text=reference_text if i == 0 else None,  # 只在第一段使用参考文本
                        global_token_ids=global_tokens,
                        semantic_token_ids=semantic_tokens,
                    )
                    
                    with open(self.log_file, "a") as f:
                        f.write(f"[{datetime.datetime.now()}] 段落{i+1}拼接后的输入长度: {len(prompt)}\n")
                    
                    # Tokenize prompt for LLM
                    model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
                    input_ids = model_inputs.input_ids.to(torch.int32)
                    
                    with open(self.log_file, "a") as f:
                        f.write(f"[{datetime.datetime.now()}] 段落{i+1}分词后的输入token数量: {input_ids.shape[1]}\n")
                    
                    # Generate semantic tokens with LLM
                    generated_ids = self.forward_llm(input_ids)
                    
                    # Decode and extract semantic token IDs from generated text
                    predicted_text = self.tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0]
                    
                    with open(self.log_file, "a") as f:
                        f.write(f"[{datetime.datetime.now()}] 段落{i+1}生成文本长度: {len(predicted_text)}\n")
                    
                    pred_semantic_ids = (
                        torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicted_text)])
                        .unsqueeze(0).to(torch.int32)
                    )
                    
                    with open(self.log_file, "a") as f:
                        f.write(f"[{datetime.datetime.now()}] 段落{i+1}提取的语义token数量: {pred_semantic_ids.shape[1]}\n")
                    
                    # Generate audio with vocoder
                    segment_audio = self.forward_vocoder(
                        global_token_ids.to(self.device),
                        pred_semantic_ids.to(self.device),
                    )
                    
                    with open(self.log_file, "a") as f:
                        f.write(f"[{datetime.datetime.now()}] 段落{i+1}生成的音频shape: {segment_audio.shape}\n")
                    
                    all_audio_segments.append(segment_audio)
                
                # 合并所有音频段落
                with open(self.log_file, "a") as f:
                    f.write(f"[{datetime.datetime.now()}] 合并{len(all_audio_segments)}个音频段落\n")
                
                # 简单拼接音频段落
                audio = torch.cat(all_audio_segments, dim=1)
                
                with open(self.log_file, "a") as f:
                    f.write(f"[{datetime.datetime.now()}] 合并后的音频shape: {audio.shape}\n")
            else:
                # 原始处理逻辑，不分段
                prompt, global_token_ids = process_prompt(
                    text=target_text,
                    prompt_text=reference_text,
                    global_token_ids=global_tokens,
                    semantic_token_ids=semantic_tokens,
                )
                
                with open(self.log_file, "a") as f:
                    f.write(f"[{datetime.datetime.now()}] 拼接后的输入长度: {len(prompt)}\n")
                    f.write(f"[{datetime.datetime.now()}] 拼接后的输入前100个字符: {prompt[:100]}...\n")
                    if len(prompt) > 200:
                        f.write(f"[{datetime.datetime.now()}] 拼接后的输入后100个字符: ...{prompt[-100:]}\n")
                
                # Tokenize prompt for LLM
                model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
                input_ids = model_inputs.input_ids.to(torch.int32)
                
                with open(self.log_file, "a") as f:
                    f.write(f"[{datetime.datetime.now()}] 分词后的输入token数量: {input_ids.shape[1]}\n")
                
                # Generate semantic tokens with LLM
                generated_ids = self.forward_llm(input_ids)
                
                # Decode and extract semantic token IDs from generated text
                predicted_text = self.tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0]
                
                with open(self.log_file, "a") as f:
                    f.write(f"[{datetime.datetime.now()}] 生成文本长度: {len(predicted_text)}\n")
                    f.write(f"[{datetime.datetime.now()}] 生成文本前100个字符: {predicted_text[:100]}...\n")
                
                pred_semantic_ids = (
                    torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicted_text)])
                    .unsqueeze(0).to(torch.int32)
                )
                
                with open(self.log_file, "a") as f:
                    f.write(f"[{datetime.datetime.now()}] 提取的语义token数量: {pred_semantic_ids.shape[1]}\n")
                
                # Generate audio with vocoder
                audio = self.forward_vocoder(
                    global_token_ids.to(self.device),
                    pred_semantic_ids.to(self.device),
                )
            
            # Prepare response
            audio_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(audio))
            inference_response = pb_utils.InferenceResponse(output_tensors=[audio_tensor])
            responses.append(inference_response)
            
            with open(self.log_file, "a") as f:
                f.write(f"[{datetime.datetime.now()}] 请求 {request_idx+1} 处理完成\n")
                            
        return responses

    def execute_orig(self, requests):
        """Execute inference on the batched requests.
        
        Args:
            requests: List of inference requests
            
        Returns:
            List of inference responses containing generated audio
        """
        responses = []
        
        for request in requests:
            # Extract input tensors
            wav = pb_utils.get_input_tensor_by_name(request, "reference_wav")
            wav_len = pb_utils.get_input_tensor_by_name(request, "reference_wav_len")
            
            # Process reference audio through audio tokenizer
            global_tokens, semantic_tokens = self.forward_audio_tokenizer(wav, wav_len)
            
            # Extract text inputs
            reference_text = pb_utils.get_input_tensor_by_name(request, "reference_text").as_numpy()
            reference_text = reference_text[0][0].decode('utf-8')
            
            target_text = pb_utils.get_input_tensor_by_name(request, "target_text").as_numpy()
            target_text = target_text[0][0].decode('utf-8')
            
            # Prepare prompt for LLM
            prompt, global_token_ids = process_prompt(
                text=target_text,
                prompt_text=reference_text,
                global_token_ids=global_tokens,
                semantic_token_ids=semantic_tokens,
            )
            
            
            # Tokenize prompt for LLM
            model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            input_ids = model_inputs.input_ids.to(torch.int32)
            
            # Generate semantic tokens with LLM
            generated_ids = self.forward_llm(input_ids)
            
            # Decode and extract semantic token IDs from generated text
            predicted_text = self.tokenizer.batch_decode([generated_ids], skip_special_tokens=True)[0]
            pred_semantic_ids = (
                torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicted_text)])
                .unsqueeze(0).to(torch.int32)
            )
            

            # Generate audio with vocoder
            audio = self.forward_vocoder(
                global_token_ids.to(self.device),
                pred_semantic_ids.to(self.device),
            )
            
            # Prepare response
            audio_tensor = pb_utils.Tensor.from_dlpack("waveform", to_dlpack(audio))
            inference_response = pb_utils.InferenceResponse(output_tensors=[audio_tensor])
            responses.append(inference_response)
                             
        return responses
