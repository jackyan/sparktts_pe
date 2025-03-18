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
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

from sparktts.utils.token_parser import TASK_TOKEN_MAP

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

    def execute(self, requests):
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
                    segment_audio = self.forward_vocoder_with_log(
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
                audio = self.forward_vocoder_with_log(
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
