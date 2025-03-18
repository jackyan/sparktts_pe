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
import requests
import soundfile as sf
import json
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default="localhost:8000",
        help="Address of the server",
    )

    parser.add_argument(
        "--reference-audio",
        type=str,
        default="../../example/prompt_audio.wav",
        help="Path to a single audio file. It can't be specified at the same time with --manifest-dir",
    )

    parser.add_argument(
        "--reference-text",
        type=str,
        default="吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
        help="",
    )

    parser.add_argument(
        "--target-text",
        type=str,
        default="""随着大模型为主的生成式AI技术取得快速发展，各大PC厂商不仅在积极探索全新的AI PC形态，为推动大模型推理快速高效实现也在积极采纳和部署强劲的AI芯片。
传统AI PC解决方案是在CPU中嵌入iNPU，在运行大语言模型时，通常依赖GPU进行加速，iNPU只有在特定的场景中才能被调用。然而，GPU在处理大模型时可能会面临一些性能瓶颈，如GPU的架构虽然适合并行计算，但在处理深度学习任务时，会导致资源利用率不足或延迟较高。此外，GPU在推理阶段的功耗相对较高。""",
        help="",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="spark_tts",
        choices=[
            "f5_tts", "spark_tts"
        ],
        help="triton model_repo module name to request: transducer for k2, attention_rescoring for wenet offline, streaming_wenet for wenet streaming, infer_pipeline for paraformer large offline",
    )

    parser.add_argument(
        "--output-audio",
        type=str,
        default="output.wav",
        help="Path to save the output audio",
    )
    return parser.parse_args()

def prepare_request(
    waveform,
    reference_text,
    target_text,
    sample_rate=16000,
    padding_duration: int = None,
    audio_save_dir: str = "./",
):
    assert len(waveform.shape) == 1, "waveform should be 1D"
    lengths = np.array([[len(waveform)]], dtype=np.int32)
    if padding_duration:
        # padding to nearset 10 seconds
        samples = np.zeros(
            (
                1,
                padding_duration
                * sample_rate
                * ((int(duration) // padding_duration) + 1),
            ),
            dtype=np.float32,
        )

        samples[0, : len(waveform)] = waveform
    else:
        samples = waveform
        
    samples = samples.reshape(1, -1).astype(np.float32)

    data = {
        "inputs":[
            {
                "name": "reference_wav",
                "shape": samples.shape,
                "datatype": "FP32",
                "data": samples.tolist()
            },
            {
                "name": "reference_wav_len",
                "shape": lengths.shape,
                "datatype": "INT32",
                "data": lengths.tolist(),
            },
            {
                "name": "reference_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [reference_text]
            },
            {
                "name": "target_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [target_text]
            }
        ]
    }

    return data

if __name__ == "__main__":
    args = get_args()
    server_url = args.server_url
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"
    
    url = f"{server_url}/v2/models/{args.model_name}/infer"
    waveform, sr = sf.read(args.reference_audio)
    assert sr == 16000, "sample rate hardcoded in server"
    
    samples = np.array(waveform, dtype=np.float32)
    data = prepare_request(samples, args.reference_text, args.target_text)

    rsp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=data,
        verify=False,
        params={"request_id": '0'}
    )
    result = rsp.json()
    audio = result["outputs"][0]["data"]
    audio = np.array(audio, dtype=np.float32)
    sf.write(args.output_audio, audio, 16000, "PCM_16")