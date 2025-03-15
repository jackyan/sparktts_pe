import argparse
import numpy as np
import soundfile as sf
import grpc
import time
import os
from tritonclient.grpc import service_pb2, service_pb2_grpc

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--server-url",
        type=str,
        default="localhost:8001",
        help="Address of the server (note: gRPC uses port 8001 by default)",
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
        help="",
    )

    parser.add_argument(
        "--target-text",
        type=str,
        default="这是一个测试句子，用于检测文本长度限制。这是一个测试句子，用于检测文本长度限制。这是一个测试句子，用于检测文本长度限制。",
        help="Text to synthesize",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="./output_grpc.wav",
        help="Path to save output audio file",
    )
    
    return parser.parse_args()

def prepare_request(waveform, reference_text, target_text, sample_rate=16000):
    assert len(waveform.shape) == 1, "waveform should be 1D"
    lengths = np.array([[len(waveform)]], dtype=np.int32)
    samples = waveform.reshape(1, -1).astype(np.float32)

    # Create InferInput objects for the request
    inputs = []
    
    # reference_wav
    reference_wav_input = service_pb2.ModelInferRequest().InferInputTensor()
    reference_wav_input.name = "reference_wav"
    reference_wav_input.datatype = "FP32"
    reference_wav_input.shape.extend(samples.shape)
    reference_wav_bytes = samples.tobytes()
    reference_wav_input.contents.raw_contents = reference_wav_bytes
    inputs.append(reference_wav_input)
    
    # reference_wav_len
    reference_wav_len_input = service_pb2.ModelInferRequest().InferInputTensor()
    reference_wav_len_input.name = "reference_wav_len"
    reference_wav_len_input.datatype = "INT32"
    reference_wav_len_input.shape.extend(lengths.shape)
    reference_wav_len_bytes = lengths.tobytes()
    reference_wav_len_input.contents.raw_contents = reference_wav_len_bytes
    inputs.append(reference_wav_len_input)
    
    # reference_text
    reference_text_input = service_pb2.ModelInferRequest().InferInputTensor()
    reference_text_input.name = "reference_text"
    reference_text_input.datatype = "BYTES"
    reference_text_input.shape.extend([1, 1])
    reference_text_bytes = reference_text.encode('utf-8')
    reference_text_input.contents.bytes_contents.append(reference_text_bytes)
    inputs.append(reference_text_input)
    
    # target_text
    target_text_input = service_pb2.ModelInferRequest().InferInputTensor()
    target_text_input.name = "target_text"
    target_text_input.datatype = "BYTES"
    target_text_input.shape.extend([1, 1])
    target_text_bytes = target_text.encode('utf-8')
    target_text_input.contents.bytes_contents.append(target_text_bytes)
    inputs.append(target_text_input)
    
    return inputs

def main():
    args = get_args()
    
    # Create gRPC stub
    channel = grpc.insecure_channel(args.server_url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
    
    # Load reference audio
    waveform, sr = sf.read(args.reference_audio)
    assert sr == 16000, "sample rate hardcoded in server"
    
    samples = np.array(waveform, dtype=np.float32)
    
    # Prepare request
    inputs = prepare_request(samples, args.reference_text, args.target_text)
    
    # Create the inference request
    request = service_pb2.ModelInferRequest()
    request.model_name = "spark_tts"
    request.inputs.extend(inputs)
    
    # Create output specification
    output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output.name = "audio"
    request.outputs.append(output)
    
    print(f"Sending request with target text length: {len(args.target_text)} characters")
    print(f"Target text: {args.target_text}")
    
    # Time the request
    start_time = time.time()
    
    # Send request
    try:
        response = grpc_stub.ModelInfer(request)
        
        # Process response
        output = response.outputs[0]
        output_data = output.contents.raw_contents
        
        # Convert to numpy array
        audio_array = np.frombuffer(output_data, dtype=np.float32)
        
        # Save audio output
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        sf.write(args.output_file, audio_array, 16000, "PCM_16")
        
        end_time = time.time()
        
        print(f"Success! Audio saved to {args.output_file}")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        
        # Calculate audio duration
        audio_duration = len(audio_array) / 16000
        print(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Calculate expected duration based on text length
        expected_duration = len(args.target_text) / 4.5  # rough estimate for Chinese
        print(f"Expected duration: {expected_duration:.2f} seconds")
        print(f"Ratio: {audio_duration/expected_duration:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()