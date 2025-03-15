import requests
import soundfile as sf
import numpy as np
import argparse
import json
import os
import wave
import contextlib

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
        help="Path to a single audio file",
    )

    parser.add_argument(
        "--reference-text",
        type=str,
        default="吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。",
        help="",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./debug_outputs",
        help="Directory to save output files",
    )
    
    return parser.parse_args()

def prepare_request(waveform, reference_text, target_text, sample_rate=16000):
    assert len(waveform.shape) == 1, "waveform should be 1D"
    lengths = np.array([[len(waveform)]], dtype=np.int32)
    samples = waveform.reshape(1, -1).astype(np.float32)

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
    
    url = f"{server_url}/v2/models/spark_tts/infer"
    waveform, sr = sf.read(args.reference_audio)
    assert sr == 16000, "sample rate hardcoded in server"
    
    samples = np.array(waveform, dtype=np.float32)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Base text to repeat
    base_text = "这是一个测试句子，用于检测文本长度限制。"
    
    # Test with different text lengths
    for multiplier in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        test_text = base_text * multiplier
        text_length = len(test_text)
        
        print(f"Testing with text length: {text_length} characters")
        print(f"Text: {test_text}")
        
        # Prepare request
        data = prepare_request(samples, args.reference_text, test_text)
        
        # Save request for debugging
        with open(f"{args.output_dir}/request_{text_length}.json", "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Send request
        try:
            rsp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=data,
                verify=False,
                params={"request_id": str(multiplier)}
            )
            
            # Save response for debugging
            with open(f"{args.output_dir}/response_{text_length}.json", "w") as f:
                json.dump(rsp.json(), f, ensure_ascii=False, indent=2)
            
            # Save audio output
            result = rsp.json()
            audio = result["outputs"][0]["data"]
            audio = np.array(audio, dtype=np.float32)
            sf.write(f"{args.output_dir}/output_{text_length}.wav", audio, 16000, "PCM_16")
            
            # Get audio duration
            with contextlib.closing(wave.open(f"{args.output_dir}/output_{text_length}.wav", 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
            
            # Calculate expected duration based on text length
            # Assuming average speaking rate of 4-5 characters per second for Chinese
            expected_duration = text_length / 4.5  # rough estimate
            
            print(f"Success! Audio saved to {args.output_dir}/output_{text_length}.wav")
            print(f"Text length: {text_length} characters")
            print(f"Audio duration: {duration:.2f} seconds")
            print(f"Expected duration: {expected_duration:.2f} seconds")
            print(f"Ratio: {duration/expected_duration:.2f}")
            
            # Save analysis
            with open(f"{args.output_dir}/analysis_{text_length}.txt", "w") as f:
                f.write(f"Text: {test_text}\n")
                f.write(f"Text length: {text_length} characters\n")
                f.write(f"Audio duration: {duration:.2f} seconds\n")
                f.write(f"Expected duration: {expected_duration:.2f} seconds\n")
                f.write(f"Ratio: {duration/expected_duration:.2f}\n")
                
        except Exception as e:
            print(f"Error processing text length {text_length}: {e}")
            with open(f"{args.output_dir}/error_{text_length}.txt", "w") as f:
                f.write(str(e))
        
        print("-" * 50)