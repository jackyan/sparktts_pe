import os
import json
import wave
import contextlib

def analyze_debug_outputs(output_dir="./debug_outputs"):
    """Analyze the debug outputs to identify text length limitations."""
    
    results = []
    
    # Collect data from all test cases
    for file in os.listdir(output_dir):
        if file.startswith("response_") and file.endswith(".json"):
            text_length = int(file.split("_")[1].split(".")[0])
            
            # Get response data
            with open(os.path.join(output_dir, file), 'r') as f:
                response = json.load(f)
            
            # Get request data
            request_file = f"request_{text_length}.json"
            with open(os.path.join(output_dir, request_file), 'r') as f:
                request = json.load(f)
            
            # Get audio duration
            audio_file = f"output_{text_length}.wav"
            with contextlib.closing(wave.open(os.path.join(output_dir, audio_file), 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
            
            # Get input text
            input_text = request["inputs"][3]["data"][0]
            
            # Calculate expected duration (rough estimate for Chinese)
            expected_duration = len(input_text) / 4.5
            
            results.append({
                "text_length": text_length,
                "audio_duration": duration,
                "expected_duration": expected_duration,
                "ratio": duration / expected_duration,
                "input_text": input_text
            })
    
    # Sort results by text length
    results.sort(key=lambda x: x["text_length"])
    
    # Print analysis
    print("=== Text Length vs Audio Duration Analysis ===")
    print(f"{'Text Length':<12} {'Audio Duration':<15} {'Expected Duration':<18} {'Ratio':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['text_length']:<12} {result['audio_duration']:.2f}s{'':<10} {result['expected_duration']:.2f}s{'':<10} {result['ratio']:.2f}")
    
    # Save analysis to a CSV file
    with open(os.path.join(output_dir, 'analysis_results.csv'), 'w') as f:
        f.write("text_length,audio_duration,expected_duration,ratio\n")
        for result in results:
            f.write(f"{result['text_length']},{result['audio_duration']:.2f},{result['expected_duration']:.2f},{result['ratio']:.2f}\n")
    
    # Determine the likely character limit
    if len(results) > 3:
        # Look for where the ratio drops significantly
        ratios = [r["ratio"] for r in results]
        
        # If the ratio drops below 0.8 and keeps dropping, that's likely our limit
        limit_idx = None
        for i in range(1, len(ratios)):
            if ratios[i] < 0.8 * ratios[0] and ratios[i] < ratios[i-1]:
                limit_idx = i
                break
        
        if limit_idx is not None:
            limit_length = results[limit_idx]["text_length"]
            print(f"\nLikely character limit: ~{limit_length} characters")
            print(f"At this length, the audio duration is {results[limit_idx]['audio_duration']:.2f}s")
            print(f"But expected duration would be {results[limit_idx]['expected_duration']:.2f}s")
            
            # Check if the audio content is truncated
            print("\nAnalyzing audio content pattern:")
            for i in range(limit_idx-1, min(limit_idx+2, len(results))):
                r = results[i]
                print(f"Text length {r['text_length']}: duration {r['audio_duration']:.2f}s, ratio {r['ratio']:.2f}")
        else:
            print("\nNo clear character limit detected.")
    
    return results

if __name__ == "__main__":
    analyze_debug_outputs()