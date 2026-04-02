"""
Simple test script to verify the system works end-to-end.

Before running:
1. Install dependencies: pip install -r requirements.txt
2. Set API key: export GEMINI_API_KEY='your-key'
3. Prepare a test audio file
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.audio_processor import AudioProcessor
from utils.audio_io import load_audio


def test_basic_pipeline():
    """Test the basic processing pipeline."""
    print("Testing Sound Effect Generator Pipeline")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set!")
        print("Please run: export GEMINI_API_KEY='your-key'")
        return False
    
    # Create test audio (simple sine wave)
    print("\n1. Creating test audio...")
    try:
        import numpy as np
        sr = 16000
        duration = 5.0
        frequency = 440  # A4
        
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Save test audio
        test_audio_path = "/tmp/test_audio.wav"
        from utils.audio_io import save_audio
        save_audio(audio, test_audio_path, sr=sr)
        print(f"   Created test audio: {test_audio_path}")
    except Exception as e:
        print(f"   ERROR: Failed to create test audio: {str(e)}")
        return False
    
    # Initialize processor
    print("\n2. Initializing processor...")
    try:
        processor = AudioProcessor(sample_rate=sr)
        print("   Processor initialized successfully")
    except Exception as e:
        print(f"   ERROR: Failed to initialize processor: {str(e)}")
        return False
    
    # Test parameter generation
    print("\n3. Testing parameter generation...")
    test_prompts = [
        "Add a chorusing effect",
        "Apply a gentle reverb that gradually increases",
    ]
    
    for prompt in test_prompts:
        try:
            print(f"   Testing prompt: \"{prompt}\"")
            from core.llm_prompt import generate_effect_parameters
            params = generate_effect_parameters(prompt, audio_duration=duration)
            print(f"   ✓ Generated {len(params.get('effects', []))} effects")
        except Exception as e:
            print(f"   ✗ ERROR: {str(e)}")
            return False
    
    # Test full processing pipeline
    print("\n4. Testing full processing pipeline...")
    try:
        output_path = "/tmp/test_output.wav"
        prompt = "Add reverb that grows over time"
        
        output_audio, info = processor.process(
            test_audio_path,
            prompt,
            output_file=output_path,
            verbose=False,
        )
        
        print(f"   ✓ Processing successful")
        print(f"   Output shape: {output_audio.shape}")
        print(f"   Output saved to: {output_path}")
        print(f"   Processing info: {info}")
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    return True


def test_individual_modules():
    """Test individual modules."""
    print("\nTesting Individual Modules")
    print("=" * 50)
    
    # Test parameter parser
    print("\n1. Testing parameter parser...")
    try:
        from core.parameter_parser import ParameterParser
        parser = ParameterParser()
        
        test_llm_output = {
            "effects": [
                {
                    "type": "reverb",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "decay_time": {"start": 0.5, "end": 2.0},
                    "wet_dry": {"start": 0.1, "end": 0.5},
                    "width": {"start": 1.0, "end": 1.0},
                }
            ]
        }
        
        effects = parser.parse_parameters(test_llm_output, audio_duration=5.0)
        print(f"   ✓ Parser created {len(effects)} effect instances")
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")
        return False
    
    # Test effects
    print("\n2. Testing audio effects...")
    try:
        import numpy as np
        sr = 16000
        audio = np.random.randn(sr) * 0.1  # Random noise
        
        from effects.reverb import ReverbEffect
        reverb = ReverbEffect(sample_rate=sr)
        
        params = {
            "decay_time": 1.0,
            "wet_dry": 0.3,
            "width": 1.0,
        }
        
        output = reverb.process(audio, params)
        print(f"   ✓ Reverb effect processed successfully")
        print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")
        return False
    
    print("\n" + "=" * 50)
    print("Module tests passed! ✓")
    return True


if __name__ == "__main__":
    # Run tests
    success = True
    
    success = test_individual_modules() and success
    success = test_basic_pipeline() and success
    
    if success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
