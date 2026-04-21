"""
Command-line interface for Sound Effect Generator

Usage:
    python main.py --audio input.wav --prompt "your effect description" --output output.wav
    python main.py --mode extract-and-clone --audio input.wav --reference-audio reference.wav --output output.wav
"""

import argparse
import sys
import os
from pathlib import Path
from core.audio_processor import AudioProcessor
from dotenv import load_dotenv

def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Language-driven Time-Variant Sound Effect Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --audio input.wav --prompt "add reverb over 10 seconds"
  %(prog)s --audio input.wav --prompt "gradually transition to distortion" --output out.wav
  %(prog)s --batch inputs.txt --output-dir ./outputs/
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "extract-and-clone"],
        default="generate",
        help="Processing mode: generate (text-driven) or extract-and-clone (reference effect cloning)"
    )

    parser.add_argument(
        "--audio",
        type=str,
        help="Input audio file path"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        help="Text description of desired effects"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output audio file path (default: input_processed.wav)"
    )

    parser.add_argument(
        "--reference-audio",
        type=str,
        default=None,
        help="Optional reference audio file used to extract spectrogram context"
    )
    
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)"
    )
    
    parser.add_argument(
        "--batch",
        type=str,
        help="Process batch: text file with one 'audio_file|prompt' per line"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./processed_audio/",
        help="Output directory for batch processing (default: ./processed_audio/)"
    )
    
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable output normalization"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Print detailed processing information"
    )
    
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output messages"
    )
    
    return parser


def process_single_file(args):
    """Process a single audio file."""
    if not args.audio:
        print("Error: --audio is required for single file processing")
        return False
    
    if args.mode == "generate" and not args.prompt:
        print("Error: --prompt is required when --mode generate")
        return False

    if args.mode == "extract-and-clone" and not args.reference_audio:
        print("Error: --reference-audio is required when --mode extract-and-clone")
        return False
    
    # Verify input file exists
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return False
    
    # Generate output filename if not provided
    if not args.output:
        base_path = Path(args.audio)
        args.output = str(base_path.parent / f"{base_path.stem}_processed.wav")
    
    # Create processor
    processor = AudioProcessor()
    
    # Process audio
    verbose = args.verbose and not args.quiet
    try:
        output_audio, info = processor.process(
            args.audio,
            args.prompt or "Reverse-engineer and clone reference effects",
            reference_audio_file=args.reference_audio,
            output_file=args.output,
            normalize=not args.no_normalize,
            verbose=verbose,
            mode=args.mode,
        )
        
        if verbose:
            print(f"\nSuccess! Output saved to: {args.output}")
            print(f"Processing info: {info}")
        
        return True
    
    except Exception as e:
        print(f"Error processing audio: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def process_batch(args):
    """Process batch of files from a text file."""
    if not args.batch:
        print("Error: --batch file is required for batch processing")
        return False
    
    if not os.path.exists(args.batch):
        print(f"Error: Batch file not found: {args.batch}")
        return False
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read batch file
    audio_files = []
    prompts = []
    
    try:
        with open(args.batch, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('|')
                if len(parts) != 2:
                    print(f"Warning: Invalid line format (expected 'audio|prompt'): {line}")
                    continue
                
                audio_file, prompt = parts[0].strip(), parts[1].strip()
                
                if not os.path.exists(audio_file):
                    print(f"Warning: Audio file not found: {audio_file}")
                    continue
                
                audio_files.append(audio_file)
                prompts.append(prompt)
    
    except Exception as e:
        print(f"Error reading batch file: {str(e)}")
        return False
    
    if not audio_files:
        print("Error: No valid audio files in batch file")
        return False
    
    print(f"Processing {len(audio_files)} audio files...")
    
    # Create processor
    processor = AudioProcessor(sample_rate=args.sr)
    verbose = args.verbose and not args.quiet
    
    # Process batch
    results = processor.process_batch(
        audio_files,
        prompts,
        args.output_dir,
    )
    
    # Print summary
    successful = sum(1 for r in results if "error" not in r)
    failed = len(results) - successful
    
    print(f"\nBatch processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {args.output_dir}")
    
    return failed == 0


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    load_dotenv()
    # Check environment variable for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY environment variable not set.")
        print("Please set it before running: export GEMINI_API_KEY='your-key'")
        print()
    
    # Determine mode (single vs batch)
    if args.batch:
        success = process_batch(args)
    else:
        success = process_single_file(args)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
