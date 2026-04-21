"""
Audio Processor

Main audio processing engine that orchestrates LLM parameter generation,
parameter parsing, and effect application.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from core.llm_prompt import generate_effect_parameters
from core.parameter_parser import ParameterParser, EffectInstance
from effects import (
    ReverbEffect,
    ChorusEffect,
    DistortionEffect,
    LowPassFilterEffect,
)
from utils.audio_io import load_audio, save_audio, normalize_audio
from utils.evaluation import compute_spectral_distance
from utils.reference_audio import analyze_reference_audio
import config


class AudioProcessor:
    """
    Main audio processor that combines LLM, parameter parsing, and effects.
    """
    
    # Effect class mapping
    EFFECT_CLASSES = {
        "reverb": ReverbEffect,
        "chorus": ChorusEffect,
        "distortion": DistortionEffect,
        "low_pass_filter": LowPassFilterEffect,
    }
    
    def __init__(self, sample_rate: Optional[int] = None):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate or config.DEFAULT_SR
        self.parser = ParameterParser()
        self.effects = {}
        self._initialize_effects()
    
    def _initialize_effects(self) -> None:
        """Initialize all available effects."""
        for effect_name, effect_class in self.EFFECT_CLASSES.items():
            try:
                self.effects[effect_name] = effect_class(self.sample_rate)
            except Exception as e:
                print(f"Warning: Failed to initialize {effect_name}: {str(e)}")
    
    def process(
        self,
        audio_file: str,
        text_prompt: str,
        reference_audio_file: Optional[str] = None,
        output_file: Optional[str] = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Complete processing pipeline: load audio, generate parameters, apply effects.
        
        Args:
            audio_file: Path to input audio file
            text_prompt: Text description of desired effects
            reference_audio_file: Optional path to a reference audio file
            output_file: Optional path to save output audio
            normalize: Whether to normalize output audio
            verbose: Whether to print processing steps
            
        Returns:
            Tuple of (output_audio, processing_info_dict)
        """
        processing_info = {}
        
        # Step 1: Load audio
        if verbose:
            print(f"Loading audio from {audio_file}...")
        try:
            audio, sr = load_audio(audio_file, sr=self.sample_rate, mono=True)
            processing_info["audio_duration"] = len(audio) / sr
            processing_info["sample_rate"] = sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {str(e)}")

        reference_context = None
        if reference_audio_file:
            if verbose:
                print(f"Analyzing reference audio from {reference_audio_file}...")
            try:
                reference_context = analyze_reference_audio(reference_audio_file, sr=self.sample_rate)
                processing_info["reference_audio_file"] = reference_audio_file
                processing_info["reference_context"] = reference_context.to_dict()
            except Exception as e:
                raise RuntimeError(f"Failed to analyze reference audio: {str(e)}")
        
        # Step 2: Generate effect parameters from text prompt
        if verbose:
            print(f"Generating effect parameters from prompt...")
            print(f"  Prompt: \"{text_prompt}\"")
            if reference_audio_file:
                print(f"  Reference audio: {reference_audio_file}")
        try:
            llm_output = generate_effect_parameters(
                text_prompt,
                audio_duration=len(audio) / sr,
                reference_context=reference_context.to_dict() if reference_context else None,
            )
            processing_info["llm_output"] = llm_output
        except Exception as e:
            raise RuntimeError(f"Failed to generate parameters: {str(e)}")
        
        # Step 3: Parse parameters
        if verbose:
            print(f"Parsing and validating parameters...")
        try:
            effect_instances = self.parser.parse_parameters(
                llm_output,
                audio_duration=len(audio) / sr,
            )
            processing_info["num_effects"] = len(effect_instances)
            if verbose:
                for i, effect in enumerate(effect_instances):
                    print(f"  Effect {i+1}: {effect.effect_type} [{effect.start_time:.1f}s - {effect.end_time:.1f}s]")
        except Exception as e:
            raise RuntimeError(f"Failed to parse parameters: {str(e)}")
        
        # Step 4: Create parameter envelopes
        if verbose:
            print(f"Creating parameter envelopes...")
        envelope = self.parser.create_parameter_envelope(
            effect_instances,
            len(audio) / sr,
            sr,
        )
        processing_info["envelope_created"] = True
        
        # Step 5: Apply effects
        if verbose:
            print(f"Applying effects...")
        output_audio = self._apply_effects_with_envelope(audio, sr, envelope)
        processing_info["effects_applied"] = True
        
        # Step 6: Normalize if requested
        if normalize:
            if verbose:
                print(f"Normalizing output audio...")
            output_audio = normalize_audio(output_audio, target_level=-3.0)
        
        # Step 7: Save output if requested
        if output_file:
            if verbose:
                print(f"Saving output to {output_file}...")
            save_audio(output_audio, output_file, sr=sr)
            processing_info["output_file"] = output_file
        
        # Step 8: Compute evaluation metrics
        if verbose:
            print(f"\nEvaluation Metrics:")
        evaluation_report = self._compute_evaluation_metrics(audio, output_audio, effect_instances, sr)
        processing_info["evaluation"] = evaluation_report
        
        if verbose:
            self._print_evaluation_report(evaluation_report, effect_instances)
        
        if verbose:
            print("Processing complete!")
        
        return output_audio, processing_info
    
    def _apply_effects_with_envelope(
        self,
        audio: np.ndarray,
        sample_rate: int,
        envelope: Dict,
    ) -> np.ndarray:
        """
        Apply effects using parameter envelope.
        
        Args:
            audio: Input audio
            sample_rate: Sample rate
            envelope: Parameter envelope from create_parameter_envelope
            
        Returns:
            Processed audio
        """
        output = audio.copy()
        chunk_size = config.AUDIO_CHUNK_SIZE  # Process in chunks for memory efficiency
        
        for start_idx in range(0, len(audio), chunk_size):
            end_idx = min(start_idx + chunk_size, len(audio))
            chunk = output[start_idx:end_idx]
            
            # Apply effects in order
            for effect_name, effect_instance in self.effects.items():
                if effect_name not in self.EFFECT_CLASSES:
                    continue
                
                # Extract parameters for this effect from envelope
                effect_params_list = [
                    envelope.get(start_idx + i, {}).get(effect_name, {})
                    for i in range(len(chunk))
                ]
                
                # Skip if no effects for this chunk
                if not any(effect_params_list):
                    continue
                
                # Aggregate chunk-level parameters from active samples.
                # This keeps computation efficient while still tracking time variation.
                active_params = [p for p in effect_params_list if p]
                params = self._average_parameters(active_params)
                try:
                    chunk = effect_instance.process(chunk, params)
                except Exception as e:
                    print(f"Warning: Failed to apply {effect_name}: {str(e)}")
            
            output[start_idx:end_idx] = chunk
        
        return output

    @staticmethod
    def _average_parameters(param_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Average a list of parameter dictionaries.

        Args:
            param_list: List of parameter dictionaries

        Returns:
            Dictionary with averaged parameter values
        """
        if not param_list:
            return {}

        keys = set().union(*(params.keys() for params in param_list))
        averaged = {}
        for key in keys:
            values = [params[key] for params in param_list if key in params]
            if values:
                averaged[key] = float(np.mean(values))
        return averaged
    
    def _compute_evaluation_metrics(
        self,
        input_audio: np.ndarray,
        output_audio: np.ndarray,
        effect_instances: List[EffectInstance],
        sample_rate: int,
    ) -> Dict:
        """
        Compute comprehensive evaluation metrics for audio processing.
        
        Args:
            input_audio: Original audio
            output_audio: Processed audio
            effect_instances: List of applied effects
            sample_rate: Sample rate
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Quantitative Metrics
        # 1. Energy/RMS analysis
        input_rms = float(np.sqrt(np.mean(input_audio ** 2)))
        output_rms = float(np.sqrt(np.mean(output_audio ** 2)))
        metrics["input_rms_db"] = 20 * np.log10(input_rms + 1e-8)
        metrics["output_rms_db"] = 20 * np.log10(output_rms + 1e-8)
        metrics["rms_change_db"] = metrics["output_rms_db"] - metrics["input_rms_db"]
        
        # 2. Dynamic range
        input_dynamic_range = 20 * np.log10(np.max(np.abs(input_audio)) / (np.sqrt(np.mean(input_audio ** 2)) + 1e-8))
        output_dynamic_range = 20 * np.log10(np.max(np.abs(output_audio)) / (np.sqrt(np.mean(output_audio ** 2)) + 1e-8))
        metrics["input_dynamic_range_db"] = float(input_dynamic_range)
        metrics["output_dynamic_range_db"] = float(output_dynamic_range)
        
        # 3. Peak values
        metrics["input_peak"] = float(np.max(np.abs(input_audio)))
        metrics["output_peak"] = float(np.max(np.abs(output_audio)))
        
        # 4. Spectral changes (simple frequency distribution estimation)
        try:
            input_fft = np.abs(np.fft.rfft(input_audio))
            output_fft = np.abs(np.fft.rfft(output_audio))
            
            # Ensure same length for spectral distance
            min_len = min(len(input_fft), len(output_fft))
            spectral_distance = compute_spectral_distance(
                input_fft[:min_len],
                output_fft[:min_len]
            )
            metrics["spectral_change"] = float(spectral_distance)
        except Exception as e:
            metrics["spectral_change_error"] = str(e)
        
        # 5. Effect parameter statistics
        effect_summary = {}
        for effect in effect_instances:
            effect_name = effect.effect_type
            if effect_name not in effect_summary:
                effect_summary[effect_name] = {
                    "count": 0,
                    "time_ranges": [],
                    "param_ranges": {}
                }
            
            effect_summary[effect_name]["count"] += 1
            effect_summary[effect_name]["time_ranges"].append({
                "start": effect.start_time,
                "end": effect.end_time,
                "duration": effect.end_time - effect.start_time
            })
            
            # Collect parameter value ranges
            for param_name, time_varying_param in effect.parameters.items():
                if param_name not in effect_summary[effect_name]["param_ranges"]:
                    effect_summary[effect_name]["param_ranges"][param_name] = {
                        "min": time_varying_param.start_value,
                        "max": time_varying_param.start_value
                    }
                
                effect_summary[effect_name]["param_ranges"][param_name]["min"] = min(
                    effect_summary[effect_name]["param_ranges"][param_name]["min"],
                    time_varying_param.start_value,
                    time_varying_param.end_value
                )
                effect_summary[effect_name]["param_ranges"][param_name]["max"] = max(
                    effect_summary[effect_name]["param_ranges"][param_name]["max"],
                    time_varying_param.start_value,
                    time_varying_param.end_value
                )
        
        metrics["effect_summary"] = effect_summary
        
        # Qualitative observations
        metrics["observations"] = {
            "volume_increased": metrics["rms_change_db"] > 2.0,
            "volume_decreased": metrics["rms_change_db"] < -2.0,
            "dynamic_range_compressed": metrics["output_dynamic_range_db"] < metrics["input_dynamic_range_db"] - 1.0,
            "spectral_changed": metrics.get("spectral_change", 0) > 0.1,
            "no_clipping": metrics["output_peak"] < 1.0,
        }
        
        return metrics
    
    @staticmethod
    def _print_evaluation_report(metrics: Dict, effect_instances: List[EffectInstance]) -> None:
        """
        Print formatted evaluation report to console.
        
        Args:
            metrics: Evaluation metrics dictionary
            effect_instances: List of applied effects
        """
        print("=" * 60)
        print("QUANTITATIVE EVALUATION")
        print("=" * 60)
        
        # Energy metrics
        print("\nENERGY ANALYSIS:")
        print(f"  Input RMS:        {metrics['input_rms_db']:>8.2f} dB")
        print(f"  Output RMS:       {metrics['output_rms_db']:>8.2f} dB")
        print(f"  RMS Change:       {metrics['rms_change_db']:>8.2f} dB")
        
        # Dynamic range
        print("\nDYNAMIC RANGE:")
        print(f"  Input:            {metrics['input_dynamic_range_db']:>8.2f} dB")
        print(f"  Output:           {metrics['output_dynamic_range_db']:>8.2f} dB")
        
        # Peak levels
        print("\nPEAK LEVELS:")
        print(f"  Input Peak:       {metrics['input_peak']:>8.4f}")
        print(f"  Output Peak:      {metrics['output_peak']:>8.4f}")
        
        # Spectral changes
        if "spectral_change" in metrics:
            print("\nSPECTRAL CHANGES:")
            print(f"  Spectral Distance: {metrics['spectral_change']:>8.4f}")
        
        # Effect parameters
        print("\nEFFECT PARAMETERS APPLIED:")
        for effect in effect_instances:
            print(f"\n  [{effect.effect_type.upper()}] @ {effect.start_time:.1f}s - {effect.end_time:.1f}s")
            for param_name, time_param in effect.parameters.items():
                if time_param.start_value == time_param.end_value:
                    print(f"    • {param_name}: {time_param.start_value:.3f}")
                else:
                    print(f"    • {param_name}: {time_param.start_value:.3f} → {time_param.end_value:.3f}")
        
        # Qualitative observations
        print("\n" + "=" * 60)
        print("QUALITATIVE OBSERVATIONS")
        print("=" * 60)
        obs = metrics["observations"]
        
        observations_list = []
        if obs["volume_increased"]:
            observations_list.append("✓ Volume increased significantly")
        if obs["volume_decreased"]:
            observations_list.append("✓ Volume decreased significantly")
        if obs["dynamic_range_compressed"]:
            observations_list.append("✓ Dynamic range compressed (effects applied)")
        if obs["spectral_changed"]:
            observations_list.append("✓ Spectral content significantly modified")
        if obs["no_clipping"]:
            observations_list.append("✓ No digital clipping detected")
        else:
            observations_list.append("⚠ Warning: Some clipping detected")
        
        if observations_list:
            for obs_str in observations_list:
                print(f"  {obs_str}")
        
        print("\n" + "=" * 60)
    
    def process_batch(
        self,
        audio_files: List[str],
        prompts: List[str],
        output_dir: str,
    ) -> List[Dict]:
        """
        Process multiple audio files with different prompts.
        
        Args:
            audio_files: List of input audio file paths
            prompts: List of text prompts (one per audio file)
            output_dir: Directory to save output files
            
        Returns:
            List of processing info dictionaries
        """
        results = []
        
        for audio_file, prompt in zip(audio_files, prompts):
            print(f"\nProcessing {audio_file}...")
            
            # Generate output filename
            import os
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_processed.wav")
            
            try:
                _, info = self.process(
                    audio_file,
                    prompt,
                    output_file=output_file,
                    verbose=True,
                )
                results.append(info)
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                results.append({"error": str(e)})
        
        return results
