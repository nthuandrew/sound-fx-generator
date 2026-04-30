"""
Global configuration for the Sound Effect Generator system.
"""

# LLM Configuration
LLM_MODEL = "gemini-2.5-flash"  # For latest Google Gemini API
LLM_MODEL_VISION = "gemini-2.5-pro"  # Multimodal model for spectrogram analysis
LLM_MODEL_VISION_FALLBACK = "gemini-2.5-flash"  # Fallback when pro quota is exhausted
LLM_TEMPERATURE = 0.5  # Lower temp for more consistent JSON output
LLM_MAX_TOKENS = 4096  # Increased for complete JSON responses
LLM_MAX_RETRIES = 3
LLM_JSON_PREVIEW_CHARS = 300
LLM_API_MAX_ATTEMPTS = 4
LLM_API_RETRY_BASE_SECONDS = 2.0
LLM_API_RETRY_MAX_SECONDS = 60.0

# Audio Configuration
DEFAULT_SR = 44100 # Default sample rate
AUDIO_CHUNK_SIZE = 4096

# Effect Intensity Controls
# Increase this value to make effects more aggressive globally.
EFFECT_INTENSITY_MULTIPLIER = 1.5

# Effects Configuration
SUPPORTED_EFFECTS = ["reverb", "chorus", "distortion", "low_pass_filter"]

# Parameter Constraints
PARAM_CONSTRAINTS = {
    "reverb": {
        "decay_time": (0.1, 5.0),
        "wet_dry": (0.0, 1.0),
        "width": (0.0, 1.0),
    },
    "chorus": {
        "rate": (0.1, 10.0),
        "depth": (0.0, 1.0),
        "wet_dry": (0.0, 1.0),
    },
    "distortion": {
        "gain": (0.0, 10.0),
        "tone": (0.0, 1.0),
    },
    "low_pass_filter": {
        "cutoff_freq": (20, 20000),
        "resonance": (0.0, 1.0),
    },
}

# Output Configuration
OUTPUT_QUALITY = "high"  # 'low', 'medium', 'high'
