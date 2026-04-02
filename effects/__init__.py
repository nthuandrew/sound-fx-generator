"""
Audio effects module.
Provides various time-varying audio effects for signal processing.
"""

from .base_effect import AudioEffect
from .reverb import ReverbEffect
from .chorus import ChorusEffect
from .distortion import DistortionEffect
from .low_pass_filter import LowPassFilterEffect

__all__ = [
    "AudioEffect",
    "ReverbEffect", 
    "ChorusEffect",
    "DistortionEffect",
    "LowPassFilterEffect",
]
