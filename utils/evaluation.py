"""
Evaluation Metrics

Functions to evaluate the quality of processed audio against expected transformations.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.stats import pearsonr


def compute_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Mean Squared Error between two signals.
    
    Args:
        prediction: Predicted parameter curve
        target: Ground truth parameter curve
        
    Returns:
        MSE value
    """
    if len(prediction) != len(target):
        raise ValueError("Prediction and target must have same length")
    
    mse = np.mean((prediction - target) ** 2)
    return float(mse)


def compute_pearson_correlation(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Pearson correlation coefficient between two signals.
    
    Args:
        prediction: Predicted parameter curve
        target: Ground truth parameter curve
        
    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    if len(prediction) != len(target):
        raise ValueError("Prediction and target must have same length")
    
    correlation, p_value = pearsonr(prediction, target)
    return float(correlation), float(p_value)


def compute_clapscore(
    text_embedding: np.ndarray,
    audio_embedding: np.ndarray,
) -> float:
    """
    Compute CLAPscore as cosine similarity between text and audio embeddings.
    
    This is inspired by the CLAPscore metric used in audio-text alignment evaluation.
    Higher scores indicate better alignment between the text prompt and processed audio.
    
    Args:
        text_embedding: Text embedding vector (1D numpy array)
        audio_embedding: Audio embedding vector (1D numpy array)
        
    Returns:
        Cosine similarity score (0.0 to 1.0 range after softmax)
    """
    if len(text_embedding) != len(audio_embedding):
        raise ValueError("Text and audio embeddings must have same dimension")
    
    # Normalize embeddings
    text_norm = text_embedding / (np.linalg.norm(text_embedding) + 1e-8)
    audio_norm = audio_embedding / (np.linalg.norm(audio_embedding) + 1e-8)
    
    # Compute cosine similarity
    cosine_sim = float(np.dot(text_norm, audio_norm))
    
    # Convert to 0-1 range (cosine similarity is -1 to 1)
    clapscore = (cosine_sim + 1.0) / 2.0
    
    return clapscore


def compute_windowed_clapscore(
    text_embedding: np.ndarray,
    audio_embeddings: np.ndarray,
    window_size: int = 100,
    hop_size: Optional[int] = None,
) -> float:
    """
    Compute average CLAPscore across windowed audio segments.
    
    Useful for evaluating temporal alignment between text and audio.
    
    Args:
        text_embedding: Text embedding vector
        audio_embeddings: Array of audio embeddings (num_windows, embedding_dim)
        window_size: Size of audio window in frames
        hop_size: Hop size between windows (default = window_size / 2)
        
    Returns:
        Average CLAPscore across all windows
    """
    if hop_size is None:
        hop_size = window_size // 2
    
    if audio_embeddings.ndim == 1:
        # Single embedding
        return compute_clapscore(text_embedding, audio_embeddings)
    
    scores = []
    for i in range(audio_embeddings.shape[0]):
        score = compute_clapscore(text_embedding, audio_embeddings[i])
        scores.append(score)
    
    return float(np.mean(scores))


def compute_spectral_distance(
    spectrum1: np.ndarray,
    spectrum2: np.ndarray,
) -> float:
    """
    Compute spectral distance between two spectrograms.
    
    Useful for evaluating the similarity of the output audio to expected characteristics.
    
    Args:
        spectrum1: First spectrum (1D or 2D array)
        spectrum2: Second spectrum (same shape as spectrum1)
        
    Returns:
        Euclidean distance in spectral domain
    """
    if spectrum1.shape != spectrum2.shape:
        raise ValueError("Spectra must have the same shape")
    
    # Normalize to 0-1 range
    spec1_norm = (spectrum1 - spectrum1.min()) / (spectrum1.max() - spectrum1.min() + 1e-8)
    spec2_norm = (spectrum2 - spectrum2.min()) / (spectrum2.max() - spectrum2.min() + 1e-8)
    
    # Compute Euclidean distance
    distance = np.sqrt(np.mean((spec1_norm - spec2_norm) ** 2))
    
    return float(distance)
