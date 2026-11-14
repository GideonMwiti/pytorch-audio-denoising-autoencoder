#!/usr/bin/env python3
"""
Interactive Audio Denoising CLI
===============================
Restores degraded audio files (.wav, .mp3) by removing acoustic noise
using a trained DenoiseUNet Autoencoder and Griffin-Lim inversion.

Usage:
    python denoise_audio.py --audio outputs/sample_noisy.wav --output outputs/my_cleaned.wav
"""

import os
import sys
import argparse
import numpy as np
import soundfile as sf
import torch
import torchaudio

from main import (
    DenoiseAutoencoder,
    MelSpectrogramPipeline,
    SAMPLE_RATE,
    DURATION,
    TIME_STEPS
)


def load_audio(audio_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Loads audio from disk and ensures mono at target sample rate."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    data, sr = sf.read(audio_path, dtype='float32')
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        tensor_data = torch.from_numpy(data).unsqueeze(0)
        data = resampler(tensor_data).squeeze(0).numpy()
    return data


def denoise_audio_file(audio_path: str, output_path: str, model_path: str, device_str: str = 'cpu'):
    device = torch.device(device_str)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}. Run main.py first!")

    ckpt = torch.load(model_path, map_location=device)
    model = DenoiseAutoencoder().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    waveform = load_audio(audio_path, target_sr=SAMPLE_RATE)
    duration_sec = len(waveform) / SAMPLE_RATE

    pipeline = MelSpectrogramPipeline(sample_rate=SAMPLE_RATE)
    chunk_samples = int(SAMPLE_RATE * DURATION)
    hop_samples = int(chunk_samples * 0.5)

    # Segment slicing
    chunks = []
    if len(waveform) <= chunk_samples:
        padded = np.pad(waveform, (0, max(0, chunk_samples - len(waveform))), mode='constant')
        chunks.append(padded)
    else:
        for start in range(0, len(waveform) - chunk_samples + 1, hop_samples):
            chunks.append(waveform[start:start + chunk_samples])
        if len(waveform) % hop_samples != 0 and len(chunks) > 0:
            chunks.append(waveform[-chunk_samples:])

    print("\n" + "=" * 60)
    print("      DENOISEUNET AUDIO DENOISING & RESTORATION")
    print("=" * 60)
    print(f"Input File       : {audio_path}")
    print(f"Duration         : {duration_sec:.2f} seconds ({len(waveform):,} samples)")
    print(f"Processed Chunks : {len(chunks)} overlapping windows ({DURATION:.2f}s each)")
    print(f"Model Checkpoint : {model_path}")
    print("-" * 60)

    # Process chunks through Autoencoder
    denoised_chunks = []
    with torch.no_grad():
        for ch in chunks:
            ch_t = torch.from_numpy(ch)
            noisy_mel = pipeline.audio_to_mel(ch_t).unsqueeze(0).to(device)
            clean_mel = model(noisy_mel).squeeze(0).cpu()
            recon_audio = pipeline.mel_to_audio(clean_mel).numpy()
            denoised_chunks.append(recon_audio)

    # Overlap-add reconstruction
    if len(denoised_chunks) == 1:
        restored = denoised_chunks[0][:len(waveform)]
    else:
        out_len = (len(chunks) - 1) * hop_samples + chunk_samples
        out_buf = np.zeros(out_len, dtype=np.float32)
        weight_buf = np.zeros(out_len, dtype=np.float32)
        window = np.hanning(chunk_samples)

        for i, ch_out in enumerate(denoised_chunks):
            start = i * hop_samples
            out_buf[start:start + chunk_samples] += ch_out * window
            weight_buf[start:start + chunk_samples] += window

        valid_mask = weight_buf > 1e-4
        out_buf[valid_mask] /= weight_buf[valid_mask]
        restored = out_buf[:len(waveform)]

    # Peak normalization
    max_val = np.max(np.abs(restored)) + 1e-6
    restored = (restored / max_val * 0.90).astype(np.float32)

    # Estimate noise reduction
    noise_reduction_db = 10.0 * np.log10(np.mean(waveform ** 2) / (np.mean(restored ** 2) + 1e-9) + 1e-6)

    sf.write(output_path, restored, SAMPLE_RATE)

    print(f"[✓] Denoised audio saved to: {output_path}")
    print(f"[✓] Estimated Noise Attenuation: {max(0.0, noise_reduction_db):.2f} dB")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Denoise audio file using DenoiseUNet Autoencoder")
    parser.add_argument('--audio', type=str, required=True, help='Path to noisy audio file (.wav, .mp3)')
    parser.add_argument('--output', type=str, default='outputs/denoised_audio.wav', help='Path to output cleaned audio')
    parser.add_argument('--model', type=str, default='outputs/denoise_autoencoder_best.pth', help='Path to trained model')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()

    denoise_audio_file(args.audio, args.output, args.model, args.device)


if __name__ == '__main__':
    main()

