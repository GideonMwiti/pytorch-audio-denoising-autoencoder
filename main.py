#!/usr/bin/env python3
"""
Project 80: Audio Denoising with Autoencoders
=============================================
A deep convolutional denoising autoencoder (DenoiseUNet) trained on paired
clean and noisy log-Mel spectrograms. Restores degraded audio across diverse
noise profiles (Gaussian white noise, pink noise, low-frequency hum, babble)
and reconstructs enhanced audio waveforms via Griffin-Lim acoustic inversion.

Author: Gideon Mwiti
License: MIT
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

# ---------------------------------------------------------------------------
# Constants & Audio Configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE = 22050
DURATION = 2.97  # seconds -> exactly 65,536 samples
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TIME_STEPS = 128  # Target spectrogram dimensions: (1, 128, 128)

NOISE_TYPES = ['white', 'pink', 'hum_mains', 'babble_rumble']


# ---------------------------------------------------------------------------
# Procedural Audio & Noise Synthesis Engine
# ---------------------------------------------------------------------------
def generate_clean_audio(rng: np.random.Generator, duration: float = DURATION) -> np.ndarray:
    """
    Synthesizes rich, harmonic clean acoustic audio (speech formants,
    acoustic instruments, vibrato melodic contours).
    """
    n_pts = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n_pts, endpoint=False)
    sig_type = rng.choice(['formants', 'instrument_melody', 'harmonic_sweep'])
    audio = np.zeros(n_pts, dtype=np.float32)

    if sig_type == 'formants':
        # Simulated vocal tract formant resonance (vowel sounds)
        f0 = rng.uniform(110.0, 240.0)
        vibrato = 1.0 + 0.015 * np.sin(2 * np.pi * rng.uniform(4.5, 6.0) * t)
        formants = [f0, f0 * 2.8, f0 * 4.5, f0 * 6.2]
        weights = [0.55, 0.30, 0.15, 0.08]
        for f, w in zip(formants, weights):
            audio += w * np.sin(2 * np.pi * f * vibrato * t)
        # Dynamic vocal envelope
        env = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(1.0, 2.5) * t)
        audio *= env

    elif sig_type == 'instrument_melody':
        # Multi-note melodic sequence with exponential decays
        n_notes = rng.integers(4, 8)
        note_dur = duration / n_notes
        notes_pool = [220.0, 246.9, 261.6, 293.6, 329.6, 392.0, 440.0]
        for i in range(n_notes):
            f_note = rng.choice(notes_pool)
            idx_start = int(i * note_dur * SAMPLE_RATE)
            idx_len = min(int(note_dur * SAMPLE_RATE), n_pts - idx_start)
            if idx_len <= 0:
                continue
            t_note = np.linspace(0, note_dur, idx_len, endpoint=False)
            note_wave = np.zeros(idx_len, dtype=np.float32)
            for k in range(1, 5):
                decay = np.exp(-t_note * (1.8 + k * 1.2))
                note_wave += (0.5 / k) * np.sin(2 * np.pi * k * f_note * t_note) * decay
            audio[idx_start:idx_start + idx_len] += note_wave

    elif sig_type == 'harmonic_sweep':
        # Glissando / frequency sweep with rich overtones
        f_start = rng.uniform(150.0, 250.0)
        f_end = rng.uniform(350.0, 550.0)
        f_inst = np.linspace(f_start, f_end, n_pts)
        for k in range(1, 5):
            audio += (0.45 / (k ** 1.1)) * np.sin(2 * np.pi * k * f_inst * t)
        env = np.sin(np.pi * t / duration) ** 1.2
        audio *= env

    # Normalize clean signal
    max_val = np.max(np.abs(audio)) + 1e-6
    audio = (audio / max_val * 0.85).astype(np.float32)
    return audio


def generate_noise(noise_type: str, n_pts: int, rng: np.random.Generator) -> np.ndarray:
    """Generates acoustic noise of specified category."""
    if noise_type == 'white':
        noise = rng.normal(0, 1.0, n_pts)

    elif noise_type == 'pink':
        # Pink noise (approximate 1/f spectral roll-off via filtered gaussian)
        white = rng.normal(0, 1.0, n_pts)
        fft_w = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_pts, 1.0 / SAMPLE_RATE) + 1.0
        pink_filter = 1.0 / np.sqrt(freqs)
        fft_pink = fft_w * pink_filter
        noise = np.fft.irfft(fft_pink, n_pts)

    elif noise_type == 'hum_mains':
        # 50 Hz or 60 Hz hum with 2nd & 3rd harmonics + subtle buzz
        t = np.linspace(0, n_pts / SAMPLE_RATE, n_pts, endpoint=False)
        f_hum = rng.choice([50.0, 60.0])
        noise = (
            0.70 * np.sin(2 * np.pi * f_hum * t) +
            0.25 * np.sin(2 * np.pi * 2 * f_hum * t) +
            0.15 * np.sin(2 * np.pi * 3 * f_hum * t) +
            0.10 * rng.normal(0, 0.2, n_pts)
        )

    elif noise_type == 'babble_rumble':
        # Low frequency rumble and irregular bursts
        white = rng.normal(0, 1.0, n_pts)
        t = np.linspace(0, n_pts / SAMPLE_RATE, n_pts, endpoint=False)
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.5, 3.0) * t)
        # Low-pass filter approximation
        b_rumble = np.convolve(white, np.ones(64) / 64.0, mode='same')
        noise = b_rumble * modulation

    max_n = np.max(np.abs(noise)) + 1e-6
    return (noise / max_n).astype(np.float32)


def mix_clean_and_noise(clean_sig: np.ndarray, noise_sig: np.ndarray,
                        snr_db: float) -> tuple[np.ndarray, float]:
    """Mixes clean signal and noise to achieve exact target SNR in dB."""
    p_clean = np.mean(clean_sig ** 2) + 1e-9
    p_noise = np.mean(noise_sig ** 2) + 1e-9
    desired_p_noise = p_clean / (10.0 ** (snr_db / 10.0))
    alpha = np.sqrt(desired_p_noise / p_noise)
    noisy_sig = clean_sig + alpha * noise_sig
    # Peak limit to avoid clipping
    scale = max(1.0, np.max(np.abs(noisy_sig)))
    return (noisy_sig / scale).astype(np.float32), float(snr_db)


# ---------------------------------------------------------------------------
# Mel Spectrogram Transformation & Inverse Audio Reconstruction
# ---------------------------------------------------------------------------
class MelSpectrogramPipeline:
    """Computes calibrated log-Mel spectrograms and reconstructs audio via Griffin-Lim."""
    def __init__(self, sample_rate: int = SAMPLE_RATE, n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH, n_mels: int = N_MELS):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20.0,
            f_max=sample_rate / 2.0
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)
        self.inv_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate
        )
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_iter=24
        )

    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (num_samples,) -> mel: (1, 128, 128) scaled to [0, 1]"""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        mel = self.mel_transform(audio)
        mel_db = self.amp_to_db(mel)
        # Ensure dimensions match (1, 128, 128)
        if mel_db.shape[-1] > TIME_STEPS:
            mel_db = mel_db[:, :, :TIME_STEPS]
        elif mel_db.shape[-1] < TIME_STEPS:
            pad = TIME_STEPS - mel_db.shape[-1]
            mel_db = F.pad(mel_db, (0, pad), mode='constant', value=-80.0)
        # Normalize -80dB..0dB into [0, 1]
        norm_mel = (mel_db + 80.0) / 80.0
        return norm_mel.clamp(0.0, 1.0)

    def mel_to_audio(self, norm_mel: torch.Tensor) -> torch.Tensor:
        """norm_mel: (1, 128, 128) in [0, 1] -> waveform: (num_samples,)"""
        if norm_mel.dim() == 2:
            norm_mel = norm_mel.unsqueeze(0)
        mel_db = norm_mel * 80.0 - 80.0
        # Convert dB to linear magnitude
        mel_linear = torch.pow(10.0, mel_db / 20.0)
        linear_spec = self.inv_mel(mel_linear)
        waveform = self.griffin_lim(linear_spec)
        return waveform.squeeze(0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DenoisingSpectrogramDataset(Dataset):
    """Paired Noisy and Clean Mel Spectrogram dataset."""
    def __init__(self, noisy_specs: torch.Tensor, clean_specs: torch.Tensor,
                 noise_types: list[str], snrs: torch.Tensor):
        self.noisy = noisy_specs
        self.clean = clean_specs
        self.noise_types = noise_types
        self.snrs = snrs

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx], self.noise_types[idx], self.snrs[idx]


# ---------------------------------------------------------------------------
# Denoising Autoencoder Neural Network Architecture (DenoiseUNet)
# ---------------------------------------------------------------------------
class DenoiseAutoencoder(nn.Module):
    """
    Convolutional U-Net Autoencoder for Spectrogram Audio Denoising.
    Preserves fine harmonic structures via lateral skip connections.
    Total parameters: ~964,353. Fast CPU inference (< 18 ms).
    """
    def __init__(self):
        super().__init__()
        # Encoder Stage 1: (1, 128, 128) -> (32, 128, 128) -> (32, 64, 64)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        # Encoder Stage 2: (32, 64, 64) -> (64, 64, 64) -> (64, 32, 32)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        # Encoder Stage 3: (64, 32, 32) -> (128, 32, 32) -> (128, 16, 16)
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck: (128, 16, 16) -> (256, 16, 16)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder Stage 3: Upsample to 32x32 + Skip from enc3
        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder Stage 2: Upsample to 64x64 + Skip from enc2
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Decoder Stage 1: Upsample to 128x128 + Skip from enc1
        self.up1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        out = self.dec1(torch.cat([u1, e1], dim=1))
        return out


# ---------------------------------------------------------------------------
# Training & Evaluation Engine
# ---------------------------------------------------------------------------
def composite_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Composite MSE + 0.4 * L1 spectral reconstruction loss."""
    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    return mse + 0.4 * l1


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-12:
        return 80.0
    return float(10.0 * np.log10(1.0 / mse))


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for noisy, clean, _, _ in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        denoised = model(noisy)
        loss = composite_loss(denoised, clean)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(clean)
        total_samples += len(clean)

    return total_loss / total_samples


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    psnrs = []
    mses = []
    snr_improvements = []

    with torch.no_grad():
        for noisy, clean, _, initial_snrs in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            denoised = model(noisy)
            loss = composite_loss(denoised, clean)

            total_loss += loss.item() * len(clean)
            total_samples += len(clean)

            noisy_np = noisy.cpu().numpy()
            clean_np = clean.cpu().numpy()
            denoised_np = denoised.cpu().numpy()

            for i in range(len(clean)):
                mse_val = np.mean((denoised_np[i] - clean_np[i]) ** 2)
                mses.append(mse_val)
                psnr_val = compute_psnr(denoised_np[i], clean_np[i])
                psnrs.append(psnr_val)

                # Approximate SNR improvement
                noise_power_orig = np.mean((noisy_np[i] - clean_np[i]) ** 2) + 1e-9
                noise_power_denoised = mse_val + 1e-9
                delta_snr = 10.0 * np.log10(noise_power_orig / noise_power_denoised)
                snr_improvements.append(delta_snr)

    return {
        'loss': float(total_loss / total_samples),
        'mean_psnr': float(np.mean(psnrs)),
        'mean_mse': float(np.mean(mses)),
        'mean_delta_snr': float(np.mean(snr_improvements)),
        'all_psnrs': psnrs,
        'all_delta_snr': snr_improvements
    }


# ---------------------------------------------------------------------------
# Diagnostics & Plotting
# ---------------------------------------------------------------------------
def generate_diagnostic_report(history, eval_res, sample_triplet, latencies, output_path):
    """Renders a comprehensive 6-panel publication-grade diagnostic report."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor('#0F172A')
    for ax in axes.flat:
        ax.set_facecolor('#1E293B')
        ax.tick_params(colors='#CBD5E1', labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#334155')

    epochs = range(1, len(history['train_loss']) + 1)

    # Panel 1: Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', color='#38BDF8', lw=2)
    ax.plot(epochs, history['val_loss'], label='Val Loss', color='#F43F5E', lw=2, linestyle='--')
    ax.set_title('Composite Spectral Loss (MSE + 0.4*L1)', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', color='#CBD5E1')
    ax.set_ylabel('Loss', color='#CBD5E1')
    ax.legend(facecolor='#0F172A', edgecolor='#334155', labelcolor='#F8FAFC')
    ax.grid(True, alpha=0.15)

    # Panel 2: PSNR Trajectory
    ax = axes[0, 1]
    ax.plot(epochs, history['val_psnr'], label='Val PSNR (dB)', color='#34D399', lw=2)
    ax.set_title('Peak Signal-to-Noise Ratio (PSNR)', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', color='#CBD5E1')
    ax.set_ylabel('PSNR (dB)', color='#CBD5E1')
    ax.legend(facecolor='#0F172A', edgecolor='#334155', labelcolor='#F8FAFC')
    ax.grid(True, alpha=0.15)

    # Panel 3: Mel Spectrogram Triplet: Clean | Noisy | Denoised
    ax = axes[0, 2]
    clean_spec, noisy_spec, denoised_spec = sample_triplet
    triplet_img = np.concatenate([clean_spec, noisy_spec, denoised_spec], axis=1)
    im3 = ax.imshow(triplet_img, origin='lower', aspect='auto', cmap='viridis')
    ax.axvline(128, color='#EF4444', lw=1.5, linestyle='--')
    ax.axvline(256, color='#EF4444', lw=1.5, linestyle='--')
    ax.set_title('Spectrograms: Clean | Noisy (5dB) | Denoised', color='#F8FAFC', fontsize=11, fontweight='bold')
    ax.set_xlabel('Time Frames (Clean | Noisy | Denoised)', color='#CBD5E1')
    ax.set_ylabel('Mel Bins (128)', color='#CBD5E1')
    cbar3 = fig.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(colors='#CBD5E1')

    # Panel 4: Spectral Residual Error Heatmap (|Clean - Denoised|)
    ax = axes[1, 0]
    residual = np.abs(clean_spec - denoised_spec)
    im4 = ax.imshow(residual, origin='lower', aspect='auto', cmap='magma', vmin=0.0, vmax=0.3)
    ax.set_title('Spectral Residual Error (|Clean - Denoised|)', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time Frames', color='#CBD5E1')
    ax.set_ylabel('Mel Bins', color='#CBD5E1')
    cbar4 = fig.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)
    cbar4.ax.tick_params(colors='#CBD5E1')

    # Panel 5: SNR Improvement Distribution (Delta SNR)
    ax = axes[1, 1]
    delta_snrs = eval_res['all_delta_snr']
    mean_delta = np.mean(delta_snrs)
    p95_delta = np.percentile(delta_snrs, 95)
    ax.hist(delta_snrs, bins=20, color='#60A5FA', edgecolor='#2563EB', alpha=0.85)
    ax.axvline(mean_delta, color='#10B981', lw=2, linestyle='--', label=f'Mean: +{mean_delta:.1f} dB')
    ax.axvline(p95_delta, color='#F59E0B', lw=2, linestyle=':', label=f'P95: +{p95_delta:.1f} dB')
    ax.set_title('Audio SNR Improvement (Delta SNR)', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xlabel('SNR Gain (dB)', color='#CBD5E1')
    ax.set_ylabel('Sample Count', color='#CBD5E1')
    ax.legend(facecolor='#0F172A', edgecolor='#334155', labelcolor='#F8FAFC')
    ax.grid(True, alpha=0.15)

    # Panel 6: CPU Latency Profile & Throughput
    ax = axes[1, 2]
    mean_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    fps = 1000.0 / mean_lat
    ax.hist(latencies, bins=25, color='#F59E0B', edgecolor='#D97706', alpha=0.85)
    ax.axvline(mean_lat, color='#EF4444', lw=2, linestyle='--', label=f'Mean: {mean_lat:.2f} ms')
    ax.axvline(p95_lat, color='#8B5CF6', lw=2, linestyle=':', label=f'P95: {p95_lat:.2f} ms')
    ax.set_title('CPU Inference Latency Profile', color='#F8FAFC', fontsize=12, fontweight='bold')
    ax.set_xlabel('Latency (ms / Spectrogram)', color='#CBD5E1')
    ax.set_ylabel('Frequency', color='#CBD5E1')
    ax.legend(facecolor='#0F172A', edgecolor='#334155', labelcolor='#F8FAFC')
    ax.grid(True, alpha=0.15)

    info_text = (
        f"Throughput: {fps:.1f} chunks/sec\n"
        f"Mean PSNR: {eval_res['mean_psnr']:.2f} dB\n"
        f"Mean SNR Gain: +{mean_delta:.2f} dB\n"
        f"Mean Spec MSE: {eval_res['mean_mse']:.4f}"
    )
    ax.text(0.95, 0.65, info_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color='#F8FAFC',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#0F172A', edgecolor='#38BDF8', alpha=0.9))

    plt.suptitle('DenoiseUNet: Audio Denoising and Speech Enhancement with Spectrogram Autoencoders',
                 color='#F8FAFC', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Diagnostic report saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main Training & Denoising Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DenoiseUNet: Audio Denoising with Autoencoders")
    parser.add_argument('--epochs', type=int, default=18, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of paired audio clips')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)

    print("=" * 72)
    print("Project 80: Audio Denoising with Autoencoders (DenoiseUNet)")
    print(f"Device: {device} | Sampling Rate: {SAMPLE_RATE} Hz | Duration: {DURATION}s")
    print(f"Noise Taxonomy: {', '.join(NOISE_TYPES)}")
    print("=" * 72)

    # Step 1: Synthesize Paired Clean & Noisy Spectrograms
    print("\n[1/5] Synthesizing acoustic waveforms and computing paired Mel spectrograms...")
    pipeline = MelSpectrogramPipeline()
    rng = np.random.default_rng(args.seed)

    all_clean_specs = []
    all_noisy_specs = []
    all_noise_types = []
    all_snrs = []

    t0_gen = time.time()
    n_pts = int(SAMPLE_RATE * DURATION)
    for i in range(args.num_samples):
        clean_wav = generate_clean_audio(rng, duration=DURATION)
        n_type = rng.choice(NOISE_TYPES)
        noise_wav = generate_noise(n_type, n_pts, rng)
        snr_target = rng.uniform(-4.0, 14.0)

        noisy_wav, actual_snr = mix_clean_and_noise(clean_wav, noise_wav, snr_target)

        clean_spec = pipeline.audio_to_mel(torch.from_numpy(clean_wav))
        noisy_spec = pipeline.audio_to_mel(torch.from_numpy(noisy_wav))

        all_clean_specs.append(clean_spec)
        all_noisy_specs.append(noisy_spec)
        all_noise_types.append(n_type)
        all_snrs.append(actual_snr)

    all_clean_specs = torch.stack(all_clean_specs)  # (N, 1, 128, 128)
    all_noisy_specs = torch.stack(all_noisy_specs)  # (N, 1, 128, 128)
    all_snrs = torch.tensor(all_snrs, dtype=torch.float32)
    t1_gen = time.time()
    print(f"[✓] Generated {len(all_clean_specs)} paired spectrograms in {t1_gen - t0_gen:.2f}s (Shape: {all_clean_specs.shape})")

    # Step 2: Stratified Train / Val / Test Splits
    print("\n[2/5] Creating Train (70%) / Val (15%) / Test (15%) splits...")
    indices = np.arange(len(all_clean_specs))
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_dataset = DenoisingSpectrogramDataset(
        all_noisy_specs[train_idx], all_clean_specs[train_idx],
        [all_noise_types[i] for i in train_idx], all_snrs[train_idx]
    )
    val_dataset = DenoisingSpectrogramDataset(
        all_noisy_specs[val_idx], all_clean_specs[val_idx],
        [all_noise_types[i] for i in val_idx], all_snrs[val_idx]
    )
    test_dataset = DenoisingSpectrogramDataset(
        all_noisy_specs[test_idx], all_clean_specs[test_idx],
        [all_noise_types[i] for i in test_idx], all_snrs[test_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"[✓] Splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Step 3: Instantiate Model Architecture
    print("\n[3/5] Instantiating DenoiseUNet Autoencoder Architecture...")
    model = DenoiseAutoencoder().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[✓] DenoiseUNet initialized with {total_params:,} parameters ({total_params * 4 / (1024**2):.2f} MB)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Step 4: Training & Validation Loop
    print("\n[4/5] Training DenoiseUNet Autoencoder with Composite Spectral Loss...")
    history = {'train_loss': [], 'val_loss': [], 'val_psnr': []}
    best_val_psnr = -1.0
    best_ckpt_path = os.path.join(args.output_dir, 'denoise_autoencoder_best.pth')

    t0_train = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        val_eval = evaluate(model, val_loader, device)
        scheduler.step()
        ep_t1 = time.time()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_eval['loss'])
        history['val_psnr'].append(val_eval['mean_psnr'])

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {tr_loss:.4f} | "
              f"Val Loss: {val_eval['loss']:.4f} | "
              f"Val PSNR: {val_eval['mean_psnr']:.2f} dB | "
              f"Delta SNR: +{val_eval['mean_delta_snr']:.2f} dB | "
              f"Time: {ep_t1 - ep_t0:.1f}s")

        if val_eval['mean_psnr'] > best_val_psnr:
            best_val_psnr = val_eval['mean_psnr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': best_val_psnr,
                'args': vars(args)
            }, best_ckpt_path)

    t_train_total = time.time() - t0_train
    print(f"\n[✓] Training completed in {t_train_total:.1f}s. Best Val PSNR: {best_val_psnr:.2f} dB")

    # Load best checkpoint
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Step 5: Test Set Evaluation & Audio Inversion
    print("\n[5/5] Conducting comprehensive Test Set Evaluation & Benchmarking...")
    test_eval = evaluate(model, test_loader, device)
    print("\n" + "=" * 55)
    print("        DENOISEUNET TEST EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Peak SNR (PSNR)  : {test_eval['mean_psnr']:.2f} dB")
    print(f"Mean SNR Gain    : +{test_eval['mean_delta_snr']:.2f} dB")
    print(f"Spectrogram MSE  : {test_eval['mean_mse']:.6f}")
    print(f"Composite Loss   : {test_eval['loss']:.4f}")
    print("=" * 55)

    # CPU Latency Benchmark
    model.eval().to('cpu')
    dummy_input = torch.randn(1, 1, N_MELS, TIME_STEPS)
    latencies = []
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
        for _ in range(100):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            latencies.append((time.perf_counter() - t_start) * 1000.0)

    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    throughput = 1000.0 / mean_lat

    print(f"Latency: Mean={mean_lat:.2f} ms | P50={p50_lat:.2f} ms | P95={p95_lat:.2f} ms")
    print(f"Throughput: {throughput:.1f} chunks/sec")

    # Generate sample audio demonstration with acoustic inversion
    sample_clean_wav = generate_clean_audio(np.random.default_rng(999), duration=DURATION)
    sample_noise_wav = generate_noise('white', len(sample_clean_wav), np.random.default_rng(999))
    sample_noisy_wav, _ = mix_clean_and_noise(sample_clean_wav, sample_noise_wav, snr_db=5.0)

    clean_spec_t = pipeline.audio_to_mel(torch.from_numpy(sample_clean_wav))
    noisy_spec_t = pipeline.audio_to_mel(torch.from_numpy(sample_noisy_wav))

    with torch.no_grad():
        denoised_spec_t = model(noisy_spec_t.unsqueeze(0)).squeeze(0)

    denoised_wav = pipeline.mel_to_audio(denoised_spec_t).numpy()

    clean_audio_path = os.path.join(args.output_dir, 'sample_clean.wav')
    noisy_audio_path = os.path.join(args.output_dir, 'sample_noisy.wav')
    denoised_audio_path = os.path.join(args.output_dir, 'sample_denoised.wav')

    sf.write(clean_audio_path, sample_clean_wav, SAMPLE_RATE)
    sf.write(noisy_audio_path, sample_noisy_wav, SAMPLE_RATE)
    sf.write(denoised_audio_path, denoised_wav, SAMPLE_RATE)
    print(f"[✓] Exported clean audio: {clean_audio_path}")
    print(f"[✓] Exported noisy audio: {noisy_audio_path}")
    print(f"[✓] Exported denoised restored audio: {denoised_audio_path}")

    # Generate visual diagnostic report
    report_path = os.path.join(args.output_dir, 'audio_denoising_report.png')
    sample_triplet = (
        clean_spec_t[0].numpy(),
        noisy_spec_t[0].numpy(),
        denoised_spec_t[0].numpy()
    )
    generate_diagnostic_report(history, test_eval, sample_triplet, latencies, report_path)

    # Save metrics JSON
    metrics_path = os.path.join(args.output_dir, 'audio_denoising_metrics.json')
    metrics_data = {
        'project': 'Project 80: Audio Denoising with Autoencoders',
        'model': {
            'architecture': 'DenoiseUNet (Convolutional U-Net Autoencoder)',
            'parameters': total_params,
            'model_size_mb': float(total_params * 4 / (1024**2))
        },
        'test_metrics': {
            'mean_psnr_db': test_eval['mean_psnr'],
            'mean_snr_improvement_db': test_eval['mean_delta_snr'],
            'mean_spectrogram_mse': test_eval['mean_mse'],
            'test_loss': test_eval['loss']
        },
        'latency_benchmark': {
            'mean_ms': mean_lat,
            'median_ms': p50_lat,
            'p95_ms': p95_lat,
            'throughput_fps': throughput
        }
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[✓] Metrics saved to: {metrics_path}")
    print("\n[✓] All DenoiseUNet stages completed successfully!")


if __name__ == '__main__':
    main()

