"""
voice_engine.py  —  Voice analysis + acoustic voice cloning
============================================================
Pipeline:
  1. Analyze recorded samples → extract pitch (F0) and timbre fingerprint
  2. Generate a phrase with gTTS (Google Neural TTS, free)
  3. Apply librosa pitch-shifting to warp gTTS voice toward the recorded voice's
     pitch profile (F0 ratio = recorded_mean_F0 / gTTS_mean_F0)
  4. Optionally apply a gentle low-pass filter to match recorded voice warmth

This is a DSP-based voice adaptation, not a neural clone — but it produces
a noticeably different, personalised sound compared to raw gTTS.
Full neural cloning (XTTS-v2) requires Python ≤ 3.11.
"""

import io
import os
import logging
import tempfile
import shutil
from pathlib import Path

import numpy as np

log = logging.getLogger("voice_engine")

# ── Default gTTS fundamental frequency (en, female, slow) in Hz ──────────────
GTTS_F0_ESTIMATE = 220.0   # approximate mean F0 for gTTS English female voice

# ── Soothing speed ratio ──────────────────────────────────────────────────────
SLOW_RATE = 0.88   # slightly slower than natural for soothing effect


# ─────────────────────────────────────────────────────────────────────────────
# Sample analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyze_voice(sample_paths: list[str]) -> dict:
    """
    Analyze a list of WAV recordings and return a voice fingerprint dict:
        {
            "mean_f0":    float  (Hz),
            "f0_std":     float,
            "mean_rms":   float,
            "brightness": float  (spectral centroid ratio vs gTTS),
            "n_samples":  int,
            "duration_s": float,
        }
    Returns reasonable defaults if librosa is unavailable or samples are empty.
    """
    try:
        import librosa

        all_f0   = []
        all_rms  = []
        all_sc   = []
        total_dur = 0.0

        for path in sample_paths:
            if not Path(path).exists():
                continue
            try:
                y, sr = librosa.load(path, sr=16000, mono=True)
                total_dur += len(y) / sr

                # Fundamental frequency via pyin (accurate, works offline)
                f0, voiced_flag, _ = librosa.pyin(
                    y, fmin=50, fmax=600,
                    sr=sr, frame_length=2048,
                )
                voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
                if len(voiced_f0) > 0:
                    all_f0.extend(voiced_f0.tolist())

                # RMS
                rms = librosa.feature.rms(y=y)
                all_rms.append(float(rms.mean()))

                # Spectral centroid (brightness proxy)
                sc = librosa.feature.spectral_centroid(y=y, sr=sr)
                all_sc.append(float(sc.mean()))

            except Exception as e:
                log.warning(f"Could not analyze {path}: {e}")

        if not all_f0:
            log.warning("No voiced frames found in samples — using defaults")
            return _default_fingerprint(len(sample_paths))

        mean_f0 = float(np.median(all_f0))
        f0_std  = float(np.std(all_f0))
        brightness = (np.mean(all_sc) / 2200.0) if all_sc else 1.0  # norm vs gTTS baseline

        log.info(f"Voice fingerprint: F0={mean_f0:.1f}Hz ± {f0_std:.1f}, "
                 f"brightness={brightness:.2f}, dur={total_dur:.1f}s")

        return {
            "mean_f0":    mean_f0,
            "f0_std":     f0_std,
            "mean_rms":   float(np.mean(all_rms)) if all_rms else 0.03,
            "brightness": float(brightness),
            "n_samples":  len(sample_paths),
            "duration_s": total_dur,
        }

    except ImportError:
        log.warning("librosa not available — returning default fingerprint")
        return _default_fingerprint(len(sample_paths))


def _default_fingerprint(n=0):
    return {
        "mean_f0": GTTS_F0_ESTIMATE,
        "f0_std":  30.0,
        "mean_rms": 0.03,
        "brightness": 1.0,
        "n_samples": n,
        "duration_s": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TTS generation + voice adaptation
# ─────────────────────────────────────────────────────────────────────────────
def generate_adapted_phrase(
    text: str,
    out_path: str,
    fingerprint: dict | None = None,
) -> bool:
    """
    Generate TTS phrase and adapt it toward the recorded voice's pitch/timbre.

    Steps:
      1. gTTS → mp3 (temp file)
      2. Load with librosa / soundfile
      3. Compute pitch shift semitones = 12 * log2(target_f0 / gTTS_f0)
      4. Apply librosa.effects.pitch_shift
      5. Apply time-stretch if needed to match SLOW_RATE
      6. Optionally apply EQ brightness boost/cut via spectral shaping
      7. Export WAV

    Returns True on success.
    """
    try:
        from gtts import gTTS
        import librosa
        import soundfile as sf

        fp = fingerprint or _default_fingerprint()

        # ── Step 1: gTTS → temp mp3 ────────────────────────────────────────
        log.info(f"gTTS generating: '{text[:60]}'")
        tts = gTTS(text=text, lang="en", slow=True)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            mp3_path = tmp.name
        tts.save(mp3_path)

        # ── Step 2: Load audio ─────────────────────────────────────────────
        y, sr = librosa.load(mp3_path, sr=22050, mono=True)
        os.unlink(mp3_path)

        # ── Step 3: Compute pitch shift ────────────────────────────────────
        target_f0 = fp.get("mean_f0", GTTS_F0_ESTIMATE)
        # Clamp to a sane soothing range
        target_f0 = np.clip(target_f0, 140, 480)
        n_steps   = 12 * np.log2(target_f0 / GTTS_F0_ESTIMATE)
        n_steps   = float(np.clip(n_steps, -8, 8))  # max ±8 semitones
        log.info(f"Pitch shift: {n_steps:+.2f} semitones (target F0={target_f0:.0f}Hz)")

        # ── Step 4: Pitch shift ────────────────────────────────────────────
        if abs(n_steps) > 0.3:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

        # ── Step 5: Gentle time-stretch for soothing pace ─────────────────
        if abs(SLOW_RATE - 1.0) > 0.02:
            y = librosa.effects.time_stretch(y, rate=SLOW_RATE)

        # ── Step 6: Brightness shaping ────────────────────────────────────
        brightness = fp.get("brightness", 1.0)
        y = _apply_brightness(y, sr, brightness)

        # ── Step 7: Normalize + export ────────────────────────────────────
        peak = np.abs(y).max()
        if peak > 0:
            y = y / peak * 0.85   # normalize to 85% peak

        os.makedirs(Path(out_path).parent, exist_ok=True)
        sf.write(out_path, y, sr)
        log.info(f"Saved adapted phrase → {out_path}")
        return True

    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        return _fallback_gtts(text, out_path)
    except Exception as e:
        log.error(f"Voice adaptation failed: {e}")
        return _fallback_gtts(text, out_path)


def _apply_brightness(y: np.ndarray, sr: int, brightness: float) -> np.ndarray:
    """
    Gentle high-frequency boost/cut via scipy IIR filter.
    brightness < 1.0 → warmer (cut highs)
    brightness > 1.0 → brighter (boost highs)
    """
    try:
        from scipy.signal import butter, sosfilt

        if 0.85 < brightness < 1.15:
            return y  # close enough, skip filter

        # Simple high-shelf: butterworth HPF blended in
        cutoff = 3000.0
        nyq = sr / 2.0
        if cutoff >= nyq:
            return y

        b_order = 2
        sos = butter(b_order, cutoff / nyq, btype='high', output='sos')
        highs = sosfilt(sos, y)

        # blend: bright voice gets +highs, warm voice gets -highs
        gain = np.clip(brightness - 1.0, -0.4, 0.4)
        return np.clip(y + gain * highs, -1.0, 1.0)

    except Exception:
        return y


def _fallback_gtts(text: str, out_path: str) -> bool:
    """Last-resort: save raw gTTS mp3 renamed as .wav (browsers can play it)."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=True)
        # Save as mp3 alongside the expected wav path
        mp3 = out_path.replace(".wav", ".mp3")
        tts.save(mp3)
        if not out_path.endswith(".mp3"):
            # Copy so caller finds the .wav path (actually mp3 bytes — browser doesn't care)
            shutil.copy(mp3, out_path)
        return True
    except Exception as e:
        log.error(f"Fallback gTTS also failed: {e}")
        return False
