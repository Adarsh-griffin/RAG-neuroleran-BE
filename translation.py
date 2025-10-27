from faster_whisper import WhisperModel
import requests
import json
import os
import time
import argparse
from typing import Optional

try:
    import torch
except Exception:
    torch = None  # torch may be missing; we will handle gracefully

def parse_args():
    parser = argparse.ArgumentParser(description="Speech-to-text then translate pipeline using Whisper + LibreTranslate")
    parser.add_argument("--audio", default="test_audio.mp3", help="Path to input audio file")
    parser.add_argument("--model", default="medium", help="Whisper model: tiny, base, small, medium, large")
    parser.add_argument("--source", default="auto", help="Source language code (e.g., hi, en). Use 'auto' to detect")
    parser.add_argument("--target", default="en", help="Target language code (e.g., en, hi)")
    parser.add_argument("--libre-url", default="http://localhost:5000/translate", help="LibreTranslate endpoint URL")
    parser.add_argument("--stt-only", action="store_true", help="Only perform speech-to-text (no translation)")
    parser.add_argument("--translate-only", action="store_true", help="Only translate provided text via --text (skip STT)")
    parser.add_argument("--text", default=None, help="Text to translate when using --translate-only")
    parser.add_argument("--quiet", action="store_true", help="Print only the final translation")
    return parser.parse_args()


def run_stt(model, audio_path, source_language):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Logging
    print(f"-> Transcribing audio file: {audio_path}...")
    start_time = time.time()

    # Faster-Whisper transcription
    fw_kwargs = {
        "beam_size": 5,
        "temperature": 0.0,
    }
    if source_language and source_language.lower() != "auto":
        fw_kwargs["language"] = source_language

    segments, info = model.transcribe(audio_path, **fw_kwargs)
    transcribed_text_parts = []
    for seg in segments:
        transcribed_text_parts.append(seg.text)
    transcribed_text = " ".join(part.strip() for part in transcribed_text_parts).strip()
    detected_language = getattr(info, "language", None)

    asr_time = time.time() - start_time
    print(f"-> ASR Time: {asr_time:.2f} seconds")

    lang_for_print = source_language if (source_language and source_language != "auto") else (detected_language or "unknown")
    print(f"-> Transcribed Text ({lang_for_print.upper()}): {transcribed_text}")

    return transcribed_text, (detected_language or source_language)


def translate_via_libre(text, source_lang, target_lang, libre_url):
    print(f"-> Calling local LibreTranslate API for translation ({source_lang} -> {target_lang})...")
    data = {"q": text, "source": source_lang, "target": target_lang, "format": "text"}
    response = requests.post(libre_url, headers={"Content-Type": "application/json"}, data=json.dumps(data), timeout=30)
    response.raise_for_status()
    payload = response.json()
    return payload.get("translatedText")


def translate_with_whisper_fallback(model, audio_path, text, source_lang, target_lang):
    # Whisper can only translate to English (task='translate')
    if target_lang.lower() != "en":
        print("-> Whisper fallback only supports translation to English. Skipping fallback.")
        return None

    if text:
        # No direct text->text translation via Whisper; needs audio. So skip.
        print("-> Whisper fallback requires audio input; cannot translate raw text. Skipping.")
        return None

    print("-> Using Faster-Whisper translate task as fallback to English...")
    fw_kwargs = {"beam_size": 5, "temperature": 0.0, "task": "translate"}
    if source_lang and source_lang.lower() != "auto":
        fw_kwargs["language"] = source_lang
    segments, info = model.transcribe(audio_path, **fw_kwargs)
    parts = [seg.text.strip() for seg in segments]
    return " ".join(parts).strip()


def main():
    args = parse_args()

    # Load Faster-Whisper model on best available device
    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    if not args.quiet:
        print(f"Loading Faster-Whisper model: {args.model} on {device} ({compute_type})...")
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    if not args.quiet:
        print("✅ Faster-Whisper model loaded.")

    transcribed_text = None
    source_used = args.source

    if not args.translate_only:
        transcribed_text, source_used = run_stt(model, args.audio, args.source)
    else:
        if not args.text:
            raise ValueError("--translate-only requires --text to be provided")
        transcribed_text = args.text

    if args.stt_only:
        if args.quiet:
            # In quiet mode for STT-only, print the transcript and exit
            print(transcribed_text)
        return

    # Try LibreTranslate first
    translated_text = None
    try:
        translated_text = translate_via_libre(transcribed_text, source_used if source_used != "auto" else "auto", args.target, args.libre_url)
    except requests.exceptions.RequestException as e:
        if not args.quiet:
            print(f"❌ ERROR: Could not connect to LibreTranslate API at {args.libre_url}")
            print("Ensure LibreTranslate is running in another terminal window.")
            print(f"Details: {e}")
    except Exception as e:
        if not args.quiet:
            print(f"❌ ERROR during translation: {e}")

    # Fallback to Whisper translate when target is English and audio is available
    if translated_text is None:
        translated_text = translate_with_whisper_fallback(model, args.audio, args.text if args.translate_only else None, source_used, args.target)

    if translated_text is None:
        if not args.quiet:
            print("❌ Unable to produce translation. See errors above.")
        return

    # Output
    if args.quiet:
        print(translated_text)
    else:
        print(f"-> FINAL TRANSLATION ({args.target.upper()}): {translated_text}")


if __name__ == "__main__":
    main()