#!/usr/bin/env python3

import os
import subprocess
import shutil
import argparse


KNOWN_DIR = "known"
OUTPUT_DIR = "speech_mp3"
PIPER_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "voices", "nl_BE-nathalie-medium.onnx")


def load_names(known_dir):
    names = []
    if not os.path.isdir(known_dir):
        return names

    for fn in os.listdir(known_dir):
        if fn.lower().endswith(".npz"):
            name = os.path.splitext(fn)[0]
            names.append(name)

    return sorted(names)


def sanitize_filename(text: str) -> str:
    text = text.strip().replace(" ", "_")
    allowed = "-_.()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    cleaned = "".join(c for c in text if c in allowed)

    if not cleaned:
        cleaned = "speech"

    return cleaned


def generate_sentences(name):
    return [
        ("left", f"{name} staat links van je"),
        ("right", f"{name} staat rechts van je"),
        ("front", f"{name} staat voor je"),
    ]


def generate_wav_with_piper(text: str, wav_path: str, model_path: str):
    cmd = [
        "piper",
        "--model", model_path,
        "--output_file", wav_path,
    ]

    result = subprocess.run(
        cmd,
        input=text + "\n",
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Piper fout voor '{text}':\n{result.stderr}"
        )


def convert_wav_to_mp3(wav_path: str, mp3_path: str):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", wav_path,
        "-codec:a", "libmp3lame",
        "-qscale:a", "2",
        mp3_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg fout bij omzetting naar mp3:\n{result.stderr}"
        )


def generate_mp3(text: str, output_file: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    safe_file = sanitize_filename(output_file)

    wav_path = os.path.join(OUTPUT_DIR, f"{safe_file}.wav")
    mp3_path = os.path.join(OUTPUT_DIR, f"{safe_file}.mp3")

    print(f"[INFO] Genereer WAV: {text}")
    generate_wav_with_piper(text, wav_path, PIPER_MODEL)

    print(f"[INFO] Converteer naar MP3: {mp3_path}")
    convert_wav_to_mp3(wav_path, mp3_path)

    os.remove(wav_path)

    print(f"[OK] MP3 aangemaakt: {mp3_path}")


def check_dependencies():
    if not os.path.exists(PIPER_MODEL):
        raise FileNotFoundError(f"Piper model niet gevonden: {PIPER_MODEL}")

    if shutil.which("piper") is None:
        raise RuntimeError("Het commando 'piper' is niet gevonden in PATH.")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Het commando 'ffmpeg' is niet gevonden in PATH.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genereer Nederlandse spraak als MP3 met Piper."
    )

    parser.add_argument(
        "--text",
        type=str,
        help="De tekst die uitgesproken moet worden."
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Naam van het uitvoerbestand zonder extensie."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    check_dependencies()

    # Modus 1: eigen tekst + eigen bestandsnaam
    if args.text:
        if args.file:
            output_file = args.file
        else:
            output_file = sanitize_filename(args.text[:40])

        generate_mp3(args.text, output_file)
        return

    # Modus 2: oude werking, automatisch op basis van known/*.npz
    names = load_names(KNOWN_DIR)

    if not names:
        print(f"[INFO] Geen .npz bestanden gevonden in '{KNOWN_DIR}'")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name in names:
        safe_name = sanitize_filename(name)

        for pos_key, sentence in generate_sentences(name):
            output_file = f"{safe_name}_{pos_key}"
            generate_mp3(sentence, output_file)

    print(f"[OK] Klaar. MP3-bestanden staan in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()