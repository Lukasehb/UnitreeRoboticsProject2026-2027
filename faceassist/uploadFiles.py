#!/usr/bin/env python3

import os
from ftplib import FTP, error_perm
from getpass import getpass


DEFAULT_HOST = "ftp.botopiabe.webhosting.be"
DEFAULT_USERNAME = "botopiabe@botopiabe"
DEFAULT_REMOTE_DIR = "/subsites/stayontrails.botopia.be/audio"

LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speech_mp3")


def ask_with_default(prompt: str, default: str) -> str:
    value = input(f"{prompt} [{default}]: ").strip()
    return value if value else default


def ensure_remote_dir(ftp: FTP, remote_dir: str):
    """
    Zorgt ervoor dat de remote directory bestaat (maakt ze indien nodig)
    """
    parts = remote_dir.strip("/").split("/")
    for part in parts:
        try:
            ftp.cwd(part)
        except error_perm:
            print(f"[INFO] Maak map aan: {part}")
            ftp.mkd(part)
            ftp.cwd(part)


def upload_directory(ftp: FTP, local_dir: str):
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Lokale map niet gevonden: {local_dir}")

    files = [
        f for f in os.listdir(local_dir)
        if os.path.isfile(os.path.join(local_dir, f)) and f.lower().endswith(".mp3")
    ]

    if not files:
        print(f"[INFO] Geen MP3 bestanden gevonden in {local_dir}")
        return

    print(f"[INFO] {len(files)} MP3 bestand(en) gevonden")

    for filename in files:
        local_path = os.path.join(local_dir, filename)
        print(f"[UPLOAD] {filename}")

        with open(local_path, "rb") as f:
            ftp.storbinary(f"STOR {filename}", f)

        print(f"[OK] {filename} opgeladen")


def main():
    print("FTP upload script")
    print("-----------------")

    host = ask_with_default("FTP host", DEFAULT_HOST)
    username = ask_with_default("FTP username", DEFAULT_USERNAME)
    remote_dir = ask_with_default("Remote map", DEFAULT_REMOTE_DIR)

    password = getpass("FTP password: ")

    print("\n[INFO] Verbinden met FTP...")
    ftp = FTP(host, timeout=30)
    ftp.login(user=username, passwd=password)
    print("[OK] Ingelogd")

    try:
        print(f"[INFO] Ga naar remote map: {remote_dir}")
        ftp.cwd("/")  # start van root
        ensure_remote_dir(ftp, remote_dir)

        upload_directory(ftp, LOCAL_DIR)

    finally:
        try:
            ftp.quit()
        except Exception:
            ftp.close()

    print("[KLAAR] Upload voltooid.")


if __name__ == "__main__":
    main()