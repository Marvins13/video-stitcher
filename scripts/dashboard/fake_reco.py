#!/usr/bin/env python3
"""Test double for the `reco` binary, used to exercise reco_dashboard.py.

Accepts the same command line as the real recorder (unknown options are
ignored). Once per second it writes preview.jpg and status.json into
--dashboard-dir, grows the -o file, and prints carriage-return separated
progress lines plus ANSI-coloured log lines and the libOpenCL noise line.
It honours --end-time (seconds). On SIGINT it writes state "finished" and
exits 0; a second SIGINT exits at once without finalizing, like the real
recorder. No cameras are touched.

Environment switches (dashboard tests only):
  FAKE_RECO_FAIL=1            print an error and exit 1 after 2 s
  FAKE_RECO_FAIL=start        print an error and exit 1 before any status is written
  FAKE_RECO_HANG=1            stop updating status.json and progress after 3 s
                              (the process stays alive)
  FAKE_RECO_IGNORE_SIGINT=1   ignore every SIGINT (tests the SIGKILL escalation)
"""

import argparse
import base64
import json
import os
import random
import signal
import sys
import threading
import time

# A 96x27 test-pattern JPEG (1221 bytes). The stdlib cannot encode JPEG, so the
# bytes are embedded here.
PREVIEW_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAgAAAQABAAD//gAQTGF2YzYwLjMxLjEwMgD/2wBDAAgMDA4MDhAQEBAQ"
    "EBMSExQUFBMTExMUFBQVFRUZGRkVFRUUFBUVGBgZGRscGxoaGRocHB4eHiQkIiIqKiszMz7/"
    "xACOAAEBAAMBAQAAAAAAAAAAAAAABQYEAwcCAQADAQEAAAAAAAAAAAAAAAAFBAYHAxAAAQMD"
    "AwEHAwUBAQAAAAAAARIDAhMRADEEIRRBUWEzMiMFFUIigmKzUrI0hBEAAQIEBAMFBwUBAAAA"
    "AAAAAQIRABIxAyIhMhNBBDMVYnEFUmFTokKBcrFzNFFjskP/wAARCAAaAGADASIAAhEAAxEA"
    "/9oADAMBAAIRAxEAPwDy6R3Lj822eUxUfQBGIiCZSlKwiB3k2z7T8hUp/cipf2kU7XqVPLR+"
    "5Vr8a5UZMZdeymrOfTTizcxLsWpKcjE96ewfl3A5XW30HR0/e87pFFdPqKtNWq6f2+ZblOD/"
    "ADDzS7zd11IdM+FKjPIm4AoqckYU6dIGWqLVKF8vcupTfuDEoYVKSDIogJYPnxqfCJTvyXyb"
    "EhGbtiQJD8GSDE6GMoxMSD3g4a+S+TfkYwduQDI/gyAIjUylKIiAO8nOm6kgbFuKWHITcmmR"
    "vRDjok2skHQcm/NtRjayWN83JL7k5tzTE2rBt0ycQQBqORbm2gwM1vbm2bT/AGj1yzNRmxav"
    "rxjOuzOU3pNtDM/TRXankeV5psGh34PlGRfWvnVor8pXentkotda0IT43t2Zxc+f+ZZNp7i1"
    "wJD29uQQdCCGyCPEHOqodJ0yPc8zp1FSKq0K1Ujs9fhkvcSSNm3GzLkJTkmRvSW4JQWSDoOT"
    "fm2owai3ZUWPL2KkdNFACZuORpQj2xVXLCUpcLVRJrQlQElBiFag92PPGIbjcSMWwoiJkfTG"
    "MYxFzKUpWjEDvJAzZ6TeVaSPyp1b3bRTtepUvTR+5Vr8a5a2cqkfkmZp3LrjjU0QlYbgNPGb"
    "tOQA1jchIUR6Rl9bX07oqXved0Syul1FWkvVdL7fMtynNHv+c89bXKF3DjQnqLySpAUb2a0m"
    "RJMuaUjLNYpClvk7S0uT8qj4kKIFvScRAepPdMY69ut8xIRcnYmIkOGyJRlpKMogxkD3gnDO"
    "63z8jFudyImR4bAjGOspSkBGIHeSM299FTey20YIdU6aKrlsPOAtwlI9tv7WI7bY2MUt73bS"
    "gt1TRoqsXAy4S5CMh22/rcnsvnPtHmNifcVO7NOdO7JuNPpbFqZvm4we3b87b15m9aqyvL48"
    "KP7ItfUfmV06nKKl0sJRa61pQnxva/GuPqPzK6dTlFS6WEotda0oT43tfjXN1UOi6Wn7vmdM"
    "oqp1qiFaqp9nrtzbCodF0tP3fM6ZRVTrVEK1VT7PXbm2BO0+Y/j/AKSVOn32vR8PfiK7E5D0"
    "W+nPoTq9x09fxf1x5xvPPn+n/IzRze3nnz/T/kZo51t6E/aPxFpzf7m/+rc/0YpYxjOMZzFP"
    "GMYnB+MOxjGaBCsU8YxicH4pZ3aZm9IxgLkAyPIAERqZSJAA8Sc4Zb23/Dvv/P8AyHB15Zto"
    "cM5UhOdMawl/o7tEbYti4ti7BK1FqmRClNxZ2Z82j//Z"
)

FPS = 30
CHUNK = 512 * 1024  # bytes appended to the output file per second
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RESET = "\x1b[0m"
NOISE = "libOpenCL.so.1: no version information available (required by libavutil.so.58)"


def say(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def atomic_write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("command", nargs="?")
    ap.add_argument("--left-device")
    ap.add_argument("--right-device")
    ap.add_argument("-c", "--calibration")
    ap.add_argument("-o", "--output")
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--encoder")
    ap.add_argument("--container")
    ap.add_argument("--quality", default="balanced")
    ap.add_argument("--dashboard-dir")
    ap.add_argument("--end-time", type=float)
    args, _unknown = ap.parse_known_args(argv)

    fail = os.environ.get("FAKE_RECO_FAIL", "")
    hang = os.environ.get("FAKE_RECO_HANG") == "1"
    ignore_sigint = os.environ.get("FAKE_RECO_IGNORE_SIGINT") == "1"
    out_path = os.path.abspath(args.output) if args.output else None
    dash = args.dashboard_dir

    stop = threading.Event()
    sigint_count = [0]

    def on_sigint(signum, frame):
        if ignore_sigint:
            return
        sigint_count[0] += 1
        if sigint_count[0] >= 2:
            os._exit(130)  # second SIGINT: force exit, file not finalized
        stop.set()

    signal.signal(signal.SIGINT, on_sigint)

    say(f"{GREEN}INFO{RESET} fake reco: {args.quality} {args.width}x{args.height} -> {out_path}\n")
    say(NOISE + "\n")
    if fail == "start":
        say(f"{YELLOW}ERROR{RESET} camera 0: device not found\n")
        return 1

    out = open(out_path, "wb") if out_path else None
    preview = base64.b64decode(PREVIEW_JPEG_B64)
    fps_recent = float(FPS)
    start = time.monotonic()

    def write_status(state, elapsed, frames):
        if not dash:
            return
        doc = {
            "state": state,
            "output": out_path,
            "width": args.width,
            "height": args.height,
            "frames": frames,
            "elapsed_s": round(elapsed, 1),
            "fps_avg": round(frames / elapsed, 1) if elapsed > 0 else 0.0,
            "fps_recent": fps_recent,
            "updated_unix": round(time.time(), 3),
        }
        atomic_write(os.path.join(dash, "status.json"), (json.dumps(doc) + "\n").encode())

    while True:
        elapsed = time.monotonic() - start
        frames = int(elapsed * FPS)
        if not (hang and elapsed >= 3.0):
            fps_recent = round(FPS + random.uniform(-0.4, 0.2), 1)
            avg = round(frames / elapsed, 1) if elapsed > 0 else 0.0
            write_status("recording", elapsed, frames)
            if dash:
                atomic_write(os.path.join(dash, "preview.jpg"), preview)
            say(f"\rProcessed {frames} frames ({avg} / {fps_recent} fps)")
        if fail == "1" and elapsed >= 2.0:
            say(f"\n{YELLOW}ERROR{RESET} simulated failure: camera 0 stopped delivering frames\n")
            return 1
        if out is not None:
            out.write(b"\0" * CHUNK)
            out.flush()
        if args.end_time is not None and elapsed >= args.end_time:
            break
        if stop.wait(1.0):
            break

    elapsed = time.monotonic() - start
    frames = int(elapsed * FPS)
    if out is not None:
        out.close()
    write_status("finished", elapsed, frames)
    say(f"\n{GREEN}INFO{RESET} recording finished: {frames} frames\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
