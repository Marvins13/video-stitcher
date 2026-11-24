# Reco dashboard

A small web dashboard for the `reco` panoramic recorder, meant for a phone (iPhone Safari)
on the same Wi-Fi as the board. Python standard library only, no pip packages. The page is
one HTML file with inline CSS and JavaScript and loads nothing from the internet, so it works
on a network without internet access.

## Files

- `reco_dashboard.py` - HTTP server; starts and stops `reco camera ...`, reports its status.
- `index.html` - the UI (German).
- `fake_reco.py` - test double with the same command line as `reco`; it touches no cameras.

## Run

    python3 reco_dashboard.py

| Option | Default | Meaning |
|---|---|---|
| `--port` | 8080 | HTTP port, bound on all interfaces |
| `--reco` | `/home/arduino/bin/reco` | reco binary |
| `--calibration` | `/home/arduino/match2.json` | passed to reco as `-c` |
| `--recordings` | `/home/arduino/recordings` | output directory, created if missing |
| `--runtime-dir` | `/dev/shm/reco-dashboard` | tmpfs directory, passed to reco as `--dashboard-dir`; also holds `reco.log` |

Open `http://<board-ip>:<port>/` on the phone; the IPv4 addresses are shown in the footer.
`scripts/gameday.py` also defaults to port 8080, so run only one of them or pass `--port`.

Ctrl+C or SIGTERM stops a running recording first (SIGINT, up to 20 s, then SIGKILL) so that
the file is finalized, and then exits. SIGHUP is ignored, so a dropped SSH session does not
stop the dashboard.

## Endpoints

JSON responses are UTF-8 with `Cache-Control: no-store`.

| Method | Path | Notes |
|---|---|---|
| GET | `/` | the UI |
| GET | `/preview.jpg` | last preview frame; 404 if none exists yet |
| GET | `/api/status` | `state`, `stale`, `error`, `run`, `reco` (contents of status.json), `file_size_bytes`, `log`, `system`, `recordings`, `now_unix` |
| POST | `/api/start` | body `{"quality": "fast" \| "balanced" \| "high", "minutes": null \| 1..180}`; returns the status. 400 on invalid input; 409 if a recording runs, another `reco camera` process runs, or less than 2 GB is free |
| POST | `/api/stop` | SIGINT to reco; a second SIGINT after 20 s and SIGKILL 10 s later if it is still alive; returns state `stopping`; 409 if nothing runs |
| GET | `/recordings/<name>` | `.mkv` or `.mp4` in the recordings directory; `Range` requests get 206; `?download=1` sets `attachment` |

`state` is `idle`, `starting`, `recording`, `stopping` or `error`. `stale: true` means a
recording runs but status.json has not been updated for more than 5 s. `error` holds the
reason and the last log lines when a run exited with a non-zero code or died during start.

## Notes

- Recordings are written as fragmented MP4 (`--container fmp4`, ~1 s fragments): they play
  in iOS Safari, and a crash or power loss only loses the last fragment. Older `.mkv` files
  are still listed and downloadable, but iOS Safari cannot play Matroska.
- The dashboard only controls recordings it started. After a dashboard restart, a recording
  that is still running blocks new starts (409) but cannot be stopped from the UI.
- Sizes use GiB (1024^3), matching `df -h`.

## Testing without cameras

    python3 reco_dashboard.py --port 8090 --reco ./fake_reco.py \
        --recordings /tmp/reco-test/rec --runtime-dir /tmp/reco-test/rt

`fake_reco.py` reads these environment variables (set them for the dashboard process):

- `FAKE_RECO_FAIL=1` - error exit after 2 s; `FAKE_RECO_FAIL=start` - error exit at once
- `FAKE_RECO_HANG=1` - stops updating status.json after 3 s (shows `stale`)
- `FAKE_RECO_IGNORE_SIGINT=1` - ignores SIGINT (exercises the SIGKILL escalation)
