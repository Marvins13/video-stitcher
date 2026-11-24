#!/usr/bin/env python3
"""Web dashboard for the reco panoramic camera recorder.

Standard library only, no internet needed. Serves index.html and a small JSON
API on all interfaces, starts and stops `reco camera ...` as a child process
and reports its status.json and preview.jpg. See README.md in this directory.
"""

import argparse
import json
import os
import re
import shutil
import signal
import socket
import socketserver
import stat
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GIB = 1024 ** 3
MIN_FREE_BYTES = 2 * GIB        # refuse to start with less free space
STATUS_MAX_AGE_S = 5.0          # a status.json older than this during a recording is stale
STOP_GRACE_S = 20.0             # stop request: wait before the second SIGINT
KILL_GRACE_S = 10.0             # after the second SIGINT: wait before SIGKILL
SHUTDOWN_GRACE_S = 20.0         # dashboard exit: wait for reco after SIGINT
IP_CACHE_S = 10.0
QUALITIES = ("fast", "balanced", "high")
MINUTES_MIN, MINUTES_MAX = 1, 180
LOG_TAIL_BYTES = 64 * 1024      # only the tail of reco.log is read on each poll
LOG_LINES = 30
LOG_ERROR_LINES = 6
RECORDINGS_LIMIT = 30
STREAM_CHUNK = 1024 * 1024
MAX_BODY_BYTES = 4096
JSON_TYPE = "application/json; charset=utf-8"
RECO_NAMES = {"reco", "fake_reco.py"}  # executables recognised as a running recorder
NAME_RE = re.compile(r"[A-Za-z0-9._-]+\.(?:mkv|mp4)")
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[@-Z\\-_])")
NOISE = "libOpenCL.so.1: no version information available"
RANGE_UNSATISFIABLE = object()


class ApiError(Exception):
    """Turned into a JSON response {"error": message} with this HTTP status."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def de_number(value, digits=1):
    """Format a number with a German decimal comma for messages shown to the user."""
    return f"{value:.{digits}f}".replace(".", ",")


def json_bytes(obj):
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def as_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _signal(proc, sig):
    try:
        proc.send_signal(sig)  # Popen ignores this once the child has been reaped
    except OSError:
        pass


def read_status_file(runtime_dir):
    """Parsed status.json written by reco, or None if missing or unreadable."""
    try:
        with open(os.path.join(runtime_dir, "status.json"), "rb") as f:
            doc = json.loads(f.read())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def read_log_lines(path):
    """Last meaningful lines of the reco log.

    ANSI escapes are stripped, empty lines and libOpenCL noise are dropped, and
    consecutive "Processed ..." progress lines collapse to the last one.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - LOG_TAIL_BYTES))
            data = f.read()
    except OSError:
        return []
    text = data.decode("utf-8", errors="replace")
    if size > LOG_TAIL_BYTES:
        cut = re.search(r"[\r\n]", text)  # drop the partial first line
        text = text[cut.end():] if cut else ""
    text = ANSI_RE.sub("", text)
    lines = []
    for piece in re.split(r"[\r\n]+", text):  # reco uses \r for progress, \n for logs
        line = piece.strip()
        if not line or NOISE in line:
            continue
        if line.startswith("Processed ") and lines and lines[-1].startswith("Processed "):
            lines[-1] = line
        else:
            lines.append(line)
    return lines[-LOG_LINES:]


def find_foreign_reco():
    """PID of another running `reco camera` process, else None.

    Matches only when an argument whose basename is `reco` (or the test
    double `fake_reco.py`) is directly followed by `camera`, so shells or
    editors that merely mention both words do not block a start.
    """
    me = os.getpid()
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for entry in entries:
        if not entry.isdigit() or int(entry) == me:
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                raw = f.read()
        except OSError:
            continue
        argv = [a.decode("utf-8", errors="replace") for a in raw.split(b"\0")]
        for i, arg in enumerate(argv[:-1]):
            if os.path.basename(arg) in RECO_NAMES and argv[i + 1] == "camera":
                return int(entry)
    return None


def read_meminfo():
    """(MemTotal, MemAvailable) in MB; either may be None."""
    total_mb = avail_mb = None
    try:
        with open("/proc/meminfo", encoding="ascii", errors="replace") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_mb = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    avail_mb = int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return total_mb, avail_mb


def read_uptime_s():
    try:
        with open("/proc/uptime", encoding="ascii") as f:
            return int(float(f.read().split()[0]))
    except (OSError, ValueError, IndexError):
        return None


def read_temps():
    """Temperatures in degrees C.

    cpu, gpu and camera are the maxima over the matching thermal zone types;
    max is the maximum over all usable zones. Unreadable zones and values
    outside (0, 150] are ignored.
    """
    base = "/sys/class/thermal"
    try:
        names = os.listdir(base)
    except OSError:
        names = []
    readings = []
    for name in names:
        if not name.startswith("thermal_zone"):
            continue
        try:
            with open(os.path.join(base, name, "type"), encoding="utf-8") as f:
                zone_type = f.read().strip().lower()
            with open(os.path.join(base, name, "temp"), encoding="utf-8") as f:
                celsius = int(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            continue
        if 0 < celsius <= 150:
            readings.append((zone_type, celsius))

    def hottest(matches):
        values = [celsius for zone_type, celsius in readings if matches(zone_type)]
        return round(max(values), 1) if values else None

    return {
        "cpu": hottest(lambda t: t.startswith("cpu")),
        "gpu": hottest(lambda t: "gpu" in t),
        "camera": hottest(lambda t: t.startswith("camss")),
        "max": hottest(lambda t: True),
    }


def query_ipv4():
    """IPv4 addresses from `ip -4 -o addr`, without loopback and docker's 172.17.* bridge."""
    exe = shutil.which("ip")
    if exe is None:
        exe = next((p for p in ("/usr/sbin/ip", "/sbin/ip") if os.path.exists(p)), None)
    if exe is None:
        return []
    try:
        out = subprocess.run([exe, "-4", "-o", "addr"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    ips = []
    for ip in re.findall(r"\binet (\d+\.\d+\.\d+\.\d+)/", out):
        if ip.startswith(("127.", "172.17.")) or ip in ips:
            continue
        ips.append(ip)
    return ips


class IpCache:
    """Caches query_ipv4() so that every status poll does not spawn `ip`."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stamp = None
        self._ips = []

    def get(self):
        with self._lock:
            now = time.monotonic()
            if self._stamp is None or now - self._stamp > IP_CACHE_S:
                self._ips = query_ipv4()
                self._stamp = now
            return list(self._ips)


def collect_system(recordings_dir, ip_cache):
    mem_total_mb, mem_avail_mb = read_meminfo()
    try:
        usage = shutil.disk_usage(recordings_dir)
        disk_free_gb = round(usage.free / GIB, 2)
        disk_total_gb = round(usage.total / GIB, 2)
    except OSError:
        disk_free_gb = disk_total_gb = None
    try:
        loadavg = round(os.getloadavg()[0], 2)
    except OSError:
        loadavg = None
    return {
        "hostname": socket.gethostname(),
        "ips": ip_cache.get(),
        "uptime_s": read_uptime_s(),
        "loadavg": loadavg,
        "mem_total_mb": mem_total_mb,
        "mem_available_mb": mem_avail_mb,
        "disk_free_gb": disk_free_gb,
        "disk_total_gb": disk_total_gb,
        "temps": read_temps(),
    }


def list_recordings(rec_dir):
    """Newest-first list of recordings (regular files only, names as the download route accepts)."""
    items = []
    try:
        with os.scandir(rec_dir) as entries:
            for entry in entries:
                if not NAME_RE.fullmatch(entry.name):
                    continue
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISREG(st.st_mode):
                    items.append({
                        "name": entry.name,
                        "size_bytes": st.st_size,
                        "mtime_unix": round(st.st_mtime, 1),
                    })
    except OSError:
        return []
    items.sort(key=lambda item: item["mtime_unix"], reverse=True)
    return items[:RECORDINGS_LIMIT]


def public_run(run):
    if run is None:
        return None
    return {key: run.get(key) for key in ("output", "started_unix", "quality", "minutes", "pid", "exit_code")}


def build_reco_command(args, outfile, quality, minutes):
    """The reco invocation. minutes (1..180) becomes --end-time in seconds."""
    command = [
        args.reco, "camera",
        "--left-device", "0", "--right-device", "1",
        "-c", args.calibration,
        "-o", outfile,
        "--width", "3840", "--height", "1080",
        "--encoder", "gst-v4l2h264",
        "--container", "fmp4",  # fragmented MP4: plays in iOS Safari, crash loses only ~1 s
        "--quality", quality,
        "--dashboard-dir", args.runtime_dir,
    ]
    if minutes is not None:
        command += ["--end-time", str(minutes * 60)]
    return command


def parse_start_body(doc):
    """Validate the POST /api/start body. Returns (quality, minutes)."""
    unknown = sorted(set(doc) - {"quality", "minutes"})
    if unknown:
        raise ApiError(400, "Unbekannte Felder: " + ", ".join(unknown) + ".")
    quality = doc.get("quality")
    if not isinstance(quality, str) or quality not in QUALITIES:
        raise ApiError(400, "Ungültige Qualität. Erlaubt sind fast, balanced und high.")
    minutes = doc.get("minutes")
    if minutes is not None:
        is_int = isinstance(minutes, int) and not isinstance(minutes, bool)
        if not is_int or not MINUTES_MIN <= minutes <= MINUTES_MAX:
            raise ApiError(400, "Ungültige Dauer. Erlaubt sind null (unbegrenzt) oder 1 bis 180 Minuten.")
    return quality, minutes


def parse_range(header, size):
    """Interpret a Range header for a single byte range.

    Returns (start, end) inclusive, RANGE_UNSATISFIABLE, or None to ignore the
    header and serve the whole file. Multi-range and malformed headers are
    ignored, as RFC 9110 allows.
    """
    match = RANGE_RE.fullmatch(header.strip())
    if match is None:
        return None
    first, last = match.group(1), match.group(2)
    if first == "" and last == "":
        return None
    if size == 0:
        return RANGE_UNSATISFIABLE
    if first == "":  # suffix range: the last N bytes
        suffix = int(last)
        if suffix == 0:
            return RANGE_UNSATISFIABLE
        return max(0, size - suffix), size - 1
    start = int(first)
    if last != "" and int(last) < start:
        return None
    if start >= size:
        return RANGE_UNSATISFIABLE
    end = size - 1 if last == "" else min(int(last), size - 1)
    return start, end


class RecoController:
    """Owns the reco child process. All mutable state is guarded by self.lock."""

    def __init__(self, args):
        self.args = args
        self.runtime = args.runtime_dir
        self.lock = threading.Lock()
        self.ip_cache = IpCache()
        self.proc = None              # the running reco child (subprocess.Popen) or None
        self.run = None               # current or last run, see public_run()
        self.error = None             # error text of the last run, if it failed
        self.stop_requested = False
        self.seen_recording = False   # this run has reported state recording or finished
        self.closing = False          # the dashboard is exiting: refuse new starts

    def start(self, quality, minutes):
        with self.lock:
            if self.closing:
                raise ApiError(503, "Das Dashboard wird gerade beendet.")
            if self.proc is not None:
                raise ApiError(409, "Eine Aufnahme läuft bereits.")
            other = find_foreign_reco()
            if other is not None:
                raise ApiError(409, f"Ein anderer reco-Prozess läuft bereits (PID {other}).")
            free = shutil.disk_usage(self.args.recordings).free
            if free < MIN_FREE_BYTES:
                raise ApiError(409, f"Zu wenig Speicher frei: {de_number(free / GIB)} GB, mindestens 2 GB nötig.")
            outfile = self._new_outfile()
            for name in ("status.json", "preview.jpg"):  # so that stale files cannot look fresh
                try:
                    os.remove(os.path.join(self.runtime, name))
                except FileNotFoundError:
                    pass
            run = {
                "output": outfile,
                "started_unix": round(time.time(), 3),
                "quality": quality,
                "minutes": minutes,
                "pid": None,
                "exit_code": None,
            }
            command = build_reco_command(self.args, outfile, quality, minutes)
            try:
                with open(os.path.join(self.runtime, "reco.log"), "wb") as log:
                    proc = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
            except OSError as exc:
                self.run = run
                self.error = f"reco konnte nicht gestartet werden: {exc}"
                raise ApiError(500, self.error)
            run["pid"] = proc.pid
            self.run = run
            self.proc = proc
            self.error = None
            self.stop_requested = False
            self.seen_recording = False
        threading.Thread(target=self._watch, args=(proc,), name="reco-watch", daemon=True).start()

    def stop(self):
        with self.lock:
            proc = self.proc
            if proc is None:
                raise ApiError(409, "Keine Aufnahme aktiv.")
            if self.stop_requested:
                return  # already stopping; the escalation thread is running
            self.stop_requested = True
            _signal(proc, signal.SIGINT)
        threading.Thread(target=self._escalate, args=(proc,), name="reco-stop", daemon=True).start()

    def _escalate(self, proc):
        """After a stop request: a second SIGINT if reco is still alive after 20 s, SIGKILL 10 s later."""
        try:
            proc.wait(timeout=STOP_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            pass
        _signal(proc, signal.SIGINT)
        try:
            proc.wait(timeout=KILL_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            pass
        _signal(proc, signal.SIGKILL)

    def _watch(self, proc):
        """Reaps the child and records how the run ended."""
        code = proc.wait()
        with self.lock:
            if self.proc is not proc:
                return
            self._observe(read_status_file(self.runtime))
            reached = self.seen_recording
            self.proc = None
            self.run["exit_code"] = code
            if code == 0 and (reached or self.stop_requested):
                self.error = None
            else:
                self.error = self._failure_text(code, reached)

    def _failure_text(self, code, reached):
        if code < 0:
            try:
                head = f"reco wurde durch Signal {signal.Signals(-code).name} beendet."
            except ValueError:
                head = f"reco wurde durch Signal {-code} beendet."
        elif code == 0:
            head = "reco hat sich während des Starts beendet."
        else:
            head = f"reco beendete sich mit Exit-Code {code}."
        if reached:
            head += " Die Datei ist eventuell unvollständig."
        tail = read_log_lines(os.path.join(self.runtime, "reco.log"))[-LOG_ERROR_LINES:]
        return "\n".join([head] + tail)

    def _observe(self, reco):
        """Note that this run has reached recording (status.json is per run: removed before spawn)."""
        if reco is not None and reco.get("state") in ("recording", "finished"):
            self.seen_recording = True

    def _new_outfile(self):
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")  # local time of the board
        path = os.path.join(self.args.recordings, stamp + ".mp4")
        counter = 2
        while os.path.exists(path):  # never overwrite a file from the same second
            path = os.path.join(self.args.recordings, f"{stamp}_{counter}.mp4")
            counter += 1
        return path

    def status(self):
        now = time.time()
        with self.lock:
            proc = self.proc
            run = dict(self.run) if self.run else None
            error = self.error
            reco = read_status_file(self.runtime)
            if proc is not None:
                self._observe(reco)
            seen = self.seen_recording
            stop_requested = self.stop_requested

        stale = False
        if proc is None:
            state = "error" if error else "idle"
        else:
            rstate = reco.get("state") if reco else None
            updated = as_number(reco.get("updated_unix")) if reco else None
            fresh = rstate == "recording" and updated is not None and now - updated <= STATUS_MAX_AGE_S
            if stop_requested or rstate == "finished":
                state = "stopping"
            elif fresh or seen:
                state = "recording"  # seen before but not fresh now: stale, see below
            else:
                state = "starting"
            stale = seen and not fresh and rstate != "finished"

        size = None
        if run is not None:
            try:
                size = os.path.getsize(run["output"])
            except OSError:
                size = None
        return {
            "state": state,
            "stale": stale,
            "error": error if state == "error" else None,
            "run": public_run(run),
            "reco": reco,
            "file_size_bytes": size,
            "log": read_log_lines(os.path.join(self.runtime, "reco.log")),
            "system": collect_system(self.args.recordings, self.ip_cache),
            "recordings": list_recordings(self.args.recordings),
            "now_unix": round(now, 3),
        }

    def shutdown(self):
        """Dashboard exit: refuse new starts and stop a running recording gracefully."""
        with self.lock:
            self.closing = True
            proc = self.proc
            if proc is not None and not self.stop_requested:
                self.stop_requested = True
                _signal(proc, signal.SIGINT)
        if proc is None:
            return
        try:
            proc.wait(timeout=SHUTDOWN_GRACE_S)
        except subprocess.TimeoutExpired:
            _signal(proc, signal.SIGKILL)  # do not leave an unmanaged recorder running
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def kill_now(self):
        with self.lock:
            proc = self.proc
        if proc is not None:
            _signal(proc, signal.SIGKILL)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "RecoDashboard/1.0"
    timeout = 60          # seconds before an idle or stalled connection is dropped
    head_only = False
    responded = False

    def version_string(self):
        return self.server_version

    def log_request(self, code="-", size="-"):
        """Quiet: successful requests are not logged, error responses are."""
        try:
            status = int(code)
        except (TypeError, ValueError):
            status = 0
        if status == 0 or status >= 400:
            sys.stderr.write(f'{self.address_string()} "{self.requestline}" {code}\n')

    def log_message(self, fmt, *args):
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("HEAD")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def do_OPTIONS(self):
        self._dispatch("OPTIONS")

    def _dispatch(self, method):
        self.head_only = method == "HEAD"
        self.responded = False
        try:
            self._route(method)
        except ApiError as exc:
            self._reply_error(exc.status, exc.message)
        except (BrokenPipeError, ConnectionError, TimeoutError):
            pass  # the client went away; nothing to report
        except Exception as exc:  # one bad request must not take the server down
            sys.stderr.write(f"error while handling {method} {self.path}: {exc!r}\n")
            self._reply_error(500, "Interner Fehler im Dashboard.")

    def _reply_error(self, status, message):
        if self.responded:
            return
        try:
            self._send(status, json_bytes({"error": message}), JSON_TYPE)
        except OSError:
            pass

    def _route(self, method):
        url = urlsplit(self.path)
        path = url.path
        ctrl = self.server.controller
        if method in ("GET", "HEAD"):
            if path == "/":
                return self._send_static(os.path.join(SCRIPT_DIR, "index.html"),
                                         "text/html; charset=utf-8", "Nicht gefunden.")
            if path == "/preview.jpg":
                return self._send_static(os.path.join(self.server.args.runtime_dir, "preview.jpg"),
                                         "image/jpeg", "Noch keine Vorschau vorhanden.")
            if path == "/api/status":
                return self._send(200, json_bytes(ctrl.status()), JSON_TYPE)
            if path.startswith("/recordings/"):
                return self._send_recording(path[len("/recordings/"):], parse_qs(url.query))
        elif method == "POST":
            if path == "/api/start":
                quality, minutes = parse_start_body(self._read_json_object())
                ctrl.start(quality, minutes)
                return self._send(200, json_bytes(ctrl.status()), JSON_TYPE)
            if path == "/api/stop":
                ctrl.stop()
                return self._send(200, json_bytes(ctrl.status()), JSON_TYPE)
        raise ApiError(404, "Nicht gefunden.")

    def _send(self, code, body, ctype, extra=()):
        self.responded = True
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()
        if body and not self.head_only:
            self.wfile.write(body)

    def _send_static(self, path, ctype, missing_message):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            raise ApiError(404, missing_message)
        self._send(200, body, ctype)

    def _read_json_object(self):
        length = self.headers.get("Content-Length", "")
        if not length.isdigit():
            raise ApiError(400, "Content-Length fehlt oder ist ungültig.")
        size = int(length)
        if size > MAX_BODY_BYTES:
            raise ApiError(400, "Der Request-Body ist zu groß.")
        raw = self.rfile.read(size) if size else b""
        try:
            doc = json.loads(raw.decode("utf-8"))
        except ValueError:  # includes UnicodeDecodeError
            raise ApiError(400, "Ungültiges JSON.")
        if not isinstance(doc, dict):
            raise ApiError(400, "Ein JSON-Objekt wird erwartet.")
        return doc

    def _send_recording(self, raw_name, query):
        name = unquote(raw_name)
        if not NAME_RE.fullmatch(name):
            raise ApiError(404, "Aufnahme nicht gefunden.")
        path = os.path.join(self.server.args.recordings, name)
        try:
            if not stat.S_ISREG(os.lstat(path).st_mode):  # rejects symlinks and directories
                raise ApiError(404, "Aufnahme nicht gefunden.")
            handle = open(path, "rb")
        except OSError:
            raise ApiError(404, "Aufnahme nicht gefunden.")
        with handle:
            size = os.fstat(handle.fileno()).st_size
            ctype = "video/x-matroska" if name.endswith(".mkv") else "video/mp4"
            disposition = "attachment" if query.get("download") == ["1"] else "inline"
            extra = [
                ("Accept-Ranges", "bytes"),
                ("Content-Disposition", f'{disposition}; filename="{name}"'),
            ]
            start, end, code = 0, size - 1, 200
            requested = self.headers.get("Range")
            if requested is not None:
                parsed = parse_range(requested, size)
                if parsed is RANGE_UNSATISFIABLE:
                    self._send(416, json_bytes({"error": "Ungültiger Byte-Bereich."}), JSON_TYPE,
                               extra + [("Content-Range", f"bytes */{size}")])
                    return
                if parsed is not None:
                    start, end = parsed
                    code = 206
                    extra.append(("Content-Range", f"bytes {start}-{end}/{size}"))
            length = max(0, end - start + 1)
            self.responded = True
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            for key, value in extra:
                self.send_header(key, value)
            self.end_headers()
            if self.head_only or length == 0:
                return
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(STREAM_CHUNK, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class DashboardServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, args, controller):
        self.args = args
        self.controller = controller
        super().__init__(("0.0.0.0", args.port), DashboardHandler)

    def server_bind(self):
        # HTTPServer would reverse-resolve 0.0.0.0; there is no DNS on the pitch, so skip it.
        socketserver.TCPServer.server_bind(self)
        self.server_name = self.server_address[0]
        self.server_port = self.server_address[1]

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionError, TimeoutError)):
            return
        sys.stderr.write(f"connection error from {client_address[0]}: {exc!r}\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Web dashboard for the reco panoramic camera recorder.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument("--reco", default="/home/arduino/bin/reco", help="reco binary")
    parser.add_argument("--calibration", default="/home/arduino/match2.json",
                        help="calibration JSON passed to reco as -c")
    parser.add_argument("--recordings", default="/home/arduino/recordings",
                        help="recordings directory (created if missing)")
    parser.add_argument("--runtime-dir", default="/dev/shm/reco-dashboard",
                        help="tmpfs runtime directory, passed to reco as --dashboard-dir; also holds reco.log")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.recordings, exist_ok=True)
    os.makedirs(args.runtime_dir, exist_ok=True)
    controller = RecoController(args)
    try:
        server = DashboardServer(args, controller)
    except OSError as exc:
        sys.stderr.write(f"cannot listen on port {args.port}: {exc}\n")
        return 1

    exiting = threading.Event()

    def graceful_exit():
        controller.shutdown()
        server.shutdown()

    def on_signal(signum, frame):
        if exiting.is_set():  # a second signal: stop waiting for reco
            sys.stderr.write("second signal: killing reco and exiting now\n")
            controller.kill_now()
            os._exit(1)
        exiting.set()
        sys.stderr.write(f"{signal.Signals(signum).name}: stopping the recording and the dashboard\n")
        threading.Thread(target=graceful_exit, name="dashboard-exit").start()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # a dropped SSH session must not stop a recording

    sys.stderr.write(f"reco dashboard on port {args.port}; recordings: {args.recordings}; "
                     f"runtime: {args.runtime_dir}\n")
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
