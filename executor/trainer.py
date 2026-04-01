import os
import re
import subprocess
import sys
import threading
import time
import signal
from config import Config

class Trainer:
    PROGRESS_RE = re.compile(
        r"(?:[\u2501\u2578\u2579\u257a\u257b\u254b\u2503\u2500\u2580\u2584\u2588]"
        r"|\b\d+/\d+\s+\d+\.\d+it/s"
        r"|\d+%\s*[|\u2500-\u259f])"
    )
    TERM_WIDTH = 120

    def __init__(self, num_gpus: int, cuda_devices: str = None, quiet: bool = False):
        if cuda_devices:
            self.num_gpus = num_gpus or len(cuda_devices.split(","))
        else:
            self.num_gpus = num_gpus or self._detect_gpus()
        self.quiet = quiet

    def _detect_gpus(self) -> int:
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL)
            return max(len([l for l in out.strip().splitlines() if l.strip()]), 1)
        except Exception:
            return 1

    def _build_cmd(self) -> list[str]:
        py = sys.executable
        train = str(Config.TRAIN_FILE)
        if self.num_gpus <= 1:
            return [py, "-u", train]
        return [py, "-u", "-m", "torch.distributed.run", f"--nproc_per_node={self.num_gpus}", "--master_port=29500", train]

    def _force_unbuffered_stdout(self):
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                try: stream.reconfigure(line_buffering=True)
                except: pass

    def _kill_process_tree(self, proc: subprocess.Popen):
        """Forces the process and all zombie child workers to release GPU memory."""
        try:
            if sys.platform != "win32":
                # Kill the entire process group (Linux/Mac)
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                # Kill process tree (Windows)
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        except Exception:
            proc.kill()

    def run(self, timeout: float = None) -> tuple[bool, float]:
        cmd = self._build_cmd()
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}

        # IMPORTANT: Create a new process group so we can cleanly kill all zombie workers
        popen_kwargs = {}
        if sys.platform != "win32":
            popen_kwargs["start_new_session"] = True

        if self.quiet:
            t0 = time.time()
            proc = subprocess.Popen(cmd, cwd=str(Config.ROOT_DIR), stdout=open(Config.LOG_FILE, "w"), stderr=subprocess.STDOUT, env=env, **popen_kwargs)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._kill_process_tree(proc)
                return False, time.time() - t0
            finally:
                self._kill_process_tree(proc) # Ensure VRAM is cleared even on success
            return proc.returncode == 0, time.time() - t0

        self._force_unbuffered_stdout()
        with open(Config.LOG_FILE, "w", buffering=1) as log_f:
            proc = subprocess.Popen(cmd, cwd=str(Config.ROOT_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, **popen_kwargs)
            return self._stream_process(proc, log_f, timeout)

    def _stream_process(self, proc: subprocess.Popen, log_f, timeout: float | None) -> tuple[bool, float]:
        t0 = time.time()
        timed_out, on_prog_line = False, False

        def _watchdog():
            nonlocal timed_out
            if timeout:
                time.sleep(timeout)
                if proc.poll() is None:
                    timed_out = True
                    self._kill_process_tree(proc)

        if timeout: threading.Thread(target=_watchdog, daemon=True).start()

        try:
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                
                if bool(line.strip() and self.PROGRESS_RE.search(line.strip())):
                    display = line.rstrip("\n\r")[:self.TERM_WIDTH].ljust(self.TERM_WIDTH)
                    sys.stdout.write("\r" + display)
                    sys.stdout.flush()
                    on_prog_line = True
                else:
                    if on_prog_line:
                        sys.stdout.write("\n")
                        on_prog_line = False
                    sys.stdout.write(line)
                    sys.stdout.flush()
        except BrokenPipeError: pass

        if on_prog_line:
            sys.stdout.write("\n")
            sys.stdout.flush()

        rc = proc.wait()
        
        # Absolute guarantee to clear GPU VRAM after experiment ends
        self._kill_process_tree(proc)
        
        elapsed = time.time() - t0
        if timed_out: print(f"\n  [TIMEOUT] killed after {timeout:.0f}s")
        return rc == 0, elapsed