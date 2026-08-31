import sys
import time
import threading
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class ResourceMonitor:
    """
    Background resource monitor for CPU %, Peak Process RAM, RAM Delta, and System Memory.
    Designed for profiling any module (voice_app, summarizer, LLM, DB, etc.).
    """

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self.cpu_samples = []
        self.ram_samples = []
        self.running = False
        self.thread = None
        if HAS_PSUTIL:
            self.process = psutil.Process()
            # Prime cpu_percent initial sample
            self.process.cpu_percent(interval=None)
        else:
            self.process = None

    def start(self):
        if not HAS_PSUTIL or self.running:
            return
        self.cpu_samples.clear()
        self.ram_samples.clear()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                cpu = self.process.cpu_percent(interval=self.interval)
                ram = self.process.memory_info().rss / (1024 * 1024)  # MB
                self.cpu_samples.append(cpu)
                self.ram_samples.append(ram)
            except Exception:
                break

    def stop(self) -> dict:
        if not HAS_PSUTIL:
            return {}
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        max_cpu = max(self.cpu_samples) if self.cpu_samples else 0.0
        peak_ram = max(self.ram_samples) if self.ram_samples else 0.0
        min_ram = min(self.ram_samples) if self.ram_samples else 0.0
        sys_ram = psutil.virtual_memory()

        return {
            "avg_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "peak_ram_mb": peak_ram,
            "ram_delta_mb": peak_ram - min_ram,
            "sys_ram_percent": sys_ram.percent,
            "sys_ram_used_mb": sys_ram.used / (1024 * 1024),
            "sys_ram_total_mb": sys_ram.total / (1024 * 1024),
        }


def get_audio_duration(file_path: str) -> float:
    """Helper to retrieve audio file duration in seconds."""
    try:
        info = sf.info(file_path)
        return info.duration
    except Exception:
        return 0.0


def get_process_memory_mb() -> float:
    """Returns current process RAM usage in MB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    return 0.0


class PerformanceProfiler:
    """
    Context manager / helper class for timing and profiling execution metrics.
    Can be used for any workload (Whisper, LLM, DB queries, etc.).
    """

    def __init__(self, name: str = "Task", audio_path: str = None):
        self.name = name
        self.audio_path = audio_path
        self.monitor = ResourceMonitor()
        self.start_time = 0.0
        self.elapsed_time = 0.0
        self.initial_ram_mb = 0.0
        self.results = {}

    def __enter__(self):
        self.initial_ram_mb = get_process_memory_mb()
        self.monitor.start()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.time() - self.start_time
        res_stats = self.monitor.stop()
        final_ram_mb = get_process_memory_mb()

        audio_dur = get_audio_duration(self.audio_path) if self.audio_path else 0.0

        self.results = {
            "task_name": self.name,
            "elapsed_seconds": self.elapsed_time,
            "audio_duration_seconds": audio_dur,
            "speed_factor": (audio_dur / self.elapsed_time) if (audio_dur > 0 and self.elapsed_time > 0) else 0.0,
            "rtf": (self.elapsed_time / audio_dur) if audio_dur > 0 else 0.0,
            "initial_ram_mb": self.initial_ram_mb,
            "final_ram_mb": final_ram_mb,
            "ram_allocated_mb": max(0.0, final_ram_mb - self.initial_ram_mb),
            **res_stats
        }

    def print_summary(self):
        print(f"\n----------- PERFORMANCE METRICS [{self.name}] -----------", flush=True)
        if self.results.get("audio_duration_seconds", 0) > 0:
            dur = self.results["audio_duration_seconds"]
            print(f"Audio Duration   : {dur:.2f} seconds", flush=True)
        print(f"Execution Time   : {self.results.get('elapsed_seconds', 0.0):.2f} seconds", flush=True)

        if self.results.get("speed_factor", 0) > 0:
            sf_val = self.results["speed_factor"]
            rtf_val = self.results["rtf"]
            print(f"Speed Factor     : {sf_val:.2f}x Real-time (RTF: {rtf_val:.3f})", flush=True)

        if HAS_PSUTIL and self.results:
            print("\n----------- RESOURCE USAGE (CPU & RAM) -----------", flush=True)
            print(f"Initial Process RAM: {self.results.get('initial_ram_mb', 0):.1f} MB", flush=True)
            print(f"Peak Process RAM   : {self.results.get('peak_ram_mb', 0):.1f} MB", flush=True)
            print(f"RAM Delta / Alloc  : ~{self.results.get('ram_allocated_mb', 0):.1f} MB", flush=True)
            print(f"Avg CPU Usage      : {self.results.get('avg_cpu_percent', 0):.1f}% (across active cores)", flush=True)
            print(f"Max CPU Peak       : {self.results.get('max_cpu_percent', 0):.1f}%", flush=True)
            print(f"System RAM Used    : {self.results.get('sys_ram_used_mb', 0):.0f} MB / {self.results.get('sys_ram_total_mb', 0):.0f} MB ({self.results.get('sys_ram_percent', 0)}%)", flush=True)
        else:
            print("\n[INFO] Install 'psutil' to view detailed process CPU & RAM metrics (pip install psutil)", flush=True)
        print("--------------------------------------------------", flush=True)
