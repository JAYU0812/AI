"""Step 6D — Resource Monitoring.

Tracks CPU and RAM usage during training on CPU-only hardware.
Runs in a background thread and writes periodic snapshots.
"""

from __future__ import annotations

import json
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_memory_mb() -> float:
    """Get current process RSS memory in MB (Windows compatible)."""
    try:
        import psutil
        proc = psutil.Process()
        return proc.memory_info().rss / (1024 ** 2)
    except ImportError:
        pass

    try:
        # Windows fallback using ctypes
        import ctypes
        import ctypes.wintypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize / (1024 ** 2)
    except Exception:
        return -1.0


def _get_cpu_percent() -> float:
    """Get approximate CPU percent (cross-platform)."""
    try:
        import psutil
        return psutil.cpu_percent(interval=0.1)
    except ImportError:
        return -1.0


class ResourceMonitor:
    """Background resource monitor for CPU/RAM during training.

    Usage:
        monitor = ResourceMonitor(interval_sec=10)
        monitor.start()
        # ... training loop ...
        monitor.stop()
        summary = monitor.get_summary()
    """

    def __init__(self, interval_sec: float = 10.0, output_path: Optional[str] = None):
        self.interval_sec = interval_sec
        self.output_path = output_path
        self._samples: List[Dict[str, Any]] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_time: float = 0.0
        self._peak_ram_mb: float = 0.0

    def start(self) -> None:
        """Start background monitoring thread."""
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background monitoring thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _monitor_loop(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            elapsed = time.time() - self._start_time
            ram = _get_memory_mb()
            cpu = _get_cpu_percent()
            sample = {
                "elapsed_sec": round(elapsed, 1),
                "ram_mb": round(ram, 1),
                "cpu_percent": round(cpu, 1),
            }
            self._samples.append(sample)
            if ram > self._peak_ram_mb:
                self._peak_ram_mb = ram

    def record_event(self, label: str) -> None:
        """Record a named event with current resource snapshot."""
        elapsed = time.time() - self._start_time
        ram = _get_memory_mb()
        cpu = _get_cpu_percent()
        self._samples.append({
            "elapsed_sec": round(elapsed, 1),
            "event": label,
            "ram_mb": round(ram, 1),
            "cpu_percent": round(cpu, 1),
        })

    def get_summary(self) -> Dict[str, Any]:
        """Return resource usage summary."""
        ram_values = [s["ram_mb"] for s in self._samples if s.get("ram_mb", -1) >= 0]
        cpu_values = [s["cpu_percent"] for s in self._samples if s.get("cpu_percent", -1) >= 0]
        total_sec = time.time() - self._start_time if self._start_time > 0 else 0.0

        return {
            "total_duration_sec": round(total_sec, 1),
            "total_duration_hms": _format_duration(total_sec),
            "peak_ram_mb": round(max(ram_values) if ram_values else 0, 1),
            "mean_ram_mb": round(sum(ram_values) / len(ram_values) if ram_values else 0, 1),
            "mean_cpu_percent": round(sum(cpu_values) / len(cpu_values) if cpu_values else 0, 1),
            "samples": len(self._samples),
        }

    def save(self, path: Optional[str] = None) -> str:
        """Save all samples and summary to JSON."""
        p = Path(path or self.output_path or "resource_monitor.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "summary": self.get_summary(),
            "samples": self._samples,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return str(p)


def _format_duration(seconds: float) -> str:
    """Format seconds as H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def get_environment_info() -> Dict[str, Any]:
    """Collect reproducibility environment information (Step 6D.27)."""
    import importlib

    def pkg_version(name: str) -> str:
        try:
            return importlib.import_module(name).__version__
        except (ImportError, AttributeError):
            return "not installed"

    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count()
        cuda_ver = torch.version.cuda if cuda_avail else None
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "Intel UHD Graphics 620 (no CUDA)"
        torch_ver = torch.__version__
    except ImportError:
        cuda_avail = False
        cuda_count = 0
        cuda_ver = None
        gpu_name = "Intel UHD Graphics 620 (no CUDA)"
        torch_ver = "not installed"

    return {
        "os": f"{platform.system()} {platform.release()} {platform.version()}",
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "ram_gb": _get_ram_gb(),
        "cuda_available": cuda_avail,
        "cuda_device_count": cuda_count,
        "cuda_version": cuda_ver,
        "gpu": gpu_name,
        "acceleration": "CUDA" if cuda_avail else "CPU_ONLY",
        "torch": torch_ver,
        "transformers": pkg_version("transformers"),
        "peft": pkg_version("peft"),
        "accelerate": pkg_version("accelerate"),
        "datasets": pkg_version("datasets"),
        "tokenizers": pkg_version("tokenizers"),
    }


def _get_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    try:
        # Windows ctypes fallback
        import ctypes
        kernel32 = ctypes.windll.kernel32
        status = ctypes.c_ulonglong()
        kernel32.GlobalMemoryStatusEx.restype = ctypes.c_bool
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return round(stat.ullTotalPhys / (1024 ** 3), 1)
    except Exception:
        return 17052073984 / (1024 ** 3)  # Known from hardware inspection
