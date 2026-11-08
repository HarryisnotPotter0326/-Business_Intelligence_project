"""即時資源監控：給 tqdm 進度條顯示 CPU、記憶體、GPU 使用狀況。

log() 印出的每一行也會同時寫進 logs/<腳本名>_<日期時間>.log。
"""
import sys
import time
from pathlib import Path

import psutil
from tqdm.auto import tqdm

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _GPU = None

_PROC = psutil.Process()
psutil.cpu_percent(None)  # 第一次呼叫只是初始化


def usage():
    """回傳 dict，例如 {'CPU': '35%', 'RAM': '4.1G', 'GPU': '98%', 'VRAM': '3.2/11.0G'}。"""
    info = {
        "CPU": f"{psutil.cpu_percent(None):.0f}%",
        "RAM": f"{_PROC.memory_info().rss / 2**30:.1f}G",
    }
    if _GPU is not None:
        mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU)
        info["GPU"] = f"{pynvml.nvmlDeviceGetUtilizationRates(_GPU).gpu}%"
        info["VRAM"] = f"{mem.used / 2**30:.1f}/{mem.total / 2**30:.1f}G"
        info["TEMP"] = f"{pynvml.nvmlDeviceGetTemperature(_GPU, 0)}C"
    return info


def usage_str():
    return "  ".join(f"{k} {v}" for k, v in usage().items())


class Bar(tqdm):
    """會自動在右側顯示資源使用量的進度條（每 1 秒更新一次）。"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("dynamic_ncols", True)
        super().__init__(*args, **kwargs)
        self._last = 0.0
        self.extra = {}  # 額外要顯示的數字，例如 {"loss": "0.231"}

    def update(self, n=1):
        now = time.time()
        if now - self._last > 1.0:
            self.set_postfix({**self.extra, **usage()}, refresh=False)
            self._last = now
        return super().update(n)


_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
_LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = _LOG_DIR / f"{Path(sys.argv[0]).stem}_{time.strftime('%Y%m%d_%H%M%S')}.log"


def _write_file(text):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def log(msg):
    """印出一行訊息（不打斷進度條），開頭加上時間與資源使用量，並寫入記錄檔。"""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}    ({usage_str()})"
    tqdm.write(line)
    _write_file(line)


def log_text(text):
    """原樣印出多行文字（例如結果表格），並寫入記錄檔。"""
    tqdm.write(text)
    _write_file(text)


_write_file(f"# 指令：{' '.join(sys.argv)}\n# 開始：{time.strftime('%Y-%m-%d %H:%M:%S')}")


def _excepthook(etype, value, tb):
    """程式出錯時，把完整錯誤訊息也寫進記錄檔。"""
    import traceback
    _write_file("\n# 錯誤：\n" + "".join(traceback.format_exception(etype, value, tb)))
    sys.__excepthook__(etype, value, tb)


sys.excepthook = _excepthook
