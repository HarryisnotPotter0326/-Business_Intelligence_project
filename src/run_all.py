"""一次跑完剩下的所有實驗（適合睡前執行）。

- 依序執行下面 STAGES 的每一段；每段跑完都會更新 自動結果摘要.md 與 logs/run_all_state.json
- 某段失敗會自動重試（若有備用設定），再失敗就記錄下來、繼續跑下一段
- 中斷後重新執行，會跳過已完成的段落，從沒完成的地方接著跑
- 執行期間阻止 Windows 進入睡眠

用法（在專案資料夾）：
  .venv/Scripts/python.exe src/run_all.py           # 全部跑
  .venv/Scripts/python.exe src/run_all.py --list    # 只列出階段與狀態
  .venv/Scripts/python.exe src/run_all.py --only rq2_llm_zs   # 只跑某幾段（可多個）
  .venv/Scripts/python.exe src/run_all.py --redo    # 忽略之前的狀態，全部重跑
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
STATE = LOGS / "run_all_state.json"
PY = sys.executable

# (代號, 說明, 指令參數, 預估分鐘, 失敗時的備用參數)
FT = "src/rq1_finetune.py"
LLM = "src/rq2_llm.py"
STAGES = [
    ("rq1_ft_base_mbert", "RQ1 微調 BERTimbau Base + m-BERT（各 3 種子）", [FT, "bertimbau-base", "mbert"], 10, None),
    ("rq1_ft_large", "RQ1 微調 BERTimbau Large（3 種子）", [FT, "bertimbau-large"], 20,
     [FT, "bertimbau-large", "--batch", "16"]),
    ("rq2_curve_classic", "RQ2 學習曲線：TF-IDF 與凍結 BERT", ["src/rq2_learning_curve.py"], 8, None),
    *[(f"rq2_curve_ft_{n}", f"RQ2 學習曲線：微調 Base，{n} 筆（3 epoch × 3 種子）",
       [FT, "bertimbau-base", "--train_size", str(n), "--epochs", "3", "--out", "rq2_finetune.csv"], m, None)
      for n, m in [(1000, 2), (5000, 5), (10000, 8), (30464, 12)]],
    ("rq2_llm_zs", "RQ2 LLM 零樣本：Qwen2.5 0.5B 與 1.5B", None, 5, None),
    ("rq2_lora_0.5b", "RQ2 LoRA：Qwen2.5-0.5B，全部訓練資料", [LLM, "qwen-0.5b", "--mode", "lora"], 10, None),
    *[(f"rq2_lora_1.5b_{n}", f"RQ2 LoRA：Qwen2.5-1.5B，{n} 筆",
       [LLM, "qwen-1.5b", "--mode", "lora", "--train_size", str(n)], m,
       [LLM, "qwen-1.5b", "--mode", "lora", "--train_size", str(n), "--batch", "8"])
      for n, m in [(1000, 3), (5000, 6), (30464, 25)]],
]
# 有些階段要跑多個指令
MULTI = {"rq2_llm_zs": [[LLM, "qwen-0.5b", "--mode", "zeroshot"], [LLM, "qwen-1.5b", "--mode", "zeroshot"]]}


def keep_awake(on=True):
    """阻止 Windows 睡眠（ES_CONTINUOUS | ES_SYSTEM_REQUIRED）。"""
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
    except Exception:
        pass


def load_state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def save_state(st):
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def newest_log(script, since):
    stem = Path(script).stem
    logs = [p for p in LOGS.glob(f"{stem}_*.log") if p.stat().st_mtime >= since]
    return [str(p.relative_to(ROOT)) for p in sorted(logs, key=lambda p: p.stat().st_mtime)]


def run_cmd(args):
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
    return subprocess.run([PY, *args], cwd=ROOT, env=env).returncode


def summarize():
    sys.path.insert(0, str(ROOT / "src"))
    try:
        import importlib
        import summarize as s
        importlib.reload(s)
        s.main()
    except Exception as e:
        print(f"[run_all] 更新摘要失敗：{e!r}")


def banner(text):
    line = "=" * 100
    print(f"\n{line}\n{text}\n{line}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()

    st = {} if a.redo else load_state()
    todo = [s for s in STAGES if (not a.only or s[0] in a.only)]
    if a.list:
        for sid, title, _, est, _ in todo:
            print(f"{st.get(sid, {}).get('status', '未執行'):8s} {sid:22s} 約 {est:3d} 分  {title}")
        return

    pending = [s for s in todo if a.only or st.get(s[0], {}).get("status") != "完成"]
    total_est = sum(s[3] for s in pending)
    banner(f"[run_all] 共 {len(pending)} 段，預估約 {total_est} 分鐘（{total_est / 60:.1f} 小時）。"
           f"\n每段完成後會更新：自動結果摘要.md、logs/{STATE.name}")
    keep_awake(True)
    t_all = time.time()
    try:
        for k, (sid, title, args, est, fallback) in enumerate(pending, 1):
            banner(f"[run_all] 第 {k}/{len(pending)} 段：{sid}  {title}（預估 {est} 分）  開始 {time.strftime('%H:%M:%S')}")
            t0 = time.time()
            st[sid] = dict(title=title, status="執行中", start=time.strftime("%m-%d %H:%M"))
            save_state(st)
            cmds = MULTI.get(sid, [args])
            ok = all(run_cmd(c) == 0 for c in cmds)
            if not ok and fallback:
                print(f"[run_all] {sid} 失敗，改用備用設定重試：{' '.join(fallback)}", flush=True)
                ok = run_cmd(fallback) == 0
            minutes = round((time.time() - t0) / 60, 1)
            logs = sorted({l for c in cmds + ([fallback] if fallback else []) for l in newest_log(c[0], t0)})
            st[sid].update(status="完成" if ok else "失敗", minutes=minutes, log="；".join(logs))
            save_state(st)
            summarize()
            print(f"[run_all] {sid} {'完成' if ok else '失敗（繼續下一段）'}，花了 {minutes} 分；"
                  f"已更新 自動結果摘要.md", flush=True)
    except KeyboardInterrupt:
        print("\n[run_all] 手動中斷。下次執行會從沒完成的段落接著跑。")
        for v in st.values():
            if v["status"] == "執行中":
                v["status"] = "中斷"
        save_state(st)
        summarize()
    finally:
        keep_awake(False)
    done = sum(v["status"] == "完成" for v in st.values())
    failed = [k for k, v in st.items() if v["status"] == "失敗"]
    banner(f"[run_all] 結束，總共 {(time.time() - t_all) / 60:.0f} 分鐘。完成 {done} 段"
           + (f"，失敗：{', '.join(failed)}" if failed else "，全部成功") + "。\n請看 自動結果摘要.md")


if __name__ == "__main__":
    main()
