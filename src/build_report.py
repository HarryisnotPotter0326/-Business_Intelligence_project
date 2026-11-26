"""把 report/*.md（簡易標記）轉成學術格式的 Word 報告，再用 LibreOffice 轉 PDF，並回填目錄頁碼。

標記規則（每章一個檔案，依檔名排序）：
  # 第一章　緒論          第 1 層標題（新的一頁開始）
  #! AI 工具使用聲明       不編號的第 1 層標題
  ## 1.1　研究背景         第 2 層標題
  ### 1.1.1　…             第 3 層標題
  一般段落                 連續多行合併成一段；空行分段；首行縮排 2 字
  - 項目                   項目符號
  （1）…／1. …             編號段落（保留原本的編號文字，懸掛縮排）
  > TABLE 表 5-1　標題      接著是 Markdown 表格（三線表）；欄位對齊依 :--- / :---: / ---:
  > NOTE 註：…              表或圖下方的註（小字）
  > FIG 路徑 | 寬度cm | 圖 5-1　標題
  > EQ 公式 | (3-1)
  > GEN 名稱                由程式從 results/ 產生的附錄表格（見 GEN_TABLES）
  > REF 參考文獻一筆（*斜體*）
  > PAGEBREAK
行內：**粗體**、*斜體*、`程式碼`、[文字](網址)（可點的超連結）

用法：python src/build_report.py              （產生 Word ＋ PDF）
      python src/build_report.py --no-pdf     （只產生 Word）
      python src/build_report.py --src report_精簡版 --out 報告/精簡版 --name 商業智慧期中報告_Olist評論情感分析與負評決策_精簡版
                                              （精簡版）
      加上 --github 網址，封面會多一行可點的 GitHub 連結（目前只有精簡版使用）。
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report"
OUT = ROOT / "報告"
RESULTS = ROOT / "results"
NAME = "商業智慧期中報告_Olist評論情感分析與負評決策"
WINWORD = [Path(r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"),
           Path(r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINWORD.EXE")]
SOFFICE = [Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
           Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe")]

ZH, EN, MONO = "標楷體", "Times New Roman", "Consolas"
TEXT_W = 15.92  # A4 寬 21 cm − 左右邊界 2.54 cm × 2
TITLE_ZH = "以預訓練語言模型重現與延伸電商評論情感分析：\nOlist 巴西電商的負評成因、預警與改善決策"
TITLE_EN = ("Reproducing and Extending Sentiment Analysis of E-commerce Reviews with Pre-trained Language Models: "
            "Causes, Early Warning, and Improvement Decisions for Negative Reviews at Olist")
SHORT_TITLE = "以預訓練語言模型重現與延伸電商評論情感分析"
# 封面資料
COVER_INFO = [("授課教師", "陳彥良"), ("系所", "資訊管理學系碩士班"), ("學號", "115423029"), ("姓名", "羅傳勛")]
GITHUB = None  # --github 給的網址；有給才印在封面
LOGO = SRC / "ncu_logo.png"  # 國立中央大學校徽（Wikimedia Commons：File:NCULogo.svg，公有領域）


# ============================================================ 基本格式工具
def set_run_font(run, size=None, bold=None, italic=None, zh=ZH, en=EN, color=None):
    f = run.font
    f.name = en
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for k in ["w:ascii", "w:hAnsi", "w:cs"]:
        rfonts.set(qn(k), en)
    rfonts.set(qn("w:eastAsia"), zh)
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color:
        f.color.rgb = RGBColor.from_string(color)


def style_font(style, size, bold=False, zh=ZH, en=EN):
    style.font.name = en
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for k in ["w:ascii", "w:hAnsi", "w:cs"]:
        rfonts.set(qn(k), en)
    rfonts.set(qn("w:eastAsia"), zh)
    for k in ["w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"]:
        if rfonts.get(qn(k)) is not None:
            del rfonts.attrib[qn(k)]


def para_fmt(p, align=None, first=None, left=None, hanging=None, before=0, after=0, spacing=1.5, keep_next=None):
    pf = p.paragraph_format
    if align is not None:
        pf.alignment = align
    if first is not None:
        pf.first_line_indent = Pt(first)
    if left is not None:
        pf.left_indent = Pt(left)
    if hanging is not None:
        pf.left_indent = Pt((left or 0) + hanging)
        pf.first_line_indent = Pt(-hanging)
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = spacing
    if keep_next is not None:
        pf.keep_with_next = keep_next


def setup_styles(doc):
    st = doc.styles
    normal = st["Normal"]
    style_font(normal, 12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(0)
    # 東亞文字與英數字之間不自動加空白、標點壓縮
    for name, size, before, after, align in [("Heading 1", 16, 0, 18, WD_ALIGN_PARAGRAPH.CENTER),
                                             ("Heading 2", 14, 12, 6, WD_ALIGN_PARAGRAPH.LEFT),
                                             ("Heading 3", 12, 8, 4, WD_ALIGN_PARAGRAPH.LEFT)]:
        s = st[name]
        style_font(s, size, bold=True)
        pf = s.paragraph_format
        pf.space_before, pf.space_after, pf.alignment = Pt(before), Pt(after), align
        pf.line_spacing = 1.3
        pf.keep_with_next = True
        pf.first_line_indent = Pt(0)
    from docx.enum.style import WD_STYLE_TYPE
    for name, size, align in [("TableCaption", 11, WD_ALIGN_PARAGRAPH.CENTER), ("FigureCaption", 11, WD_ALIGN_PARAGRAPH.CENTER),
                              ("TableText", 10, WD_ALIGN_PARAGRAPH.CENTER), ("NoteText", 9.5, WD_ALIGN_PARAGRAPH.LEFT),
                              ("TOCLevel1", 12, WD_ALIGN_PARAGRAPH.LEFT), ("TOCLevel2", 12, WD_ALIGN_PARAGRAPH.LEFT),
                              ("TOCLevel3", 11, WD_ALIGN_PARAGRAPH.LEFT), ("RefText", 12, WD_ALIGN_PARAGRAPH.LEFT)]:
        s = st.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = normal
        style_font(s, size, bold=name == "TableCaption" or name == "FigureCaption")
        s.paragraph_format.alignment = align
        s.paragraph_format.first_line_indent = Pt(0)
        s.paragraph_format.line_spacing = 1.0 if name in ("TableText", "NoteText") else 1.5
    st["TableCaption"].paragraph_format.keep_with_next = True
    st["TableCaption"].paragraph_format.space_before = Pt(10)
    st["TableCaption"].paragraph_format.space_after = Pt(4)
    st["FigureCaption"].paragraph_format.space_after = Pt(12)
    st["NoteText"].paragraph_format.space_before = Pt(3)
    st["NoteText"].paragraph_format.space_after = Pt(8)
    for n in ["TableCaption", "FigureCaption"]:
        st[n].font.bold = False
    lb = st["List Bullet"]
    style_font(lb, 12)
    lb.paragraph_format.line_spacing = 1.5


def add_field(p, instr, placeholder=""):
    """在段落中插入 Word 欄位（例如 PAGE）。"""
    r = p.add_run()
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "begin")
    r._element.append(fc)
    r2 = p.add_run()
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = f" {instr} "
    r2._element.append(it)
    r3 = p.add_run()
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "separate")
    r3._element.append(fc)
    r4 = p.add_run(placeholder)
    r5 = p.add_run()
    fc = OxmlElement("w:fldChar")
    fc.set(qn("w:fldCharType"), "end")
    r5._element.append(fc)
    return [r, r2, r3, r4, r5]


def set_pgnum(section, fmt, start=1):
    sectPr = section._sectPr
    pg = sectPr.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType")
        sectPr.append(pg)
    pg.set(qn("w:fmt"), fmt)
    pg.set(qn("w:start"), str(start))


def setup_section(section):
    section.page_height, section.page_width = Cm(29.7), Cm(21.0)
    section.top_margin = section.bottom_margin = Cm(2.54)
    section.left_margin = section.right_margin = Cm(2.54)
    section.header_distance = section.footer_distance = Cm(1.5)


def footer_page_number(section):
    f = section.footer
    f.is_linked_to_previous = False
    p = f.paragraphs[0] if f.paragraphs else f.add_paragraph()
    p.text = ""
    para_fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0)
    for r in add_field(p, "PAGE", "1"):
        set_run_font(r, size=10)


def header_text(section, text):
    h = section.header
    h.is_linked_to_previous = False
    p = h.paragraphs[0] if h.paragraphs else h.add_paragraph()
    p.text = ""
    para_fmt(p, align=WD_ALIGN_PARAGRAPH.RIGHT, spacing=1.0)
    if text:
        r = p.add_run(text)
        set_run_font(r, size=9, color="595959")
        pPr = p._p.get_or_add_pPr()
        bdr = OxmlElement("w:pBdr")
        b = OxmlElement("w:bottom")
        for k, v in [("w:val", "single"), ("w:sz", "4"), ("w:space", "1"), ("w:color", "808080")]:
            b.set(qn(k), v)
        bdr.append(b)
        pPr.append(bdr)


# ============================================================ 行內格式
INLINE = re.compile(r"(\[[^\]]+\]\([^)\s]+\)|\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
REL_LINK = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"


def add_link(p, text, url, size=None):
    """可點的超連結：藍字加底線，轉成 PDF 後仍可點。"""
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), p.part.relate_to(url, REL_LINK, is_external=True))
    r = p.add_run(text)
    set_run_font(r, size=size, color="1F4E9A")
    r.font.underline = True
    h.append(r._element)  # 把 run 移進 hyperlink
    p._p.append(h)


def add_inline(p, text, size=None, bold=False):
    for part in INLINE.split(text):
        if not part:
            continue
        m = LINK.fullmatch(part)
        if m:
            add_link(p, m.group(1), m.group(2), size)
        elif part.startswith("**"):
            r = p.add_run(part[2:-2])
            set_run_font(r, size=size, bold=True)
        elif part.startswith("*"):
            r = p.add_run(part[1:-1])
            set_run_font(r, size=size, italic=True, bold=bold or None)
        elif part.startswith("`"):
            r = p.add_run(part[1:-1])
            set_run_font(r, size=(size or 12) - 1.5, en=MONO, bold=bold or None)
        else:
            r = p.add_run(part)
            set_run_font(r, size=size, bold=bold or None)


def disp_len(s):
    s = re.sub(r"\*\*|`", "", str(s))
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def min_len(s, cjk_keep=8):
    """儲存格裡不能斷行的最小寬度（半形字數），回傳 (硬下限, 軟下限)。
    硬下限：沒有漢字的字串（數字、百分比、英文單字）不能斷；超過 25 字的亂碼字串本來就會被強制斷開，只算 12。
    軟下限：含漢字的詞可在字間斷行，但 cjk_keep 個半形寬度以內盡量不斷。"""
    s = re.sub(r"\*\*|`", "", str(s))
    hard = soft = 0
    for tok in s.split():
        if re.search(r"\d", tok) and re.fullmatch(r"[\d.,%／/±+\-−–()\[\]]+", tok):  # 數字（含 a／b）整段不斷
            hard = max(hard, disp_len(tok))
            continue
        for piece in re.split(r"[　-〿！-／：-？]", tok):  # 中文標點後可斷行
            if not piece:
                continue
            if not re.search(r"[㐀-鿿]", piece):
                L = disp_len(piece)
                hard = max(hard, L if L <= 25 else 12)
            else:
                latin = max((len(x) for x in re.findall(r"[^⺀-￿]+", piece)), default=0)
                hard = max(hard, latin)
                soft = max(soft, min(disp_len(piece), cjk_keep))
    return hard, max(hard, soft)


def col_widths(header, rows, font):
    """欄寬（cm）。依序嘗試：內容全放得下 → 中文欄縮到 4 字一行（表頭 5 字）→ 3 字一行（表頭 2 字）→ 只保留數字與英文單字；
    數字與英文單字永遠不壓縮。回傳 (欄寬, 是否不必降字級)。"""
    n = len(header)
    cells = [[header[j]] + [r[j] for r in rows] for j in range(n)]
    cm_per = font * 0.0185 * 1.08  # 每個半形字約占的寬度（cm）
    w = lambda L, k=1.06: min(L, 46) * cm_per * k + 0.35

    def level(head_keep, body_keep):
        out = []
        for col in cells:
            lim = [min_len(col[0], head_keep)] + [min_len(v, body_keep) for v in col[1:]]
            out.append(w(max(max(x) for x in lim)))
        return out

    hard = [w(max(max(min_len(v)[0] for v in col), 2)) for col in cells]
    l2 = [max(a, b) for a, b in zip(level(4, 6), hard)]
    l1 = [max(a, b) for a, b in zip(level(10, 8), l2)]
    need = [max(w(max(max(disp_len(v) for v in col), 3), 1.0), b) for col, b in zip(cells, l1)]
    hard[0], l2[0] = l1[0], l1[0]  # 第一欄是列標題，至少保留 4 個字寬
    levels = [need, l1, l2, hard]

    def fill(lo, hi):
        extra = TEXT_W - sum(lo)
        flex = [b - a for a, b in zip(lo, hi)]
        return [a + extra * f / sum(flex) for a, f in zip(lo, flex)] if sum(flex) > 0 else lo

    for k, lv in enumerate(levels):
        if sum(lv) <= TEXT_W:
            return (lv if k == 0 else fill(lv, levels[k - 1])), k <= 1
    return [x * TEXT_W / sum(hard) for x in hard], False


# ============================================================ 三線表
def border(el, edge, sz):
    b = OxmlElement(f"w:{edge}")
    if sz:
        for k, v in [("w:val", "single"), ("w:sz", str(sz)), ("w:space", "0"), ("w:color", "000000")]:
            b.set(qn(k), v)
    else:
        b.set(qn("w:val"), "nil")
    el.append(b)


def three_line_table(doc, header, rows, aligns=None, font=10, keep=True):
    n = len(header)
    aligns = aligns or (["l"] + ["c"] * (n - 1))
    t = doc.add_table(rows=len(rows) + 1, cols=n)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = t._tbl.tblPr
    bd = OxmlElement("w:tblBorders")
    for edge, sz in [("top", 12), ("left", 0), ("bottom", 12), ("right", 0), ("insideH", 0), ("insideV", 0)]:
        border(bd, edge, sz)
    tblPr.append(bd)
    mar = OxmlElement("w:tblCellMar")
    for edge, w in [("left", 85), ("right", 85), ("top", 20), ("bottom", 20)]:
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:w"), str(w))
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)
    # 欄寬：放不下時先把字級降為 9 號再算
    widths, ok = col_widths(header, rows, font)
    if not ok and font > 9:
        font = 9
        widths, ok = col_widths(header, rows, font)
    al = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT}
    for i, row in enumerate([header] + rows):
        tr = t.rows[i]
        if i == 0:
            trPr = tr._tr.get_or_add_trPr()
            th = OxmlElement("w:tblHeader")
            trPr.append(th)
        if keep and len(rows) <= 18:
            trPr = tr._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit")
            trPr.append(cs)
        for j, val in enumerate(row):
            c = tr.cells[j]
            c.width = Cm(widths[j])
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = c.paragraphs[0]
            p.style = doc.styles["TableText"]
            para_fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER if i == 0 else al[aligns[j]], spacing=1.1, before=1, after=1,
                     keep_next=(i < 3 or (keep and len(rows) <= 18 and i < len(rows))))
            add_inline(p, str(val), size=font, bold=(i == 0))
            if i == 0:
                tcPr = c._tc.get_or_add_tcPr()
                tb = OxmlElement("w:tcBorders")
                border(tb, "bottom", 6)
                tcPr.append(tb)
    for j, col in enumerate(t.columns):
        col.width = Cm(widths[j])
    # 固定欄寬：Word 預設會依內容自動調整欄寬，長的外文欄會把中文欄擠成一個字寬
    t.autofit = False
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW")
        tblPr.append(tblW)
    tblW.set(qn("w:w"), str(int(round(sum(widths) / 2.54 * 1440))))
    tblW.set(qn("w:type"), "dxa")
    return t


# ============================================================ 由 results/ 產生的附錄表
def fmt(x, d=2):
    if isinstance(x, str):
        return x
    if pd.isna(x):
        return "—"
    return f"{x:.{d}f}"


def gen_frozen():
    d = pd.read_csv(RESULTS / "rq1_frozen.csv")
    names = {"bertimbau-base": "BERTimbau Base", "bertimbau-large": "BERTimbau Large", "mbert": "m-BERT"}
    rows = []
    for strat in d.strategy.unique():
        r = [strat]
        for m in ["mbert", "bertimbau-base", "bertimbau-large"]:
            x = d[(d.model == m) & (d.strategy == strat)].iloc[0]
            r.append(f"{100 * x.val_auc:.2f}／{100 * x.test_auc:.2f}")
        rows.append(r)
    return ["聚合方式"] + [f"{names[m]}（驗證／測試）" for m in ["mbert", "bertimbau-base", "bertimbau-large"]], rows, None


def gen_sweep():
    s = pd.read_csv(RESULTS / "rq1_sweep.csv")
    s = s[s.epoch == s.epochs]
    g = s.groupby(["model", "lr", "dropout", "epochs"]).agg(n=("seed", "size"), val=("val_auc", "mean"), auc=("test_auc", "mean"),
                                                          sd=("test_auc", "std"), mins=("minutes", "mean")).reset_index()
    rows = []
    for _, r in g.iterrows():
        rows.append(["Base" if r.model == "bertimbau-base" else "Large", f"{r.lr:g}", f"{r.dropout:g}", str(int(r.epochs)),
                     f"{100 * r.val:.2f}", f"{100 * r.auc:.2f} ± {100 * r.sd:.2f}", f"{r.mins:.1f}"])
    return ["模型", "學習率", "dropout", "epoch 數", "驗證 AUC", "測試 AUC（平均 ± 標準差）", "訓練分鐘"], rows, None


def gen_llm():
    d = pd.read_csv(RESULTS / "rq2_llm.csv")
    mode = {"zeroshot": "零樣本", "fewshot4": "少樣本（4 則）", "fewshot8": "少樣本（8 則）", "lora": "LoRA"}
    rows = []
    for _, r in d.sort_values(["model", "mode", "train_size", "seed"]).iterrows():
        rows.append([r.model.replace("qwen-", "Qwen2.5-").upper().replace("QWEN2.5-", "Qwen2.5-"), mode[r["mode"]],
                     f"{int(r.train_size):,}" if r.train_size else "0", str(int(r.seed)),
                     f"{100 * r.test_auc:.2f}", f"{100 * r.test_acc:.2f}", f"{100 * r.test_f1_neg:.2f}",
                     f"{100 * r.pred_pos_rate:.1f}", fmt(r.train_minutes, 1) if r.train_minutes else "—", fmt(r.infer_sec_per_1k, 1)])
    return ["模型", "方式", "訓練筆數", "種子", "AUC", "準確率", "負評 F1", "判為正面 %", "訓練分鐘", "秒／千則"], rows, \
        ["l", "l", "r", "c", "c", "c", "c", "c", "c", "c"]


def gen_calib():
    d = pd.read_csv(RESULTS / "ext_llm_calibration.csv")
    mode = {"zeroshot": "零樣本", "fewshot4": "4 則範例", "fewshot8": "8 則範例"}
    rows = []
    for _, r in d.iterrows():
        rows.append([r.model.replace("qwen-", "Qwen-"), mode.get(r["mode"], r["mode"]), r.校正方式, f"{r.門檻:.2f}",
                     f"{100 * r.test_acc:.2f}", f"{100 * r.test_f1_neg:.2f}", f"{100 * r.判為正面比例:.1f}"])
    return ["模型", "提示", "校正方式", "門檻", "準確率", "負評 F1", "判為正面 %"], rows, None


def gen_cross_tfidf():
    d = pd.read_csv(RESULTS / "ext_crossdomain.csv")
    doms = ["olist", "b2w", "buscape", "utlc_apps", "utlc_movies"]
    m = d[d.method == "TF-IDF"].pivot(index="train", columns="test", values="auc").reindex(index=doms, columns=doms)
    rows = [[i] + [f"{100 * m.loc[i, j]:.1f}" for j in doms] for i in doms]
    return ["訓練＼測試"] + doms, rows, None


def gen_backtest():
    d = pd.read_csv(RESULTS / "ext_backtest.csv")
    rows = []
    for mth in d.month.unique():
        f = d[(d.month == mth) & (d.features == "full")].iloc[0]
        p = d[(d.month == mth) & (d.features == "preship")].iloc[0]
        rows.append([mth, f"{int(f.n_test):,}", f"{100 * f.負評率:.1f}", f"{f.auc_每月重訓:.3f}", f"{f.auc_不更新:.3f}",
                     f"{p.auc_每月重訓:.3f}", f"{p.auc_不更新:.3f}"])
    return ["測試月份", "訂單數", "負評率 %", "完整：每月重訓", "完整：不更新", "出貨前：每月重訓", "出貨前：不更新"], rows, None


def gen_errors():
    d = pd.read_csv(RESULTS / "ext_error_sample.csv")
    rows = []
    for i, r in d.iterrows():
        txt = re.sub(r"\s+", " ", str(r.review_text)).strip()
        if len(txt) > 110:
            txt = txt[:108] + "…"
        rows.append([str(i + 1), "假正評" if r.錯誤類型.startswith("假正") else "假負評", str(int(r.rating)), f"{r.p_bert:.3f}",
                     txt, r.人工歸類, r.歸類理由])
    return ["#", "錯誤類型", "星等", "p(正面)", "評論原文（葡萄牙文）", "歸類", "理由"], rows, ["c", "c", "c", "c", "l", "l", "l"]


def _best_epoch(df, key):
    """每個設定依「各種子平均的驗證 AUC」挑 epoch（第二輪補做的選 epoch 規則）。"""
    m = df.groupby(key + ["epoch"]).val_auc.mean().reset_index()
    return m.loc[m.groupby(key).val_auc.idxmax(), key + ["epoch"]].assign(選用=True)


def gen_lora_v2():
    v = pd.read_csv(RESULTS / "rq2_llm_v2.csv")
    key = ["model", "train_size", "lr", "epochs"]
    v = v.merge(_best_epoch(v, key), on=key + ["epoch"], how="left")
    rows = []
    for _, r in v.sort_values(["model", "train_size", "lr", "seed", "epoch"]).iterrows():
        rows.append([r.model.replace("qwen-", "Qwen-").replace("b", "B"), f"{int(r.train_size):,}", f"{r.lr:.0e}".replace("e-0", "e-"), str(int(r.batch)),
                     str(int(r.seed)), f"{int(r.epoch)}／{int(r.epochs)}", f"{r.train_loss:.3f}", f"{100 * r.val_auc:.2f}",
                     f"{100 * r.test_auc:.2f}", f"{100 * r.test_acc:.2f}", "✓" if pd.notna(r.選用) else ""])
    return ["模型", "訓練筆數", "學習率", "批次", "種子", "epoch", "訓練損失", "驗證 AUC", "測試 AUC", "準確率", "選用"], rows, \
        ["l", "r", "c", "c", "c", "c", "c", "c", "c", "c", "c"]


def gen_ft8():
    f = pd.read_csv(RESULTS / "rq2_finetune.csv")
    f = f[(f.model == "bertimbau-base") & (f.epochs == 8)]
    g = f.groupby(["train_size", "epoch"]).agg(loss=("train_loss", "mean"), val=("val_auc", "mean"), auc=("test_auc", "mean"),
                                               sd=("test_auc", "std")).reset_index()
    g = g.merge(_best_epoch(f, ["train_size"]), on=["train_size", "epoch"], how="left")
    rows = [[f"{int(r.train_size):,}", str(int(r.epoch)), f"{r.loss:.3f}", f"{100 * r.val:.2f}",
             f"{100 * r.auc:.2f} ± {100 * r.sd:.2f}", "✓" if pd.notna(r.選用) else ""] for _, r in g.iterrows()]
    return ["訓練筆數", "epoch", "訓練損失", "驗證 AUC", "測試 AUC（平均 ± 標準差）", "選用"], rows, None


def gen_preship_backtest():
    b = pd.read_csv(RESULTS / "ext_preship_backtest.csv")
    w = b.pivot(index="month", columns="features", values="auc_每月重訓")
    n = b[b.features == "preship"].set_index("month")
    rows = [[m, f"{int(n.n_test[m]):,}", f"{100 * n.負評率[m]:.1f}", f"{w.preship[m]:.3f}", f"{w.preship2[m]:.3f}",
             f"{100 * (w.preship2[m] - w.preship[m]):+.1f}".replace("-", "−")] for m in w.index]
    return ["測試月份", "訂單數", "負評率 %", "原本", "改良版", "差距（百分點）"], rows, None


def _p(x):
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def gen_robust_rq3():
    r = pd.read_csv(RESULTS / "ext_robust_rq3.csv")
    fam = {"完整特徵（依時間切分）": "完整特徵", "出貨前特徵（依時間切分）": "出貨前特徵", "出貨前特徵的改良（LightGBM）": "出貨前改良（LightGBM）"}
    rows = [[fam[x.比較族], f"{x.方法} − {x.對照}", f"{x.方法AUC:.3f}／{x.對照AUC:.3f}",
             f"{x.差距_百分點:+.2f}".replace("-", "−"), f"[{x.CI低:+.2f}, {x.CI高:+.2f}]".replace("-", "−"),
             _p(x.p值), _p(x.p值_Holm)] for x in r.itertuples()]
    return ["比較族", "比較（前者 − 後者）", "AUC", "差距（百分點）", "95% 信賴區間", "p 值", "Holm 校正後"], rows, \
        ["l", "l", "c", "c", "c", "c", "c"]


def gen_robust_rq4():
    ci = pd.read_csv(RESULTS / "ext_robust_rq4.csv")
    df = pd.read_csv(RESULTS / "ext_robust_rq4_diff.csv")

    def show(m, v):
        return f"{v:.3f}" if m == "AUC" else f"{100 * v:.1f}%"

    def dshow(m, v):
        return (f"{v:+.3f}" if m == "AUC" else f"{100 * v:+.1f}").replace("-", "−")
    rows = [[x.時點, x.指標, show(x.指標, x.點估計), f"[{show(x.指標, x.CI低)}, {show(x.指標, x.CI高)}]", "", ""]
            for x in ci.itertuples()]
    rows += [[x.比較, x.指標, dshow(x.指標, x.差距), f"[{dshow(x.指標, x.CI低)}, {dshow(x.指標, x.CI高)}]", _p(x.p值), _p(x.p值_Holm)]
             for x in df.itertuples()]
    return ["時點或比較", "指標", "估計值", "95% 信賴區間", "p 值", "Holm 校正後"], rows, ["l", "l", "c", "c", "c", "c"]


def gen_selection():
    s = pd.read_csv(RESULTS / "ext_robust_selection.csv")
    c = pd.read_csv(RESULTS / "ext_robust_comment_rate.csv")

    def pc(v):
        return "—" if pd.isna(v) else f"{100 * v:.1f}%"
    rows = [[x.群組.replace("（原始訂單表）", ""), f"{int(x.訂單數):,}", pc(x.占全部有評論訂單), pc(x.負評率), pc(x.延遲率_已送達), pc(x.未送達率), pc(x.多賣家比例)]
            for x in s.itertuples()]
    hi = c[c.分組方式 != "星等"]
    rows.append(["4–5 星的留言率", "", "", "", "", "", ""])
    rows += [[f"　{x.組別}", f"{int(x.訂單數):,}", pc(x.留言率), "", "", "", ""] for x in hi.itertuples()]
    return ["群組", "訂單數", "占比／留言率", "負評率", "延遲率（已送達）", "未送達率", "多賣家比例"], rows, \
        ["l", "r", "c", "c", "c", "c", "c"]


def gen_bridge():
    b = pd.read_csv(RESULTS / "ext_bridge_aipw.csv")
    av = pd.read_csv(RESULTS / "ext_bridge_avoidable.csv").set_index("組別")
    outs = ["星等負評（1–2 星）", "星等未反映的不滿（3–5 星、文字負面）", "任一不滿（星等或文字）"]

    def cell(x):
        return f"{x.AIPW_ATT_百分點:+.1f} [{x.CI低:+.1f}, {x.CI高:+.1f}]".replace("-", "−")
    rows = []
    for g, d in b.groupby("組別", sort=False):
        d = d.set_index("結果變數")
        rows.append([g.replace(">", "＞"), f"{int(d.處理組訂單數.iloc[0]):,}"] + [cell(d.loc[o]) for o in outs])
    rows.append(["可避免的不滿（則）", "—"] + [f"{av.loc['合計', o]:,.0f}" for o in outs])
    return ["處理組", "訂單數", "星等負評", "星等未反映的不滿", "任一不滿"], rows, ["l", "r", "c", "c", "c"]


def gen_files():
    rows = [
        ["common.py／monitor.py", "共用設定（路徑、離線模式、隨機種子）；進度條與 CPU／GPU 監控、記錄檔"],
        ["run_all.py、run_night2.py、run_night3.py", "RQ1／RQ2 批次流程（每段完成自動更新結果摘要）"],
        ["rq1_tfidf.py", "TF-IDF＋邏輯斯迴歸基準與設定敏感度"],
        ["rq1_frozen.py", "BERT 凍結向量（9 種聚合）＋邏輯斯迴歸"],
        ["rq1_finetune.py", "BERT 微調（論文設定、多種子、超參數搜尋、學習曲線）"],
        ["rq1_significance.py", "成對 bootstrap 信賴區間與顯著性檢定"],
        ["rq2_learning_curve.py", "學習曲線（TF-IDF、凍結 BERT）"],
        ["rq2_llm.py", "Qwen2.5 零樣本、少樣本與 LoRA 微調"],
        ["download_models.py", "下載 Hugging Face 模型到本機快取"],
        ["rq3_data.py", "串接 Olist 9 張表，建立訂單層級資料與特徵"],
        ["rq3_model.py", "負評預測（5 種模型 × 2 組特徵 × 2 種切分）"],
        ["rq3_explain.py", "SHAP、決策樹規則、各因素負評率"],
        ["rq3_cluster.py／rq3_topics.py", "RFM＋K-means 分群；負評文字 NMF 主題"],
        ["rq4_decision.py", "預警時點、成本敏感名單、改善優先順序、預警名單"],
        ["run_rq3.py、run_rq4.py", "RQ3／RQ4 批次流程"],
        ["ext_tfidf_gap.py、ext_error_analysis.py、ext_error_coding.py", "TF-IDF 差距分析；錯誤分析與錯誤原因歸類"],
        ["ext_ensemble.py", "集成與成對 bootstrap"],
        ["ext_backtest.py", "依月份滾動回測"],
        ["ext_causal.py", "斷點迴歸、傾向分數配對、AIPW"],
        ["ext_cluster2.py", "購物經驗分群（K-means、GMM）"],
        ["ext_montecarlo.py", "成本不確定下的蒙地卡羅模擬"],
        ["ext_apply_bert.py", "BERT 交叉評分有留言的評論（3 星、隱性不滿）"],
        ["ext_crossdomain.py", "跨資料集 5×5（TF-IDF、BERT、Qwen 零樣本）"],
        ["ext_speed.py、ext_llm_calibration.py", "推論速度與記憶體；LLM 門檻校正"],
        ["run_extra_cpu.py、run_extra_gpu.py", "加深實驗批次流程"],
        ["run_night4.py", "第二輪補做的批次流程（學習曲線延長訓練、Qwen-3B 學習率、出貨前特徵改良）"],
        ["ext_preship.py", "出貨前特徵改良：延遲歷史特徵的消融比較、SHAP、預警時點、蒙地卡羅、滾動回測"],
        ["ext_error_blind.py、ext_error_recode.py、ext_error_kappa.py", "錯誤歸類的盲測表、第二次獨立歸類與 Cohen's κ"],
        ["ext_robust.py、ext_bridge.py、run_night5.py", "補強檢驗：Holm 校正、RQ3／RQ4 的成對 bootstrap、樣本選擇檢查、以留言情緒擴充結果的 AIPW"],
        ["fig_eda.py、fig_rq12.py", "報告用圖（研究流程、EDA、RQ1／RQ2）"],
        ["summarize.py、build_report.py", "自動結果摘要；產生本報告的 Word 與 PDF"],
    ]
    return ["程式（src/）", "用途"], rows, ["l", "l"]


GEN_TABLES = {"frozen": gen_frozen, "sweep": gen_sweep, "llm": gen_llm, "calib": gen_calib, "cross_tfidf": gen_cross_tfidf,
              "backtest": gen_backtest, "errors": gen_errors, "files": gen_files, "lora_v2": gen_lora_v2, "ft8": gen_ft8,
              "preship_backtest": gen_preship_backtest, "robust_rq3": gen_robust_rq3, "robust_rq4": gen_robust_rq4,
              "selection": gen_selection, "bridge": gen_bridge}


# ============================================================ 解析與輸出
class Builder:
    def __init__(self, toc_pages=None):
        self.doc = Document()
        setup_styles(self.doc)
        self.toc_pages = toc_pages or {}
        self.headings = []  # (level, text)
        self.tables = []    # caption
        self.figures = []   # caption
        self.section_start = True

    # ---- 段落
    def body(self, text):
        p = self.doc.add_paragraph()
        label = re.fullmatch(r"\*\*[^*]+\*\*：?", text.strip())  # 只有粗體小標的一行（例如「**使用的工具**：」）不縮排
        para_fmt(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, first=0 if label else 24, spacing=1.5, after=3)
        add_inline(p, text)

    def bullet(self, text, level=0):
        p = self.doc.add_paragraph(style="List Bullet")
        para_fmt(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, hanging=12, left=12 + 18 * level, spacing=1.5, after=1)
        add_inline(p, text)

    def numbered(self, text):
        m = re.match(r"^(（\d+）|\d+\.|\(\d+\))\s*(.*)$", text)
        p = self.doc.add_paragraph()
        para_fmt(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, hanging=30, left=6, spacing=1.5, after=2)
        add_inline(p, m.group(1) + " " + m.group(2))

    def heading(self, text, level, numbered=True):
        p = self.doc.add_heading(level=level)
        if level == 1:
            # 每章從新的一頁開始；緊接在分節符後的第一章不需要
            p.paragraph_format.page_break_before = not self.section_start
            self.section_start = False
        add_inline(p, text, size={1: 16, 2: 14, 3: 12}[level], bold=True)
        self.headings.append((level, text))

    def page_break(self):
        p = self.doc.add_paragraph()
        p.add_run().add_break(WD_BREAK.PAGE)
        para_fmt(p, spacing=1.0)

    def caption(self, text, kind):
        p = self.doc.add_paragraph(style="TableCaption" if kind == "table" else "FigureCaption")
        m = re.match(r"^((?:表|圖)\s*[A-Z0-9]+-\d+)\s*(.*)$", text)
        if m:
            r = p.add_run(m.group(1).replace(" ", "\u00a0") + "\u3000")
            set_run_font(r, size=11, bold=True)
            add_inline(p, m.group(2), size=11)
        else:
            add_inline(p, text, size=11)
        (self.tables if kind == "table" else self.figures).append(text)
        return p

    def note(self, text):
        p = self.doc.add_paragraph(style="NoteText")
        add_inline(p, text, size=9.5)
        # 註記要和上方的表格（或上一則註記）在同一頁：把前一個元素設為「與下段同頁」
        prev = p._p.getprevious()
        # （用 python-docx 的屬性設定，元素順序才合乎 Word 的規格；直接 append 會被 Word 忽略）
        if prev is not None and prev.tag == qn("w:tbl"):
            last_row = prev.findall(qn("w:tr"))[-1]
            for para in last_row.iter(qn("w:p")):
                Paragraph(para, None).paragraph_format.keep_with_next = True
        elif prev is not None and prev.tag == qn("w:p"):
            st = prev.find(qn("w:pPr") + "/" + qn("w:pStyle")) if prev.find(qn("w:pPr")) is not None else None
            if st is not None and st.get(qn("w:val")) == self.doc.styles["NoteText"].style_id:
                Paragraph(prev, None).paragraph_format.keep_with_next = True

    def figure(self, path, width, cap):
        p = self.doc.add_paragraph()
        para_fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0, before=6, after=2, keep_next=True)
        p.add_run().add_picture(str(ROOT / path), width=Cm(float(width)))
        self.caption(cap, "figure")

    def equation(self, expr, num):
        p = self.doc.add_paragraph()
        para_fmt(p, spacing=1.5, before=4, after=4)
        ts = p.paragraph_format.tab_stops
        ts.add_tab_stop(Cm(TEXT_W / 2), WD_TAB_ALIGNMENT.CENTER)
        ts.add_tab_stop(Cm(TEXT_W), WD_TAB_ALIGNMENT.RIGHT)
        r = p.add_run("\t")
        add_inline(p, expr)
        r = p.add_run("\t" + num)
        set_run_font(r)

    def reference(self, text):
        p = self.doc.add_paragraph(style="RefText")
        para_fmt(p, align=WD_ALIGN_PARAGRAPH.LEFT, hanging=24, spacing=1.3, after=4)
        add_inline(p, text)

    # ---- 目錄（以 Word 欄位包住、預先填好頁碼的結果）
    def toc_block(self, entries, instr):
        """entries: [(style, text, page)]"""
        if not entries:
            return
        first = None
        for k, (style, text, page) in enumerate(entries):
            p = self.doc.add_paragraph(style=style)
            indent = {"TOCLevel1": 0, "TOCLevel2": 24, "TOCLevel3": 48}.get(style, 0)
            para_fmt(p, left=indent, spacing=1.35, after=1)
            p.paragraph_format.tab_stops.add_tab_stop(Cm(TEXT_W), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            if k == 0:
                r = p.add_run()
                fc = OxmlElement("w:fldChar")
                fc.set(qn("w:fldCharType"), "begin")
                r._element.append(fc)
                r2 = p.add_run()
                it = OxmlElement("w:instrText")
                it.set(qn("xml:space"), "preserve")
                it.text = f" {instr} "
                r2._element.append(it)
                r3 = p.add_run()
                fc = OxmlElement("w:fldChar")
                fc.set(qn("w:fldCharType"), "separate")
                r3._element.append(fc)
            add_inline(p, text.replace("\u3000", " "), size=12 if style != "TOCLevel3" else 11, bold=False)
            r = p.add_run("\t" + str(page))
            set_run_font(r, size=12 if style != "TOCLevel3" else 11)
            if k == len(entries) - 1:
                r = p.add_run()
                fc = OxmlElement("w:fldChar")
                fc.set(qn("w:fldCharType"), "end")
                r._element.append(fc)

    def front_heading(self, text, new_page=True):
        p = self.doc.add_paragraph()
        para_fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.3, after=18)
        r = p.add_run(text)
        set_run_font(r, size=16, bold=True)
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.page_break_before = new_page
        return p

    # ---- 封面與前頁
    def cover(self):
        d = self.doc
        d.add_paragraph()
        if LOGO.exists():  # 校徽放在校名上方
            p = d.add_paragraph()
            para_fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0, after=10)
            p.add_run().add_picture(str(LOGO), width=Cm(3.6))
        for text, size, bold, after in [("國立中央大學", 20, True, 6), ("商業智慧　期中報告", 18, True, 60)]:
            p = d.add_paragraph()
            para_fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.3, after=after)
            set_run_font(p.add_run(text), size=size, bold=bold)
        for line in TITLE_ZH.split("\n"):
            p = d.add_paragraph()
            para_fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.4, after=2)
            set_run_font(p.add_run(line), size=20, bold=True)
        p = d.add_paragraph()
        para_fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.2, before=14, after=70)
        set_run_font(p.add_run(TITLE_EN), size=13, italic=True)
        for line in [f"{k}：{v}" for k, v in COVER_INFO]:
            p = d.add_paragraph()
            para_fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.6)
            set_run_font(p.add_run(line), size=14)
        if GITHUB:
            p = d.add_paragraph()
            para_fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.6)
            set_run_font(p.add_run("GitHub："), size=12)
            add_link(p, GITHUB, GITHUB, size=12)
        p = d.add_paragraph()
        para_fmt(p, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.6, before=60)
        set_run_font(p.add_run("中華民國 115 年 11 月"), size=14)

    def parse_file(self, path):
        lines = path.read_text(encoding="utf-8").splitlines()
        i, buf = 0, []

        def flush():
            if buf:
                text = buf[0]
                for s in buf[1:]:  # 中文行直接接起來；英文行之間補一個空白
                    cjk = ord(text[-1]) > 0x2E80 or ord(s[0]) > 0x2E80
                    text += s if cjk else " " + s
                self.body(text)
                buf.clear()

        while i < len(lines):
            ln = lines[i].rstrip()
            s = ln.strip()
            if not s or s.startswith("<!--"):
                flush()
                i += 1
                continue
            if s.startswith("#!"):
                flush()
                self.heading(s[2:].strip(), 1, numbered=False)
            elif s.startswith("###"):
                flush()
                self.heading(s[3:].strip(), 3)
            elif s.startswith("##"):
                flush()
                self.heading(s[2:].strip(), 2)
            elif s.startswith("#"):
                flush()
                self.heading(s[1:].strip(), 1)
            elif s.startswith("> TABLE"):
                flush()
                cap = s[len("> TABLE"):].strip()
                rows = []
                i += 1
                while i < len(lines) and lines[i].strip().startswith("|"):
                    rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                    i += 1
                header, sep, data = rows[0], rows[1], rows[2:]
                aligns = []
                for c in sep:
                    aligns.append("c" if c.startswith(":") and c.endswith(":") else "r" if c.endswith(":") else
                                  "l" if c.startswith(":") else None)
                aligns = [a or ("l" if j == 0 else "c") for j, a in enumerate(aligns)]
                self.caption(cap, "table")
                three_line_table(self.doc, header, data, aligns)
                continue
            elif s.startswith("> GEN"):
                flush()
                parts = s[len("> GEN"):].strip().split("|")
                name, cap = parts[0].strip(), parts[1].strip()
                header, data, aligns = GEN_TABLES[name]()
                self.caption(cap, "table")
                three_line_table(self.doc, header, data, aligns, font=9 if len(header) > 6 or name == "errors" else 10,
                                 keep=False)
            elif s.startswith("> NOTE"):
                flush()
                self.note(s[len("> NOTE"):].strip())
            elif s.startswith("> FIG"):
                flush()
                path, width, cap = [x.strip() for x in s[len("> FIG"):].split("|", 2)]
                self.figure(path, width, cap)
            elif s.startswith("> EQ"):
                flush()
                expr, num = [x.strip() for x in s[len("> EQ"):].rsplit("|", 1)]
                self.equation(expr, num)
            elif s.startswith("> REF"):
                flush()
                self.reference(s[len("> REF"):].strip())
            elif s.startswith("> PAGEBREAK"):
                flush()
                self.page_break()
            elif re.match(r"^\s*- ", ln):
                flush()
                level = (len(ln) - len(ln.lstrip())) // 2
                self.bullet(s[2:].strip(), level)
            elif re.match(r"^(（\d+）|\d+\.\s|\(\d+\)\s)", s):
                flush()
                self.numbered(s)
            else:
                buf.append(s)
            i += 1
        flush()


def roman(n):
    vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, r in vals:
        while n >= v:
            out += r
            n -= v
    return out


def build(pages):
    """pages：{('h', 文字) 或 ('t'/'f', 圖表名): 頁碼字串}；第一輪為空。"""
    b = Builder()
    d = b.doc
    s0 = d.sections[0]
    setup_section(s0)
    b.cover()
    # 第 1 節：前頁（羅馬數字頁碼）
    s1 = d.add_section(WD_SECTION.NEW_PAGE)
    setup_section(s1)
    set_pgnum(s1, "lowerRoman", 1)
    footer_page_number(s1)
    header_text(s1, "")
    s0.footer.is_linked_to_previous = False
    s0.header.is_linked_to_previous = False
    front = sorted(SRC.glob("0*.md"))
    chapters = sorted(p for p in SRC.glob("*.md") if not p.name.startswith("0") and not p.name.startswith("_"))
    front_entries = []
    for k, f in enumerate(front):  # 摘要、Abstract
        text = f.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("#! ").strip()
        b.front_heading(title, new_page=k > 0)
        front_entries.append(("TOCLevel1", title, pages.get(("h", title), "i")))
        tmp = SRC / "_tmp_front.md"
        tmp.write_text("\n".join(text.splitlines()[1:]), encoding="utf-8")
        b.parse_file(tmp)
        tmp.unlink()
    # 先解析正文，收集標題與圖表清單（另建一份暫存文件）
    probe = Builder()
    for f in chapters:
        probe.parse_file(f)
    toc = list(front_entries)
    toc += [("TOCLevel1", "表目錄", pages.get(("h", "表目錄"), "i")), ("TOCLevel1", "圖目錄", pages.get(("h", "圖目錄"), "i"))]
    for lvl, text in probe.headings:
        toc.append((f"TOCLevel{lvl}", text, pages.get(("h", text), "0")))
    b.front_heading("目　錄")
    b.toc_block(toc, 'TOC \\o "1-3" \\h \\z \\u')
    b.front_heading("表目錄")
    b.toc_block([("TOCLevel2", t, pages.get(("t", t), "0")) for t in probe.tables], 'TOC \\h \\z \\t "TableCaption,1"')
    b.front_heading("圖目錄")
    b.toc_block([("TOCLevel2", t, pages.get(("f", t), "0")) for t in probe.figures], 'TOC \\h \\z \\t "FigureCaption,1"')
    # 第 2 節：正文（阿拉伯數字頁碼）
    s2 = d.add_section(WD_SECTION.NEW_PAGE)
    setup_section(s2)
    set_pgnum(s2, "decimal", 1)
    footer_page_number(s2)
    header_text(s2, SHORT_TITLE)
    b.section_start = True
    for f in chapters:
        b.parse_file(f)
    # 文件設定：開啟時提示更新欄位（Word）
    settings = d.settings.element
    uf = OxmlElement("w:updateFields")
    uf.set(qn("w:val"), "false")
    settings.append(uf)
    return d, probe


# ============================================================ PDF 與頁碼
def soffice():
    for p in SOFFICE:
        if p.exists():
            return p
    w = shutil.which("soffice")
    return Path(w) if w else None


def winword():
    for p in WINWORD:
        if p.exists():
            return p
    return None


# Word 轉 PDF：用 PowerShell 呼叫 Word（COM），路徑由參數傳入，腳本本身只有英文，避免編碼問題
WORD_PS1 = r"""
param([string]$src, [string]$dst)
$ErrorActionPreference = 'Stop'
$w = New-Object -ComObject Word.Application
try {
    $w.Visible = $false
    $w.DisplayAlerts = 0
    $d = $w.Documents.Open($src, $false, $true, $false)
    try {
        $d.Repaginate()
        # 17 = PDF; CreateBookmarks 1 = bookmarks from headings
        $d.ExportAsFixedFormat($dst, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1, $true, $true, $false)
    } finally {
        $d.Close(0)
    }
} finally {
    $w.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($w)
}
"""


def to_pdf(docx, outdir, engine="auto"):
    pdf = outdir / (docx.stem + ".pdf")
    word = winword() if engine in ("auto", "word") else None
    if pdf.exists():
        pdf.unlink()
    if word is not None:
        ps1 = outdir / "_word2pdf.ps1"
        ps1.write_text(WORD_PS1, encoding="ascii")
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(ps1),
                            str(docx.resolve()), str(pdf.resolve())], capture_output=True, timeout=900)
        ps1.unlink()
        if r.returncode != 0 or not pdf.exists():
            msg = r.stderr.decode("utf-8", "replace") or r.stdout.decode("cp950", "replace")
            raise SystemExit(f"Word 轉 PDF 失敗（PDF 若正開著請先關閉）：\n{msg}")
        return pdf
    if engine == "word":
        raise SystemExit("找不到 Word（WINWORD.EXE）。")
    exe = soffice()
    if exe is None:
        raise SystemExit("找不到 Word 或 LibreOffice，無法轉成 PDF。")
    subprocess.run([str(exe), "--headless", "--norestore", "--convert-to", "pdf", "--outdir", str(outdir), str(docx)],
                   check=True, capture_output=True, timeout=600)
    return pdf


def norm(s):
    return re.sub(r"[\s\u3000\u00a0]+", "", s)


def find_pages(pdf, probe, front_titles):
    from pypdf import PdfReader
    r = PdfReader(str(pdf))
    pages = [[norm(x) for x in (pg.extract_text() or "").splitlines()] for pg in r.pages]
    def first_page(pred, start=0):
        for k in range(start, len(pages)):
            if any(pred(x) for x in pages[k]):
                return k
        return None
    abs_idx = first_page(lambda x: x == norm(front_titles[0]))
    ch1 = norm(probe.headings[0][1])
    body = None
    for k in range(len(pages)):
        if any(x == ch1 for x in pages[k]):
            body = k  # 取最後一次「整行等於第一章標題」的頁（目錄裡的那一行後面有頁碼，不會整行相等）
            break
    out = {}
    for t in front_titles + ["表目錄", "圖目錄"]:
        k = first_page(lambda x, t=t: x == norm(t) or x == norm(t.replace("　", "")))
        if k is not None and abs_idx is not None:
            out[("h", t)] = roman(k - abs_idx + 1)
    k_toc = first_page(lambda x: x == norm("目　錄"))
    for lvl, text in probe.headings:
        k = first_page(lambda x, t=norm(text): x == t, body)
        out[("h", text)] = str(k - body + 1) if k is not None else "?"
    for kind, caps in [("t", probe.tables), ("f", probe.figures)]:
        for c in caps:
            m = re.match(r"^((?:表|圖)\s*[A-Z0-9]+-\d+)\s*(.*)$", c)
            key = norm(m.group(1)) + norm(m.group(2))[:6]
            k = first_page(lambda x, key=key: x.startswith(key), body)
            out[(kind, c)] = str(k - body + 1) if k is not None else "?"
    missing = [k for k, v in out.items() if v == "?"]
    return out, missing, len(pages)


def main():
    global SRC, OUT, NAME, GITHUB
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--engine", choices=["auto", "word", "libreoffice"], default="auto",
                    help="PDF 轉檔：auto＝有 Word 用 Word，否則用 LibreOffice")
    ap.add_argument("--src", default="report", help="章節 .md 所在資料夾（相對於專案根目錄）")
    ap.add_argument("--out", default="報告", help="輸出資料夾（相對於專案根目錄）")
    ap.add_argument("--name", default=NAME, help="輸出檔名（不含副檔名）")
    ap.add_argument("--github", help="GitHub 網址；有給就印在封面")
    a = ap.parse_args()
    SRC, OUT, NAME, GITHUB = ROOT / a.src, ROOT / a.out, a.name, a.github
    OUT.mkdir(parents=True, exist_ok=True)
    tmpdir = OUT / "_build"
    tmpdir.mkdir(exist_ok=True)
    front_titles = [f.read_text(encoding="utf-8").splitlines()[0].lstrip("#! ").strip() for f in sorted(SRC.glob("0*.md"))]
    doc, probe = build({})
    docx1 = tmpdir / f"{NAME}.docx"
    doc.save(docx1)
    print(f"第一輪 Word：{docx1}（標題 {len(probe.headings)}、表 {len(probe.tables)}、圖 {len(probe.figures)}）")
    if a.no_pdf:
        shutil.copy(docx1, OUT / docx1.name)
        return
    eng = "Word" if a.engine != "libreoffice" and winword() else "LibreOffice"
    print(f"PDF 轉檔：{eng}")
    pdf1 = to_pdf(docx1, tmpdir, a.engine)
    pages, missing, n = find_pages(pdf1, probe, front_titles)
    print(f"第一輪 PDF {n} 頁；找不到頁碼的項目 {len(missing)} 個" + (f"：{missing[:8]}" if missing else ""))
    doc, probe = build(pages)
    final = OUT / f"{NAME}.docx"
    doc.save(final)
    pdf2 = to_pdf(final, OUT, a.engine)
    pages2, missing2, n2 = find_pages(pdf2, probe, front_titles)
    diff = [k for k in pages if pages.get(k) != pages2.get(k)]
    print(f"完成：{final}\n      {pdf2}（{n2} 頁）；與第一輪頁碼不同的項目 {len(diff)} 個" + (f"：{diff[:5]}" if diff else ""))


if __name__ == "__main__":
    sys.exit(main())
