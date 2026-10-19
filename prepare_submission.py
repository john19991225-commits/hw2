"""Submission packaging script for Data Mining HW2.

Student ID: 7115056206
Workflow:
1. Checks for genuine NotebookLM summary (>= 100 Chinese characters).
2. Generates professional 7115056206_HW2_Report.pdf.
3. Checks for 7115056206_ChatGPT_Conversation.pdf.
4. Prepares local submission directory.
5. Packages into 7115056206.zip once all prerequisites are satisfied.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# Ensure UTF-8 stdout on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
STUDENT_ID = "7115056206"
SUBMISSION_DIR = ROOT / f"submission_{STUDENT_ID}"
ZIP_PATH = ROOT / f"{STUDENT_ID}.zip"
REPORT_MD = ROOT / "HW2_Report.md"
REPORT_PDF = ROOT / f"{STUDENT_ID}_HW2_Report.pdf"
EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

# Expected files in submission
CORE_FILES = [
    "winequality-red.csv",
    "requirements.txt",
    "evaluation_metrics.json",
    "feature_selection_results.csv",
    "model_comparison_metrics.csv",
    "regression_coefficients.csv",
    "learning_curve.csv",
    "red_wine_model.pkl.gz",
    "actual_vs_predicted.png",
    "feature_selection_metrics.png",
    "model_comparison.png",
    "residual_plot.png",
    "correlation_heatmap.png",
    "exploratory_scatter_plots.png",
    "learning_curve.png",
    "prediction_interval.png",
    "README.md",
    "HW2_Report.md",
]


def check_notebooklm_summary() -> tuple[bool, int, str]:
    """Check if genuine NotebookLM research summary has been inserted."""
    if not REPORT_MD.exists():
        return False, 0, "HW2_Report.md not found"

    with open(REPORT_MD, "r", encoding="utf-8") as f:
        content = f.read()

    # Locate section 8
    match = re.search(r"## 第八階段：NotebookLM 摘要(.*?)(?:---|\Z)", content, re.DOTALL)
    if not match:
        return False, 0, "找不到『第八階段：NotebookLM 摘要』區塊"

    sec_text = match.group(1)

    # Check if still the default placeholder
    if "[NotebookLM 摘要佔位符號]" in sec_text or "核心發現待填寫" in sec_text:
        return False, 0, "目前仍為佔位符號（Placeholder），尚未填入真實 NotebookLM 摘要"

    # Count Chinese characters
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", sec_text)
    char_count = len(chinese_chars)

    if char_count < 100:
        return False, char_count, f"中文字數不足 100 字（目前僅 {char_count} 字）"

    return True, char_count, f"已填入真實 NotebookLM 摘要（共 {char_count} 個中文字）"


def generate_report_pdf() -> bool:
    """Generate high-quality PDF from HW2_Report.md using Edge headless."""
    try:
        import markdown
    except ImportError:
        print("[PDF Export] Installing markdown package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "markdown"], check=True)
        import markdown

    if not REPORT_MD.exists():
        print(f"[PDF Export] Error: {REPORT_MD} does not exist.")
        return False

    with open(REPORT_MD, "r", encoding="utf-8") as f:
        md_text = f.read()

    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<title>Data Mining HW2 Report - {STUDENT_ID}</title>
<style>
@page {{
    size: A4;
    margin: 18mm 15mm;
}}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Microsoft JhengHei', Arial, sans-serif;
    color: #24292e;
    line-height: 1.6;
    font-size: 10.5pt;
}}
h1 {{ font-size: 18pt; border-bottom: 2px solid #eaecef; padding-bottom: 8px; color: #1a365d; }}
h2 {{ font-size: 14pt; border-bottom: 1px solid #eaecef; padding-bottom: 5px; margin-top: 22px; color: #2b6cb0; page-break-after: avoid; }}
h3 {{ font-size: 11.5pt; margin-top: 16px; color: #2d3748; page-break-after: avoid; }}
h4 {{ font-size: 10.5pt; color: #4a5568; page-break-after: avoid; }}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 14px 0;
    font-size: 9pt;
    page-break-inside: avoid;
}}
th, td {{
    border: 1px solid #d2d6dc;
    padding: 6px 8px;
    text-align: left;
}}
th {{
    background-color: #f7fafc;
    font-weight: 600;
}}
img {{
    max-width: 95%;
    height: auto;
    display: block;
    margin: 12px auto;
    page-break-inside: avoid;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
}}
pre, code {{
    font-family: Consolas, Monaco, monospace;
    background-color: #f6f8fa;
    border-radius: 3px;
}}
pre {{
    padding: 10px;
    font-size: 8.5pt;
    overflow-x: auto;
    page-break-inside: avoid;
}}
blockquote {{
    border-left: 4px solid #3182ce;
    padding: 6px 12px;
    color: #4a5568;
    background: #ebf8ff;
    margin: 12px 0;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    temp_html = ROOT / "_temp_report.html"
    with open(temp_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    if not Path(EDGE_EXE).exists():
        print(f"[PDF Export] Warning: Edge executable not found at {EDGE_EXE}")
        return False

    cmd = [
        EDGE_EXE,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={REPORT_PDF.resolve()}",
        str(temp_html.resolve()),
    ]
    subprocess.run(cmd, capture_output=True)

    if temp_html.exists():
        temp_html.unlink()

    success = REPORT_PDF.exists() and REPORT_PDF.stat().st_size > 1000
    if success:
        print(f"[PDF Export] Generated {REPORT_PDF.name} ({REPORT_PDF.stat().st_size / 1024 / 1024:.2f} MB)")
    return success


def find_chatgpt_pdf() -> Path | None:
    """Find ChatGPT conversation PDF in root or submission folder."""
    candidates = [
        ROOT / f"{STUDENT_ID}_ChatGPT_Conversation.pdf",
        ROOT / "ChatGPT_Conversation.pdf",
        ROOT / f"{STUDENT_ID}_chatgpt.pdf",
        ROOT / "chatgpt.pdf",
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 500:
            return c
    return None


def prepare_submission_folder() -> None:
    """Prepare and populate local submission directory."""
    if SUBMISSION_DIR.exists():
        shutil.rmtree(SUBMISSION_DIR)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # Copy Python script renamed as student ID
    shutil.copy2(ROOT / "hw2.py", SUBMISSION_DIR / f"{STUDENT_ID}.py")

    # Copy core project files
    for fname in CORE_FILES:
        fpath = ROOT / fname
        if fpath.exists():
            shutil.copy2(fpath, SUBMISSION_DIR / fname)

    # Copy report PDF if generated
    if REPORT_PDF.exists():
        shutil.copy2(REPORT_PDF, SUBMISSION_DIR / REPORT_PDF.name)

    # Copy ChatGPT conversation PDF if present
    chat_pdf = find_chatgpt_pdf()
    if chat_pdf:
        shutil.copy2(chat_pdf, SUBMISSION_DIR / f"{STUDENT_ID}_ChatGPT_Conversation.pdf")

    print(f"[Submission] Prepared submission staging folder: {SUBMISSION_DIR.name}")


def create_zip_archive() -> bool:
    """Create submission ZIP archive."""
    if not SUBMISSION_DIR.exists():
        return False

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root_dir, _, files in os.walk(SUBMISSION_DIR):
            for file in files:
                full_path = Path(root_dir) / file
                rel_path = full_path.relative_to(SUBMISSION_DIR)
                zf.write(full_path, arcname=str(rel_path))

    print(f"[Archive] Created {ZIP_PATH.name} ({ZIP_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
    return True


def main() -> None:
    print("=" * 70)
    print(f"HW2 Submission Status Check - Student ID: {STUDENT_ID}")
    print("=" * 70)

    # 1. Check NotebookLM
    has_nlm, nlm_chars, nlm_msg = check_notebooklm_summary()
    print(f"[1] NotebookLM 摘要狀態: {nlm_msg}")

    # 2. Check ChatGPT PDF
    chat_pdf = find_chatgpt_pdf()
    has_chat_pdf = chat_pdf is not None
    print(f"[2] ChatGPT 對話紀錄 PDF: {'已找到 (' + chat_pdf.name + ')' if has_chat_pdf else '尚未匯入 (待提供)'}")

    # 3. Always prepare submission folder structure
    prepare_submission_folder()

    # 4. Generate report PDF if NotebookLM is ready
    if has_nlm:
        generate_report_pdf()
        if REPORT_PDF.exists():
            shutil.copy2(REPORT_PDF, SUBMISSION_DIR / REPORT_PDF.name)

    # 5. Final check for ZIP packaging
    if has_nlm and has_chat_pdf and REPORT_PDF.exists():
        create_zip_archive()
        print("\n>>> 所有必備項目皆已齊全！提交包已打包完成。")
    else:
        print("\n>>> 【尚未完成最終打包】")
        print("    原因：依作業規範，在實際填入真實 NotebookLM 摘要與匯入 ChatGPT 對話 PDF 之前，")
        print("    不得虛構研究或提前標記為完成。")
        print("=" * 70)


if __name__ == "__main__":
    main()
