# ============================================================
#  SYBCA-AI (Sem-III) | 303 Machine Learning
#  Practical 6 — Automated EDA
# ============================================================
#  Unit Covered : Unit-2
#  Tool Used    : ydata-profiling  (fallback: sweetviz)
# ============================================================

import pandas as pd
import os

df = pd.read_csv('../Dataset/dataset_clean.csv')
os.makedirs('../Reports', exist_ok=True)

print("=" * 60)
print("PRACTICAL 6 : Automated EDA")
print("=" * 60)
print(f"Dataset shape: {df.shape}")

try:
    from ydata_profiling import ProfileReport
    print("\n[ydata-profiling] Generating EDA report...")
    profile = ProfileReport(df, title="Advertising Sales – EDA Report", explorative=True)
    profile.to_file("../Reports/EDA_Report.html")
    print("[✓] Saved: Reports/EDA_Report.html")

except ImportError:
    try:
        import sweetviz as sv
        print("\n[Sweetviz] Generating EDA report...")
        report = sv.analyze(df)
        report.show_html("../Reports/EDA_Report.html", open_browser=False)
        print("[✓] Saved: Reports/EDA_Report.html")

    except ImportError:
        print("\n[!] Please install a profiling library first:")
        print("    pip install ydata-profiling")
        print("    OR")
        print("    pip install sweetviz")
