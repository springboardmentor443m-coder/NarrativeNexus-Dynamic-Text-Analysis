# ===============================
# core/report.py – Week 4
# ===============================

import json, time, os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def save_json_report(payload: dict, out_dir="data/outputs"):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"report-{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def save_pdf_report(payload: dict, out_dir="data/outputs"):
    """Generate a simple PDF summary of results."""
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"report-{ts}.pdf")

    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    flow = [
        Paragraph("<b>NarrativeNexus Analysis Report</b>", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Model Used: {payload.get('model_type','N/A')}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph(f"<b>Sentiment:</b> {payload['sentiment'].get('label','N/A')}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("<b>Summary</b>", styles["Heading2"]),
        Paragraph(payload.get("summary",""), styles["Normal"]),
        Spacer(1, 12),
        Paragraph("<b>Topics</b>", styles["Heading2"]),
    ]
    for t in payload.get("topics", []):
        flow.append(Paragraph(f"Topic {t['topic_id']}: {', '.join(t['keywords'])}", styles["Normal"]))
    doc.build(flow)
    return path
