"""
Report Generator — Creates comprehensive PDF analysis reports.
Supports HTML-to-PDF rendering with clinical formatting.
"""

import base64
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import get_config


class ReportGenerator:
    """
    Generates HTML and PDF reports from analysis results.
    Template-based with clinical formatting and branding.
    """

    def __init__(self):
        self._config = get_config().report

    def generate_html(
        self,
        analysis: Dict[str, Any],
        patient: Optional[Dict] = None,
        comparison: Optional[Dict] = None,
        doctor_notes: str = "",
        annotations_base64: Optional[str] = None,
    ) -> str:
        """
        Generate a complete HTML report.

        Args:
            analysis: Full analysis results from pipeline
            patient: Patient info dict
            comparison: Optional before/after comparison data
            doctor_notes: Free-text notes from doctor
            annotations_base64: Base64 annotation overlay image

        Returns:
            Complete HTML string
        """
        patient = patient or {}
        now = datetime.utcnow().strftime("%B %d, %Y")
        session_id = analysis.get("session_id", "N/A")

        sections = [
            self._header_section(patient, now, session_id),
            self._overall_score_section(analysis),
            self._heatmaps_section(analysis.get("heatmaps", {})),
            self._zone_scores_section(analysis.get("zone_scores", {})),
            self._measurements_section(analysis.get("measurements", {})),
            self._symmetry_section(analysis.get("symmetry", {})),
            self._conditions_section(analysis.get("conditions", [])),
        ]

        if comparison:
            sections.append(self._comparison_section(comparison))

        sections.extend([
            self._recommendations_section(analysis.get("recommendations", [])),
            self._doctor_notes_section(doctor_notes, annotations_base64),
            self._footer_section(),
        ])

        return self._wrap_html("\n".join(sections))

    def generate_pdf_bytes(self, html: str) -> bytes:
        """
        Convert HTML report to PDF bytes.
        Falls back to HTML if weasyprint not available.
        """
        try:
            from weasyprint import HTML
            pdf_bytes = HTML(string=html).write_pdf()
            return pdf_bytes
        except ImportError:
            # Fallback: return HTML as bytes (client can render)
            return html.encode("utf-8")

    def save_report(
        self, analysis: Dict, patient: Optional[Dict] = None,
        output_path: Optional[str] = None, **kwargs
    ) -> str:
        """Generate and save report to disk."""
        html = self.generate_html(analysis, patient, **kwargs)

        if output_path is None:
            os.makedirs(self._config.output_dir, exist_ok=True)
            session_id = analysis.get("session_id", "unknown")
            output_path = os.path.join(
                self._config.output_dir,
                f"report_{session_id}.html"
            )

        with open(output_path, "w") as f:
            f.write(html)

        return output_path

    # ── Section Generators ──────────────────────────────────────────────────

    def _header_section(self, patient: Dict, date: str, session_id: str) -> str:
        name = patient.get("name", "—")
        age = patient.get("age", "—")
        skin_type = patient.get("skin_type", "—")
        fitzpatrick = patient.get("fitzpatrick", "—")

        return f"""
        <div class="header">
            <div class="logo-area">
                <h1>🏥 {self._config.clinic_name}</h1>
                <p class="subtitle">Comprehensive Skin Analysis Report</p>
            </div>
            <div class="patient-info">
                <table>
                    <tr><td><strong>Patient:</strong></td><td>{name}</td>
                        <td><strong>Date:</strong></td><td>{date}</td></tr>
                    <tr><td><strong>Age:</strong></td><td>{age}</td>
                        <td><strong>Session:</strong></td><td>{session_id[:8]}</td></tr>
                    <tr><td><strong>Skin Type:</strong></td><td>{skin_type}</td>
                        <td><strong>Fitzpatrick:</strong></td><td>Type {fitzpatrick}</td></tr>
                </table>
            </div>
        </div>
        """

    def _overall_score_section(self, analysis: Dict) -> str:
        score = analysis.get("overall_score", 0)
        skin_age = analysis.get("skin_age", "—")
        time_ms = analysis.get("processing_time_ms", 0)

        color = "#22c55e" if score >= 75 else "#eab308" if score >= 50 else "#ef4444"
        status = "Excellent" if score >= 85 else "Good" if score >= 70 else "Fair" if score >= 50 else "Needs Attention"

        bar_width = min(score, 100)

        return f"""
        <div class="section">
            <h2>📊 Overall Skin Health</h2>
            <div class="score-card">
                <div class="big-score" style="color: {color}">{score:.0f} / 100</div>
                <div class="score-bar">
                    <div class="score-fill" style="width: {bar_width}%; background: {color}"></div>
                </div>
                <p>Skin Age: <strong>{skin_age}</strong> | Status: <strong>{status}</strong></p>
                <p class="small">Analysis completed in {time_ms:.0f}ms</p>
            </div>
        </div>
        """

    def _heatmaps_section(self, heatmaps: Dict) -> str:
        if not heatmaps:
            return ""

        cards = ""
        labels = {
            "wrinkle": "Wrinkles", "pore": "Pores",
            "pigmentation": "Pigmentation", "redness": "Redness",
            "texture": "Texture", "symmetry": "Symmetry",
        }
        for key, b64 in heatmaps.items():
            label = labels.get(key, key.title())
            cards += f"""
            <div class="heatmap-card">
                <img src="data:image/png;base64,{b64}" alt="{label}" />
                <p>{label}</p>
            </div>
            """

        return f"""
        <div class="section">
            <h2>🗺️ Analysis Heatmaps</h2>
            <div class="heatmap-grid">{cards}</div>
            <p class="legend">🟢 Healthy &nbsp; 🟡 Mild &nbsp; 🔴 Needs Attention</p>
        </div>
        """

    def _zone_scores_section(self, zone_scores: Dict) -> str:
        if not zone_scores:
            return ""

        rows = ""
        for zone, scores in sorted(zone_scores.items()):
            if isinstance(scores, dict):
                row_cells = "".join(
                    f"<td class='{self._score_class(v)}'>{v:.0f}</td>"
                    for k, v in scores.items()
                    if k in ["wrinkles", "pores", "pigmentation", "redness", "texture", "overall"]
                    and isinstance(v, (int, float))
                )
                # Fallback: use dict keys in order
                if not row_cells:
                    for k in ["wrinkle", "pore", "pigmentation", "redness", "texture", "overall"]:
                        v = scores.get(k, 50)
                        row_cells += f"<td class='{self._score_class(v)}'>{v:.0f}</td>"

                rows += f"<tr><td><strong>{zone.replace('_', ' ').title()}</strong></td>{row_cells}</tr>"

        return f"""
        <div class="section">
            <h2>📏 Zone-by-Zone Scoring</h2>
            <table class="scores-table">
                <thead>
                    <tr><th>Zone</th><th>Wrinkles</th><th>Pores</th>
                    <th>Pigment</th><th>Redness</th><th>Texture</th><th>Overall</th></tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    def _measurements_section(self, measurements: Dict) -> str:
        if not measurements:
            return ""

        rows = ""
        for key, data in measurements.items():
            if isinstance(data, dict) and "value" in data:
                name = key.replace("_", " ").title()
                value = data["value"]
                unit = data.get("unit", "")
                ref = data.get("reference", "—")
                rows += f"<tr><td>{name}</td><td><strong>{value} {unit}</strong></td><td>{ref}</td></tr>"

        # Facial thirds
        thirds = measurements.get("facial_thirds", {})
        if isinstance(thirds, dict) and "upper" in thirds:
            rows += f"""<tr><td>Facial Thirds</td>
                <td>Upper: {thirds['upper']}% | Middle: {thirds['middle']}% | Lower: {thirds['lower']}%</td>
                <td>Ideal: 33.3% each</td></tr>"""

        return f"""
        <div class="section">
            <h2>📐 Facial Measurements</h2>
            <table class="measurements-table">
                <thead><tr><th>Measurement</th><th>Value</th><th>Reference Range</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    def _symmetry_section(self, symmetry: Dict) -> str:
        if not symmetry:
            return ""

        def bar(label, val):
            color = "#22c55e" if val >= 90 else "#eab308" if val >= 80 else "#ef4444"
            return f"""
            <div class="sym-row">
                <span>{label}</span>
                <div class="sym-bar"><div style="width:{val}%; background:{color}"></div></div>
                <span>{val:.0f}%</span>
            </div>"""

        bars = ""
        for key, label in [
            ("overall_score", "Overall"), ("eye_alignment", "Eyes"),
            ("cheek_balance", "Cheeks"), ("jawline_symmetry", "Jawline"),
            ("lip_symmetry", "Lips"),
        ]:
            val = symmetry.get(key, 50)
            bars += bar(label, val)

        deviation = symmetry.get("midline_deviation_mm", 0)

        return f"""
        <div class="section">
            <h2>🔄 Symmetry Analysis</h2>
            {bars}
            <p>Midline deviation: <strong>{deviation:.1f}mm</strong></p>
        </div>
        """

    def _conditions_section(self, conditions: List) -> str:
        if not conditions:
            return '<div class="section"><h2>🎯 Conditions Detected</h2><p>No significant conditions detected.</p></div>'

        items = ""
        for c in conditions:
            icon = {"severe": "🔴", "moderate": "🟡", "mild": "🟢"}.get(c.get("severity", ""), "⚪")
            items += f"""
            <div class="condition-card">
                <span class="icon">{icon}</span>
                <div>
                    <strong>{c.get('type', '').replace('_', ' ').title()}</strong>
                    <span class="severity">({c.get('severity', 'unknown')})</span>
                    <br/>Zone: {c.get('zone', '—')}
                    {f" | Count: {c['count']}" if c.get('count') else ""}
                </div>
            </div>"""

        return f'<div class="section"><h2>🎯 Conditions Detected</h2>{items}</div>'

    def _comparison_section(self, comparison: Dict) -> str:
        delta = comparison.get("overall_delta", 0)
        direction = "↑ Improved" if delta > 0 else "↓ Declined" if delta < 0 else "→ Stable"
        color = "#22c55e" if delta > 0 else "#ef4444" if delta < 0 else "#6b7280"

        improvements = comparison.get("improvements", [])[:5]
        regressions = comparison.get("regressions", [])[:5]

        imp_rows = "".join(
            f"<tr><td>✅ {i['zone']}</td><td>{i['metric']}</td><td>{i['before']:.0f} → {i['after']:.0f}</td><td style='color:#22c55e'>+{i['delta']:.1f}</td></tr>"
            for i in improvements
        )
        reg_rows = "".join(
            f"<tr><td>⚠️ {r['zone']}</td><td>{r['metric']}</td><td>{r['before']:.0f} → {r['after']:.0f}</td><td style='color:#ef4444'>{r['delta']:.1f}</td></tr>"
            for r in regressions
        )

        return f"""
        <div class="section">
            <h2>📈 Before & After Comparison</h2>
            <p style="font-size:1.3em; color:{color}"><strong>{direction}</strong> (Overall: {delta:+.1f})</p>
            {"<h3>Improvements</h3><table class='compare-table'><tbody>" + imp_rows + "</tbody></table>" if imp_rows else ""}
            {"<h3>Needs Attention</h3><table class='compare-table'><tbody>" + reg_rows + "</tbody></table>" if reg_rows else ""}
        </div>
        """

    def _recommendations_section(self, recommendations: List) -> str:
        if not recommendations:
            return ""

        items = ""
        for r in recommendations:
            icon = "🔴" if r.get("priority", 5) <= 2 else "🟡" if r.get("priority", 5) <= 4 else "🟢"
            items += f"""
            <div class="rec-card">
                <span class="priority">{icon} #{r.get('priority', '')}</span>
                <strong>{r.get('area', '')}</strong>
                <p>{r.get('suggestion', '')}</p>
            </div>"""

        return f'<div class="section"><h2>💡 Recommendations</h2>{items}</div>'

    def _doctor_notes_section(self, notes: str, annotations_b64: Optional[str]) -> str:
        content = ""
        if notes:
            content += f'<div class="notes-box"><p>{notes}</p></div>'
        if annotations_b64:
            content += f'<img src="data:image/png;base64,{annotations_b64}" class="annotations-img" />'
        if not content:
            content = "<p>No additional notes.</p>"

        return f'<div class="section"><h2>✍️ Doctor\'s Notes</h2>{content}</div>'

    def _footer_section(self) -> str:
        return f"""
        <div class="footer">
            <p class="disclaimer">{self._config.disclaimer}</p>
            <p>Generated by {self._config.clinic_name} | © {datetime.utcnow().year}</p>
        </div>
        """

    @staticmethod
    def _score_class(value) -> str:
        if isinstance(value, (int, float)):
            if value >= 80:
                return "score-good"
            elif value >= 60:
                return "score-fair"
            return "score-poor"
        return ""

    def _wrap_html(self, body: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Skin Analysis Report — {self._config.clinic_name}</title>
<style>
{self._get_styles()}
</style>
</head>
<body>
<div class="report-container">
{body}
</div>
</body>
</html>"""

    @staticmethod
    def _get_styles() -> str:
        return """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #f8fafc; color: #1e293b; line-height: 1.6; }
.report-container { max-width: 900px; margin: 0 auto; background: white; box-shadow: 0 4px 24px rgba(0,0,0,0.08); }
.header { background: linear-gradient(135deg, #1e293b, #334155); color: white; padding: 2rem; }
.header h1 { font-size: 1.8rem; margin-bottom: 0.3rem; }
.header .subtitle { opacity: 0.8; font-size: 1rem; margin-bottom: 1rem; }
.header table { width: 100%; color: white; }
.header td { padding: 0.2rem 0.5rem; font-size: 0.9rem; }
.section { padding: 1.5rem 2rem; border-bottom: 1px solid #e2e8f0; }
.section h2 { font-size: 1.3rem; margin-bottom: 1rem; color: #1e293b; }
.score-card { text-align: center; padding: 1rem; }
.big-score { font-size: 3rem; font-weight: 700; }
.score-bar { width: 100%; height: 12px; background: #e2e8f0; border-radius: 6px; margin: 1rem 0; overflow: hidden; }
.score-fill { height: 100%; border-radius: 6px; transition: width 0.5s ease; }
.small { font-size: 0.8rem; color: #94a3b8; }
.heatmap-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
.heatmap-card { text-align: center; }
.heatmap-card img { width: 100%; border-radius: 8px; border: 1px solid #e2e8f0; }
.heatmap-card p { margin-top: 0.5rem; font-weight: 600; font-size: 0.9rem; }
.legend { text-align: center; margin-top: 0.5rem; font-size: 0.85rem; color: #64748b; }
.scores-table, .measurements-table, .compare-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.scores-table th, .measurements-table th { background: #f1f5f9; padding: 0.6rem; text-align: left; font-weight: 600; }
.scores-table td, .measurements-table td, .compare-table td { padding: 0.5rem 0.6rem; border-bottom: 1px solid #f1f5f9; }
.score-good { color: #22c55e; font-weight: 600; }
.score-fair { color: #eab308; font-weight: 600; }
.score-poor { color: #ef4444; font-weight: 600; }
.sym-row { display: flex; align-items: center; gap: 0.5rem; margin: 0.4rem 0; }
.sym-row span:first-child { width: 80px; font-size: 0.9rem; }
.sym-bar { flex: 1; height: 10px; background: #e2e8f0; border-radius: 5px; overflow: hidden; }
.sym-bar div { height: 100%; border-radius: 5px; }
.condition-card { display: flex; align-items: flex-start; gap: 0.7rem; padding: 0.7rem; border-radius: 8px; background: #f8fafc; margin-bottom: 0.5rem; }
.condition-card .icon { font-size: 1.3rem; }
.severity { color: #64748b; font-size: 0.85rem; }
.rec-card { padding: 0.8rem; border-left: 4px solid #3b82f6; margin-bottom: 0.5rem; background: #f0f9ff; border-radius: 0 8px 8px 0; }
.rec-card .priority { margin-right: 0.5rem; }
.rec-card p { font-size: 0.9rem; color: #475569; margin-top: 0.3rem; }
.notes-box { background: #fffbeb; border: 1px solid #fbbf24; border-radius: 8px; padding: 1rem; }
.annotations-img { width: 100%; max-width: 500px; margin-top: 1rem; border-radius: 8px; }
.footer { padding: 1.5rem 2rem; text-align: center; color: #94a3b8; font-size: 0.8rem; }
.disclaimer { font-style: italic; margin-bottom: 0.5rem; }
@media print { body { background: white; } .report-container { box-shadow: none; } }
"""
