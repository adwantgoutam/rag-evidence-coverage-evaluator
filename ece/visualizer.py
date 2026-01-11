"""Module for generating HTML visualization reports.

Author: Goutam Adwant (gadwant)

This module creates interactive HTML reports for evidence coverage
evaluation results, including color-coded claim analyses, highlighted
answer spans, and citation quality metrics.
"""

from typing import List, Optional
from pathlib import Path
from datetime import datetime
from ece.models import EvaluationResult, ClaimAnalysis, UnsupportedClaim


class HTMLVisualizer:
    """Generates HTML visualization reports for evaluation results.
    
    Creates comprehensive HTML reports with visual indicators for
    claim support status, evidence snippets, and actionable feedback.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        pass
    
    def generate_report(
        self,
        result: EvaluationResult,
        answer: str,
        output_path: str,
        title: str = "Evidence Coverage Evaluation Report"
    ) -> str:
        """Generate an HTML report.
        
        Args:
            result: Evaluation result to visualize
            answer: Original answer text
            output_path: Path to save HTML file
            title: Report title
            
        Returns:
            Path to generated HTML file
        """
        html_content = self._generate_html(result, answer, title)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        return str(output_file)
    
    def _generate_html(
        self,
        result: EvaluationResult,
        answer: str,
        title: str
    ) -> str:
        """Generate HTML content.
        
        Args:
            result: Evaluation result
            answer: Original answer text
            title: Report title
            
        Returns:
            Complete HTML document string
        """
        coverage_color = self._get_coverage_color(result.coverage_score)
        
        summary_section = self._generate_summary_section(result, coverage_color)
        answer_section = self._generate_answer_section(answer, result)
        claims_section = self._generate_claims_section(result)
        feedback_section = self._generate_feedback_section(result)
        citation_section = self._generate_citation_section(result)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        {self._get_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="timestamp">Generated on {self._get_timestamp()}</p>
        </header>
        
        {summary_section}
        {answer_section}
        {claims_section}
        {feedback_section}
        {citation_section}
    </div>
</body>
</html>"""
        
        return html
    
    def _generate_summary_section(self, result: EvaluationResult, color: str) -> str:
        """Generate summary section with key metrics."""
        coverage_percent = result.coverage_score * 100
        
        return f"""
        <section class="summary">
            <h2>Summary</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value" style="color: {color};">
                        {coverage_percent:.1f}%
                    </div>
                    <div class="metric-label">Evidence Coverage</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{result.total_claims}</div>
                    <div class="metric-label">Total Claims</div>
                </div>
                <div class="metric">
                    <div class="metric-value" style="color: #4CAF50;">
                        {result.supported_claims}
                    </div>
                    <div class="metric-label">Supported</div>
                </div>
                <div class="metric">
                    <div class="metric-value" style="color: #f44336;">
                        {len(result.unsupported_claims)}
                    </div>
                    <div class="metric-label">Unsupported</div>
                </div>
            </div>
        </section>"""
    
    def _generate_answer_section(self, answer: str, result: EvaluationResult) -> str:
        """Generate answer section with highlighted unsupported spans."""
        highlighted_answer = self._highlight_unsupported_spans(answer, result.unsupported_claims)
        
        return f"""
        <section class="answer">
            <h2>Answer Analysis</h2>
            <div class="answer-text">
                {highlighted_answer}
            </div>
            <div class="legend">
                <span class="legend-item">
                    <span class="legend-color supported"></span>
                    Supported
                </span>
                <span class="legend-item">
                    <span class="legend-color unsupported"></span>
                    Unsupported
                </span>
            </div>
        </section>"""
    
    def _highlight_unsupported_spans(
        self,
        answer: str,
        unsupported: List[UnsupportedClaim]
    ) -> str:
        """Highlight unsupported spans in the answer.
        
        Args:
            answer: Original answer text
            unsupported: List of unsupported claims
            
        Returns:
            HTML with highlighted spans
        """
        sorted_unsupported = sorted(unsupported, key=lambda x: x.span[0], reverse=True)
        
        highlighted = answer
        for claim in sorted_unsupported:
            start, end = claim.span
            span_text = answer[start:end]
            highlighted_span = f'<mark class="unsupported" title="{self._escape_html(claim.missing_info or "No evidence")}">{self._escape_html(span_text)}</mark>'
            highlighted = highlighted[:start] + highlighted_span + highlighted[end:]
        
        return highlighted
    
    def _generate_claims_section(self, result: EvaluationResult) -> str:
        """Generate detailed claims section."""
        claims_html = []
        
        for i, analysis in enumerate(result.claim_analysis, 1):
            status_class = "supported" if analysis.supported else "unsupported"
            status_icon = "✓" if analysis.supported else "✗"
            status_text = "Supported" if analysis.supported else "Unsupported"
            
            support_snippets_html = ""
            if analysis.supporting_snippets:
                support_snippets_html = "<div class='supporting-snippets'><strong>Supporting Evidence:</strong><ul>"
                for snippet in analysis.supporting_snippets[:3]:
                    support_snippets_html += f"""
                    <li>
                        <span class="passage-id">[{snippet.passage_id}]</span>
                        <span class="snippet-text">{self._escape_html(snippet.text[:200])}...</span>
                        <span class="score">(score: {snippet.score:.2f})</span>
                    </li>"""
                support_snippets_html += "</ul></div>"
            
            missing_info_html = ""
            if analysis.missing_info and not analysis.supported:
                missing_info_html = f'<div class="missing-info"><strong>Missing:</strong> {self._escape_html(analysis.missing_info)}</div>'
            
            claims_html.append(f"""
            <div class="claim-item {status_class}">
                <div class="claim-header">
                    <span class="claim-number">Claim {i}</span>
                    <span class="claim-status {status_class}">
                        {status_icon} {status_text} (score: {analysis.support_score:.2f})
                    </span>
                </div>
                <div class="claim-text">{self._escape_html(analysis.claim.text)}</div>
                {support_snippets_html}
                {missing_info_html}
            </div>""")
        
        return f"""
        <section class="claims">
            <h2>Claim-by-Claim Analysis</h2>
            <div class="claims-list">
                {''.join(claims_html)}
            </div>
        </section>"""
    
    def _generate_feedback_section(self, result: EvaluationResult) -> str:
        """Generate feedback section."""
        if not result.feedback:
            return ""
        
        feedback_items = "".join([
            f'<li>{self._escape_html(fb)}</li>' for fb in result.feedback
        ])
        
        return f"""
        <section class="feedback">
            <h2>Actionable Feedback</h2>
            <ul class="feedback-list">
                {feedback_items}
            </ul>
        </section>"""
    
    def _generate_citation_section(self, result: EvaluationResult) -> str:
        """Generate citation analysis section."""
        citation_analysis = result.metadata.get("citation_analysis")
        if not citation_analysis:
            return ""
        
        overall_quality = citation_analysis.get("overall_citation_quality", 0.0)
        spam_score = citation_analysis.get("citation_spam_score", 0.0)
        total_citations = citation_analysis.get("total_citations", 0)
        
        quality_color = self._get_coverage_color(overall_quality)
        spam_color = "#f44336" if spam_score > 0.5 else "#4CAF50"
        
        return f"""
        <section class="citations">
            <h2>Citation Quality Analysis</h2>
            <div class="citation-metrics">
                <div class="metric">
                    <div class="metric-value" style="color: {quality_color};">
                        {overall_quality * 100:.1f}%
                    </div>
                    <div class="metric-label">Citation Quality</div>
                </div>
                <div class="metric">
                    <div class="metric-value" style="color: {spam_color};">
                        {spam_score * 100:.1f}%
                    </div>
                    <div class="metric-label">Citation Spam Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{total_citations}</div>
                    <div class="metric-label">Total Citations</div>
                </div>
            </div>
        </section>"""
    
    def _get_coverage_color(self, score: float) -> str:
        """Get color based on coverage score.
        
        Args:
            score: Coverage score (0-1)
            
        Returns:
            CSS color string
        """
        if score >= 0.8:
            return "#4CAF50"  # Green
        elif score >= 0.5:
            return "#FF9800"  # Orange
        else:
            return "#f44336"  # Red
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters.
        
        Args:
            text: Raw text
            
        Returns:
            HTML-escaped text
        """
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#x27;"))
    
    def _get_css(self) -> str:
        """Get CSS styles for the report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        header {
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        
        h1 {
            color: #2196F3;
            margin-bottom: 10px;
        }
        
        .timestamp {
            color: #666;
            font-size: 0.9em;
        }
        
        section {
            margin-bottom: 40px;
        }
        
        h2 {
            color: #2196F3;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .metric {
            text-align: center;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 8px;
        }
        
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .metric-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
        }
        
        .answer-text {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            line-height: 1.8;
            font-size: 1.1em;
        }
        
        mark.unsupported {
            background-color: #ffebee;
            color: #c62828;
            padding: 2px 4px;
            border-radius: 3px;
            cursor: help;
        }
        
        .legend {
            margin-top: 15px;
            display: flex;
            gap: 20px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }
        
        .legend-color.supported {
            background-color: #e8f5e9;
        }
        
        .legend-color.unsupported {
            background-color: #ffebee;
        }
        
        .claims-list {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .claim-item {
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid;
        }
        
        .claim-item.supported {
            background-color: #e8f5e9;
            border-left-color: #4CAF50;
        }
        
        .claim-item.unsupported {
            background-color: #ffebee;
            border-left-color: #f44336;
        }
        
        .claim-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .claim-number {
            font-weight: bold;
            color: #666;
        }
        
        .claim-status {
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        
        .claim-status.supported {
            background-color: #4CAF50;
            color: white;
        }
        
        .claim-status.unsupported {
            background-color: #f44336;
            color: white;
        }
        
        .claim-text {
            margin-bottom: 15px;
            font-size: 1.05em;
        }
        
        .supporting-snippets {
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 4px;
        }
        
        .supporting-snippets ul {
            list-style: none;
            margin-top: 10px;
        }
        
        .supporting-snippets li {
            padding: 10px;
            margin-bottom: 10px;
            background: #f5f5f5;
            border-radius: 4px;
            border-left: 3px solid #2196F3;
        }
        
        .passage-id {
            font-weight: bold;
            color: #2196F3;
            margin-right: 10px;
        }
        
        .snippet-text {
            color: #555;
        }
        
        .score {
            float: right;
            color: #999;
            font-size: 0.9em;
        }
        
        .missing-info {
            margin-top: 15px;
            padding: 10px;
            background: #fff3cd;
            border-left: 3px solid #ffc107;
            border-radius: 4px;
        }
        
        .feedback-list {
            list-style: none;
            padding: 0;
        }
        
        .feedback-list li {
            padding: 15px;
            margin-bottom: 10px;
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
        }
        
        .citation-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        """
