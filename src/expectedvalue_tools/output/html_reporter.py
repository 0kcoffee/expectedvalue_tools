"""HTML report generator for test outputs."""

import os
import base64
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import io

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None

# Keep matplotlib import for backward compatibility with existing charts
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Figure = None

import pandas as pd
import numpy as np


class HTMLReporter:
    """Generate professional HTML reports from test outputs."""

    def __init__(self, logo_path: str, output_dir: str = "reports_output"):
        """
        Initialize HTML reporter.

        Args:
            logo_path: Path to logo image file
            output_dir: Directory to save HTML reports
        """
        self.logo_path = logo_path
        self.output_dir = output_dir
        self.sections: List[Dict[str, Any]] = []
        self.test_name: Optional[str] = None
        self.strategy_name: Optional[str] = None
        self.test_kwargs: Dict[str, Any] = {}
        self.logo_base64: Optional[str] = None
        self.icon_base64: Optional[str] = None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Load logo as base64
        self._load_logo()
        
        # Load icon as base64 (try to find icon.png in same directory as logo)
        self._load_icon()

    def _load_logo(self) -> None:
        """Load logo image and convert to base64."""
        try:
            if os.path.exists(self.logo_path):
                with open(self.logo_path, "rb") as f:
                    logo_data = f.read()
                    self.logo_base64 = base64.b64encode(logo_data).decode("utf-8")
            else:
                self.logo_base64 = None
        except Exception:
            self.logo_base64 = None

    def _load_icon(self) -> None:
        """Load icon image and convert to base64 for favicon."""
        try:
            # Try to find icon.png in same directory as logo
            if self.logo_path:
                logo_dir = os.path.dirname(self.logo_path)
                icon_path = os.path.join(logo_dir, "icon.png")
                if os.path.exists(icon_path):
                    with open(icon_path, "rb") as f:
                        icon_data = f.read()
                        self.icon_base64 = base64.b64encode(icon_data).decode("utf-8")
                    return
            
            # Try common locations
            possible_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "images", "icon.png"),
                "images/icon.png",
                os.path.join(os.getcwd(), "images", "icon.png"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        icon_data = f.read()
                        self.icon_base64 = base64.b64encode(icon_data).decode("utf-8")
                    return
            
            self.icon_base64 = None
        except Exception:
            self.icon_base64 = None

    def start_report(
        self, test_name: str, strategy_name: str, test_kwargs: Optional[Dict] = None
    ) -> None:
        """
        Start a new report.

        Args:
            test_name: Name of the test
            strategy_name: Name of the strategy
            test_kwargs: Test parameters
        """
        self.test_name = test_name
        self.strategy_name = strategy_name
        self.test_kwargs = test_kwargs or {}
        self.sections = []

    def add_section(self, title: str, content: str, section_type: str = "default") -> None:
        """
        Add a content section to the report.

        Args:
            title: Section title
            content: HTML content
            section_type: Type of section (default, box, table, chart, etc.)
        """
        self.sections.append({
            "title": title,
            "content": content,
            "type": section_type,
        })

    def add_chart(self, figure: Figure, title: str) -> None:
        """
        Add a matplotlib chart to the report.

        Args:
            figure: Matplotlib figure object
            title: Chart title
        """
        if not MATPLOTLIB_AVAILABLE or figure is None:
            return

        try:
            # Convert figure to base64
            img_base64 = self._figure_to_base64(figure)
            if img_base64:
                img_html = f'<img src="data:image/png;base64,{img_base64}" alt="{title}" class="chart-image">'
                self.add_section(title, img_html, "chart")
        except Exception:
            pass

    def add_table(self, data: pd.DataFrame, title: str, color_columns: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """
        Add a DataFrame as an HTML table with optional color coding.

        Args:
            data: DataFrame to convert
            title: Table title
            color_columns: Dict mapping column names to color logic:
                - "positive_negative": Green for positive, red for negative
                - "positive": Always green
                - "negative": Always red
                - "warning": Yellow
                - "info": Cyan
            **kwargs: Additional arguments for to_html()
        """
        if data is None or len(data) == 0:
            return

        try:
            # Format numeric columns and apply color coding
            formatted_df = data.copy()
            original_df = data.copy()  # Keep original for color logic
            
            for col in formatted_df.columns:
                if formatted_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    if formatted_df[col].dtype in [np.float64, np.float32]:
                        # Format floats with 2 decimal places and apply color
                        formatted_df[col] = formatted_df[col].apply(
                            lambda x: self._format_number_with_color(x, col, color_columns, original_df) if pd.notna(x) else ""
                        )
                    else:
                        # Format integers with commas and apply color
                        formatted_df[col] = formatted_df[col].apply(
                            lambda x: self._format_number_with_color(x, col, color_columns, original_df) if pd.notna(x) else ""
                        )

            # Check if any cells contain HTML (for color coding)
            has_html = False
            for col in formatted_df.columns:
                if formatted_df[col].dtype == 'object':
                    try:
                        if formatted_df[col].astype(str).str.contains('<span', na=False).any():
                            has_html = True
                            break
                    except:
                        # Check first value as fallback
                        if len(formatted_df) > 0:
                            sample = str(formatted_df[col].iloc[0])
                            if '<span' in sample:
                                has_html = True
                                break
            
            html_table = formatted_df.to_html(
                classes="data-table",
                table_id=None,
                escape=False,  # Always don't escape to preserve HTML spans
                index=False,
                border=0,  # Remove default border attribute that creates black lines
                **kwargs
            )
            # Remove any border attribute that might have been added by pandas
            import re
            html_table = re.sub(r'\s+border=["\']?\d+["\']?', '', html_table)
            # Also remove any inline border styles that might override CSS
            html_table = re.sub(r'style="[^"]*border[^"]*"', '', html_table)
            # Generate unique ID for this table
            import uuid
            table_id = f"table_{uuid.uuid4().hex[:8]}"
            # Wrap table in scrollable container with fullscreen button
            wrapped_table = f'''
            <div class="table-wrapper" id="{table_id}_wrapper">
                <button class="table-fullscreen-btn" onclick="toggleTableFullscreen('{table_id}')">⛶ Full Screen</button>
                {html_table}
            </div>
            <div class="table-fullscreen-overlay" id="{table_id}_overlay">
                <div class="table-fullscreen-content">
                    <button class="table-fullscreen-close" onclick="toggleTableFullscreen('{table_id}')">✕ Close</button>
                    <h3 style="margin-bottom: 20px;">{title}</h3>
                    {html_table}
                </div>
            </div>
            '''
            self.add_section(title, wrapped_table, "table")
        except Exception:
            pass

    def _format_number_with_color(self, value: float, column: str, color_columns: Optional[Dict], original_df: pd.DataFrame) -> str:
        """Format a number with HTML color coding."""
        if color_columns is None or column not in color_columns:
            # Default: color code if it's a difference column
            if "diff" in column.lower() or "difference" in column.lower():
                if value >= 0:
                    return f'<span class="value-positive">${value:,.2f}</span>' if "$" not in str(value) else f'<span class="value-positive">{value:,.2f}</span>'
                else:
                    return f'<span class="value-negative">${value:,.2f}</span>' if "$" not in str(value) else f'<span class="value-negative">{value:,.2f}</span>'
            # For P/L columns
            if "p/l" in column.lower() or "pl" in column.lower():
                if value >= 0:
                    return f'<span class="value-positive">${value:,.2f}</span>'
                else:
                    return f'<span class="value-negative">${value:,.2f}</span>'
            # Default formatting
            if abs(value) >= 1:
                return f"${value:,.2f}" if "$" not in str(value) else f"{value:,.2f}"
            else:
                return f"${value:,.2f}" if "$" not in str(value) else f"{value:,.2f}"
        
        color_logic = color_columns[column]
        
        if color_logic == "positive_negative":
            if value >= 0:
                formatted = f"${value:,.2f}" if "$" not in str(value) else f"{value:,.2f}"
                return f'<span class="value-positive">{formatted}</span>'
            else:
                formatted = f"${value:,.2f}" if "$" not in str(value) else f"{value:,.2f}"
                return f'<span class="value-negative">{formatted}</span>'
        elif color_logic == "positive":
            formatted = f"${value:,.2f}" if "$" not in str(value) else f"{value:,.2f}"
            return f'<span class="value-positive">{formatted}</span>'
        elif color_logic == "negative":
            formatted = f"${value:,.2f}" if "$" not in str(value) else f"{value:,.2f}"
            return f'<span class="value-negative">{formatted}</span>'
        elif color_logic == "warning":
            formatted = f"${value:,.2f}" if "$" not in str(value) else f"{value:,.2f}"
            return f'<span class="value-warning">{formatted}</span>'
        elif color_logic == "info":
            formatted = f"${value:,.2f}" if "$" not in str(value) else f"{value:,.2f}"
            return f'<span class="value-info">{formatted}</span>'
        else:
            # Default formatting
            if abs(value) >= 1:
                return f"${value:,.2f}" if "$" not in str(value) else f"{value:,.2f}"
            else:
                return f"${value:,.2f}" if "$" not in str(value) else f"{value:,.2f}"

    def add_statistics_box(self, title: str, stats: List[tuple]) -> None:
        """
        Add a statistics box with key-value pairs.

        Args:
            title: Box title
            stats: List of (label, value, color_class) tuples
        """
        lines = []
        for stat in stats:
            if len(stat) == 3:
                label, value, color_class = stat
            elif len(stat) == 2:
                label, value = stat
                color_class = "value-default"
            else:
                continue

            lines.append(
                f'<div class="stat-line">'
                f'<span class="stat-label">{label}:</span> '
                f'<span class="stat-value {color_class}">{value}</span>'
                f'</div>'
            )

        content = '<div class="statistics-box">' + "\n".join(lines) + '</div>'
        self.add_section(title, content, "box")

    def add_ascii_visualization(self, ascii_content: str, title: str) -> None:
        """
        Add ASCII visualization content.

        Args:
            ascii_content: ASCII art/visualization text
            title: Visualization title
        """
        # Escape HTML and wrap in pre tag
        escaped = ascii_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        content = f'<pre class="ascii-visualization">{escaped}</pre>'
        self.add_section(title, content, "ascii")

    def _convert_numpy_to_list(self, obj):
        """
        Recursively convert numpy arrays and numpy scalars to Python lists/values.
        This is needed because JSON serialization of Plotly figures with numpy arrays
        creates binary data that JavaScript cannot decode.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_to_list(item) for item in obj)
        else:
            return obj

    def add_distribution_chart(self, data: np.ndarray, title: str, is_percentage: bool = False) -> None:
        """
        Add a distribution histogram chart from numpy array data using Plotly.

        Args:
            data: Array of numeric values
            title: Chart title
            is_percentage: If True, format as percentages
        """
        if not PLOTLY_AVAILABLE:
            return
        
        # Convert to numpy array if needed
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if len(data) == 0:
            return

        try:
            # Calculate statistics
            mean_val = np.mean(data)
            std_val = np.std(data)
            n = len(data)
            
            # Create histogram using Plotly
            # Convert numpy array to list to avoid binary serialization
            data_list = data.tolist() if isinstance(data, np.ndarray) else list(data)
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(go.Histogram(
                x=data_list,
                nbinsx=30,
                marker_color='#3d15e5',
                marker_line_color='#3d15e5',
                marker_line_width=0.5,
                opacity=0.7,
                name='Frequency',
                hovertemplate='Value: %{x}<br>Frequency: %{y}<extra></extra>'
            ))
            
            # Add mean line
            mean_label = f'Mean: ${mean_val:.2f}' if not is_percentage else f'Mean: {mean_val:.2f}%'
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="#3d15e5",
                line_width=2,
                annotation_text=mean_label,
                annotation_position="top"
            )
            
            # Add zero line
            fig.add_vline(
                x=0,
                line_dash="solid",
                line_color="#999999",
                line_width=1,
                opacity=0.4,
                annotation_text="Zero",
                annotation_position="bottom"
            )
            
            # Update layout
            xlabel = "Value (%)" if is_percentage else "Value ($)"
            stats_text = f'Mean: ${mean_val:.2f}<br>Std: ${std_val:.2f}<br>N: {n}' if not is_percentage else f'Mean: {mean_val:.2f}%<br>Std: {std_val:.2f}%<br>N: {n}'
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title="Frequency",
                showlegend=True,
                hovermode='x unified',
                template='plotly_white',
                annotations=[
                    dict(
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        xanchor="left", yanchor="top",
                        text=stats_text,
                        showarrow=False,
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="rgba(61, 21, 229, 0.2)",
                        borderwidth=1,
                        font=dict(size=11)
                    )
                ],
                height=350,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            # Convert to HTML div (Plotly.js will be loaded from CDN in head)
            chart_div_id = f"plotly_chart_{len(self.sections)}"
            # Get the dict representation and convert to JSON
            # Convert numpy arrays to lists to avoid binary encoding that JavaScript can't decode
            import json
            fig_dict = fig.to_dict()
            # Convert numpy arrays to lists to avoid binary encoding
            fig_dict = self._convert_numpy_to_list(fig_dict)
            fig_json = json.dumps(fig_dict)
            
            # Create the div and script tag manually
            chart_html = f'''
            <div id="{chart_div_id}" style="width:100%;height:350px;"></div>
            <script>
                var figure = {fig_json};
                Plotly.newPlot('{chart_div_id}', figure.data, figure.layout, figure.config || {{}});
            </script>
            '''
            
            self.add_section(title, chart_html, "chart")
        except Exception as e:
            # Log error but don't fail silently
            import sys
            print(f"Warning: Failed to create distribution chart '{title}': {e}", file=sys.stderr)

    def _figure_to_base64(self, figure: Figure) -> Optional[str]:
        """
        Convert matplotlib figure to base64 string.

        Args:
            figure: Matplotlib figure object

        Returns:
            Base64 encoded image string or None
        """
        if not MATPLOTLIB_AVAILABLE or figure is None:
            return None

        try:
            buf = io.BytesIO()
            figure.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            img_data = buf.read()
            buf.close()
            return base64.b64encode(img_data).decode("utf-8")
        except Exception:
            return None

    def _get_css(self) -> str:
        """Get CSS styles for the report."""
        return """
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f5f5f5;
                padding: 20px;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
                border-radius: 12px;
                border: 1px solid #e5e7eb;
            }

            header {
                background-color: #ffffff;
                border-bottom: 1px solid #e5e7eb;
                padding: 32px 30px;
                text-align: center;
            }

            .logo-container {
                margin-bottom: 20px;
            }

            .logo-container a {
                display: inline-block;
                transition: opacity 0.2s;
            }

            .logo-container a:hover {
                opacity: 0.8;
            }

            .logo-container img {
                max-height: 80px;
                max-width: 300px;
                cursor: pointer;
            }

            header h1 {
                font-size: 2em;
                margin-bottom: 10px;
                font-weight: 600;
                color: #3d15e5;
            }

            header .subtitle {
                font-size: 1.1em;
                color: #6c757d;
            }

            .metadata {
                background-color: #fafafa;
                padding: 24px;
                border-bottom: 1px solid #e5e7eb;
            }

            .metadata-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }

            .metadata-item {
                display: flex;
                flex-direction: column;
            }

            .metadata-label {
                font-size: 0.85em;
                color: #6c757d;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 5px;
            }

            .metadata-value {
                font-size: 1.1em;
                font-weight: 600;
                color: #212529;
            }

            main {
                padding: 30px;
                overflow-x: visible;
            }

            .section {
                margin-bottom: 40px;
            }

            .section-title {
                font-size: 1.375rem;
                font-weight: 600;
                color: #111827;
                margin-bottom: 24px;
                padding-bottom: 12px;
                border-bottom: 1px solid #e5e7eb;
            }

            .section-content {
                margin-top: 15px;
            }

            .statistics-box {
                background-color: #fafafa;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 24px;
                margin: 20px 0;
            }

            .stat-line {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 0;
                border-bottom: 1px solid #f3f4f6;
            }

            .stat-line:last-child {
                border-bottom: none;
            }

            .stat-label {
                font-weight: 500;
                color: #6b7280;
                font-size: 0.9375rem;
            }

            .stat-value {
                font-weight: 600;
                font-size: 1rem;
            }

            .value-default {
                color: #212529;
            }

            .value-positive {
                color: #28a745;
            }

            .value-negative {
                color: #dc3545;
            }

            .value-warning {
                color: #ffc107;
            }

            .value-info {
                color: #17a2b8;
            }

            .table-wrapper {
                overflow-x: auto;
                margin: 24px 0;
                -webkit-overflow-scrolling: touch;
                border-radius: 8px;
                border: 1px solid #e8e8e8;
                background-color: white;
                position: relative;
            }
            
            .table-fullscreen-btn {
                position: absolute;
                top: 8px;
                right: 8px;
                background-color: #3d15e5;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 0.75rem;
                cursor: pointer;
                z-index: 10;
                opacity: 0.8;
                transition: opacity 0.2s;
            }
            
            .table-fullscreen-btn:hover {
                opacity: 1;
            }
            
            .table-fullscreen-overlay {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.9);
                z-index: 9999;
                padding: 20px;
                overflow: auto;
            }
            
            .table-fullscreen-overlay.active {
                display: block;
            }
            
            .table-fullscreen-content {
                max-width: 95%;
                margin: 0 auto;
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                position: relative;
            }
            
            .table-fullscreen-close {
                position: absolute;
                top: 10px;
                right: 10px;
                background-color: #3d15e5;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 0.875rem;
                cursor: pointer;
                z-index: 10;
            }
            
            .table-fullscreen-close:hover {
                background-color: #2d0fb8;
            }

            .data-table {
                width: 100%;
                min-width: 100%;
                border-collapse: collapse;
                font-size: 0.875rem;
                background-color: white;
                border: none !important;
            }
            
            /* Ensure no black borders anywhere in tables */
            .data-table,
            .data-table th,
            .data-table td,
            .data-table tr {
                border-color: #e8e8e8 !important;
            }
            
            /* Override any default browser table styling */
            table.data-table {
                border: none !important;
            }
            
            table.data-table * {
                border-color: #e8e8e8 !important;
            }

            .data-table thead {
                background-color: #f9fafb;
                border-bottom: 1px solid #e8e8e8 !important;
            }

            .data-table thead tr {
                background-color: #f9fafb;
            }

            .data-table th {
                padding: 0.75rem 0.875rem;
                text-align: left;
                font-weight: 600;
                font-size: 0.6875rem;
                letter-spacing: 0.025em;
                text-transform: uppercase;
                color: #6b7280;
                border: none !important;
                border-bottom: 1px solid #e8e8e8 !important;
            }

            .data-table tbody tr {
                border: none !important;
                border-bottom: 1px solid #f0f0f0 !important;
                transition: background-color 0.1s ease-in-out;
            }

            .data-table tbody tr:last-child {
                border-bottom: none !important;
            }

            .data-table tbody tr:hover {
                background-color: #f9fafb;
            }

            .data-table td {
                padding: 0.875rem 1rem;
                color: #111827;
                font-size: 0.875rem;
                border: none !important;
                border-bottom: 1px solid #f0f0f0 !important;
                white-space: nowrap;
            }

            .data-table tbody tr:nth-child(even) {
                background-color: #fafafa;
            }

            .data-table tbody tr:nth-child(even):hover {
                background-color: #f5f5f5;
            }

            .chart-image {
                max-width: 100%;
                height: auto;
                display: block;
                margin: 20px auto;
                border-radius: 6px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }

            .ascii-visualization {
                background-color: #1e1e1e;
                color: #d4d4d4;
                padding: 20px;
                border-radius: 6px;
                font-family: 'Courier New', monospace;
                font-size: 0.85em;
                line-height: 1.4;
                overflow-x: auto;
                white-space: pre;
            }

            .calendar-grid {
                margin: 20px 0;
            }

            .calendar-row {
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                margin-bottom: 12px;
            }

            .calendar-cell {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-width: 110px;
                padding: 10px 8px;
                border-radius: 6px;
                border: 2px solid;
                background-color: #fafafa;
                transition: transform 0.2s, box-shadow 0.2s;
                cursor: pointer;
            }

            .calendar-cell:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }

            .calendar-cell.value-positive {
                border-color: #28a745;
                background-color: #d4edda;
            }

            .calendar-cell.value-negative {
                border-color: #dc3545;
                background-color: #f8d7da;
            }

            .calendar-symbol {
                font-size: 1.5em;
                font-weight: bold;
                margin-bottom: 4px;
            }

            .calendar-cell.value-positive .calendar-symbol {
                color: #28a745;
            }

            .calendar-cell.value-negative .calendar-symbol {
                color: #dc3545;
            }

            .calendar-label {
                font-size: 0.75rem;
                font-weight: 600;
                color: #333;
                text-align: center;
                margin-bottom: 4px;
            }

            .calendar-pl {
                font-size: 0.7rem;
                font-weight: 600;
                text-align: center;
                margin-top: 2px;
            }

            .calendar-cell.value-positive .calendar-pl {
                color: #28a745;
            }

            .calendar-cell.value-negative .calendar-pl {
                color: #dc3545;
            }

            footer {
                background-color: #fafafa;
                padding: 24px;
                text-align: center;
                color: #6b7280;
                font-size: 0.875rem;
                border-top: 1px solid #e5e7eb;
            }

            .disclaimer {
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #dee2e6;
                font-style: italic;
            }

            @media print {
                body {
                    background-color: white;
                    padding: 0;
                }

                .container {
                    box-shadow: none;
                }

                .section {
                    page-break-inside: avoid;
                }
            }

            @media (max-width: 768px) {
                body {
                    padding: 10px;
                }

                header h1 {
                    font-size: 1.5em;
                }

                .metadata-grid {
                    grid-template-columns: 1fr;
                }

                .data-table {
                    font-size: 0.8em;
                }

                .data-table th,
                .data-table td {
                    padding: 8px;
                }
            }
        </style>
        """

    def _generate_html(self) -> str:
        """Generate complete HTML document."""
        # Header with logo
        logo_html = ""
        if self.logo_base64:
            logo_html = f'<div class="logo-container"><a href="https://expectedvalue.trade" target="_blank" rel="noopener noreferrer"><img src="data:image/png;base64,{self.logo_base64}" alt="Logo"></a></div>'

        header_html = f"""
        <header>
            {logo_html}
            <h1>Test Report: {self.test_name}</h1>
            <div class="subtitle">{self.strategy_name}</div>
        </header>
        """

        # Metadata section
        metadata_items = []
        metadata_items.append(('<div class="metadata-item">', '<span class="metadata-label">Test</span>', f'<span class="metadata-value">{self.test_name}</span>', '</div>'))
        metadata_items.append(('<div class="metadata-item">', '<span class="metadata-label">Strategy</span>', f'<span class="metadata-value">{self.strategy_name}</span>', '</div>'))
        metadata_items.append(('<div class="metadata-item">', '<span class="metadata-label">Generated</span>', f'<span class="metadata-value">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</span>', '</div>'))

        if self.test_kwargs:
            for key, value in self.test_kwargs.items():
                if value is not None:
                    # Format boolean values
                    if isinstance(value, bool):
                        display_value = "Yes" if value else "No"
                    else:
                        display_value = value
                    metadata_items.append(('<div class="metadata-item">', f'<span class="metadata-label">{key.replace("_", " ").title()}</span>', f'<span class="metadata-value">{display_value}</span>', '</div>'))

        metadata_html = f"""
        <div class="metadata">
            <div class="metadata-grid">
                {''.join(''.join(item) for item in metadata_items)}
            </div>
        </div>
        """

        # Sections
        sections_html = ""
        for section in self.sections:
            sections_html += f"""
            <div class="section">
                <h2 class="section-title">{section['title']}</h2>
                <div class="section-content">
                    {section['content']}
                </div>
            </div>
            """

        # Footer
        footer_html = """
        <footer>
            <div>Generated by Expected Value Tools</div>
            <div class="disclaimer">
                DISCLAIMER: This tool is provided for educational purposes only.
                The analysis and results should not be considered as financial advice.
                Always perform your own due diligence and consult with qualified
                professionals before making any trading or investment decisions.
            </div>
        </footer>
        """

        # Complete HTML
        plotly_script = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>' if PLOTLY_AVAILABLE else ''
        table_fullscreen_script = """
    <script>
        function toggleTableFullscreen(tableId) {
            const overlay = document.getElementById(tableId + '_overlay');
            if (overlay) {
                overlay.classList.toggle('active');
            }
        }
        
        // Close fullscreen on Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                const overlays = document.querySelectorAll('.table-fullscreen-overlay.active');
                overlays.forEach(overlay => {
                    overlay.classList.remove('active');
                });
            }
        });
    </script>
"""
        # Favicon
        favicon_html = ""
        if self.icon_base64:
            favicon_html = f'<link rel="icon" type="image/png" href="data:image/png;base64,{self.icon_base64}">'
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Report - {self.test_name}</title>
    {favicon_html}
    {plotly_script}
    {self._get_css()}
</head>
<body>
    <div class="container">
        {header_html}
        {metadata_html}
        <main>
            {sections_html}
        </main>
        {footer_html}
    </div>
    {table_fullscreen_script}
</body>
</html>
"""

        return html

    def save_report(self, filename: Optional[str] = None) -> str:
        """
        Save the HTML report to a file.

        Args:
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved report file
        """
        if filename is None:
            # Generate filename from test name, strategy, and timestamp
            safe_test = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in self.test_name)
            safe_strategy = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in self.strategy_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_test}_{safe_strategy}_{timestamp}.html"

        # Ensure filename ends with .html
        if not filename.endswith(".html"):
            filename += ".html"

        filepath = os.path.join(self.output_dir, filename)
        html_content = self._generate_html()

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return filepath
