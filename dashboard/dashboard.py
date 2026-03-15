"""
Plotly Dash Dashboard — real-time performance monitoring.

Displays:
  • Portfolio equity curve
  • Win rate / Sharpe / Profit Factor / Brier Score
  • Open positions table
  • Trade history chart
  • Sentiment distribution
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data.trade_logger import TradeLogger
from utils.config import config
from utils.helpers import get_logger, max_drawdown, sharpe_ratio

logger = get_logger("dashboard")

try:
    import dash
    from dash import dcc, html, dash_table
    from dash.dependencies import Input, Output
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logger.warning("Dash not installed. Run: pip install dash")


def build_dashboard() -> "dash.Dash":
    """Construct the Dash application object."""
    if not DASH_AVAILABLE:
        raise ImportError("Install dash: pip install dash")

    app = dash.Dash(
        __name__,
        title="PM Bot Dashboard",
        update_title=None,
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )

    app.layout = html.Div(
        style={"fontFamily": "Inter, sans-serif", "backgroundColor": "#0f1117", "minHeight": "100vh"},
        children=[
            # ── Header ────────────────────────────────────────────────────────
            html.Div(
                style={"backgroundColor": "#1a1d2e", "padding": "20px 32px",
                       "borderBottom": "1px solid #2d3148"},
                children=[
                    html.H1("🤖 Prediction Market Bot",
                            style={"color": "#e2e8f0", "margin": 0, "fontSize": "24px"}),
                    html.P("Paper Trading Dashboard — Live Performance Monitoring",
                           style={"color": "#94a3b8", "margin": "4px 0 0 0", "fontSize": "13px"}),
                ],
            ),

            # ── Auto-refresh ──────────────────────────────────────────────────
            dcc.Interval(id="refresh", interval=10_000, n_intervals=0),

            # ── KPI Cards ─────────────────────────────────────────────────────
            html.Div(id="kpi-cards", style={"padding": "20px 32px 0"}),

            # ── Charts row ────────────────────────────────────────────────────
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                       "gap": "20px", "padding": "20px 32px"},
                children=[
                    html.Div([
                        html.H3("Equity Curve", style={"color": "#94a3b8", "fontSize": "14px",
                                                        "margin": "0 0 12px 0"}),
                        dcc.Graph(id="equity-chart", style={"height": "280px"},
                                  config={"displayModeBar": False}),
                    ], style={"backgroundColor": "#1a1d2e", "borderRadius": "12px",
                              "padding": "20px", "border": "1px solid #2d3148"}),

                    html.Div([
                        html.H3("PnL Distribution", style={"color": "#94a3b8", "fontSize": "14px",
                                                             "margin": "0 0 12px 0"}),
                        dcc.Graph(id="pnl-chart", style={"height": "280px"},
                                  config={"displayModeBar": False}),
                    ], style={"backgroundColor": "#1a1d2e", "borderRadius": "12px",
                              "padding": "20px", "border": "1px solid #2d3148"}),
                ],
            ),

            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                       "gap": "20px", "padding": "0 32px 20px"},
                children=[
                    html.Div([
                        html.H3("Edge vs Outcome", style={"color": "#94a3b8", "fontSize": "14px",
                                                           "margin": "0 0 12px 0"}),
                        dcc.Graph(id="edge-chart", style={"height": "280px"},
                                  config={"displayModeBar": False}),
                    ], style={"backgroundColor": "#1a1d2e", "borderRadius": "12px",
                              "padding": "20px", "border": "1px solid #2d3148"}),

                    html.Div([
                        html.H3("Sentiment Distribution", style={"color": "#94a3b8", "fontSize": "14px",
                                                                   "margin": "0 0 12px 0"}),
                        dcc.Graph(id="sentiment-chart", style={"height": "280px"},
                                  config={"displayModeBar": False}),
                    ], style={"backgroundColor": "#1a1d2e", "borderRadius": "12px",
                              "padding": "20px", "border": "1px solid #2d3148"}),
                ],
            ),

            # ── Trade table ───────────────────────────────────────────────────
            html.Div(
                style={"padding": "0 32px 32px"},
                children=[
                    html.Div([
                        html.H3("Recent Trades", style={"color": "#94a3b8", "fontSize": "14px",
                                                         "margin": "0 0 12px 0"}),
                        html.Div(id="trade-table"),
                    ], style={"backgroundColor": "#1a1d2e", "borderRadius": "12px",
                              "padding": "20px", "border": "1px solid #2d3148"}),
                ],
            ),
        ],
    )

    # ── Callbacks ──────────────────────────────────────────────────────────────

    @app.callback(
        [Output("kpi-cards", "children"),
         Output("equity-chart", "figure"),
         Output("pnl-chart", "figure"),
         Output("edge-chart", "figure"),
         Output("sentiment-chart", "figure"),
         Output("trade-table", "children")],
        Input("refresh", "n_intervals"),
    )
    def update_dashboard(_):
        import os
        _db = "data/demo_trades.db" if os.path.exists("data/demo_trades.db") else config.db_path
        logger_db = TradeLogger(_db)
        trades_df = logger_db.get_trades()
        research_df = logger_db.get_research()
        closed = trades_df[trades_df["status"] == "closed"].copy()
        open_pos = trades_df[trades_df["status"] == "open"].copy()

        # ── KPI Cards ─────────────────────────────────────────────────────────
        total_pnl = closed["pnl"].sum() if len(closed) else 0.0
        win_rate = (closed["pnl"] > 0).mean() if len(closed) else 0.0
        pnl_arr = closed["pnl"].values if len(closed) else np.array([0.0])
        pf = (pnl_arr[pnl_arr > 0].sum() / abs(pnl_arr[pnl_arr < 0].sum())
              if any(pnl_arr < 0) else float("inf"))
        sr = sharpe_ratio(pnl_arr / 100) if len(pnl_arr) > 1 else 0.0
        brier = None
        if len(closed) and "model_prob" in closed.columns and "outcome" in closed.columns:
            valid = closed.dropna(subset=["model_prob", "outcome"])
            if len(valid):
                brier = float(np.mean((valid["model_prob"] - valid["outcome"]) ** 2))

        def kpi_card(title, value, colour="#6366f1"):
            return html.Div([
                html.P(title, style={"color": "#94a3b8", "fontSize": "12px",
                                     "margin": "0 0 4px 0", "textTransform": "uppercase"}),
                html.H2(value, style={"color": colour, "fontSize": "28px", "margin": 0}),
            ], style={"backgroundColor": "#1a1d2e", "borderRadius": "12px",
                      "padding": "16px 20px", "border": "1px solid #2d3148",
                      "flex": "1", "minWidth": "140px"})

        kpis = html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
            kpi_card("Total PnL", f"${total_pnl:+,.2f}",
                     "#22c55e" if total_pnl >= 0 else "#ef4444"),
            kpi_card("Win Rate", f"{win_rate:.1%}"),
            kpi_card("Closed Trades", str(len(closed))),
            kpi_card("Open Positions", str(len(open_pos)), "#f59e0b"),
            kpi_card("Profit Factor", f"{pf:.2f}" if pf != float("inf") else "∞"),
            kpi_card("Sharpe Ratio", f"{sr:.2f}"),
            kpi_card("Brier Score", f"{brier:.3f}" if brier is not None else "—", "#a78bfa"),
        ])

        # ── Equity curve ───────────────────────────────────────────────────────
        fig_equity = go.Figure()
        if len(closed):
            closed_sorted = closed.sort_values("opened_at")
            equity = config.bankroll + closed_sorted["pnl"].cumsum()
            fig_equity.add_trace(go.Scatter(
                x=list(range(len(equity))), y=equity.values,
                fill="tozeroy", line=dict(color="#6366f1", width=2),
                name="Equity",
            ))
        _style_fig(fig_equity)
        fig_equity.update_yaxes(tickprefix="$")

        # ── PnL histogram ──────────────────────────────────────────────────────
        fig_pnl = go.Figure()
        if len(closed):
            colours = ["#22c55e" if p > 0 else "#ef4444" for p in closed["pnl"]]
            fig_pnl.add_trace(go.Bar(
                x=list(range(len(closed))),
                y=closed.sort_values("opened_at")["pnl"].values,
                marker_color=colours,
                name="PnL",
            ))
        _style_fig(fig_pnl)
        fig_pnl.update_yaxes(tickprefix="$")

        # ── Edge vs outcome scatter ────────────────────────────────────────────
        fig_edge = go.Figure()
        if len(closed) and "edge" in closed.columns:
            won = closed[closed["pnl"] > 0]
            lost = closed[closed["pnl"] <= 0]
            fig_edge.add_trace(go.Scatter(
                x=won["edge"], y=won["pnl"], mode="markers",
                marker=dict(color="#22c55e", size=8), name="Win",
            ))
            fig_edge.add_trace(go.Scatter(
                x=lost["edge"], y=lost["pnl"], mode="markers",
                marker=dict(color="#ef4444", size=8), name="Loss",
            ))
        _style_fig(fig_edge)
        fig_edge.update_xaxes(title_text="Edge", title_font=dict(color="#94a3b8"))
        fig_edge.update_yaxes(title_text="PnL ($)", title_font=dict(color="#94a3b8"),
                               tickprefix="$")

        # ── Sentiment pie ──────────────────────────────────────────────────────
        fig_sent = go.Figure()
        if len(research_df):
            counts = research_df["sentiment_label"].value_counts()
            fig_sent.add_trace(go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                hole=0.5,
                marker=dict(colors=["#22c55e", "#ef4444", "#f59e0b"]),
            ))
        _style_fig(fig_sent)

        # ── Trade table ───────────────────────────────────────────────────────
        display_df = trades_df.head(20)
        cols = [c for c in ["trade_id", "market_id", "direction", "entry_price",
                              "size_usd", "pnl", "status", "opened_at"]
                if c in display_df.columns]
        table = dash_table.DataTable(
            data=display_df[cols].to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in cols],
            style_header={"backgroundColor": "#2d3148", "color": "#e2e8f0",
                          "fontWeight": "600", "border": "none", "fontSize": "12px"},
            style_cell={"backgroundColor": "#1a1d2e", "color": "#cbd5e1",
                        "border": "1px solid #2d3148", "fontSize": "12px", "padding": "8px"},
            style_data_conditional=[
                {"if": {"filter_query": "{pnl} > 0", "column_id": "pnl"},
                 "color": "#22c55e"},
                {"if": {"filter_query": "{pnl} < 0", "column_id": "pnl"},
                 "color": "#ef4444"},
            ],
            page_size=10,
        ) if len(display_df) else html.P("No trades yet.", style={"color": "#94a3b8"})

        return kpis, fig_equity, fig_pnl, fig_edge, fig_sent, table

    return app


def _style_fig(fig: go.Figure) -> None:
    """Apply dark theme styling to a figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=11),
        margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        showlegend=True,
    )
    fig.update_xaxes(gridcolor="#2d3148", zerolinecolor="#2d3148")
    fig.update_yaxes(gridcolor="#2d3148", zerolinecolor="#2d3148")


if __name__ == "__main__":
    if not DASH_AVAILABLE:
        print("Dash not installed. Run: pip install dash")
        sys.exit(1)
    dashboard = build_dashboard()
    print(f"Dashboard running at http://localhost:{config.dashboard_port}")
    dashboard.run(debug=False, port=config.dashboard_port, host="0.0.0.0")
