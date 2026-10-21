"""Frontend CSS styles and theme definitions."""

DASHBOARD_CSS = """
<style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2638 0%, #151a28 100%);
        border: 1px solid #2e3852;
        border-radius: 10px;
        padding: 20px;
        color: white;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 5px;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 5px;
    }
    .badge-success {
        background-color: #065f46;
        color: #34d399;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #78350f;
        color: #fbbf24;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-info {
        background-color: #1e3a8a;
        color: #60a5fa;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
"""
