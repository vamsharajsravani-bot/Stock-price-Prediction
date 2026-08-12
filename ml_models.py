import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
import warnings
warnings.filterwarnings('ignore')


# ── Styling ────────────────────────────────────────────────────────────────────
BG      = '#0d0d0d'
SURFACE = '#161616'
ACCENT  = '#e8e8e8'
DIM     = '#555555'
GREEN   = '#4ade80'
RED     = '#f87171'
BLUE    = '#60a5fa'


def _base_fig(w=12, h=4.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=DIM, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a2a')
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
    ax.grid(color='#1e1e1e', linewidth=0.6, linestyle='--')
    return fig, ax


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def fetch_data(ticker: str, period: str = '1y') -> pd.DataFrame:
    """Fetch stock data and return a clean single-column DataFrame."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Please check the symbol.")

    # Handle MultiIndex columns (newer yfinance versions return MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if 'Close' not in df.columns:
        raise ValueError(f"Could not find 'Close' price column for '{ticker}'.")

    result = df[['Close']].copy().dropna()
    result.columns = ['Close']

    # Flatten any remaining multi-dimensional values
    result['Close'] = result['Close'].values.flatten()
    return result


def _get_closes(df: pd.DataFrame) -> np.ndarray:
    """Safely extract close prices as a 1D float numpy array."""
    vals = df['Close'].values
    if vals.ndim > 1:
        vals = vals.flatten()
    return vals.astype(float)


def plot_history(df: pd.DataFrame, ticker: str) -> str:
    closes = _get_closes(df)
    fig, ax = _base_fig()
    color = GREEN if closes[-1] >= closes[0] else RED
    ax.plot(df.index, closes, color=color, linewidth=1.4, alpha=0.9)
    ax.fill_between(df.index, closes, closes.min(), color=color, alpha=0.06)
    ax.set_title(f'{ticker.upper()}  —  Price History',
                 color=ACCENT, fontsize=11, pad=12, fontweight='500')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate(rotation=30, ha='right')
    return _fig_to_b64(fig)


# ── Linear Regression ──────────────────────────────────────────────────────────
def linear_regression_predict(df: pd.DataFrame, days: int = 30):
    closes = _get_closes(df).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(closes)

    X = np.arange(len(scaled)).reshape(-1, 1)
    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr = scaled[:split]

    model = LinearRegression()
    model.fit(X_tr, y_tr)

    test_pred   = scaler.inverse_transform(model.predict(X_te))
    future_X    = np.arange(len(X), len(X) + days).reshape(-1, 1)
    future_pred = scaler.inverse_transform(model.predict(future_X))

    mae  = mean_absolute_error(closes[split:], test_pred)
    rmse = np.sqrt(mean_squared_error(closes[split:], test_pred))

    # chart
    fig, ax = _base_fig()
    ax.plot(df.index, closes, color=DIM, linewidth=1, label='Actual')
    ax.plot(df.index[split:], test_pred, color=BLUE, linewidth=1.4, label='Test Fit')
    last_date   = df.index[-1]
    freq        = pd.infer_freq(df.index) or 'B'
    future_idx  = pd.date_range(last_date, periods=days + 1, freq=freq)[1:]
    ax.plot(future_idx, future_pred, color=GREEN, linewidth=1.6,
            linestyle='--', label=f'{days}d Forecast')
    ax.axvline(last_date, color='#333333', linewidth=0.8, linestyle=':')
    ax.legend(frameon=False, labelcolor=ACCENT, fontsize=8)
    ax.set_title('Linear Regression Forecast', color=ACCENT,
                 fontsize=11, pad=12, fontweight='500')
    fig.autofmt_xdate(rotation=30, ha='right')
    chart = _fig_to_b64(fig)

    return {
        'model': 'Linear Regression',
        'mae': round(float(mae), 4),
        'rmse': round(float(rmse), 4),
        'last_price': round(float(closes[-1]), 2),
        'predicted_price': round(float(future_pred[-1]), 2),
        'chart': chart,
        'future_dates':  [d.strftime('%Y-%m-%d') for d in future_idx],
        'future_prices': [round(float(p), 2) for p in future_pred.flatten()],
    }


# ── ARIMA ──────────────────────────────────────────────────────────────────────
def arima_predict(df: pd.DataFrame, days: int = 30):
    closes = _get_closes(df)
    split  = int(len(closes) * 0.8)
    train, test = closes[:split], closes[split:]

    model = ARIMA(train, order=(5, 1, 0))
    fit   = model.fit()

    test_pred   = fit.forecast(steps=len(test))
    future_pred = fit.forecast(steps=len(test) + days)[-days:]

    mae  = mean_absolute_error(test, test_pred)
    rmse = np.sqrt(mean_squared_error(test, test_pred))

    # chart
    fig, ax = _base_fig()
    ax.plot(df.index, closes, color=DIM, linewidth=1, label='Actual')
    ax.plot(df.index[split:], test_pred, color=BLUE, linewidth=1.4, label='Test Fit')
    last_date  = df.index[-1]
    freq       = pd.infer_freq(df.index) or 'B'
    future_idx = pd.date_range(last_date, periods=days + 1, freq=freq)[1:]
    ax.plot(future_idx, future_pred, color=GREEN, linewidth=1.6,
            linestyle='--', label=f'{days}d Forecast')
    ax.axvline(last_date, color='#333333', linewidth=0.8, linestyle=':')
    ax.legend(frameon=False, labelcolor=ACCENT, fontsize=8)
    ax.set_title('ARIMA(5,1,0) Forecast', color=ACCENT,
                 fontsize=11, pad=12, fontweight='500')
    fig.autofmt_xdate(rotation=30, ha='right')
    chart = _fig_to_b64(fig)

    return {
        'model': 'ARIMA',
        'mae': round(float(mae), 4),
        'rmse': round(float(rmse), 4),
        'last_price': round(float(closes[-1]), 2),
        'predicted_price': round(float(future_pred[-1]), 2),
        'chart': chart,
        'future_dates':  [d.strftime('%Y-%m-%d') for d in future_idx],
        'future_prices': [round(float(p), 2) for p in future_pred.flatten()],
    }


# ── Moving Average ─────────────────────────────────────────────────────────────
def moving_average_predict(df: pd.DataFrame, days: int = 30, window: int = 20):
    closes = _get_closes(df)
    ma     = pd.Series(closes).rolling(window=window).mean().values

    last_ma     = float(np.nanmean(closes[-window:]))
    future_pred = np.full(days, last_ma)

    split      = int(len(closes) * 0.8)
    test_pred  = ma[split:]
    valid_mask = ~np.isnan(test_pred)

    mae  = mean_absolute_error(closes[split:][valid_mask], test_pred[valid_mask])
    rmse = np.sqrt(mean_squared_error(closes[split:][valid_mask], test_pred[valid_mask]))

    # chart
    fig, ax = _base_fig()
    ax.plot(df.index, closes, color=DIM, linewidth=1, alpha=0.7, label='Actual')
    ax.plot(df.index, ma, color=BLUE, linewidth=1.4, label=f'MA({window})')
    last_date  = df.index[-1]
    freq       = pd.infer_freq(df.index) or 'B'
    future_idx = pd.date_range(last_date, periods=days + 1, freq=freq)[1:]
    ax.plot(future_idx, future_pred, color=GREEN, linewidth=1.6,
            linestyle='--', label=f'{days}d Forecast')
    ax.axvline(last_date, color='#333333', linewidth=0.8, linestyle=':')
    ax.legend(frameon=False, labelcolor=ACCENT, fontsize=8)
    ax.set_title(f'Moving Average MA({window}) Forecast', color=ACCENT,
                 fontsize=11, pad=12, fontweight='500')
    fig.autofmt_xdate(rotation=30, ha='right')
    chart = _fig_to_b64(fig)

    return {
        'model': f'Moving Average (window={window})',
        'mae': round(float(mae), 4),
        'rmse': round(float(rmse), 4),
        'last_price': round(float(closes[-1]), 2),
        'predicted_price': round(float(future_pred[-1]), 2),
        'chart': chart,
        'future_dates':  [d.strftime('%Y-%m-%d') for d in future_idx],
        'future_prices': [round(float(p), 2) for p in future_pred.flatten()],
    }
