from django.shortcuts import render
from .ml_models import (
    fetch_data,
    plot_history,
    linear_regression_predict,
    arima_predict,
    moving_average_predict,
)

POPULAR = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']


def index(request):
    return render(request, 'predictor/index.html', {'popular': POPULAR})


def predict(request):
    if request.method != 'POST':
        return render(request, 'predictor/index.html', {'popular': POPULAR})

    ticker = request.POST.get('ticker', '').strip().upper()
    period = request.POST.get('period', '1y')
    days   = int(request.POST.get('days', 30))
    model  = request.POST.get('model', 'all')

    if not ticker:
        return render(request, 'predictor/index.html', {
            'popular': POPULAR,
            'error': 'Please enter a ticker symbol.',
        })

    try:
        df      = fetch_data(ticker, period)
        history = plot_history(df, ticker)

        results = []
        if model in ('lr',    'all'): results.append(linear_regression_predict(df, days))
        if model in ('arima', 'all'): results.append(arima_predict(df, days))
        if model in ('ma',    'all'): results.append(moving_average_predict(df, days))

        ctx = {
            'ticker':        ticker,
            'period':        period,
            'days':          days,
            'history':       history,
            'results':       results,
            'popular':       POPULAR,
            'current_price': results[0]['last_price'] if results else 'N/A',
        }
        return render(request, 'predictor/result.html', ctx)

    except Exception as e:
        import traceback
        traceback.print_exc()          # prints full error to terminal
        return render(request, 'predictor/index.html', {
            'popular': POPULAR,
            'error': str(e),
        })
