import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def get_user_inputs():
    #Collect user inputs for backtesting parameters
    print("=== MACD Backtest Configuration ===")
    
    # Get ticker symbol
    ticker = input("Enter ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper().strip()
    
    # Get timeframe
    print("\nTimeframe options:")
    print("1. 1 months")
    print("2. 3 months")
    print("3. 6 months")
    print("4. 1 year") 
    print("5. 2 years")
    print("6. 5 years")
    print("7. Custom")
    
    timeframe_choice = input("Select timeframe (1-7): ").strip()
    
    match timeframe_choice:
        case "1":
            period = '1mo'
        case '2':
            period = '3mo'
        case '3':
            period = '6mo'
        case '4':
            period = '1y'
        case '5':
            period = '2y'
        case '6':
            period = '5y'
        case '7':
            start_date = input("Enter start date (YYYY-MM-DD): ")
            end_date = input("Enter end date (YYYY-MM-DD): ")
            period = None
        case _:
            print("Invalid choice, defaulting to 1 year")
            period = "1y"
    
    # Get candle duration
    print("\nCandle duration options:")
    print("1. 5 minutes")
    print("2. 30 minutes")
    print("3. 1 hour")
    print("4. 1 day")
    print("5. 1 week")
    print("6. 1 month")
    
    duration_choice = input("Select duration (1-6): ").strip()
    
    match duration_choice:
        case '1':
            interval = '5m'
        case '2':
            interval = '30m'
        case '3':
            interval = '1h'
        case '4':
            interval = '1d'
        case '5':
            interval = '1wk'
        case '6':
            interval = '1mo'
        case _:
            print("Invalid choice, defaulting to daily")
            interval = "1d"

    # Validate timeframe compatibility with intervals
    if interval in ['5m', '30m', '1h'] and period in ['2y', '5y']:
        print("Warning: Intraday intervals (5m, 30m, 1h) are limited to 60 days of data.")
        print("Adjusting timeframe to 60 days for intraday data.")
        period = '60d'
    elif interval in ['5m', '30m', '1h'] and timeframe_choice == '7':
        print("Note: Intraday data may be limited. Yahoo Finance provides ~60 days for 5m/30m intervals.")

    if timeframe_choice == "7":
        return ticker, period, interval, start_date, end_date
    else:
        return ticker, period, interval, None, None

def calculate_macd(data, fast=12, slow=26, signal=9):
    # Calculate MACD indicator components
    # Calculate exponential moving averages
    ema_fast = data['Close'].ewm(span=fast).mean()
    ema_slow = data['Close'].ewm(span=slow).mean()
    
    # MACD line = Fast EMA - Slow EMA
    macd_line = ema_fast - ema_slow
    
    # Signal line = EMA of MACD line
    signal_line = macd_line.ewm(span=signal).mean()
    
    # Histogram = MACD - Signal
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def generate_macd_signals(macd_line, signal_line):
    # Generate buy/sell signals based on MACD crossovers
    signals = pd.DataFrame(index=macd_line.index)
    signals['signal'] = 0
    
    # Buy signal: MACD crosses above signal line
    signals.loc[macd_line > signal_line, 'signal'] = 1
    # Sell/Short signal: MACD crosses below signal line  
    signals.loc[macd_line <= signal_line, 'signal'] = -1
    
    # Generate position changes (1 for buy, -1 for sell/short, 0 for hold)
    signals['position'] = signals['signal'].diff()
    
    # Clean up the signals - only keep actual crossover points
    # Buy signal (2): MACD crosses above signal (was -1, now 1) -> change of 2
    # Sell signal (-2): MACD crosses below signal (was 1, now -1) -> change of -2
    signals.loc[signals['position'] == 2, 'position'] = 1   # Buy signal
    signals.loc[signals['position'] == -2, 'position'] = -1 # Sell/Short signal
    signals.loc[signals['position'] == 0, 'position'] = 0   # No change
    
    return signals

def backtest_strategy(data, signals):
    # Backtest the MACD strategy with corrected long/short positions
    initial_capital = 10000
    positions = pd.DataFrame(index=data.index)
    positions['holdings'] = 0.0
    positions['cash'] = 0.0
    positions['total'] = 0.0
    
    current_position = 0  # 1 = long position, -1 = short position, 0 = no position
    shares_held = 0.0
    entry_price = 0.0  # Track entry price for short positions
    cash = float(initial_capital)
    
    for i in range(len(data)):
        current_price = data['Close'].iloc[i]
        
        # Check for buy signal (go long or cover short and go long)
        if signals['position'].iloc[i] == 1:
            if current_position == -1:
                # Cover short position: profit = shares_held * (entry_price - current_price)
                short_profit = shares_held * (entry_price - current_price)
                cash = cash + short_profit
                print(f"COVER SHORT: {data.index[i]} at ${current_price:.2f}, profit: ${short_profit:.2f}, cash: ${cash:.2f}")
            
            # Enter long position with available cash
            if cash > 0:
                shares_held = cash / current_price
                cash = 0.0
                current_position = 1
                entry_price = current_price
                print(f"BUY (LONG): {data.index[i]} at ${current_price:.2f}, shares: {shares_held:.2f}")
            
        # Check for sell signal (go short or sell long and go short)
        elif signals['position'].iloc[i] == -1:
            if current_position == 1:
                # Sell long position
                cash = shares_held * current_price
                print(f"SELL LONG: {data.index[i]} at ${current_price:.2f}, cash: ${cash:.2f}")
            
            # Enter short position with available cash
            if cash > 0:
                shares_held = cash / current_price  # Number of shares to short
                entry_price = current_price  # Remember short entry price
                current_position = -1
                print(f"SHORT: {data.index[i]} at ${current_price:.2f}, shares shorted: {shares_held:.2f}")
        
        # Calculate portfolio value based on current position
        if current_position == 1:  # Long position
            portfolio_value = shares_held * current_price
        elif current_position == -1:  # Short position
            # Short P&L = shares_held * (entry_price - current_price)
            short_pnl = shares_held * (entry_price - current_price)
            portfolio_value = cash + short_pnl
        else:  # No position
            portfolio_value = cash
            
        positions.iloc[i, positions.columns.get_loc('total')] = portfolio_value
    
    # Handle final position
    final_price = data['Close'].iloc[-1]
    if current_position == 1:  # Close long position
        final_value = shares_held * final_price
        print(f"Final LONG position closed at ${final_price:.2f}, final value: ${final_value:.2f}")
    elif current_position == -1:  # Close short position
        short_pnl = shares_held * (entry_price - final_price)
        final_value = cash + short_pnl
        print(f"Final SHORT position closed at ${final_price:.2f}, final value: ${final_value:.2f}")
    else:
        final_value = cash
        
    positions.iloc[-1, positions.columns.get_loc('total')] = final_value
    
    # Calculate strategy returns
    positions['returns'] = positions['total'].pct_change()
    positions['cumulative_returns'] = positions['total'] / initial_capital
    
    return positions

def get_sp500_data(start_date, end_date, interval):
    # Download S&P 500 data for benchmark comparison
    try:
        # For intraday intervals, use daily data for S&P 500 comparison
        sp500_interval = '1d' if interval in ['5m', '30m', '1h'] else interval
        
        sp500 = yf.download('^GSPC', start=start_date, end=end_date, interval=sp500_interval)
        
        # Handle MultiIndex columns
        if isinstance(sp500.columns, pd.MultiIndex):
            sp500 = sp500.droplevel(1, axis=1)
            
        # Calculate buy and hold returns
        sp500['returns'] = sp500['Close'].pct_change()
        sp500['cumulative_returns'] = (1 + sp500['returns']).cumprod()
        return sp500
    except Exception as e:
        print(f"Error downloading S&P 500 data: {e}")
        return None

def calculate_performance_metrics(strategy_returns, benchmark_returns):
    # Calculate key performance metrics including alpha
    # Remove NaN values
    strategy_clean = strategy_returns.dropna()
    benchmark_clean = benchmark_returns.dropna()
    
    # For different time intervals, align on dates only (not times)
    if len(strategy_clean) > 0 and len(benchmark_clean) > 0:
        # Get the overlapping date range
        start_date = max(strategy_clean.index[0].date(), benchmark_clean.index[0].date())
        end_date = min(strategy_clean.index[-1].date(), benchmark_clean.index[-1].date())
        
        # Calculate total returns over the period
        strategy_start_val = strategy_clean.iloc[0] if len(strategy_clean) > 0 else 1
        strategy_end_val = strategy_clean.iloc[-1] if len(strategy_clean) > 0 else 1
        benchmark_start_val = benchmark_clean.iloc[0] if len(benchmark_clean) > 0 else 1  
        benchmark_end_val = benchmark_clean.iloc[-1] if len(benchmark_clean) > 0 else 1
    else:
        strategy_start_val = strategy_end_val = 1
        benchmark_start_val = benchmark_end_val = 1
    
    # Calculate metrics
    strategy_total_return = (strategy_end_val - strategy_start_val) / strategy_start_val * 100
    benchmark_total_return = (benchmark_end_val - benchmark_start_val) / benchmark_start_val * 100
    
    # Calculate annualized volatility (adjust for different intervals)
    periods_per_year = 252  # Default for daily
    if any('m' in str(idx) for idx in strategy_clean.index[:5]):  # Intraday data
        if '5m' in str(strategy_clean.index[0]):
            periods_per_year = 252 * 78  # 78 five-minute periods per trading day
        elif '30m' in str(strategy_clean.index[0]):
            periods_per_year = 252 * 13  # 13 thirty-minute periods per trading day
        elif '1h' in str(strategy_clean.index[0]):
            periods_per_year = 252 * 6.5  # 6.5 hours per trading day
    
    strategy_vol = strategy_clean.pct_change().std() * np.sqrt(periods_per_year) * 100 if len(strategy_clean) > 1 else 0
    benchmark_vol = benchmark_clean.pct_change().std() * np.sqrt(252) * 100 if len(benchmark_clean) > 1 else 0
    
    # Calculate Sharpe ratio (assuming 0% risk-free rate)
    strategy_sharpe = (strategy_clean.pct_change().mean() * periods_per_year) / (strategy_clean.pct_change().std() * np.sqrt(periods_per_year)) if len(strategy_clean) > 1 else 0
    benchmark_sharpe = (benchmark_clean.pct_change().mean() * 252) / (benchmark_clean.pct_change().std() * np.sqrt(252)) if len(benchmark_clean) > 1 else 0
    
    # Calculate alpha (simple version: strategy return - benchmark return)
    alpha = strategy_total_return - benchmark_total_return
    
    return {
        'strategy_return': strategy_total_return,
        'benchmark_return': benchmark_total_return,
        'alpha': alpha,
        'strategy_volatility': strategy_vol,
        'benchmark_volatility': benchmark_vol,
        'strategy_sharpe': strategy_sharpe,
        'benchmark_sharpe': benchmark_sharpe
    }

def plot_backtest_results(data, positions, sp500_data, macd_line, signal_line, signals, ticker, metrics):
    # Create comprehensive visualization of backtest results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'MACD Long/Short Strategy Backtest: {ticker}', fontsize=16, fontweight='bold')
    
    # Plot 1: Price chart with buy/sell signals
    ax1.plot(data.index, data['Close'], label=f'{ticker} Price', linewidth=1.5)
    
    # Mark buy/sell points
    buy_signals = signals[signals['position'] == 1]
    sell_signals = signals[signals['position'] == -1]
    
    for date in buy_signals.index:
        if date in data.index:
            ax1.scatter(date, data.loc[date, 'Close'], color='green', marker='^', s=100, label='Buy/Long' if date == buy_signals.index[0] else "")
    
    for date in sell_signals.index:
        if date in data.index:
            ax1.scatter(date, data.loc[date, 'Close'], color='red', marker='v', s=100, label='Sell/Short' if date == sell_signals.index[0] else "")
    
    ax1.set_title(f'{ticker} Price with MACD Long/Short Signals')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MACD indicator
    ax2.plot(data.index, macd_line, label='MACD Line', linewidth=1.5)
    ax2.plot(data.index, signal_line, label='Signal Line', linewidth=1.5)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.fill_between(data.index, macd_line - signal_line, 0, alpha=0.3, label='Histogram')
    ax2.set_title('MACD Indicator')
    ax2.set_ylabel('MACD Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Strategy vs S&P 500 performance
    ax3.plot(positions.index, (positions['cumulative_returns'] - 1) * 100, 
             label=f'{ticker} MACD Long/Short Strategy', linewidth=2, color='blue')
    
    if sp500_data is not None:
        ax3.plot(sp500_data.index, (sp500_data['cumulative_returns'] - 1) * 100, 
                 label='S&P 500 Buy & Hold', linewidth=2, color='orange')
    
    ax3.set_title('Cumulative Returns Comparison')
    ax3.set_ylabel('Cumulative Return (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance metrics table
    ax4.axis('off')
    
    # Create performance summary
    perf_text = f"""
PERFORMANCE SUMMARY

Strategy Total Return: {metrics['strategy_return']:.2f}%
S&P 500 Total Return: {metrics['benchmark_return']:.2f}%
Alpha (Excess Return): {metrics['alpha']:.2f}%

Strategy Volatility: {metrics['strategy_volatility']:.2f}%
S&P 500 Volatility: {metrics['benchmark_volatility']:.2f}%

Strategy Sharpe Ratio: {metrics['strategy_sharpe']:.3f}
S&P 500 Sharpe Ratio: {metrics['benchmark_sharpe']:.3f}

Alpha Interpretation:
{'✓ POSITIVE ALPHA - Strategy outperformed!' if metrics['alpha'] > 0 else '✗ NEGATIVE ALPHA - Strategy underperformed'}

Strategy Type: Long/Short (flips positions on signals)
    """
    
    ax4.text(0.1, 0.9, perf_text, transform=ax4.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

    return fig

def main():
    # Main execution function
    print("MACD Long/Short Strategy Backtester with Alpha Analysis")
    print("=" * 50)
    
    # Get user inputs
    user_inputs = get_user_inputs()
    ticker = user_inputs[0]
    period = user_inputs[1] 
    interval = user_inputs[2]
    
    print(f"\nDownloading data for {ticker}...")
    
    try:
        # Download stock data
        if len(user_inputs) == 5 and user_inputs[3]:  # Custom date range
            start_date, end_date = user_inputs[3], user_inputs[4]
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        else:
            data = yf.download(ticker, period=period, interval=interval)
        
        # Handle MultiIndex columns (when downloading single ticker)
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)
        
        if data.empty:
            print(f"No data found for ticker {ticker}")
            return
        
        print(f"Downloaded {len(data)} data points")
        
        # Calculate MACD indicator
        print("Calculating MACD indicator...")
        macd_line, signal_line, histogram = calculate_macd(data)
        
        # Generate trading signals
        print("Generating trading signals...")
        signals = generate_macd_signals(macd_line, signal_line)
        
        # Backtest strategy
        print("Running backtest...")
        positions = backtest_strategy(data, signals)
        
        # Download S&P 500 benchmark data
        print("Downloading S&P 500 benchmark data...")
        start_date = data.index[0]
        end_date = data.index[-1]
        sp500_data = get_sp500_data(start_date, end_date, interval)
        
        # Calculate performance metrics
        if sp500_data is not None:
            metrics = calculate_performance_metrics(
                positions['cumulative_returns'], 
                sp500_data['cumulative_returns']
            )
        else:
            print("Warning: Could not download S&P 500 data for comparison")
            metrics = {
                'strategy_return': (positions['cumulative_returns'].iloc[-1] - 1) * 100,
                'benchmark_return': 0,
                'alpha': 0,
                'strategy_volatility': 0,
                'benchmark_volatility': 0,
                'strategy_sharpe': 0,
                'benchmark_sharpe': 0
            }
        
        # Create visualization
        print("Creating visualization...")
        plot_backtest_results(data, positions, sp500_data, macd_line, signal_line, signals, ticker, metrics)
        
        print("\nBacktest completed successfully!")
        print(f"Final Alpha: {metrics['alpha']:.2f}%")
        again = input("Try another symbol? (y/n)")
        if again == 'y' or again == 'Y':
            main()
        elif again =='n' or again =='N':
            exit()
        else:
            print("Not a valid input. Exitting...")
            exit()
        
    except Exception as e:
        print(f"Error during backtesting: {e}")
        return

if __name__ == "__main__":
    main()