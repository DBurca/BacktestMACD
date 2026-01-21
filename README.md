# Backtest MACD Indicator
A Python script to backtest the Moving Average Convergence Divergence (MACD) indicator against the buy-and-hold S&P 500. The backtesting strategy involves long/short position flipping based on the crossover signals. I made this project to learn more about algorithmic backtesting with real market data. Claude.ai was used in the making of this repository

## Features

### Strategy Implementation
- **MACD Indicator**: 12/26/9 EMA configuration (Fast EMA, Slow EMA, Signal Line)
- **Long/Short Strategy**: Automatically flips between long and short positions on MACD crossovers
- **Signal Generation**: Buy when MACD crosses above signal line, sell short when MACD crosses below
- **Trade Log**: All trades taken are logged in the console

### Performance Analysis
- **Alpha Calculation**: Measures returns over S&P 500 buy-and-hold
- **Risk Metrics**: Volatility, Sharpe ratio, and total return calculations
- **Benchmark Comparison**: Side-by-side performance visualization against market index
- **Trade Logging**: Detailed console output of all buy/sell/short/cover transactions

### Multiple Timeframes
- **Intraday**: 5-minute, 30-minute, 1-hour intervals
- **Daily/Weekly/Monthly**: Standard time periods (Day, Week, Month)
- **Custom Date Ranges**: User-defined start and end dates

### Comprehensive Visualization
Four-panel visual chart:
1. **Price Chart**: Stock price with buy/sell signal markers
2. **MACD Indicator**: MACD line, signal line, and histogram
3. **Performance Comparison**: Strategy vs S&P 500 cumulative returns
4. **Metrics Summary**: Key performance statistics and alpha interpretation

## Installation

### Prerequisites
```bash
pip install yfinance pandas numpy matplotlib
```

### Required Libraries
- `yfinance`: Financial data download from Yahoo Finance
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization

## Usage

### Basic Usage
1. Run the script:
```bash
python main.py
```

2. Follow the prompts:
   - **Ticker Symbol**: Enter any valid stock symbol (e.g., AAPL, TSLA, NVDA)
   - **Timeframe**: Choose from predefined periods or custom date range
   - **Interval**: Select candle duration from 5 minutes to 1 month

### Example Session
```
=== MACD Backtest Configuration ===
Enter ticker symbol (e.g., AAPL, TSLA, MSFT): NVDA
Select timeframe (1-7): 4
Select duration (1-6): 4

Downloading data for NVDA...
Downloaded 252 data points
Calculating MACD indicator...
Generating trading signals...
Running backtest...
```

## Strategy Logic

### Signal Generation
- **Buy Signal**: MACD line crosses above signal line → Go long (or cover short and go long)
- **Sell Signal**: MACD line crosses below signal line → Go short (or sell long and go short)

### Position Management
- **Long Position**: Profit from price increases
- **Short Position**: Profit from price decreases
- **Position Flipping**: Automatically switches between long/short on every signal
- **Cash Management**: Maintains proper cash flow during position transitions

### Performance Metrics
- **Total Return**: Overall strategy performance vs buy-and-hold
- **Alpha**: Excess return over S&P 500 benchmark
- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return metric

## Output

### Console Output
- Real-time trade execution logs
- Performance summary with key metrics
- Alpha interpretation (positive/negative)

### Visual Dashboard
- **Interactive Charts**: Price action with signal overlays
- **MACD Visualization**: Technical indicator with crossover points
- **Comparative Analysis**: Strategy vs benchmark performance
- **Statistical Summary**: Comprehensive metrics table

## Technical Details

### Data Limitations
- **Intraday Data**: Yahoo Finance limits 5m/30m intervals to the last 60 days
- **Market Hours**: Intraday data reflects regular trading hours only

### Calculation Methods
- **MACD**: Standard 12/26/9 EMA configuration
- **Short P&L**: `shares_held × (entry_price - current_price)`
- **Portfolio Value**: Dynamic calculation based on current position type (long or short)
- **Annualized Metrics**: Adjusted for different time intervals

## Risk Considerations

**Important Disclaimers**:
- This tool is for educational and research purposes only
- Past performance does not guarantee future results
- Short selling involves unlimited loss potential
- Always consider transaction costs and slippage in real trading
- Backtest results may not reflect real-world trading conditions

## Example Results

### Positive Alpha Scenario
```
Strategy Total Return: 15.67%
S&P 500 Total Return: 8.45%
Alpha (Excess Return): 7.22%
✓ POSITIVE ALPHA - Strategy outperformed!
```

### Negative Alpha Scenario
```
Strategy Total Return: 3.21%
S&P 500 Total Return: 12.34%
Alpha (Excess Return): -9.13%
✗ NEGATIVE ALPHA - Strategy underperformed
```

## Future Updates
- Transitioning to a fully GUI-based program
- Changing the chart style to bar chart
- Adding additional indicators, making a "super backtesting" tool 

## License

This project is open source

Part of this repository was created with the help of Claude.ai

---

**Developed for quantitative analysis and trading strategy research**
