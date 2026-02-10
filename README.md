[<img src="/images/logo.png" alt="Alt text" width="300"/>](https://expectedvalue.trade)

# EVTools

A comprehensive toolkit for quantitative analysis, power testing, and statistical validation of trading strategies. This CLI provides robust tools for parsing, normalization, enrichment, significance testing, and visualization, enabling deep evaluation of trading strategy performance and reliability across a wide range of data sources.

## Overview

Expected Value Tools helps traders and quantitative analysts assess the statistical significance of their trading strategies by:

- Analyzing trade data from various sources
- Performing statistical power analysis
- Providing recommendations for sample sizes
- Visualizing results with histograms and distributions

## Installation

### From Source

```bash
git clone <repository-url>
cd expectedvalue_tools
pip install -e .
```

### With Visualization Support

```bash
pip install -e ".[viz]"
```

### Development Installation

```bash
pip install -e ".[dev,viz]"
```

## Quick Start

```bash
# Run power analysis on a single strategy
evtools test power example_data/oo/backtests/single_strategy/Monday-2-4-DC.csv

# Run with custom target power
evtools test power data.csv --target-power 0.90

# Save histograms to output directory
evtools test power data.csv --output-dir ./output

# List all available tests
evtools list-tests
```

## Available Tests

### Power Analysis (`power`)

**Description**: Estimates the sample size needed for an observed edge to be statistically stable. Uses bootstrap sampling with replacement to test statistical power.

**Statistical Power**: The probability that mean return > 0 in a random sample. This is independent of position sizing and risk management parameters.

**Logic**:
1. **Bootstrap Sampling**: The test uses Monte Carlo simulation with bootstrap sampling (sampling with replacement) to generate thousands of random samples from your observed trade data.
2. **Power Calculation**: For each sample size tested, it calculates the percentage of simulations where the mean return remains positive.
3. **Binary Search**: Once a range is found, binary search is used to find the minimum sample size that achieves the target power level.
4. **Recommendation**: Provides a recommended minimum sample size to achieve your target confidence level.

**Key Metrics**:
- **Observed Edge**: Mean return per contract from your backtest
- **Current Power**: Probability that a random sample of your current size would have positive mean return
- **Recommended N**: Minimum number of trades needed to achieve target power

**Usage**:
```bash
evtools test power <file> [OPTIONS]

Options:
  --target-power, -p FLOAT    Target power level (0-1, default: 0.95)
  --simulations, -s INTEGER   Number of Monte Carlo simulations (default: 10000)
  --output-dir, -o PATH       Directory to save histogram images
  --html-report               Generate HTML report (saved to reports_output/)
```

**Example**:
```bash
evtools test power data.csv --target-power 0.95 --simulations 50000 --output-dir ./output
```

### Live vs Backtest Comparison (`compare`)

**Description**: Compares real live trading execution against backtest data for the same period. Matches trades by time window and strategy, calculates P/L differences, premium differences, and identifies missed/over trades. Provides slippage analysis for fully matched trades.

**Logic**:
1. **Trade Matching**: Matches trades between backtest and live data using:
   - Time window matching (configurable, default ±10 minutes)
   - Strategy matching (handles empty strategy columns in backtest)
   - Full matching criteria: same time, strategy, legs, and reason for close
2. **Overall Metrics**: Calculates total P/L difference and premium difference (normalized to one contract)
3. **Missed/Over Trades**: Identifies trades in backtest not executed live, and trades executed live but not in backtest
4. **Slippage Analysis**: For fully matched trades, calculates detailed statistics on execution differences

**Key Metrics**:
- **Overall P/L Difference**: Total difference between live and backtest P/L
- **Overall Premium Difference**: Total difference in entry premiums (normalized per contract)
- **Match Rate**: Percentage of trades successfully matched
- **Fully Matched Trades**: Count of trades matching on all criteria (time, strategy, legs, reason for close)
- **Slippage Statistics**: Mean, median, and standard deviation of P/L and premium differences for fully matched trades
- **Win Rate Comparison**: Win rates for backtest vs live for fully matched trades

**Usage**:
```bash
evtools test compare <backtest_file> --live-file <live_file> [OPTIONS]

Options:
  --live-file, -l PATH      Path to live trading data file (required)
  --window-minutes, -w INT   Time window in minutes for matching trades (default: 10)
  --output-dir, -o PATH     Directory to save histogram images
  --html-report               Generate HTML report (saved to reports_output/)
```

**Example**:
```bash
evtools test compare example_data/oo/backtests/single_strategy/RIC-IntraDay-Crush-OOS.csv \
  --live-file example_data/oo/live/OO-RIC.csv \
  --window-minutes 10
```

**Matching Details**:
- Trades are matched if they occur within the specified time window (±window_minutes)
- Strategy matching: If backtest has empty Strategy column and live has single strategy, matching is allowed
- For portfolios with multiple strategies, matching occurs per strategy
- Fully matched trades require exact match on: time window, strategy, legs, and reason for close
- Premium differences are normalized to one contract for fair comparison

### Drawdown Analysis (`drawdown`)

**Description**: Analyzes drawdowns for trading strategies including margin requirement validation, maximum drawdown, drawdown periods, and calendar visualizations of profitable/losing periods.

**Logic**:
1. **"Can You Run It" Check**: Validates margin requirements against desired allocation (backtest data only)
   - Calculates margin per contract for each trade
   - Shows mean and max allocation percentages
   - Displays histogram of margin allocations
   - Warns if max allocation exceeds desired allocation
   - Shows biggest loss per contract and its percentage of portfolio
2. **Drawdown Calculation**: Calculates comprehensive drawdown metrics from portfolio value over time
   - Portfolio value = starting portfolio + cumulative P/L
   - Tracks peak portfolio value and calculates drawdowns from peaks
   - Identifies all drawdown periods (start/end dates)
3. **Calendar Visualizations**: Generates ASCII calendars showing profitable/losing periods
   - Weekly calendar: Shows each week as green (profitable) or red (losing)
   - Monthly calendar: Shows each month as green (profitable) or red (losing)

**Key Metrics**:
- **Max Drawdown ($)**: Largest drop from peak in dollars
- **Max Drawdown (%)**: Largest drop as percentage of peak
- **Longest Drawdown**: Period with start/end dates and length in days
- **Average Drawdown Length**: Mean duration of all drawdown periods
- **Percent Time in Drawdown**: Percentage of total time spent in drawdown
- **Margin Allocation Stats**: Mean and max margin allocation percentages
- **Biggest Loss**: Largest loss per contract and its percentage of portfolio

**Usage**:
```bash
evtools test drawdown <file> [OPTIONS]

Options:
  --portfolio-size, -p FLOAT   Initial portfolio size (default: 100000)
  --allocation, -a FLOAT        Desired allocation percentage (default: 1.0)
  --output-dir, -o PATH        Directory to save outputs
  --html-report               Generate HTML report (saved to reports_output/)
```

**Example**:
```bash
evtools test drawdown data.csv --portfolio-size 100000 --allocation 1.5
```

**Multi-Strategy Handling**:
- If portfolio detected (multiple strategies), the tool will interactively prompt for allocation % for each strategy
- Each strategy is analyzed separately with its own allocation percentage

### Tail Events Overfitting (`tail_overfitting`)

**Description**: Detects whether a strategy's performance is disproportionately driven by a small number of extreme tail events. Identifies tail events, calculates overfitting scores, and shows baseline performance metrics without tail events to assess strategy robustness.

**Logic**:
1. **Tail Event Identification**: Identifies the top X% of trades by absolute P/L value as "tail events"
   - Sorts all trades by absolute P/L (descending)
   - Selects the top tail_count trades (based on tail_percentage)
   - Can filter by direction: "all", "positive" (profits only), or "negative" (losses only)
2. **Overfitting Score Calculation**: Compares tail event magnitude to typical trades
   - Calculates average absolute P/L across all trades
   - Calculates average absolute P/L for tail events only
   - Overfitting score = avg_tail_abs_pnl / avg_abs_pnl
   - Higher scores indicate more dependence on extreme events
3. **Tail Contribution Analysis**: Shows what percentage of total absolute P/L comes from tail events
   - Tail contribution = (sum of tail absolute P/L / sum of all absolute P/L) * 100
4. **Baseline Performance Analysis**: Removes tail events and calculates performance metrics without them
   - Shows total P/L, mean P/L per contract, and win rate without tail events
   - Compares baseline metrics to full dataset metrics
   - Helps assess strategy robustness - if baseline is still profitable, strategy is less dependent on extreme events

**Key Metrics**:
- **Overfitting Score**: Ratio of tail event magnitude to average trade magnitude (lower is better)
- **Tail Contribution**: Percentage of total absolute P/L from tail events
- **Total Trades**: Number of trades analyzed (after direction filtering)
- **Tail Events Count**: Number of trades classified as tail events
- **Average Absolute P/L (All)**: Average absolute P/L per contract across all trades
- **Average Absolute P/L (Tail)**: Average absolute P/L per contract for tail events only
- **Baseline Total P/L**: Total P/L without tail events removed
- **Baseline Mean P/L**: Mean P/L per contract without tail events
- **Baseline Win Rate**: Win rate without tail events
- **P/L Difference**: Difference between P/L with and without tail events (tail contribution)

**Usage**:
```bash
evtools test tail_overfitting <file> [OPTIONS]

Options:
  --tail-percentage, -t FLOAT   Percentage of trades to consider as tail events (default: 1.0, range: 0.1-50.0)
  --max-score, -m FLOAT        Maximum acceptable overfitting score (default: 12.0, min: 1.0)
  --tail-direction, -d STR     Which tail events to analyze: 'all', 'positive', or 'negative' (default: 'all')
  --output-dir, -o PATH        Directory to save outputs
  --html-report               Generate HTML report (saved to reports_output/)
```

**Example**:
```bash
evtools test tail_overfitting data.csv --tail-percentage 1.0 --max-score 12.0
evtools test tail_overfitting data.csv --tail-direction positive --tail-percentage 2.0
```

**Interpretation**:
- **Passing Test** (overfitting_score < max_score): Strategy has more consistent performance, less dependent on rare extreme events
- **Failing Test** (overfitting_score >= max_score): Strategy relies heavily on rare extreme events, may be overfitted to specific historical conditions
- **Baseline Analysis**: If baseline performance (without tail events) is still profitable, the strategy is more robust. If baseline is unprofitable, the strategy depends heavily on catching extreme moves.

### Portfolio Track Analysis (`track`)

**Description**: Analyzes live trading portfolio performance including equity curve, CAGR, Sharpe/Sortino ratios, drawdown analysis, and strategy-level statistics with calendar visualizations.

**Logic**:
1. **Equity Curve Calculation**: Builds equity curve from starting capital and cumulative P/L
   - Sorts trades by datetime
   - Starts with specified starting capital
   - Accumulates P/L for each trade to build equity curve over time
2. **Portfolio Metrics**: Calculates comprehensive performance metrics
   - Net P/L: Sum of all P/L values
   - Extra Fees: Optional monthly fees (e.g., automation costs) - calculated based on months with trades
   - Net P/L After Fees: Net P/L minus total fees
   - CAGR: Compound Annual Growth Rate from first to last trade
   - Sharpe Ratio: Risk-adjusted return using standard deviation (annualized)
   - Sortino Ratio: Risk-adjusted return using downside deviation only (annualized)
   - Total Premium: Sum of all premium collected
   - PCR (Premium Capture Rate): Net P/L divided by Total Premium
   - MAR: CAGR divided by Max Drawdown
3. **Drawdown Analysis**: Comprehensive drawdown metrics from equity curve
   - Max Drawdown: Largest drop from peak (dollars and percentage)
   - Longest/Shortest Drawdown: Periods with dates, lengths, and depths
   - Average Drawdown Length and Depth
   - Current Drawdown status
4. **Strategy Statistics**: Per-strategy analysis table
   - Number of trades, Net P/L, Average P/L per contract/trade
   - Average Win/Loss per contract (separate calculations)
   - Win rate, PCR (strategy-level)
   - Last trade date and days since (color-coded: yellow=recent, red=old)
   - Sorted by most recent to oldest
5. **Calendar Visualizations**: Weekly and monthly profitability calendars
   - Green = profitable period, Red = losing period

**Key Metrics**:
- **Net P/L**: Total profit/loss across all trades
- **Extra Fees**: Monthly fees (if specified) - total calculated based on months with trades
- **Net P/L After Fees**: Net P/L minus total fees
- **CAGR**: Compound Annual Growth Rate
- **Max Drawdown**: Largest drop from peak (dollars and %)
- **MAR**: CAGR / Max Drawdown ratio
- **Sharpe Ratio**: Risk-adjusted return metric (annualized)
- **Sortino Ratio**: Downside risk-adjusted return metric (annualized)
- **Total Premium**: Sum of all premium collected
- **PCR**: Premium Capture Rate (Net P/L / Total Premium)
- **Drawdown Statistics**: Longest, shortest, average lengths and depths
- **Current Drawdown**: Status of current drawdown if in one
- **Strategy Table**: Comprehensive per-strategy statistics

**Usage**:
```bash
evtools test track <file> [OPTIONS]

Options:
  --starting-capital, -c FLOAT   Starting capital for equity curve calculation (default: 100000)
  --extra-fees, -f FLOAT         Monthly extra fees (e.g., automation costs) (default: 0.0)
  --output-dir, -o PATH          Directory to save outputs
  --html-report               Generate HTML report (saved to reports_output/)
```

**Example**:
```bash
evtools test track example_data/oo/live/portfolio-track.csv --starting-capital 100000
evtools test track portfolio-track.csv -c 50000 -f 150 -o ./output
```

**Output Includes**:
- Portfolio-level metrics box (Net P/L, CAGR, Max DD, MAR, Sharpe, Sortino, Total Premium, PCR)
- Equity curve chart (with drawdown overlay)
- ASCII equity curve and drawdown visualizations
- Drawdown analysis box (max, longest, shortest, average, current)
- Strategy statistics table (color-coded by recency)
- Weekly and monthly profitability calendars

### Portfolio Stress Test (`portfolio_stress`)

**Description**: Performs Monte Carlo stress testing on portfolio backtests. Generates thousands of simulations by sampling with replacement from each strategy's trade distribution, combines them into portfolio equity curves using dynamic allocation, and analyzes worst-case scenarios including max drawdown and comprehensive risk metrics.

**Logic**:
1. **Strategy Distribution Extraction**: For each strategy in the portfolio, extracts the P/L per contract distribution
2. **Monte Carlo Simulation**: For each simulation:
   - Samples trades with replacement from each strategy's distribution (maintaining original trade count per strategy)
   - Randomizes the order of sampled trades
   - Combines trades from all strategies chronologically
   - Generates portfolio equity curve using dynamic allocation (percentage-based per strategy)
3. **Metrics Calculation**: For each simulation, calculates:
   - Max drawdown (dollars and percentage)
   - Final portfolio value
   - Total return percentage
   - CAGR (Compound Annual Growth Rate)
   - Sharpe and Sortino ratios
   - Drawdown period statistics
4. **Aggregation**: Across all simulations, calculates:
   - Worst-case scenarios (maximum drawdown, minimum final portfolio, etc.)
   - Distribution statistics (mean, standard deviation)
   - Percentiles (5th, 25th, 50th, 75th, 95th) for all key metrics

**Key Metrics**:
- **Worst Max Drawdown**: Maximum drawdown across all simulations (dollars and %)
- **Worst Final Portfolio**: Minimum final portfolio value across all simulations
- **Worst Total Return**: Minimum total return percentage across all simulations
- **Worst CAGR**: Minimum CAGR across all simulations
- **Worst Sharpe/Sortino**: Minimum risk-adjusted return metrics
- **Distribution Statistics**: Mean and standard deviation for all metrics
- **Percentiles**: 5th, 25th, 50th (median), 75th, and 95th percentiles for:
  - Max drawdown (dollars and %)
  - Final portfolio value
  - Total return
  - CAGR
  - Sharpe and Sortino ratios

**Usage**:
```bash
evtools test portfolio_stress <file> [OPTIONS]

Options:
  --portfolio-size, -p FLOAT    Initial portfolio size (default: 100000)
  --allocation, -a FLOAT        Default allocation percentage (default: 1.0)
  --simulations, -s INT         Number of Monte Carlo simulations (default: 10000)
  --force-one-lot, -f          Force at least 1 contract even when allocation is insufficient
  --output-dir, -o PATH        Directory to save outputs
  --html-report                Generate HTML report (saved to reports_output/)
```

**Example**:
```bash
evtools test portfolio_stress example_data/oo/backtests/portfolios/LESS-FILTERED-0-0-2.csv --portfolio-size 100000 --allocation 1.5 --simulations 10000
evtools test portfolio_stress portfolio.csv -p 50000 -a 2.0 -s 5000
```

**Output Includes**:
- Test header with strategy information and simulation parameters
- Worst-case scenario box (worst max drawdown, final portfolio, returns, etc.)
- Distribution statistics box (mean and std dev for key metrics)
- Percentile analysis boxes (5th, 25th, 50th, 75th, 95th percentiles for max drawdown, final portfolio, and total return)

**Interpretation**:
- **Worst-case metrics** show the most adverse outcomes across thousands of simulations
- **Percentiles** help understand the distribution of outcomes - e.g., 95th percentile max drawdown means 95% of simulations had lower drawdowns
- **Distribution statistics** provide expected values and variability
- Use this test to stress test portfolio robustness and understand tail risk

### Portfolio Correlation Test (`portfolio_correlation`)

**Description**: Analyzes correlation between all portfolio strategies using Pearson and Spearman correlation coefficients on both returns and cumulative returns. Includes rolling correlation analysis, comprehensive visualizations (heatmaps, scatter plots, time series), and statistical significance testing.

**Logic**:
1. **Data Preparation**: 
   - Calculates trade-level returns per strategy (P/L per Contract)
   - Builds cumulative returns (equity curves) per strategy from starting capital
   - Aligns all strategy data to a common time index for correlation calculations
2. **Correlation Matrices**: Calculates four correlation matrices:
   - Pearson correlation for returns (linear relationships)
   - Spearman correlation for returns (rank-based, handles non-linear relationships)
   - Pearson correlation for cumulative returns
   - Spearman correlation for cumulative returns
3. **Statistical Significance**: Calculates p-values for all correlations (if scipy is available)
4. **Rolling Correlation**: Calculates time-varying correlations using a rolling window:
   - Tracks correlation stability over time
   - Identifies periods of high/low correlation
   - Helps detect regime changes in strategy relationships
5. **Statistical Metrics**: Calculates comprehensive statistics:
   - Mean, median, min, max correlations
   - Standard deviation of correlations
   - Strategy pairs with highest/lowest correlations
   - Correlation stability metrics (from rolling correlations)

**Key Metrics**:
- **Correlation Matrices**: Four matrices showing pairwise correlations between all strategies
  - Pearson Returns: Linear correlation of trade-level returns
  - Spearman Returns: Rank-based correlation of trade-level returns
  - Pearson Cumulative: Linear correlation of equity curves
  - Spearman Cumulative: Rank-based correlation of equity curves
- **P-Values**: Statistical significance of correlations (if scipy available)
- **Mean Correlation**: Average correlation across all strategy pairs
- **Min/Max Correlations**: Lowest and highest correlation pairs
- **Rolling Correlation Stability**: Mean and standard deviation of rolling correlations over time

**Usage**:
```bash
evtools test portfolio_correlation <file> [OPTIONS]

Options:
  --starting-capital, -c FLOAT    Starting capital for equity curve calculation (default: 100000)
  --rolling-window, -w INT        Rolling window size for correlation (default: 30 trades)
  --output-dir, -o PATH          Directory to save outputs
  --html-report                  Generate HTML report (saved to reports_output/)
```

**Example**:
```bash
evtools test portfolio_correlation example_data/oo/backtests/portfolios/LESS-FILTERED-0-0-2.csv --starting-capital 100000 --rolling-window 30
evtools test portfolio_correlation portfolio.csv -c 50000 -w 50 --html-report
```

**Output Includes**:
- Test header with strategy information
- Four correlation matrices (Pearson/Spearman for returns/cumulative) printed as tables
- P-value matrices (if scipy available)
- Statistical summary boxes for each correlation type:
  - Mean, median, min, max correlations
  - Standard deviation
  - Highest and lowest correlation pairs
- Visualizations (if matplotlib available):
  - Correlation heatmaps (4 types) - color-coded correlation matrices
  - Scatter plot matrix - pairwise scatter plots for all strategy pairs
  - Time series overlay - all strategy equity curves on same plot
  - Rolling correlation plots - time series of rolling correlations for key pairs

**Interpretation**:
- **High Correlation (>0.7)**: Strategies move together - may provide less diversification benefit
- **Low Correlation (<0.3)**: Strategies are relatively independent - better diversification
- **Negative Correlation**: Strategies move in opposite directions - potential hedging benefit
- **Rolling Correlations**: Show how strategy relationships change over time
  - Stable correlations indicate consistent relationships
  - Volatile correlations may indicate regime changes or strategy evolution
- **P-Values**: Indicate statistical significance (typically <0.05 considered significant)
- **Returns vs Cumulative**: 
  - Returns correlation shows short-term relationship
  - Cumulative correlation shows long-term relationship (may differ due to compounding effects)

**Best Practices**:
- Use both Pearson and Spearman correlations:
  - Pearson for linear relationships
  - Spearman for non-linear or rank-based relationships
- Analyze both returns and cumulative returns:
  - Returns show trade-level relationships
  - Cumulative shows portfolio-level relationships
- Monitor rolling correlations to detect:
  - Strategy drift (correlations changing over time)
  - Market regime changes
  - Strategy effectiveness changes
- Aim for low to moderate correlations (0.2-0.5) for better diversification
- Avoid highly correlated strategies (>0.8) unless intentionally seeking concentration

## Data Format Support

### Option Omega CSV

Currently supports Option Omega CSV format with the following required columns:
- `P/L`: Profit/Loss per trade
- `No. of Contracts`: Number of contracts traded
- `Strategy`: Strategy name (for portfolio analysis)

The tool automatically:
- Detects if the file contains a single strategy or portfolio
- Normalizes data by calculating P/L per contract
- Handles missing values and edge cases

**Future Support**: The architecture is designed to be extensible. Additional parsers can be added for:
- Other CSV formats
- JSON files
- Excel files
- Database connections

## CLI Reference

### Commands

#### `test`

Run a statistical test on trading data.

```bash
evtools test <test_name> <file_path> [OPTIONS]
```

**Arguments**:
- `test_name`: Name of the test to run (e.g., `power`, `compare`)
- `file_path`: Path to the data file (backtest file for `compare` test)

**Options**:
- `--target-power, -p`: Target power level (0-1, default: 0.95) - for `power` test
- `--simulations, -s`: Number of Monte Carlo simulations (default: 10000) - for `power` test
- `--live-file, -l`: Path to live data file (required for `compare` test)
- `--window-minutes, -w`: Time window in minutes for matching trades (default: 10) - for `compare` test
- `--portfolio-size, -p`: Initial portfolio size (default: 100000) - for `drawdown` test
- `--allocation, -a`: Desired allocation percentage (default: 1.0) - for `drawdown` test
- `--tail-percentage, -t`: Percentage of trades to consider as tail events (default: 1.0) - for `tail_overfitting` test
- `--max-score, -m`: Maximum acceptable overfitting score (default: 12.0) - for `tail_overfitting` test
- `--tail-direction, -d`: Which tail events to analyze: 'all', 'positive', or 'negative' (default: 'all') - for `tail_overfitting` test
- `--starting-capital, -c`: Starting capital for equity curve calculation (default: 100000) - for `track` test
- `--extra-fees, -f`: Monthly extra fees (e.g., automation costs) (default: 0.0) - for `track` test
- `--portfolio-size, -p`: Initial portfolio size (default: 100000) - for `portfolio_stress` test
- `--allocation, -a`: Default allocation percentage (default: 1.0) - for `portfolio_stress` test
- `--simulations, -s`: Number of Monte Carlo simulations (default: 10000) - for `portfolio_stress` test
- `--force-one-lot, -f`: Force at least 1 contract even when allocation is insufficient - for `portfolio_stress` test
- `--starting-capital, -c`: Starting capital for equity curve calculation (default: 100000) - for `portfolio_correlation` test
- `--rolling-window, -w`: Rolling window size for correlation (default: 30 trades) - for `portfolio_correlation` test
- `--output-dir, -o`: Directory to save histogram images
- `--html-report`: Generate HTML report (saved to reports_output/)

#### `list-tests`

List all available tests with descriptions.

```bash
evtools list-tests
```

## Examples

### Single Strategy Analysis

```bash
evtools test power example_data/oo/backtests/single_strategy/Monday-2-4-DC.csv
```

### Portfolio Analysis

The tool automatically detects portfolios and analyzes each strategy separately:

```bash
evtools test power example_data/oo/backtests/portfolios/LESS-FILTERED-0-0-2.csv
```

### Custom Configuration

```bash
evtools test power data.csv \
  --target-power 0.90 \
  --simulations 50000 \
  --output-dir ./analysis_results
```

### Live vs Backtest Comparison

Compare live trading execution against backtest data:

```bash
evtools test compare example_data/oo/backtests/single_strategy/RIC-IntraDay-Crush-OOS.csv \
  --live-file example_data/oo/live/OO-RIC.csv \
  --window-minutes 15
```

### Drawdown Analysis

Analyze drawdowns and margin requirements:

```bash
evtools test drawdown example_data/oo/backtests/single_strategy/RIC-IntraDay-Crush-IS.csv \
  --portfolio-size 100000 \
  --allocation 1.5
```

### Tail Events Overfitting Analysis

Detect potential overfitting to extreme tail events:

```bash
evtools test tail_overfitting example_data/oo/backtests/single_strategy/Monday-2-4-DC.csv \
  --tail-percentage 1.0 \
  --max-score 12.0

# Analyze only positive tail events
evtools test tail_overfitting data.csv \
  --tail-direction positive \
  --tail-percentage 2.0 \
  --max-score 10.0
```

### Portfolio Track Analysis

Analyze live trading portfolio performance:

```bash
evtools test track example_data/oo/live/portfolio-track.csv \
  --starting-capital 100000 \
  --extra-fees 150 \
  --output-dir ./output
```

### Portfolio Correlation Analysis

Analyze correlation between all portfolio strategies:

```bash
evtools test portfolio_correlation example_data/oo/backtests/portfolios/LESS-FILTERED-0-0-2.csv \
  --starting-capital 100000 \
  --rolling-window 30 \
  --html-report

# With custom rolling window
evtools test portfolio_correlation portfolio.csv \
  -c 50000 \
  -w 50 \
  -o ./output
```

### Portfolio Stress Test

Perform Monte Carlo stress testing on portfolio backtests:

```bash
evtools test portfolio_stress example_data/oo/backtests/portfolios/LESS-FILTERED-0-0-2.csv \
  --portfolio-size 100000 \
  --allocation 1.5 \
  --simulations 10000

# With custom parameters
evtools test portfolio_stress portfolio.csv \
  -p 50000 \
  -a 2.0 \
  -s 5000 \
  --force-one-lot
```

## Architecture

The tool follows SOLID principles with a modular architecture:

```
File → Parser → Normalizer → Enricher → Test → Output Formatter/Visualizer
```

### Components

- **Parsers**: Load and parse data from various sources
- **Normalizers**: Transform data to standard format
- **Enrichers**: Add calculated fields and metrics
- **Tests**: Perform statistical analysis
- **Output**: Format and visualize results

### Extensibility

The architecture is designed for easy extension:

1. **Adding New Parsers**: Create a class extending `BaseParser` and register it
2. **Adding New Tests**: Create a class extending `BaseTest` - auto-discovered by CLI
3. **Adding New Normalizers**: Create a class extending `BaseNormalizer`
4. **Adding New Output Formats**: Extend formatters/visualizers modules

## Extending the Tool

### Adding a New Test

1. Create a new class in `src/expectedvalue_tools/tests/` extending `BaseTest`
2. Implement required methods: `run()`, `get_name()`, `get_description()`
3. The test will be automatically discovered by the CLI

Example:
```python
from .base import BaseTest

class MyNewTest(BaseTest):
    def get_name(self) -> str:
        return "my-test"
    
    def get_description(self) -> str:
        return "Description of my test"
    
    def run(self, data: pd.DataFrame, **kwargs) -> Dict:
        # Your test logic here
        return {"result": "value"}
```

### Adding a New Parser

1. Create a new class in `src/expectedvalue_tools/parsers/` extending `BaseParser`
2. Implement `parse()`, `validate()`, and `detect_format()` methods
3. Register it in `cli/main.py` in the `PARSERS` list

## Output

### Console Output

The tool provides rich, formatted console output with:
- ASCII histograms showing data distributions
- Formatted boxes with key metrics
- Progress indicators
- Color-coded results

### Visualizations

When `--output-dir` is specified, the tool generates:
- Histogram of original P/L per contract distribution
- Histogram of Monte Carlo mean returns distribution
- Statistics annotations on charts

## Requirements

- Python 3.8+
- numpy
- pandas
- typer

Optional:
- matplotlib (for visualizations)

## License

MIT License

## Disclaimer

This tool is provided for educational purposes only. The analysis and results should not be considered as financial advice. Always perform your own due diligence and consult with qualified professionals before making any trading or investment decisions.

## Contributing

Contributions are welcome! Please ensure:
- Code follows SOLID principles
- Tests are added for new features
- README is updated with new functionality
- Architecture consistency is maintained

## Support

For issues, questions, or contributions, please visit: https://expectedvalue.trade
