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
- `--output-dir, -o`: Directory to save histogram images

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
