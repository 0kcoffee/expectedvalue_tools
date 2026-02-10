"""Portfolio correlation test for analyzing relationships between strategies."""

import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from .base import BaseTest
from ..output.formatters import print_box, print_section_box
from ..utils.colors import Colors

# Try to import scipy for statistical tests
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

# Try to import matplotlib for charts
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    mdates = None
    sns = None


class PortfolioCorrelationTest(BaseTest):
    """Test that analyzes correlation between all portfolio strategies."""

    def get_name(self) -> str:
        """Get the name of the test."""
        return "portfolio_correlation"

    def get_description(self) -> str:
        """Get a description of what the test does."""
        return (
            "Analyzes correlation between all portfolio strategies using Pearson and Spearman "
            "correlation coefficients on both returns and cumulative returns. Includes rolling "
            "correlation analysis, comprehensive visualizations (heatmaps, scatter plots, time "
            "series), and statistical significance testing."
        )

    def run(
        self,
        data: pd.DataFrame,
        starting_capital: float = 100000.0,
        rolling_window: int = 30,
        verbose: bool = True,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Run portfolio correlation analysis on the provided data.

        Args:
            data: DataFrame with normalized and enriched trade data
            starting_capital: Starting capital for equity curve calculation (default: 100000)
            rolling_window: Window size for rolling correlation (default: 30 trades)
            verbose: If True, print formatted output (default: True)
            output_dir: Directory to save visualizations (default: None)

        Returns:
            Dictionary with correlation analysis results
        """
        # Validate required columns
        self.validate_data(data, ["P/L", "datetime_opened", "Strategy"])

        # Make a copy to avoid modifying original
        df = data.copy()

        # Check if this is a portfolio (multiple strategies)
        if "Strategy" not in df.columns:
            raise ValueError("Portfolio correlation test requires 'Strategy' column")

        unique_strategies = df["Strategy"].dropna().unique()
        unique_strategies = [s for s in unique_strategies if str(s).strip() != ""]

        if len(unique_strategies) < 2:
            raise ValueError(
                f"Portfolio correlation test requires at least 2 strategies. Found: {len(unique_strategies)}"
            )

        # Sort by datetime
        if "datetime_opened" in df.columns:
            df = df.sort_values("datetime_opened").reset_index(drop=True)
        else:
            raise ValueError("Missing datetime information (datetime_opened)")

        results = {}

        # Prepare strategy returns
        strategy_returns = self._prepare_strategy_returns(df, unique_strategies)
        results["strategy_returns"] = strategy_returns

        # Prepare cumulative returns (equity curves)
        strategy_cumulative = self._prepare_cumulative_returns(
            df, unique_strategies, starting_capital
        )
        results["strategy_cumulative"] = strategy_cumulative

        # Calculate correlation matrices
        correlation_matrices = self._calculate_correlation_matrices(
            strategy_returns, strategy_cumulative
        )
        results["correlation_matrices"] = correlation_matrices

        # Calculate rolling correlations
        rolling_correlations = self._calculate_rolling_correlations(
            strategy_returns, strategy_cumulative, rolling_window
        )
        results["rolling_correlations"] = rolling_correlations

        # Calculate statistical metrics
        statistical_metrics = self._calculate_statistical_metrics(
            correlation_matrices, rolling_correlations, unique_strategies
        )
        results["statistical_metrics"] = statistical_metrics

        # Generate visualizations
        figures = {}
        if verbose:
            figures = self._generate_visualizations(
                correlation_matrices,
                strategy_returns,
                strategy_cumulative,
                rolling_correlations,
                unique_strategies,
                output_dir,
            )
        results["figures"] = figures

        if verbose:
            self._print_results(results, unique_strategies, output_dir)

        return results

    def _prepare_strategy_returns(
        self, data: pd.DataFrame, strategies: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate trade-level returns per strategy.

        Args:
            data: DataFrame with trade data
            strategies: List of strategy names

        Returns:
            Dictionary mapping strategy names to DataFrames with returns
        """
        strategy_returns = {}

        for strategy in strategies:
            strategy_df = data[data["Strategy"] == strategy].copy()
            strategy_df = strategy_df.sort_values("datetime_opened").reset_index(drop=True)

            # Calculate returns (P/L per Contract)
            if "P/L per Contract" in strategy_df.columns:
                returns = strategy_df["P/L per Contract"].values
            else:
                # Calculate from P/L and contracts
                pl = strategy_df["P/L"].values
                contracts = strategy_df["No. of Contracts"].values
                returns = np.where(contracts > 0, pl / contracts, 0)

            # Create DataFrame with datetime and returns
            returns_df = pd.DataFrame(
                {
                    "datetime": strategy_df["datetime_opened"].values,
                    "returns": returns,
                }
            )

            strategy_returns[strategy] = returns_df

        return strategy_returns

    def _prepare_cumulative_returns(
        self, data: pd.DataFrame, strategies: List[str], starting_capital: float
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate cumulative returns (equity curves) per strategy.

        Args:
            data: DataFrame with trade data
            strategies: List of strategy names
            starting_capital: Starting capital amount

        Returns:
            Dictionary mapping strategy names to DataFrames with cumulative returns
        """
        strategy_cumulative = {}

        for strategy in strategies:
            strategy_df = data[data["Strategy"] == strategy].copy()
            strategy_df = strategy_df.sort_values("datetime_opened").reset_index(drop=True)

            # Calculate cumulative equity curve
            if "P/L per Contract" in strategy_df.columns:
                pl_per_contract = strategy_df["P/L per Contract"].values
            else:
                pl = strategy_df["P/L"].values
                contracts = strategy_df["No. of Contracts"].values
                pl_per_contract = np.where(contracts > 0, pl / contracts, 0)

            # Build equity curve
            equity_curve = []
            current_value = starting_capital
            for pl in pl_per_contract:
                # For cumulative returns, we need to scale by contracts
                # Use average contracts or assume 1 contract for simplicity
                # Actually, for correlation, we want the equity curve value
                if "No. of Contracts" in strategy_df.columns:
                    contracts = strategy_df["No. of Contracts"].iloc[len(equity_curve)]
                    current_value += pl * contracts
                else:
                    current_value += pl
                equity_curve.append(current_value)

            # Calculate cumulative returns as percentage
            cumulative_returns = [
                ((eq - starting_capital) / starting_capital * 100) if starting_capital > 0 else 0.0
                for eq in equity_curve
            ]

            # Create DataFrame
            cumulative_df = pd.DataFrame(
                {
                    "datetime": strategy_df["datetime_opened"].values,
                    "cumulative_return": cumulative_returns,
                    "equity_value": equity_curve,
                }
            )

            strategy_cumulative[strategy] = cumulative_df

        return strategy_cumulative

    def _align_data_by_datetime(
        self, data_dict: Dict[str, pd.DataFrame], value_column: str
    ) -> pd.DataFrame:
        """
        Align strategy data to a common time index.

        Args:
            data_dict: Dictionary mapping strategy names to DataFrames
            value_column: Column name to extract (e.g., 'returns' or 'cumulative_return')

        Returns:
            DataFrame with strategies as columns, aligned by datetime
        """
        # Get all unique datetimes
        all_dates = set()
        for df in data_dict.values():
            all_dates.update(df["datetime"].values)

        # Create sorted datetime index
        sorted_dates = sorted(all_dates)

        # Create aligned DataFrame
        aligned_data = pd.DataFrame(index=sorted_dates)

        for strategy, df in data_dict.items():
            # Set datetime as index temporarily
            df_indexed = df.set_index("datetime")
            # Reindex to common dates and forward fill
            aligned_data[strategy] = df_indexed[value_column].reindex(
                sorted_dates, method="ffill"
            )

        return aligned_data

    def _calculate_correlation_matrices(
        self,
        strategy_returns: Dict[str, pd.DataFrame],
        strategy_cumulative: Dict[str, pd.DataFrame],
    ) -> Dict:
        """
        Calculate correlation matrices for returns and cumulative returns.

        Args:
            strategy_returns: Dictionary of strategy returns DataFrames
            strategy_cumulative: Dictionary of strategy cumulative returns DataFrames

        Returns:
            Dictionary with correlation matrices and p-values
        """
        # Align returns data
        aligned_returns = self._align_data_by_datetime(strategy_returns, "returns")
        aligned_cumulative = self._align_data_by_datetime(
            strategy_cumulative, "cumulative_return"
        )

        # Remove rows with all NaN
        aligned_returns = aligned_returns.dropna(how="all")
        aligned_cumulative = aligned_cumulative.dropna(how="all")

        results = {}

        # Pearson correlation for returns
        pearson_returns = aligned_returns.corr(method="pearson")
        results["pearson_returns"] = pearson_returns

        # Spearman correlation for returns
        spearman_returns = aligned_returns.corr(method="spearman")
        results["spearman_returns"] = spearman_returns

        # Pearson correlation for cumulative returns
        pearson_cumulative = aligned_cumulative.corr(method="pearson")
        results["pearson_cumulative"] = pearson_cumulative

        # Spearman correlation for cumulative returns
        spearman_cumulative = aligned_cumulative.corr(method="spearman")
        results["spearman_cumulative"] = spearman_cumulative

        # Calculate p-values if scipy is available
        if SCIPY_AVAILABLE:
            results["pearson_returns_pvalues"] = self._calculate_pvalues(
                aligned_returns, method="pearson"
            )
            results["spearman_returns_pvalues"] = self._calculate_pvalues(
                aligned_returns, method="spearman"
            )
            results["pearson_cumulative_pvalues"] = self._calculate_pvalues(
                aligned_cumulative, method="pearson"
            )
            results["spearman_cumulative_pvalues"] = self._calculate_pvalues(
                aligned_cumulative, method="spearman"
            )

        # Store aligned data for later use
        results["aligned_returns"] = aligned_returns
        results["aligned_cumulative"] = aligned_cumulative

        return results

    def _calculate_pvalues(
        self, data: pd.DataFrame, method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Calculate p-values for correlation matrix.

        Args:
            data: DataFrame with aligned data
            method: Correlation method ('pearson' or 'spearman')

        Returns:
            DataFrame with p-values
        """
        if not SCIPY_AVAILABLE:
            return None

        n = len(data)
        strategies = data.columns.tolist()
        pvalues = pd.DataFrame(index=strategies, columns=strategies)

        for i, strat1 in enumerate(strategies):
            for j, strat2 in enumerate(strategies):
                if i == j:
                    pvalues.loc[strat1, strat2] = 0.0
                else:
                    x = data[strat1].dropna()
                    y = data[strat2].dropna()
                    # Align by index
                    common_idx = x.index.intersection(y.index)
                    if len(common_idx) < 3:
                        pvalues.loc[strat1, strat2] = np.nan
                    else:
                        x_aligned = x.loc[common_idx]
                        y_aligned = y.loc[common_idx]
                        if method == "pearson":
                            corr, pval = stats.pearsonr(x_aligned, y_aligned)
                        else:  # spearman
                            corr, pval = stats.spearmanr(x_aligned, y_aligned)
                        pvalues.loc[strat1, strat2] = pval

        return pvalues.astype(float)

    def _calculate_rolling_correlations(
        self,
        strategy_returns: Dict[str, pd.DataFrame],
        strategy_cumulative: Dict[str, pd.DataFrame],
        window: int,
    ) -> Dict:
        """
        Calculate rolling correlations over time.

        Args:
            strategy_returns: Dictionary of strategy returns DataFrames
            strategy_cumulative: Dictionary of strategy cumulative returns DataFrames
            window: Rolling window size (number of trades)

        Returns:
            Dictionary with rolling correlation data
        """
        # Align data
        aligned_returns = self._align_data_by_datetime(strategy_returns, "returns")
        aligned_cumulative = self._align_data_by_datetime(
            strategy_cumulative, "cumulative_return"
        )

        aligned_returns = aligned_returns.dropna(how="all")
        aligned_cumulative = aligned_cumulative.dropna(how="all")

        strategies = list(strategy_returns.keys())
        results = {
            "returns_pearson": {},
            "returns_spearman": {},
            "cumulative_pearson": {},
            "cumulative_spearman": {},
            "dates": [],
        }

        # Calculate rolling correlations for each pair
        for i, strat1 in enumerate(strategies):
            for j, strat2 in enumerate(strategies):
                if i >= j:
                    continue  # Only calculate upper triangle

                pair_name = f"{strat1} vs {strat2}"

                # Rolling correlation for returns (Pearson)
                rolling_pearson_returns = []
                rolling_spearman_returns = []
                rolling_pearson_cumulative = []
                rolling_spearman_cumulative = []
                rolling_dates = []

                # Use a sliding window
                for k in range(len(aligned_returns)):
                    if k < window - 1:
                        continue

                    window_data_returns = aligned_returns.iloc[k - window + 1 : k + 1]
                    window_data_cumulative = aligned_cumulative.iloc[k - window + 1 : k + 1]

                    # Check if we have enough non-NaN data
                    valid_returns = window_data_returns[[strat1, strat2]].dropna()
                    valid_cumulative = window_data_cumulative[[strat1, strat2]].dropna()

                    if len(valid_returns) >= 3:
                        pearson_r = valid_returns[strat1].corr(valid_returns[strat2], method="pearson")
                        spearman_r = valid_returns[strat1].corr(
                            valid_returns[strat2], method="spearman"
                        )
                        rolling_pearson_returns.append(pearson_r if not np.isnan(pearson_r) else None)
                        rolling_spearman_returns.append(
                            spearman_r if not np.isnan(spearman_r) else None
                        )
                    else:
                        rolling_pearson_returns.append(None)
                        rolling_spearman_returns.append(None)

                    if len(valid_cumulative) >= 3:
                        pearson_c = valid_cumulative[strat1].corr(
                            valid_cumulative[strat2], method="pearson"
                        )
                        spearman_c = valid_cumulative[strat1].corr(
                            valid_cumulative[strat2], method="spearman"
                        )
                        rolling_pearson_cumulative.append(
                            pearson_c if not np.isnan(pearson_c) else None
                        )
                        rolling_spearman_cumulative.append(
                            spearman_c if not np.isnan(spearman_c) else None
                        )
                    else:
                        rolling_pearson_cumulative.append(None)
                        rolling_spearman_cumulative.append(None)

                    rolling_dates.append(aligned_returns.index[k])

                results["returns_pearson"][pair_name] = rolling_pearson_returns
                results["returns_spearman"][pair_name] = rolling_spearman_returns
                results["cumulative_pearson"][pair_name] = rolling_pearson_cumulative
                results["cumulative_spearman"][pair_name] = rolling_spearman_cumulative

        results["dates"] = rolling_dates

        return results

    def _calculate_statistical_metrics(
        self,
        correlation_matrices: Dict,
        rolling_correlations: Dict,
        strategies: List[str],
    ) -> Dict:
        """
        Calculate statistical metrics from correlation matrices.

        Args:
            correlation_matrices: Dictionary with correlation matrices
            rolling_correlations: Dictionary with rolling correlation data
            strategies: List of strategy names

        Returns:
            Dictionary with statistical metrics
        """
        metrics = {}

        # For each correlation matrix, calculate statistics
        for matrix_name in [
            "pearson_returns",
            "spearman_returns",
            "pearson_cumulative",
            "spearman_cumulative",
        ]:
            if matrix_name not in correlation_matrices:
                continue

            corr_matrix = correlation_matrices[matrix_name]

            # Extract off-diagonal elements
            mask = ~np.eye(len(corr_matrix), dtype=bool)
            off_diagonal = corr_matrix.values[mask]
            off_diagonal = off_diagonal[~np.isnan(off_diagonal)]

            if len(off_diagonal) > 0:
                metrics[f"{matrix_name}_mean"] = float(np.mean(off_diagonal))
                metrics[f"{matrix_name}_std"] = float(np.std(off_diagonal))
                metrics[f"{matrix_name}_min"] = float(np.min(off_diagonal))
                metrics[f"{matrix_name}_max"] = float(np.max(off_diagonal))
                metrics[f"{matrix_name}_median"] = float(np.median(off_diagonal))

                # Find strategy pairs with highest/lowest correlations
                corr_matrix_copy = corr_matrix.copy()
                np.fill_diagonal(corr_matrix_copy.values, np.nan)

                # Find max correlation pair
                max_idx = np.unravel_index(
                    np.nanargmax(corr_matrix_copy.values), corr_matrix_copy.shape
                )
                max_strat1 = corr_matrix.index[max_idx[0]]
                max_strat2 = corr_matrix.columns[max_idx[1]]
                metrics[f"{matrix_name}_max_pair"] = (max_strat1, max_strat2)
                metrics[f"{matrix_name}_max_value"] = float(corr_matrix_copy.iloc[max_idx])

                # Find min correlation pair
                min_idx = np.unravel_index(
                    np.nanargmin(corr_matrix_copy.values), corr_matrix_copy.shape
                )
                min_strat1 = corr_matrix.index[min_idx[0]]
                min_strat2 = corr_matrix.columns[min_idx[1]]
                metrics[f"{matrix_name}_min_pair"] = (min_strat1, min_strat2)
                metrics[f"{matrix_name}_min_value"] = float(corr_matrix_copy.iloc[min_idx])

        # Calculate rolling correlation stability
        for corr_type in [
            "returns_pearson",
            "returns_spearman",
            "cumulative_pearson",
            "cumulative_spearman",
        ]:
            if corr_type not in rolling_correlations:
                continue

            stability_metrics = {}
            for pair_name, rolling_values in rolling_correlations[corr_type].items():
                valid_values = [v for v in rolling_values if v is not None and not np.isnan(v)]
                if len(valid_values) > 0:
                    stability_metrics[pair_name] = {
                        "mean": float(np.mean(valid_values)),
                        "std": float(np.std(valid_values)),
                        "min": float(np.min(valid_values)),
                        "max": float(np.max(valid_values)),
                    }

            metrics[f"{corr_type}_stability"] = stability_metrics

        return metrics

    def _generate_visualizations(
        self,
        correlation_matrices: Dict,
        strategy_returns: Dict[str, pd.DataFrame],
        strategy_cumulative: Dict[str, pd.DataFrame],
        rolling_correlations: Dict,
        strategies: List[str],
        output_dir: Optional[str],
    ) -> Dict:
        """
        Generate all visualizations for correlation analysis.

        Args:
            correlation_matrices: Dictionary with correlation matrices
            strategy_returns: Dictionary of strategy returns
            strategy_cumulative: Dictionary of strategy cumulative returns
            rolling_correlations: Dictionary with rolling correlation data
            strategies: List of strategy names
            output_dir: Directory to save visualizations

        Returns:
            Dictionary with matplotlib figures
        """
        if not MATPLOTLIB_AVAILABLE:
            print(
                "\n"
                + Colors.BRIGHT_YELLOW
                + "Note: Visualizations require matplotlib. Install with: pip install matplotlib"
                + Colors.RESET
            )
            return {}

        figures = {}

        # Generate correlation heatmaps
        figures["heatmaps"] = self._create_correlation_heatmaps(
            correlation_matrices, strategies, output_dir
        )

        # Generate scatter plot matrix
        figures["scatter_matrix"] = self._create_scatter_matrix(
            correlation_matrices, strategies, output_dir
        )

        # Generate time series overlay
        figures["time_series"] = self._create_time_series_overlay(
            strategy_cumulative, strategies, output_dir
        )

        # Generate rolling correlation plots
        figures["rolling_correlations"] = self._create_rolling_correlation_plots(
            rolling_correlations, strategies, output_dir
        )

        return figures

    def _create_correlation_heatmaps(
        self,
        correlation_matrices: Dict,
        strategies: List[str],
        output_dir: Optional[str],
    ) -> Dict:
        """Create correlation heatmaps for all correlation matrices."""
        figures = {}

        matrix_types = [
            ("pearson_returns", "Pearson Correlation - Returns"),
            ("spearman_returns", "Spearman Correlation - Returns"),
            ("pearson_cumulative", "Pearson Correlation - Cumulative Returns"),
            ("spearman_cumulative", "Spearman Correlation - Cumulative Returns"),
        ]

        for matrix_key, title in matrix_types:
            if matrix_key not in correlation_matrices:
                continue

            corr_matrix = correlation_matrices[matrix_key]

            fig, ax = plt.subplots(figsize=(10, 8))

            # Create heatmap
            if sns is not None:
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    fmt=".3f",
                    cmap="coolwarm",
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"label": "Correlation"},
                    ax=ax,
                )
            else:
                # Fallback to matplotlib imshow
                im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
                ax.set_xticks(np.arange(len(corr_matrix.columns)))
                ax.set_yticks(np.arange(len(corr_matrix.index)))
                ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
                ax.set_yticklabels(corr_matrix.index)
                for i in range(len(corr_matrix.index)):
                    for j in range(len(corr_matrix.columns)):
                        text = ax.text(
                            j,
                            i,
                            f"{corr_matrix.iloc[i, j]:.3f}",
                            ha="center",
                            va="center",
                            color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white",
                        )
                plt.colorbar(im, ax=ax, label="Correlation")

            ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
            plt.tight_layout()

            figures[matrix_key] = fig

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                safe_title = title.replace(" ", "_").replace("-", "_").lower()
                filename = f"correlation_heatmap_{safe_title}.png"
                filepath = os.path.join(output_dir, filename)
                fig.savefig(filepath, dpi=150, bbox_inches="tight")
                plt.close(fig)

        return figures

    def _create_scatter_matrix(
        self,
        correlation_matrices: Dict,
        strategies: List[str],
        output_dir: Optional[str],
    ):
        """Create scatter plot matrix for strategy pairs."""
        if "aligned_returns" not in correlation_matrices:
            return None

        aligned_data = correlation_matrices["aligned_returns"]
        n_strategies = len(strategies)

        if n_strategies < 2:
            return None

        # Create subplot grid
        fig, axes = plt.subplots(n_strategies, n_strategies, figsize=(4 * n_strategies, 4 * n_strategies))
        
        # Handle axes array properly
        if n_strategies == 1:
            axes = np.array([[axes]])
        elif not isinstance(axes, np.ndarray):
            axes = np.array(axes)

        for i, strat1 in enumerate(strategies):
            for j, strat2 in enumerate(strategies):
                if n_strategies == 1:
                    ax = axes[0, 0]
                else:
                    ax = axes[i, j]

                if i == j:
                    # Diagonal: show strategy name
                    ax.text(
                        0.5,
                        0.5,
                        strat1,
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        transform=ax.transAxes,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    # Scatter plot
                    x = aligned_data[strat1].dropna()
                    y = aligned_data[strat2].dropna()
                    common_idx = x.index.intersection(y.index)
                    if len(common_idx) > 0:
                        x_aligned = x.loc[common_idx]
                        y_aligned = y.loc[common_idx]
                        ax.scatter(x_aligned, y_aligned, alpha=0.5, s=20)

                        # Add correlation coefficient
                        corr = x_aligned.corr(y_aligned)
                        if not np.isnan(corr):
                            ax.text(
                                0.05,
                                0.95,
                                f"r={corr:.3f}",
                                transform=ax.transAxes,
                                fontsize=10,
                                verticalalignment="top",
                                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                            )

                # Set labels
                if i == n_strategies - 1:
                    ax.set_xlabel(strat2, fontsize=10)
                if j == 0:
                    ax.set_ylabel(strat1, fontsize=10)

        plt.suptitle("Strategy Returns Scatter Plot Matrix", fontsize=16, fontweight="bold", y=0.995)
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, "correlation_scatter_matrix.png")
            fig.savefig(filepath, dpi=150, bbox_inches="tight")

        return fig

    def _create_time_series_overlay(
        self,
        strategy_cumulative: Dict[str, pd.DataFrame],
        strategies: List[str],
        output_dir: Optional[str],
    ):
        """Create time series overlay of all strategy equity curves."""
        fig, ax = plt.subplots(figsize=(14, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

        for i, strategy in enumerate(strategies):
            if strategy not in strategy_cumulative:
                continue

            df = strategy_cumulative[strategy]
            dates = df["datetime"].values
            equity = df["equity_value"].values

            ax.plot(dates, equity, label=strategy, color=colors[i], linewidth=2, alpha=0.8)

        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Equity Value ($)", fontsize=12)
        ax.set_title("Strategy Equity Curves Overlay", fontsize=14, fontweight="bold")
        ax.legend(loc="best", ncol=2)
        ax.grid(True, alpha=0.3)

        if mdates is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, "correlation_time_series_overlay.png")
            fig.savefig(filepath, dpi=150, bbox_inches="tight")

        return fig

    def _create_rolling_correlation_plots(
        self,
        rolling_correlations: Dict,
        strategies: List[str],
        output_dir: Optional[str],
    ):
        """Create rolling correlation plots for strategy pairs."""
        if not rolling_correlations.get("dates"):
            return None

        dates = rolling_correlations["dates"]
        n_pairs = len(rolling_correlations.get("returns_pearson", {}))

        if n_pairs == 0:
            return None

        # Create subplots
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_pairs == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()

        plot_idx = 0
        for corr_type, title_suffix in [
            ("returns_pearson", "Returns (Pearson)"),
            ("returns_spearman", "Returns (Spearman)"),
        ]:
            if corr_type not in rolling_correlations:
                continue

            for pair_name, rolling_values in rolling_correlations[corr_type].items():
                if plot_idx >= len(axes):
                    break

                ax = axes[plot_idx] if n_pairs > 1 else axes[0]

                # Filter out None values
                valid_indices = [
                    i for i, v in enumerate(rolling_values) if v is not None and not np.isnan(v)
                ]
                if len(valid_indices) > 0:
                    valid_dates = [dates[i] for i in valid_indices]
                    valid_values = [rolling_values[i] for i in valid_indices]

                    ax.plot(valid_dates, valid_values, linewidth=2, alpha=0.8)
                    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.3)
                    ax.set_title(f"{pair_name}\n{title_suffix}", fontsize=10)
                    ax.set_ylabel("Correlation", fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(-1.1, 1.1)

                    if mdates is not None and len(valid_dates) > 0:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

                plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Rolling Correlations Over Time", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, "correlation_rolling_plots.png")
            fig.savefig(filepath, dpi=150, bbox_inches="tight")

        return fig

    def _print_results(
        self,
        results: Dict,
        strategies: List[str],
        output_dir: Optional[str],
    ) -> None:
        """Print formatted results."""
        correlation_matrices = results["correlation_matrices"]
        statistical_metrics = results["statistical_metrics"]

        # Print header
        box_width = 80
        lines = [
            ("Number of Strategies:", f"{len(strategies)}", Colors.BRIGHT_CYAN),
            ("Strategies:", ", ".join(strategies), Colors.BRIGHT_WHITE),
        ]
        print_box(box_width, "PORTFOLIO CORRELATION ANALYSIS", lines)

        # Print correlation matrices
        self._print_correlation_matrices(correlation_matrices, statistical_metrics)

        # Print statistical summary
        self._print_statistical_summary(statistical_metrics)

    def _print_correlation_matrices(
        self, correlation_matrices: Dict, statistical_metrics: Dict
    ) -> None:
        """Print correlation matrices as formatted tables."""
        matrix_types = [
            ("pearson_returns", "PEARSON CORRELATION - RETURNS"),
            ("spearman_returns", "SPEARMAN CORRELATION - RETURNS"),
            ("pearson_cumulative", "PEARSON CORRELATION - CUMULATIVE RETURNS"),
            ("spearman_cumulative", "SPEARMAN CORRELATION - CUMULATIVE RETURNS"),
        ]

        for matrix_key, title in matrix_types:
            if matrix_key not in correlation_matrices:
                continue

            corr_matrix = correlation_matrices[matrix_key]

            # Print matrix
            print_section_box(title)
            print(corr_matrix.to_string())
            print()

            # Print p-values if available
            pvalue_key = f"{matrix_key}_pvalues"
            if pvalue_key in correlation_matrices:
                pvalues = correlation_matrices[pvalue_key]
                print_section_box(f"{title} - P-VALUES")
                print(pvalues.to_string())
                print()

    def _print_statistical_summary(self, statistical_metrics: Dict) -> None:
        """Print statistical summary metrics."""
        box_width = 80

        for matrix_name in [
            "pearson_returns",
            "spearman_returns",
            "pearson_cumulative",
            "spearman_cumulative",
        ]:
            mean_key = f"{matrix_name}_mean"
            if mean_key not in statistical_metrics:
                continue

            lines = [
                ("Mean Correlation:", f"{statistical_metrics[mean_key]:.4f}", Colors.BRIGHT_CYAN),
                ("Std Deviation:", f"{statistical_metrics.get(f'{matrix_name}_std', 0):.4f}", Colors.BRIGHT_CYAN),
                ("Min Correlation:", f"{statistical_metrics.get(f'{matrix_name}_min', 0):.4f}", Colors.BRIGHT_YELLOW),
                ("Max Correlation:", f"{statistical_metrics.get(f'{matrix_name}_max', 0):.4f}", Colors.BRIGHT_GREEN),
                ("Median Correlation:", f"{statistical_metrics.get(f'{matrix_name}_median', 0):.4f}", Colors.BRIGHT_CYAN),
            ]

            max_pair = statistical_metrics.get(f"{matrix_name}_max_pair")
            min_pair = statistical_metrics.get(f"{matrix_name}_min_pair")
            if max_pair:
                lines.append(
                    (
                        "Highest Correlation Pair:",
                        f"{max_pair[0]} vs {max_pair[1]} ({statistical_metrics.get(f'{matrix_name}_max_value', 0):.4f})",
                        Colors.BRIGHT_GREEN,
                    )
                )
            if min_pair:
                lines.append(
                    (
                        "Lowest Correlation Pair:",
                        f"{min_pair[0]} vs {min_pair[1]} ({statistical_metrics.get(f'{matrix_name}_min_value', 0):.4f})",
                        Colors.BRIGHT_YELLOW,
                    )
                )

            print_box(box_width, f"{matrix_name.upper().replace('_', ' ')} - SUMMARY", lines)
