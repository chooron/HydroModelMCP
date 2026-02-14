#!/usr/bin/env python3
"""
HydroModelMCP Results Visualization
Generates publication-quality plots for the three example cases
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def plot_timeseries(csv_path, output_path=None, title=None):
    """
    Plot observed vs simulated time series

    Args:
        csv_path: Path to CSV file with columns: time_step, observed, simulated
        output_path: Path to save figure (optional)
        title: Plot title (optional)
    """
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Time series plot
    ax1.plot(df['time_step'], df['observed'], label='Observed',
             color='black', linewidth=1.0, alpha=0.7)
    ax1.plot(df['time_step'], df['simulated'], label='Simulated',
             color='red', linewidth=1.0, alpha=0.7)
    ax1.set_ylabel('Flow (mm)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    if title:
        ax1.set_title(title)

    # Residual plot
    ax2.plot(df['time_step'], df['residual'], color='blue', linewidth=0.8, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Residual (mm)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_scatter(csv_path, output_path=None, title=None):
    """
    Plot observed vs simulated scatter plot with 1:1 line

    Args:
        csv_path: Path to CSV file with columns: observed, simulated
        output_path: Path to save figure (optional)
        title: Plot title (optional)
    """
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(df['observed'], df['simulated'], alpha=0.5, s=10, color='blue')

    # 1:1 line
    min_val = min(df['observed'].min(), df['simulated'].min())
    max_val = max(df['observed'].max(), df['simulated'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')

    ax.set_xlabel('Observed Flow (mm)')
    ax.set_ylabel('Simulated Flow (mm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)

    # Add R² annotation
    correlation = np.corrcoef(df['observed'], df['simulated'])[0, 1]
    r_squared = correlation ** 2
    ax.text(0.05, 0.95, f'R² = {r_squared:.4f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_sensitivity(csv_path, output_path=None, title="Parameter Sensitivity"):
    """
    Plot sensitivity analysis results (mu* and sigma)

    Args:
        csv_path: Path to CSV file with columns: parameter, mu_star, sigma
        output_path: Path to save figure (optional)
        title: Plot title
    """
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # mu* plot
    ax1.barh(df['parameter'], df['mu_star'], color='steelblue')
    ax1.set_xlabel('μ* (Elementary Effect Mean)')
    ax1.set_ylabel('Parameter')
    ax1.set_title('Parameter Importance')
    ax1.grid(True, alpha=0.3, axis='x')

    # sigma plot
    ax2.barh(df['parameter'], df['sigma'], color='coral')
    ax2.set_xlabel('σ (Elementary Effect Std)')
    ax2.set_ylabel('Parameter')
    ax2.set_title('Parameter Interaction/Non-linearity')
    ax2.grid(True, alpha=0.3, axis='x')

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_pareto_front(csv_path, output_path=None, title="Pareto Front"):
    """
    Plot Pareto front for multi-objective optimization

    Args:
        csv_path: Path to CSV file with columns: NSE, LogNSE
        output_path: Path to save figure (optional)
        title: Plot title
    """
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot of Pareto front
    scatter = ax.scatter(df['NSE'], df['LogNSE'],
                        c=range(len(df)), cmap='viridis',
                        s=50, alpha=0.7, edgecolors='black', linewidth=0.5)

    # Ideal point
    ax.plot(1.0, 1.0, 'r*', markersize=20, label='Ideal Point (1, 1)')

    ax.set_xlabel('NSE (High Flow Performance)')
    ax.set_ylabel('LogNSE (Low Flow Performance)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Solution Index')

    # Set axis limits with some padding
    ax.set_xlim(df['NSE'].min() - 0.05, 1.05)
    ax.set_ylim(df['LogNSE'].min() - 0.05, 1.05)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_comparison(csv1_path, csv2_path, labels, output_path=None, title=None):
    """
    Plot comparison of two time series

    Args:
        csv1_path: Path to first CSV file
        csv2_path: Path to second CSV file
        labels: List of two labels for the series
        output_path: Path to save figure (optional)
        title: Plot title (optional)
    """
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Observed (same for both)
    axes[0].plot(df1['time_step'], df1['observed'],
                color='black', linewidth=1.0, label='Observed')
    axes[0].set_ylabel('Flow (mm)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    if title:
        axes[0].set_title(title)

    # First simulation
    axes[1].plot(df1['time_step'], df1['observed'],
                color='black', linewidth=1.0, alpha=0.5, label='Observed')
    axes[1].plot(df1['time_step'], df1['simulated'],
                color='blue', linewidth=1.0, label=labels[0])
    axes[1].set_ylabel('Flow (mm)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Second simulation
    axes[2].plot(df2['time_step'], df2['observed'],
                color='black', linewidth=1.0, alpha=0.5, label='Observed')
    axes[2].plot(df2['time_step'], df2['simulated'],
                color='red', linewidth=1.0, label=labels[1])
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Flow (mm)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_example1(base_dir):
    """Generate all plots for Example 1"""
    print("\nGenerating Example 1 plots...")
    base_path = Path(base_dir)

    # Calibration time series
    plot_timeseries(
        base_path / "example1_calibration_timeseries.csv",
        base_path / "example1_calibration_plot.png",
        "Example 1: Calibration Period (Humid Catchment)"
    )

    # Validation time series
    plot_timeseries(
        base_path / "example1_validation_timeseries.csv",
        base_path / "example1_validation_plot.png",
        "Example 1: Validation Period (Humid Catchment)"
    )

    # Scatter plot
    plot_scatter(
        base_path / "example1_validation_timeseries.csv",
        base_path / "example1_scatter_plot.png",
        "Example 1: Observed vs Simulated (Validation)"
    )

    # Sensitivity analysis
    if (base_path / "example1_sensitivity.csv").exists():
        plot_sensitivity(
            base_path / "example1_sensitivity.csv",
            base_path / "example1_sensitivity_plot.png",
            "Example 1: Parameter Sensitivity (Morris Method)"
        )

    print("Example 1 plots complete!")


def plot_example2(base_dir):
    """Generate all plots for Example 2"""
    print("\nGenerating Example 2 plots...")
    base_path = Path(base_dir)

    # Comparison plot
    plot_comparison(
        base_path / "example2_original_timeseries.csv",
        base_path / "example2_log_timeseries.csv",
        ["Original (KGE)", "Log-transformed (LogKGE)"],
        base_path / "example2_comparison_plot.png",
        "Example 2: Log Transformation Comparison (Arid Catchment)"
    )

    # Individual scatter plots
    plot_scatter(
        base_path / "example2_original_timeseries.csv",
        base_path / "example2_original_scatter.png",
        "Example 2: Original Calibration (KGE)"
    )

    plot_scatter(
        base_path / "example2_log_timeseries.csv",
        base_path / "example2_log_scatter.png",
        "Example 2: Log-transformed Calibration (LogKGE)"
    )

    print("Example 2 plots complete!")


def plot_example3(base_dir):
    """Generate all plots for Example 3"""
    print("\nGenerating Example 3 plots...")
    base_path = Path(base_dir)

    # Pareto front
    plot_pareto_front(
        base_path / "example3_pareto_front.csv",
        base_path / "example3_pareto_front_plot.png",
        "Example 3: Pareto Front (NSE vs LogNSE)"
    )

    # Time series for each solution
    plot_timeseries(
        base_path / "example3_best_nse_timeseries.csv",
        base_path / "example3_best_nse_plot.png",
        "Example 3: Best NSE Solution (High Flow Focus)"
    )

    plot_timeseries(
        base_path / "example3_best_lognse_timeseries.csv",
        base_path / "example3_best_lognse_plot.png",
        "Example 3: Best LogNSE Solution (Low Flow Focus)"
    )

    plot_timeseries(
        base_path / "example3_balanced_timeseries.csv",
        base_path / "example3_balanced_plot.png",
        "Example 3: Balanced Solution (Compromise)"
    )

    print("Example 3 plots complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize HydroModelMCP example results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all results for Example 1
  python plot_results.py --example 1

  # Plot specific time series
  python plot_results.py --timeseries example1_validation_timeseries.csv

  # Plot Pareto front
  python plot_results.py --pareto example3_pareto_front.csv

  # Plot sensitivity analysis
  python plot_results.py --sensitivity example1_sensitivity.csv
        """
    )

    parser.add_argument('--example', type=int, choices=[1, 2, 3],
                       help='Generate all plots for specified example (1, 2, or 3)')
    parser.add_argument('--timeseries', type=str,
                       help='Plot time series from CSV file')
    parser.add_argument('--scatter', type=str,
                       help='Plot scatter plot from CSV file')
    parser.add_argument('--sensitivity', type=str,
                       help='Plot sensitivity analysis from CSV file')
    parser.add_argument('--pareto', type=str,
                       help='Plot Pareto front from CSV file')
    parser.add_argument('--comparison', nargs=2, metavar=('CSV1', 'CSV2'),
                       help='Compare two time series')
    parser.add_argument('--output', type=str,
                       help='Output file path (for single plots)')
    parser.add_argument('--dir', type=str, default='.',
                       help='Base directory for example files (default: current directory)')

    args = parser.parse_args()

    # Change to examples directory if running from utils
    if Path.cwd().name == 'utils':
        base_dir = Path('..') / args.dir
    else:
        base_dir = Path(args.dir)

    if args.example:
        if args.example == 1:
            plot_example1(base_dir)
        elif args.example == 2:
            plot_example2(base_dir)
        elif args.example == 3:
            plot_example3(base_dir)
    elif args.timeseries:
        plot_timeseries(base_dir / args.timeseries, args.output)
    elif args.scatter:
        plot_scatter(base_dir / args.scatter, args.output)
    elif args.sensitivity:
        plot_sensitivity(base_dir / args.sensitivity, args.output)
    elif args.pareto:
        plot_pareto_front(base_dir / args.pareto, args.output)
    elif args.comparison:
        labels = ["Series 1", "Series 2"]
        plot_comparison(base_dir / args.comparison[0],
                       base_dir / args.comparison[1],
                       labels, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
