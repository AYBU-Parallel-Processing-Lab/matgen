import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
from scipy.sparse import isspmatrix_coo


def get_stats(matrix):
    """Calculates comprehensive statistics for a sparse matrix."""

    # Ensure matrix is in COO format for easy access to row, col, data
    if not isspmatrix_coo(matrix):
        matrix = matrix.tocoo()

    stats = {}
    rows, cols = matrix.shape
    nnz = matrix.nnz

    # --- Basic Stats (N, NNZ, Density) ---
    stats['N'] = matrix.shape
    stats['NNZ'] = nnz
    stats['Density'] = nnz / (rows * cols) if rows * cols > 0 else 0.0

    # --- Symmetry Stats (Sim, Psym) ---
    # Only check symmetry for square matrices
    if rows == cols:
        # Sim: Value Symmetry (A == A.T)
        is_symmetric = (matrix - matrix.T).nnz == 0
        stats['Sim'] = 1.0 if is_symmetric else 0.0

        # Psym: Structural Symmetry (% of pattern)
        A_struct = matrix.astype(bool)
        A_T_struct = matrix.T.astype(bool)
        symmetric_pattern_nnz = A_struct.multiply(A_T_struct).nnz
        total_pattern_nnz = (A_struct + A_T_struct).nnz

        if total_pattern_nnz == 0:
            stats['Psym'] = 1.0
        else:
            stats['Psym'] = (symmetric_pattern_nnz / total_pattern_nnz)
    else:
        # Non-square matrices cannot be symmetric
        stats['Sim'] = 0.0
        stats['Psym'] = 0.0

    # --- Diagonal Stats (Diag, Ndg) ---
    diag_size = min(rows, cols)
    if diag_size > 0:
        diag_nnz = np.count_nonzero(matrix.diagonal())
    else:
        diag_nnz = 0

    stats['Diag'] = diag_nnz
    stats['Ndg'] = nnz - diag_nnz

    # --- Bandwidth and Profile ---
    if nnz == 0:
        stats['Band'] = 0
        stats['Profile'] = 0
    else:
        # Bandwidth: max distance from diagonal
        stats['Band'] = int(np.abs(matrix.row - matrix.col).max())

        # Profile: vectorized calculation
        # For each non-zero, calculate distance from diagonal
        distances = np.abs(matrix.col - matrix.row)

        # Group by row and find max distance per row
        # Using bincount for fast aggregation
        if rows > 0:
            max_distances = np.zeros(rows)
            np.maximum.at(max_distances, matrix.row, distances)
            stats['Profile'] = int(max_distances.sum())
        else:
            stats['Profile'] = 0

    # --- Value Stats (Rmx, Rmi, Rstd, Dist) ---
    if nnz == 0:
        stats['Rmx'] = 0.0
        stats['Rmi'] = 0.0
        stats['Mean'] = 0.0
        stats['Rstd'] = 0.0
        stats['Sum'] = 0.0
        stats['Dist'] = 0.0  # Range/spread metric
    else:
        data = matrix.data
        stats['Rmx'] = float(data.max())
        stats['Rmi'] = float(data.min())
        stats['Mean'] = float(data.mean())
        stats['Rstd'] = float(data.std())
        stats['Sum'] = float(data.sum())

        # Dist: Coefficient of Variation (normalized measure of dispersion)
        # CV = std / mean (when mean != 0)
        if abs(stats['Mean']) > 1e-10:
            stats['Dist'] = stats['Rstd'] / abs(stats['Mean'])
        else:
            stats['Dist'] = 0.0

    return stats


def detect_scaling_method(stats_orig, stats_scaled):
    """Attempts to detect which scaling method was used based on statistics."""

    orig_nnz = stats_orig['NNZ']
    scaled_nnz = stats_scaled['NNZ']

    if orig_nnz == 0:
        return 'Unknown'

    orig_rows, orig_cols = stats_orig['N']
    scaled_rows, scaled_cols = stats_scaled['N']

    row_scale = scaled_rows / orig_rows if orig_rows > 0 else 0
    col_scale = scaled_cols / orig_cols if orig_cols > 0 else 0

    expected_nn_nnz = orig_nnz * row_scale * col_scale

    # Check if NNZ matches nearest neighbor expectation
    if abs(scaled_nnz - expected_nn_nnz) < 1.0:
        return 'Nearest Neighbor'
    # Bilinear typically has more NNZ
    elif scaled_nnz > expected_nn_nnz:
        return 'Bilinear (likely)'
    else:
        return 'Unknown'


def print_comparison_table(stats_orig, stats_scaled, method='Auto-detect'):
    """Prints a comprehensive comparison table matching the requested format."""

    # Auto-detect method if not provided
    if method == 'Auto-detect':
        method = detect_scaling_method(stats_orig, stats_scaled)

    print("\n" + "="*100)
    print(" " * 35 + "MATRIX COMPARISON")
    print("="*100)

    # Header
    header = f"{'Metric':<12} | {'Original':<25} | {'Scaled':<25} | {'Ratio/Diff':<20}"
    print(header)
    print("-" * 100)

    # Method row
    print(f"{'Method':<12} | {'-':<25} | {method:<25} | {'-':<20}")
    print("-" * 100)

    # Define metrics to display
    metrics = [
        # (key, description, format_type)
        ('N', 'Dimensions', 'shape'),
        ('NNZ', 'Non-zeros', 'int'),
        ('Density', 'Density', 'sci'),
        ('sep', '', ''),
        ('Sim', 'Symmetric', 'pct'),
        ('Psym', 'Pattern Sym', 'pct'),
        ('sep', '', ''),
        ('Diag', 'Diagonal NNZ', 'int'),
        ('Ndg', 'Off-Diag NNZ', 'int'),
        ('Band', 'Bandwidth', 'int'),
        ('Profile', 'Profile', 'int'),
        ('sep', '', ''),
        ('Rmx', 'Max Value', 'float'),
        ('Rmi', 'Min Value', 'float'),
        ('Mean', 'Mean', 'float'),
        ('Rstd', 'Std Dev', 'float'),
        ('Dist', 'Coeff Var', 'float'),
        ('Sum', 'Total Sum', 'float'),
    ]

    for key, desc, fmt in metrics:
        if key == 'sep':
            print("-" * 100)
            continue

        orig_val = stats_orig[key]
        scaled_val = stats_scaled[key]

        # Format values based on type
        if fmt == 'shape':
            orig_str = f"{orig_val[0]} x {orig_val[1]}"
            scaled_str = f"{scaled_val[0]} x {scaled_val[1]}"
            if orig_val[0] > 0 and orig_val[1] > 0:
                ratio_str = f"{scaled_val[0]/orig_val[0]:.2f}x, {scaled_val[1]/orig_val[1]:.2f}x"
            else:
                ratio_str = "N/A"
        elif fmt == 'int':
            orig_str = f"{orig_val:,}"
            scaled_str = f"{scaled_val:,}"
            if orig_val > 0:
                ratio = scaled_val / orig_val
                ratio_str = f"{ratio:.4f}x"
            else:
                ratio_str = "N/A"
        elif fmt == 'float':
            orig_str = f"{orig_val:.6g}"
            scaled_str = f"{scaled_val:.6g}"
            if abs(orig_val) > 1e-10:
                ratio = scaled_val / orig_val
                ratio_str = f"{ratio:.4f}x"
            else:
                diff = scaled_val - orig_val
                ratio_str = f"Δ {diff:.6g}"
        elif fmt == 'sci':
            orig_str = f"{orig_val:.4e}"
            scaled_str = f"{scaled_val:.4e}"
            if orig_val > 0:
                ratio = scaled_val / orig_val
                ratio_str = f"{ratio:.4f}x"
            else:
                ratio_str = "N/A"
        elif fmt == 'pct':
            orig_str = f"{orig_val*100:.2f}%"
            scaled_str = f"{scaled_val*100:.2f}%"
            diff = (scaled_val - orig_val) * 100
            ratio_str = f"Δ {diff:+.2f}%"
        else:
            orig_str = str(orig_val)
            scaled_str = str(scaled_val)
            ratio_str = "N/A"

        print(f"{desc:<12} | {orig_str:<25} | {scaled_str:<25} | {ratio_str:<20}")

    print("="*100)


def analyze_scaling(stats_orig, stats_scaled):
    """Analyzes the scaling operation and returns derived metrics."""

    analysis = {}

    # --- Dimension Scaling ---
    orig_rows, orig_cols = stats_orig['N']
    scaled_rows, scaled_cols = stats_scaled['N']

    if orig_rows > 0 and orig_cols > 0:
        row_scale = scaled_rows / orig_rows
        col_scale = scaled_cols / orig_cols
        analysis['Row Scale Factor'] = row_scale
        analysis['Col Scale Factor'] = col_scale
        analysis['Scale Type'] = classify_scale_type(row_scale, col_scale)
    else:
        row_scale = 0.0
        col_scale = 0.0
        analysis['Row Scale Factor'] = row_scale
        analysis['Col Scale Factor'] = col_scale
        analysis['Scale Type'] = 'Empty Matrix'

    # --- NNZ Scaling ---
    orig_nnz = stats_orig['NNZ']
    scaled_nnz = stats_scaled['NNZ']

    if orig_nnz > 0:
        nnz_ratio = scaled_nnz / orig_nnz
        analysis['NNZ Ratio'] = nnz_ratio
        expected_nn = orig_nnz * row_scale * col_scale
        analysis['Expected NNZ (NN)'] = expected_nn
        analysis['NNZ Match NN'] = abs(scaled_nnz - expected_nn) < 1.0
    else:
        analysis['NNZ Ratio'] = 0 if scaled_nnz == 0 else float('inf')
        analysis['Expected NNZ (NN)'] = 0
        analysis['NNZ Match NN'] = (scaled_nnz == 0)

    # --- Diagonal Scaling ---
    orig_diag = stats_orig['Diag']
    scaled_diag = stats_scaled['Diag']

    if orig_diag > 0:
        diag_ratio = scaled_diag / orig_diag
        analysis['Diag NNZ Ratio'] = diag_ratio

        if abs(row_scale - col_scale) < 1e-6:
            expected_diag = orig_diag * row_scale
        else:
            min_scale = min(row_scale, col_scale)
            expected_diag = orig_diag * min_scale

        analysis['Expected Diag NNZ'] = expected_diag
        analysis['Diag Match Expected'] = abs(scaled_diag - expected_diag) < 1.0
    else:
        analysis['Diag NNZ Ratio'] = 0 if scaled_diag == 0 else float('inf')
        analysis['Expected Diag NNZ'] = 0
        analysis['Diag Match Expected'] = (scaled_diag == 0)

    # --- Value Conservation ---
    orig_sum = stats_orig['Sum']
    scaled_sum = stats_scaled['Sum']

    if abs(orig_sum) > 1e-10:
        sum_ratio = scaled_sum / orig_sum
        analysis['Sum Ratio'] = sum_ratio
        analysis['Value Conserved'] = abs(sum_ratio - 1.0) < 1e-6
    else:
        analysis['Sum Ratio'] = 0 if abs(scaled_sum) < 1e-10 else float('inf')
        analysis['Value Conserved'] = abs(scaled_sum) < 1e-10

    # --- Density Change ---
    orig_density = stats_orig['Density']
    scaled_density = stats_scaled['Density']

    if orig_density > 0:
        density_ratio = scaled_density / orig_density
        analysis['Density Ratio'] = density_ratio
    else:
        analysis['Density Ratio'] = 0 if scaled_density == 0 else float('inf')

    return analysis


def classify_scale_type(row_scale, col_scale):
    """Classifies the type of scaling operation."""

    if abs(row_scale - 1.0) < 1e-6 and abs(col_scale - 1.0) < 1e-6:
        return 'Identity (1x)'
    elif row_scale > 1 and col_scale > 1:
        if abs(row_scale - col_scale) < 1e-6:
            return f'Upscale (Square {row_scale:.2f}x)'
        else:
            return f'Upscale (Non-square {row_scale:.2f}x{col_scale:.2f})'
    elif row_scale < 1 and col_scale < 1:
        if abs(row_scale - col_scale) < 1e-6:
            return f'Downscale (Square {row_scale:.2f}x)'
        else:
            return f'Downscale (Non-square {row_scale:.2f}x{col_scale:.2f})'
    elif abs(row_scale - col_scale) < 1e-6:
        return f'Scale (Square {row_scale:.2f}x)'
    else:
        return f'Scale (Non-square {row_scale:.2f}x{col_scale:.2f})'


def print_analysis(analysis):
    """Prints the scaling analysis summary."""

    print("\n" + "="*80)
    print("--- Scaling Summary ---")
    print("="*80)

    print(f"  Scale Type: {analysis['Scale Type']}")
    print(f"  Row/Col Factors: {analysis['Row Scale Factor']:.4f} x {analysis['Col Scale Factor']:.4f}")

    expected_nn = analysis['Expected NNZ (NN)']
    print(f"  NNZ Growth: {analysis['NNZ Ratio']:.4f}x (Expected NN: {expected_nn:.4f})")

    conserved_str = '✓ YES' if analysis['Value Conserved'] else '✗ NO'
    print(f"  Value Conserved: {conserved_str} (Sum ratio: {analysis['Sum Ratio']:.6f})")

    print("="*80)


def main():
    # --- 1. Parse Command-Line Arguments ---
    parser = argparse.ArgumentParser(
        description='Compare two MTX matrices with detailed scaling analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare scaled matrix to original
  python compare_mtx.py original.mtx scaled_4x.mtx

  # With method specification
  python compare_mtx.py original.mtx scaled.mtx --method "Nearest Neighbor"

  # Skip plots (faster for batch processing)
  python compare_mtx.py original.mtx scaled.mtx --no-plot
        """
    )

    parser.add_argument('original', help='Path to original MTX file')
    parser.add_argument('scaled', help='Path to scaled MTX file')
    parser.add_argument('--method', default='Auto-detect',
                       help='Scaling method used (default: auto-detect)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip visualization (faster for batch processing)')
    parser.add_argument('--no-summary', action='store_true',
                       help='Skip scaling summary (show only comparison table)')

    args = parser.parse_args()

    # --- 2. Load Matrices ---
    try:
        matrix_orig = mmread(args.original)
        matrix_scaled = mmread(args.scaled)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An error occurred while reading a file: {e}")
        sys.exit(1)

    # --- 3. Get Statistics ---
    stats_orig = get_stats(matrix_orig)
    stats_scaled = get_stats(matrix_scaled)

    # --- 4. Print Comparison Table ---
    print_comparison_table(stats_orig, stats_scaled, method=args.method)

    # --- 5. Print Scaling Summary (optional) ---
    if not args.no_summary:
        analysis = analyze_scaling(stats_orig, stats_scaled)
        print_analysis(analysis)

    # --- 6. Create Side-by-Side Plots ---
    if not args.no_plot:
        analysis = analyze_scaling(stats_orig, stats_scaled)

        # Calculate appropriate marker sizes
        orig_size = max(stats_orig['N'])
        scaled_size = max(stats_scaled['N'])

        def calc_markersize(size):
            if size < 100:
                return 5
            elif size < 500:
                return 2
            elif size < 2000:
                return 1
            else:
                return 0.5

        orig_marker = calc_markersize(orig_size)
        scaled_marker = calc_markersize(scaled_size)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: Original Matrix
        ax1.spy(matrix_orig, markersize=orig_marker, aspect='auto')
        ax1.set_title("Original", fontsize=12, fontweight='bold')
        ax1.set_xlabel(f"Shape: {stats_orig['N']} | NNZ: {stats_orig['NNZ']:,}", fontsize=10)
        ax1.set_ylabel('Row', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Plot 2: Scaled Matrix
        ax2.spy(matrix_scaled, markersize=scaled_marker, aspect='auto')
        ax2.set_title(f"Scaled ({args.method})", fontsize=12, fontweight='bold')
        ax2.set_xlabel(f"Shape: {stats_scaled['N']} | NNZ: {stats_scaled['NNZ']:,}", fontsize=10)
        ax2.set_ylabel('Row', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # Adjust axes limits
        ax1.set_xlim(-0.5, stats_orig['N'][1] - 0.5)
        ax1.set_ylim(stats_orig['N'][0] - 0.5, -0.5)
        ax2.set_xlim(-0.5, stats_scaled['N'][1] - 0.5)
        ax2.set_ylim(stats_scaled['N'][0] - 0.5, -0.5)

        # Overall title
        conserved = '✓' if analysis['Value Conserved'] else '✗'
        nnz_info = f"NNZ: {stats_orig['NNZ']:,} → {stats_scaled['NNZ']:,} ({analysis['NNZ Ratio']:.2f}x)"
        fig.suptitle(f"Sparsity Pattern Comparison | Value Conserved: {conserved} | {nnz_info}",
                    fontsize=14, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()


if __name__ == "__main__":
    main()
