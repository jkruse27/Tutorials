import numpy as np
import pandas as pd
import os
import glob
import sys

sys.path.append(os.getcwd())

try:
    from hrv_utils import rri_utils
    from hrv_utils import nongaussian
    from hrv_utils import dma
except ImportError:
    print("❌ Could not import 'hrv_utils'.")
    sys.exit(1)


def read_raw_rri(filepath):
    """
    Reads raw RRI file.
    Assumes: No header, one value per line, simple CSV structure.
    """
    try:
        df = pd.read_csv(filepath, header=None)
        return df.values.flatten().astype(np.float64)
    except Exception as e:
        print(f"Error reading raw file {filepath}: {e}")
        return None


def read_ground_truth(filepath):
    """
    Reads the processed ground truth file.
    Robustly looks for 'RRI', 'RRI.r', or falls back to the 2nd column
    (assuming Col 1 is time, Col 2 is RRI).
    """
    try:
        df = pd.read_csv(filepath)

        return df['RRI_Resampled'].values

    except Exception as e:
        print(f"Error reading GT file {filepath}: {e}")
        return None


def run_comparison(orig_dir, proc_dir):
    print(f"{'FILENAME':<30} | {'STATUS':<10} | {'DETAILS':<20}")
    print("-" * 75)

    orig_files = glob.glob(os.path.join(orig_dir, "*.csv"))

    if not orig_files:
        print(f"No .csv files found in {orig_dir}")
        return

    for f_path in orig_files:
        filename = os.path.basename(f_path)
        proc_path = os.path.join(proc_dir, filename)

        if not os.path.exists(proc_path):
            print(f"{filename:<30} | ❌ MISSING   | Processed file not found")
            continue

        rri_raw = read_raw_rri(f_path)
        if rri_raw is None:
            continue

        try:
            result = rri_utils.clean_and_resample_pipeline(
                rri_raw, start_hour=0.0, t_sec=0.5
            )
            my_rri = result['rri']
        except Exception as e:
            print(f"{filename:<30} | ❌ ERROR     | Processing: {e}")
            continue

        gt_rri = read_ground_truth(proc_path)
        if gt_rri is None:
            print(f"{filename:<30} | ❌ ERROR     | GT Read Error")
            continue

        min_len = min(len(my_rri), len(gt_rri))

        if min_len == 0:
            print(f"{filename:<30} | ❌ EMPTY     | Result is empty")
            continue

        diff = np.abs(my_rri[:min_len] - gt_rri[:min_len])

        mean_err = np.mean(diff)
        max_err = np.max(diff)

        is_passed = (mean_err < 1.0) and (max_err < 5.0)

        status_icon = "✅ PASSED" if is_passed else "❌ FAILED"

        details = ""
        if abs(len(my_rri) - len(gt_rri)) > 1:
            details = f"(Len mismatch: {len(my_rri)} vs {len(gt_rri)})"
        elif not is_passed:
            details = f"(Mean: {mean_err:.2f}, Max: {max_err:.2f})"

        print(f"{filename:<30} | {status_icon:<10} | {details}")


def run_savgol(orig_dir, proc_dir):
    print(f"{'FILENAME':<30} | {'STATUS':<10} | {'DETAILS':<20}")
    print("-" * 75)

    orig_files = glob.glob(os.path.join(orig_dir, "*.csv"))

    if not orig_files:
        print(f"No .csv files found in {orig_dir}")
        return

    for f_path in orig_files:
        filename = os.path.basename(f_path)
        proc_path = os.path.join(proc_dir, filename)

        if not os.path.exists(proc_path):
            print(f"{filename:<30} | ❌ MISSING   | Processed file not found")
            continue

        rri_raw = read_raw_rri(f_path)
        if rri_raw is None:
            continue

        try:
            result = rri_utils.clean_and_resample_pipeline(
                rri_raw, start_hour=0.0, t_sec=0.5
            )['rri']
            result = np.cumsum(result - np.mean(result))
            result = nongaussian.sgolayfilt(result, 3, 51)
            my_rri = result
        except Exception as e:
            print(f"{filename:<30} | ❌ ERROR     | Processing: {e}")
            continue

        gt_rri = read_ground_truth(proc_path)
        if gt_rri is None:
            print(f"{filename:<30} | ❌ ERROR     | GT Read Error")
            continue

        min_len = min(len(my_rri), len(gt_rri))

        if min_len == 0:
            print(f"{filename:<30} | ❌ EMPTY     | Result is empty")
            continue

        diff = np.abs(my_rri[:min_len] - gt_rri[:min_len])

        mean_err = np.mean(diff)
        max_err = np.max(diff)

        is_passed = (mean_err < 1.0) and (max_err < 5.0)

        status_icon = "✅ PASSED" if is_passed else "❌ FAILED"

        details = ""
        if abs(len(my_rri) - len(gt_rri)) > 1:
            details = f"(Len mismatch: {len(my_rri)} vs {len(gt_rri)})"
        elif not is_passed:
            details = f"(Mean: {mean_err:.2f}, Max: {max_err:.2f})"

        print(f"{filename:<30} | {status_icon:<10} | {details}")


def load_ground_truth_lambdas(csv_path):
    """Loads {filename: lambda_val} from a summary CSV."""
    if not os.path.exists(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path)
        df.iloc[:, 0] = df.iloc[:, 0].apply(os.path.basename)
        return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    except Exception as e:
        print(f"Error reading ground truth CSV: {e}")
        return {}


def run_lambda_comparison(orig_dir, gt_csv_path, scale_sec=25, t_sec=0.5):
    """
    Computes lambda for files in orig_dir and compares with values in
    gt_csv_path.

    Parameters:
    - orig_dir: Directory containing raw RRI .csv files
    - gt_csv_path: Path to the CSV containing [filename, expected_lambda]
    - scale_sec: The scale in seconds to compute lambda at (default 25s)
    - t_sec: Resampling interval (default 0.5s)
    """
    print(f"{'FILENAME':<30} | {'STATUS':<10} | {'DETAILS':<35}")
    print("-" * 80)

    gt_map = load_ground_truth_lambdas(gt_csv_path)
    if not gt_map:
        print(f"❌ Error: Could not load ground truth from {gt_csv_path}")
        return

    orig_files = glob.glob(os.path.join(orig_dir, "*.csv"))
    if not orig_files:
        print(f"No .csv files found in {orig_dir}")
        return

    scale_samples = int(2 * np.round((scale_sec / t_sec) / 2) + 1)

    scales_arr = np.array([scale_samples], dtype=np.int64)

    for f_path in orig_files:
        filename = os.path.basename(f_path)

        if filename not in gt_map:
            print(f"{filename:<30} | ⚠️ SKIP     | Not in Ground Truth CSV")
            continue

        gt_lambda = gt_map[filename]

        rri_raw = read_raw_rri(f_path)
        if rri_raw is None:
            continue

        try:
            result = rri_utils.clean_and_resample_pipeline(
                rri_raw, start_hour=0.0, t_sec=t_sec
            )
            my_rri = result['rri']

            if len(my_rri) <= scale_samples + 2:
                print(f"{filename:<30} | ❌ SHORT    " +
                      "| Signal too short for scale")
                continue

            lambdas_sq, _ = nongaussian.nongaussian_analysis(
                my_rri,
                scales_arr,
                q=0.25,
                m=3
            )

            my_lambda = lambdas_sq[0]

        except Exception as e:
            print(f"{filename:<30} | ❌ ERROR    " +
                  f"| Processing: {str(e)[:20]}...")
            continue

        diff = abs(my_lambda - gt_lambda)
        is_passed = diff < 0.01

        status_icon = "✅ PASSED" if is_passed else "❌ FAILED"

        details = f"Got: {my_lambda:.4f} | GT: {gt_lambda:.4f} | Δ: {diff:.4f}"

        print(f"{filename:<30} | {status_icon:<10} | {details}")


def test_dma_vs_c_output(ORIG_DIR, DMA_DIR, order):
    out_files = glob.glob(os.path.join(DMA_DIR, "*.csv"))

    for c_output_file in out_files:
        df = pd.read_csv(c_output_file, header=None)
        c_log_scales = df.iloc[:, 2].values
        scales = np.round(10**c_log_scales).astype(np.int_)

        target_log_rmse = df.iloc[:, 1].values
        input_data_file = f"{ORIG_DIR}/{c_output_file.split('/')[-1]}"

        data = np.loadtxt(input_data_file)
        python_rmse = dma.dma(data, scales, order=order, integrate=1)
        python_log_rmse = np.log10(python_rmse)
        diff = np.abs(python_log_rmse - target_log_rmse)
        mean_err = np.mean(diff)
        max_err = np.max(diff)

        is_passed = (mean_err < 0.001) and (max_err < 0.001)
        status_icon = "✅ PASSED" if is_passed else "❌ FAILED"

        details = ""
        if abs(len(python_log_rmse) - len(target_log_rmse)) > 1:
            l1 = len(python_log_rmse)
            l2 = len(target_log_rmse)
            details = f"(Len mismatch: {l1} vs {l2})"
        elif not is_passed:
            details = f"(Mean: {mean_err:.2f}, Max: {max_err:.2f})"

        print(f"{c_output_file:<30} | {status_icon:<10} | {details}")


def test_synthetic_lambdas(target_lambdas):
    print(f"{'TARGET Λ':<30} | {'STATUS':<10} | {'DETAILS':<35}")
    print("-" * 80)

    K = 50
    for gt_lambda in target_lambdas:
        my_lambda = 0
        for i in range(K):
            signal = nongaussian.generate_nongaussian(
                14,
                gt_lambda,
                3
            )
            lambdas_sq = nongaussian.nongaussian_index(signal, 0.25)

            my_lambda += np.abs(lambdas_sq)

        my_lambda /= K
        diff = abs(my_lambda - gt_lambda)
        is_passed = diff < 0.05

        status_icon = "✅ PASSED" if is_passed else "❌ FAILED"

        name_display = f"Synth_Lambda_{gt_lambda:.3f}"
        details = f"Got: {my_lambda:.4f} | GT: {gt_lambda:.4f} | Δ: {diff:.4f}"

        print(f"{name_display:<30} | {status_icon:<10} | {details}")


if __name__ == "__main__":
    ORIG_DIR = "test/orig"
    PROC_DIR = "test/processed"
    SG_DIR = "test/savgol"
    DMA0_DIR = "test/DMA0"
    DMA2_DIR = "test/DMA2"
    DMA4_DIR = "test/DMA4"

    print("CLEANING DATA COMPARISON")
    run_comparison(ORIG_DIR, PROC_DIR)
    print()
    print("------------------------------------------------------------")
    print()
    print("SAVGOL COMPARISON")
    run_savgol(ORIG_DIR, SG_DIR)
    print()
    print("------------------------------------------------------------")
    print()
    print("NON-GAUSSIAN INDEX COMPARISON")
    run_lambda_comparison(ORIG_DIR, "test/lambda_results.csv")
    print()
    print("------------------------------------------------------------")
    print()
    print("NON-GAUSSIAN GENERATION EVALUATION")
    test_synthetic_lambdas(np.arange(0, 1, 0.1, dtype=np.float64))
    print()
    print("------------------------------------------------------------")
    print()
    print("DMA 0th order")
    test_dma_vs_c_output(ORIG_DIR, DMA0_DIR, 0)
    print()
    print("------------------------------------------------------------")
    print()
    print("DMA 2nd order")
    test_dma_vs_c_output(ORIG_DIR, DMA2_DIR, 2)
    print()
    print("------------------------------------------------------------")
    print()
    print("DMA 4th order")
    test_dma_vs_c_output(ORIG_DIR, DMA4_DIR, 4)
