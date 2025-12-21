"""
Analyze BRATS dataset to find optimal crop dimensions.
This script determines the minimal bounding rectangle that contains brain content.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import os


BASE_DIR = Path(__file__).resolve().parents[2]
BRATS_ROOT = Path(BASE_DIR / "data" / "brats-2021").expanduser()


def find_bounding_box(slice_2d, threshold=0.01):
    """
    Find the bounding box of non-background content in a 2D slice.
    
    Args:
        slice_2d: 2D numpy array
        threshold: Minimum intensity to consider as content (normalized 0-1)
    
    Returns:
        (min_row, max_row, min_col, max_col) or None if empty
    """
    # Normalize
    if slice_2d.max() > 0:
        normalized = slice_2d / slice_2d.max()
    else:
        return None
    
    # Find non-background pixels
    mask = normalized > threshold
    
    if not mask.any():
        return None
    
    # Find bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]
    
    if len(row_indices) == 0 or len(col_indices) == 0:
        return None
    
    min_row, max_row = row_indices[0], row_indices[-1]
    min_col, max_col = col_indices[0], col_indices[-1]
    
    return (min_row, max_row, min_col, max_col)


def process_single_case(args):
    """
    Process a single case and return bounding box statistics.
    This function is called by multiprocessing workers.
    
    Args:
        args: Tuple of (case_dir, slice_axis, slices_per_case)
    
    Returns:
        Tuple of (widths, heights, original_shape, slice_data)
    """
    case_dir, slice_axis, slices_per_case = args
    
    widths = []
    heights = []
    slice_data = []
    
    # Find FLAIR modality
    flair_files = sorted(case_dir.glob("*flair.nii*"))
    if not flair_files:
        return widths, heights, None, slice_data
    
    try:
        # Load volume
        volume = nib.load(flair_files[0]).get_fdata()
        original_shape = volume.shape
        
        # Sample slices from center (or all slices if slices_per_case <= 0)
        num_slices = volume.shape[slice_axis]
        
        if slices_per_case <= 0:
            # Use all slices
            slice_ids = np.arange(num_slices, dtype=int)
        else:
            # Sample slices_per_case slices from center
            center_slice = num_slices // 2
            offset = slices_per_case // 2
            usable = range(center_slice - offset, center_slice + offset)
            slice_ids = np.linspace(
                usable.start,
                usable.stop - 1,
                num=min(slices_per_case, len(usable)),
                dtype=int,
            )
        
        for slice_idx in slice_ids:
            slice_2d = np.take(volume, indices=slice_idx, axis=slice_axis).astype(np.float32)
            
            bbox = find_bounding_box(slice_2d)
            if bbox is not None:
                min_row, max_row, min_col, max_col = bbox
                height = max_row - min_row + 1
                width = max_col - min_col + 1
                
                heights.append(height)
                widths.append(width)
                
                # Store slice data for outlier analysis (lightweight version)
                # We'll identify outliers later and only keep necessary data
                slice_data.append({
                    'width': width,
                    'height': height,
                    'bbox': bbox,
                    'case_dir': case_dir,  # Store path instead of slice data
                    'slice_idx': slice_idx,
                    'case_name': case_dir.name,
                    'slice_axis': slice_axis,
                    'flair_path': flair_files[0],  # Store path to reload if needed
                })
        
        return widths, heights, original_shape, slice_data
    
    except Exception as e:
        print(f"Error processing {case_dir}: {e}")
        return widths, heights, None, slice_data


def analyze_dataset(brats_root, slice_axis=2, max_cases=None, slices_per_case=12, num_workers=None):
    """
    Analyze the dataset to find bounding box statistics using multiprocessing.
    
    Args:
        brats_root: Path to BRATS dataset
        slice_axis: Which axis to slice along
        max_cases: Maximum number of cases to analyze (None = all)
        slices_per_case: Number of slices to sample per case
        num_workers: Number of parallel workers (None = auto-detect)
    
    Returns:
        Dictionary with statistics
    """
    case_dirs = sorted([p for p in Path(brats_root).iterdir() if p.is_dir()])
    
    if max_cases is not None:
        case_dirs = case_dirs[:max_cases]
    
    if num_workers is None:
        # Respect SLURM CPU allocation if running under SLURM
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
        else:
            num_workers = cpu_count()
        num_workers = min(num_workers, len(case_dirs))
    
    print(f"Analyzing {len(case_dirs)} cases using {num_workers} workers...")
    
    # Prepare arguments for each case
    case_args = [(case_dir, slice_axis, slices_per_case) for case_dir in case_dirs]
    
    all_widths = []
    all_heights = []
    original_shapes = []
    all_slice_data = []
    
    # Process cases in parallel with progress bar
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_case, case_args),
            total=len(case_args),
            desc="Processing cases"
        ))
    
    # Aggregate results
    for widths, heights, original_shape, slice_data in results:
        all_widths.extend(widths)
        all_heights.extend(heights)
        if original_shape is not None:
            original_shapes.append(original_shape)
        all_slice_data.extend(slice_data)
    
    # Calculate global bounding box (union of all bboxes)
    all_min_rows = [data['bbox'][0] for data in all_slice_data if 'bbox' in data]
    all_max_rows = [data['bbox'][1] for data in all_slice_data if 'bbox' in data]
    all_min_cols = [data['bbox'][2] for data in all_slice_data if 'bbox' in data]
    all_max_cols = [data['bbox'][3] for data in all_slice_data if 'bbox' in data]
    
    global_bbox = {
        'min_row': np.min(all_min_rows) if all_min_rows else 0,
        'max_row': np.max(all_max_rows) if all_max_rows else 0,
        'min_col': np.min(all_min_cols) if all_min_cols else 0,
        'max_col': np.max(all_max_cols) if all_max_cols else 0,
    }
    global_bbox['height'] = global_bbox['max_row'] - global_bbox['min_row'] + 1
    global_bbox['width'] = global_bbox['max_col'] - global_bbox['min_col'] + 1
    
    # Calculate statistics
    stats = {
        'num_slices': len(all_widths),
        'width': {
            'min': np.min(all_widths),
            'max': np.max(all_widths),
            'mean': np.mean(all_widths),
            'median': np.median(all_widths),
            'std': np.std(all_widths),
            'percentile_5': np.percentile(all_widths, 5),
            'percentile_95': np.percentile(all_widths, 95),
        },
        'height': {
            'min': np.min(all_heights),
            'max': np.max(all_heights),
            'mean': np.mean(all_heights),
            'median': np.median(all_heights),
            'std': np.std(all_heights),
            'percentile_5': np.percentile(all_heights, 5),
            'percentile_95': np.percentile(all_heights, 95),
        },
        'global_bbox': global_bbox,  # Add global bounding box
        'original_shapes': original_shapes,
    }
    
    return stats, all_widths, all_heights, all_slice_data


def detect_and_visualize_outliers(all_slice_data, all_widths, all_heights, output_dir='.'):
    """
    Detect outliers and save comparison visualizations.
    
    Args:
        all_slice_data: List of slice data dictionaries
        all_widths: List of all width values
        all_heights: List of all height values
        output_dir: Directory to save outlier visualizations
    """
    if not all_slice_data:
        print("No slice data available for outlier detection")
        return
    
    print("\nDetecting outliers...")
    
    # Calculate IQR for outlier detection
    widths_array = np.array(all_widths)
    heights_array = np.array(all_heights)
    
    # Width outliers
    w_q1, w_q3 = np.percentile(widths_array, [25, 75])
    w_iqr = w_q3 - w_q1
    w_lower = w_q1 - 2.5 * w_iqr
    w_upper = w_q3 + 2.5 * w_iqr
    
    # Height outliers
    h_q1, h_q3 = np.percentile(heights_array, [25, 75])
    h_iqr = h_q3 - h_q1
    h_lower = h_q1 - 2.5 * h_iqr
    h_upper = h_q3 + 2.5 * h_iqr
    
    # Find outliers
    width_outliers = []
    height_outliers = []
    normal_samples = []
    
    for data in all_slice_data:
        w, h = data['width'], data['height']
        is_width_outlier = w < w_lower or w > w_upper
        is_height_outlier = h < h_lower or h > h_upper
        
        if is_width_outlier:
            width_outliers.append(data)
        if is_height_outlier:
            height_outliers.append(data)
        if not is_width_outlier and not is_height_outlier:
            normal_samples.append(data)
    
    print(f"Found {len(width_outliers)} width outliers")
    print(f"Found {len(height_outliers)} height outliers")
    
    # Determine number of examples to show based on outlier count
    max_outliers = max(len(width_outliers), len(height_outliers))
    if max_outliers <= 4:
        num_examples = max_outliers
    elif max_outliers <= 8:
        num_examples = min(6, max_outliers)
    else:
        num_examples = 8  # Show more when there are many outliers
    
    num_examples = min(num_examples, len(normal_samples))  # Can't show more than we have
    
    if num_examples == 0:
        print("Not enough samples for visualization")
        return
    
    print(f"Showing {num_examples} most extreme examples per category")
    
    # Sort and select extremes
    if width_outliers:
        width_outliers_sorted = sorted(width_outliers, key=lambda x: abs(x['width'] - np.median(widths_array)), reverse=True)[:num_examples]
    else:
        width_outliers_sorted = []
    
    if height_outliers:
        height_outliers_sorted = sorted(height_outliers, key=lambda x: abs(x['height'] - np.median(heights_array)), reverse=True)[:num_examples]
    else:
        height_outliers_sorted = []
    
    if normal_samples:
        normal_sorted = sorted(normal_samples, key=lambda x: abs(x['width'] - np.median(widths_array)) + abs(x['height'] - np.median(heights_array)))[:num_examples]
    else:
        normal_sorted = []
    
    # Now load only the slices we need for visualization
    def load_slice(data):
        """Load a single slice from disk."""
        try:
            volume = nib.load(data['flair_path']).get_fdata()
            slice_2d = np.take(volume, indices=data['slice_idx'], axis=data['slice_axis']).astype(np.float32)
            return slice_2d
        except Exception as e:
            print(f"Error loading slice: {e}")
            return None
    
    # Create visualization
    fig, axes = plt.subplots(3, num_examples, figsize=(4*num_examples, 12))
    if num_examples == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Outlier Analysis: Extreme vs Normal Slices', fontsize=16)
    
    # Plot width outliers
    for i in range(num_examples):
        if i < len(width_outliers_sorted):
            data = width_outliers_sorted[i]
            img = load_slice(data)
            if img is None:
                axes[0, i].axis('off')
                continue
            # Normalize for display
            if img.max() > 0:
                img = img / img.max()
            
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f"Width Outlier (z={data['slice_idx']})\nW={data['width']}, H={data['height']}\n{data['case_name'][:20]}", fontsize=9)
            axes[0, i].axis('off')
            
            # Draw bounding box
            min_row, max_row, min_col, max_col = data['bbox']
            rect = plt.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row,
                               fill=False, edgecolor='red', linewidth=2)
            axes[0, i].add_patch(rect)
        else:
            axes[0, i].axis('off')
    
    # Plot height outliers
    for i in range(num_examples):
        if i < len(height_outliers_sorted):
            data = height_outliers_sorted[i]
            img = load_slice(data)
            if img is None:
                axes[1, i].axis('off')
                continue
            if img.max() > 0:
                img = img / img.max()
            
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].set_title(f"Height Outlier (z={data['slice_idx']})\nW={data['width']}, H={data['height']}\n{data['case_name'][:20]}", fontsize=9)
            axes[1, i].axis('off')
            
            min_row, max_row, min_col, max_col = data['bbox']
            rect = plt.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row,
                               fill=False, edgecolor='red', linewidth=2)
            axes[1, i].add_patch(rect)
        else:
            axes[1, i].axis('off')
    
    # Plot normal samples
    for i in range(num_examples):
        if i < len(normal_sorted):
            data = normal_sorted[i]
            img = load_slice(data)
            if img is None:
                axes[2, i].axis('off')
                continue
            if img.max() > 0:
                img = img / img.max()
            
            axes[2, i].imshow(img, cmap='gray')
            axes[2, i].set_title(f"Normal Sample (z={data['slice_idx']})\nW={data['width']}, H={data['height']}\n{data['case_name'][:20]}", fontsize=9)
            axes[2, i].axis('off')
            
            min_row, max_row, min_col, max_col = data['bbox']
            rect = plt.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row,
                               fill=False, edgecolor='green', linewidth=2)
            axes[2, i].add_patch(rect)
        else:
            axes[2, i].axis('off')
    
    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'Width\nOutliers', transform=axes[0, 0].transAxes,
                   fontsize=12, va='center', ha='right', weight='bold', rotation=90)
    axes[1, 0].text(-0.1, 0.5, 'Height\nOutliers', transform=axes[1, 0].transAxes,
                   fontsize=12, va='center', ha='right', weight='bold', rotation=90)
    axes[2, 0].text(-0.1, 0.5, 'Normal\nSamples', transform=axes[2, 0].transAxes,
                   fontsize=12, va='center', ha='right', weight='bold', rotation=90)
    
    plt.tight_layout()
    outlier_path = Path(output_dir) / 'outlier_comparison.png'
    plt.savefig(outlier_path, dpi=150, bbox_inches='tight')
    print(f"Outlier comparison saved to {outlier_path}")
    plt.close()


def visualize_global_bbox(global_bbox, all_slice_data, output_dir='.'):
    """
    Visualize the global bounding box on the widest and highest slices.
    
    Args:
        global_bbox: Dictionary with global bounding box info
        all_slice_data: List of slice data dictionaries
        output_dir: Directory to save visualization
    """
    if not all_slice_data:
        print("No slice data available for global bbox visualization")
        return
    
    print("\nVisualizing global bounding box...")
    
    # Find slice with maximum width
    widest_slice = max(all_slice_data, key=lambda x: x['width'])
    # Find slice with maximum height
    highest_slice = max(all_slice_data, key=lambda x: x['height'])
    
    def load_slice(data):
        """Load a single slice from disk."""
        try:
            volume = nib.load(data['flair_path']).get_fdata()
            slice_2d = np.take(volume, indices=data['slice_idx'], axis=data['slice_axis']).astype(np.float32)
            return slice_2d
        except Exception as e:
            print(f"Error loading slice: {e}")
            return None
    
    # Load the slices
    widest_img = load_slice(widest_slice)
    highest_img = load_slice(highest_slice)
    
    if widest_img is None or highest_img is None:
        print("Failed to load slices for visualization")
        return
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Global Bounding Box Visualization', fontsize=16)
    
    # Normalize images
    if widest_img.max() > 0:
        widest_img = widest_img / widest_img.max()
    if highest_img.max() > 0:
        highest_img = highest_img / highest_img.max()
    
    # Plot widest slice
    axes[0].imshow(widest_img, cmap='gray')
    axes[0].set_title(f"Widest Slice (W={widest_slice['width']}, H={widest_slice['height']})\n"
                     f"Case: {widest_slice['case_name']}, z={widest_slice['slice_idx']}", fontsize=10)
    axes[0].axis('off')
    
    # Draw global bounding box (blue) and individual bbox (red)
    gb = global_bbox
    global_rect = plt.Rectangle((gb['min_col'], gb['min_row']), 
                                gb['width'], gb['height'],
                                fill=False, edgecolor='blue', linewidth=3, 
                                label=f"Global bbox\n({gb['width']}×{gb['height']})")
    axes[0].add_patch(global_rect)
    
    min_row, max_row, min_col, max_col = widest_slice['bbox']
    individual_rect = plt.Rectangle((min_col, min_row), 
                                   max_col-min_col, max_row-min_row,
                                   fill=False, edgecolor='red', linewidth=2,
                                   linestyle='--', label='This slice bbox')
    axes[0].add_patch(individual_rect)
    axes[0].legend(loc='upper right')
    
    # Plot highest slice
    axes[1].imshow(highest_img, cmap='gray')
    axes[1].set_title(f"Highest Slice (W={highest_slice['width']}, H={highest_slice['height']})\n"
                     f"Case: {highest_slice['case_name']}, z={highest_slice['slice_idx']}", fontsize=10)
    axes[1].axis('off')
    
    # Draw global bounding box (blue) and individual bbox (red)
    global_rect = plt.Rectangle((gb['min_col'], gb['min_row']), 
                                gb['width'], gb['height'],
                                fill=False, edgecolor='blue', linewidth=3,
                                label=f"Global bbox\n({gb['width']}×{gb['height']})")
    axes[1].add_patch(global_rect)
    
    min_row, max_row, min_col, max_col = highest_slice['bbox']
    individual_rect = plt.Rectangle((min_col, min_row), 
                                   max_col-min_col, max_row-min_row,
                                   fill=False, edgecolor='red', linewidth=2,
                                   linestyle='--', label='This slice bbox')
    axes[1].add_patch(individual_rect)
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    bbox_path = Path(output_dir) / 'global_bbox_visualization.png'
    plt.savefig(bbox_path, dpi=150, bbox_inches='tight')
    print(f"Global bbox visualization saved to {bbox_path}")
    plt.close()


def plot_statistics(stats, all_widths, all_heights, output_path='crop_analysis.png'):
    """
    Create visualization of the bounding box statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BRATS Dataset Bounding Box Analysis', fontsize=16)
    
    # Width histogram
    axes[0, 0].hist(all_widths, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(stats['width']['mean'], color='red', linestyle='--', 
                       label=f"Mean: {stats['width']['mean']:.1f}")
    axes[0, 0].axvline(stats['width']['median'], color='green', linestyle='--',
                       label=f"Median: {stats['width']['median']:.1f}")
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Bounding Box Width Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Height histogram
    axes[0, 1].hist(all_heights, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].axvline(stats['height']['mean'], color='red', linestyle='--',
                       label=f"Mean: {stats['height']['mean']:.1f}")
    axes[0, 1].axvline(stats['height']['median'], color='green', linestyle='--',
                       label=f"Median: {stats['height']['median']:.1f}")
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Bounding Box Height Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Width vs Height scatter
    axes[1, 0].scatter(all_widths, all_heights, alpha=0.3, s=10)
    axes[1, 0].plot([min(all_widths), max(all_widths)], 
                    [min(all_widths), max(all_widths)], 
                    'r--', label='Square (width=height)')
    axes[1, 0].set_xlabel('Width (pixels)')
    axes[1, 0].set_ylabel('Height (pixels)')
    axes[1, 0].set_title('Width vs Height')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')
    
    # Aspect ratio histogram
    aspect_ratios = np.array(all_widths) / np.array(all_heights)
    axes[1, 1].hist(aspect_ratios, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].axvline(1.0, color='red', linestyle='--', label='Square (ratio=1.0)')
    axes[1, 1].axvline(np.mean(aspect_ratios), color='green', linestyle='--',
                       label=f"Mean: {np.mean(aspect_ratios):.3f}")
    axes[1, 1].set_xlabel('Aspect Ratio (width/height)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Aspect Ratio Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")
    plt.close()


def print_report(stats):
    """
    Print a detailed text report of the statistics.
    """
    print("\n" + "="*70)
    print("BRATS DATASET BOUNDING BOX ANALYSIS REPORT")
    print("="*70)
    print(f"\nTotal slices analyzed: {stats['num_slices']}")
    
    # Original shapes
    unique_shapes = set(tuple(s) for s in stats['original_shapes'])
    print(f"\nOriginal volume shapes found: {unique_shapes}")
    
    print("\n" + "-"*70)
    print("WIDTH STATISTICS (pixels)")
    print("-"*70)
    for key, value in stats['width'].items():
        print(f"{key:15s}: {value:7.2f}")
    
    print("\n" + "-"*70)
    print("HEIGHT STATISTICS (pixels)")
    print("-"*70)
    for key, value in stats['height'].items():
        print(f"{key:15s}: {value:7.2f}")
    
    print("\n" + "="*70)
    print("GLOBAL BOUNDING BOX (union of all slices)")
    print("="*70)
    gb = stats['global_bbox']
    print(f"Top (min_row)   : {gb['min_row']}")
    print(f"Left (min_col)  : {gb['min_col']}")
    print(f"Height          : {gb['height']}")
    print(f"Width           : {gb['width']}")
    print(f"\nCrop coordinates for dataset.py:")
    print(f"  crop_top    = {gb['min_row']}")
    print(f"  crop_left   = {gb['min_col']}")
    print(f"  crop_height = {gb['height']}")
    print(f"  crop_width  = {gb['width']}")
    
    print("\n" + "="*70)
    print("RECOMMENDED CROP DIMENSIONS")
    print("="*70)
    
    # Recommendations based on 95th percentile to capture most content
    rec_width = int(np.ceil(stats['width']['percentile_95']))
    rec_height = int(np.ceil(stats['height']['percentile_95']))
    
    print(f"\nBased on 95th percentile (captures 95% of content):")
    print(f"  Recommended crop: {rec_height} x {rec_width}")
    print(f"  Aspect ratio: {rec_width/rec_height:.3f}")
    
    # Square recommendation
    square_size = max(rec_width, rec_height)
    print(f"\nFor square crop (to avoid distortion):")
    print(f"  Recommended crop: {square_size} x {square_size}")
    print(f"  Note: This may include some background but avoids distortion")
    
    # Tight crop recommendation
    tight_width = int(np.ceil(stats['width']['median']))
    tight_height = int(np.ceil(stats['height']['median']))
    print(f"\nFor tight crop (median, may cut some content):")
    print(f"  Recommended crop: {tight_height} x {tight_width}")
    print(f"  Note: ~50% of slices may be slightly cropped")
    
    print("\n" + "="*70)


# run with:
# sbatch --partition=normal --time=00:30:00 --cpus-per-task=8 --mem=16G --wrap="source venv/bin/activate && python experiments/ddpm_2d/analyze_crop_dimensions.py"
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze BRATS dataset for optimal crop dimensions')
    parser.add_argument('--max_cases', type=int, default=None,
                        help='Maximum number of cases to analyze (default: all)')
    parser.add_argument('--slices_per_case', type=int, default=12,
                        help='Number of slices to sample per case (default: 12, use -1 for all slices)')
    parser.add_argument('--output', type=str, default='crop_analysis.png',
                        help='Output path for visualization (default: crop_analysis.png)')
    parser.add_argument('--slice_axis', type=int, default=2,
                        help='Axis to slice along (default: 2)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto-detect)')
    
    args = parser.parse_args()
    
    if not BRATS_ROOT.exists():
        print(f"Error: BRATS dataset not found at {BRATS_ROOT}")
        return
    
    # Analyze dataset
    stats, all_widths, all_heights, all_slice_data = analyze_dataset(
        BRATS_ROOT,
        slice_axis=args.slice_axis,
        max_cases=args.max_cases,
        slices_per_case=args.slices_per_case,
        num_workers=args.workers
    )
    
    # Print report
    print_report(stats)
    
    # Visualize global bounding box
    output_dir = Path(args.output).parent
    visualize_global_bbox(stats['global_bbox'], all_slice_data, output_dir)
    
    # Detect and visualize outliers
    detect_and_visualize_outliers(all_slice_data, all_widths, all_heights, output_dir)
    
    # Create visualization
    plot_statistics(stats, all_widths, all_heights, args.output)


if __name__ == "__main__":
    # run with:
    # sbatch --partition=normal --time=01:00:00 --cpus-per-task=32 --mem=32G --wrap="source venv/bin/activate && python experiments/ddpm_2d/analyze_crop_dimensions.py --slices_per_case -1"
    main()
