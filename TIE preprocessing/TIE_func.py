import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.morphology import white_tophat, disk
from skimage.filters import gaussian
from matplotlib.patches import Rectangle
from scipy.ndimage import generic_filter, uniform_filter, gaussian_filter1d, uniform_filter1d
import re
from scipy.signal import find_peaks
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as patches
from scipy.ndimage import binary_erosion
from scipy.ndimage import label, generate_binary_structure
from scipy.ndimage import binary_dilation
from scipy.ndimage import distance_transform_edt
import os
from scipy.interpolate import PchipInterpolator



def threshold_where_pixel_count_rises(image, min_pixels=3):
    # Flatten and count occurrences of each unique value
    unique_vals, counts = np.unique(image, return_counts=True)

    # Sort values from high to low
    sort_idx = np.argsort(-unique_vals)
    sorted_vals = unique_vals[sort_idx]
    sorted_counts = counts[sort_idx]


    # Find index where count exceeds threshold
    valid_idx = np.where(sorted_counts >= min_pixels)[0]
    FWQM_idx = np.where(sorted_counts >= 30)[0]
    if valid_idx.size == 0:
        return sorted_vals[-1], sorted_vals[-1]  # fallback to lowest value
    if FWQM_idx.size == 0:
        return sorted_vals[-1], sorted_vals[-1]  # fallback to lowest value

    threshold_val = sorted_vals[valid_idx[0]]
    FWQM = sorted_vals[FWQM_idx[0]]
    return threshold_val , FWQM

def replace_outliers_with_neighbors(image, up_limit):
        image = image.astype(np.float32)

        def replace_if_high(values):
            center = values[4]
            if center > up_limit:
                neighbors = np.delete(values, 4)
                valid_neighbors = neighbors[neighbors <= up_limit]
                if valid_neighbors.size > 0:
                    return np.mean(valid_neighbors)
                else:
                    return up_limit  # fallback: clip to max limit
            else:
                return center

        image_replaced = generic_filter(image, replace_if_high, size=3, mode='mirror')
        return image_replaced


def replace_negatives_with_neighbors(image):
        image = image.astype(np.float32)
        mask = image < 0

        def replace_if_negative(values):
            center = values[4]
            if center < 0:
                # Remove center from the 3x3 window
                neighbors = np.delete(values, 4)
                valid_neighbors = neighbors[neighbors >= 0]
                if valid_neighbors.size > 0:
                    return np.mean(valid_neighbors)
                else:
                    return 0.0  # fallback if all neighbors are invalid
            else:
                return center

        # Apply 3x3 pixelwise replacement
        image_replaced = generic_filter(image, replace_if_negative, size=3, mode='mirror')
        return image_replaced



import matplotlib.pyplot as plt
import numpy as np

def plot_image(original, defocus, bias, voltage, min_plot=10, max_plot=240,radius=10,
                x_pos=0, y_pos=0,
                figsize=(20, 14)):
    height, width = original.shape

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # ===== First row: Original image =====
    img0 = axes[0, 0].imshow(original, cmap='gray', extent=(0, width, height, 0))
    cbar0 = plt.colorbar(img0, ax=axes[0, 0])
    cbar0.set_ticks(np.linspace(np.nanmin(original), np.nanmax(original), 10))
    # axes[0, 0].text(20, 20, f"Defocus: {defocus} mm - E_{bias} : {voltage}V", fontsize=12, ha='left', va='top', color='white')
    axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis('on')

    # Histogram of original
    axes[0, 1].hist(original.ravel(), bins=256, color='gray')
    axes[0, 1].set_title("Histogram", fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(np.linspace(int(original.min()), int(original.max()) + 1, 30))
    axes[0, 1].tick_params(axis='x', labelrotation=45)

    # ===== Second row: Outlier map =====
    print(original.shape)
    # print(f"Min plot value: {min_plot}, Max plot value: {max_plot}")
    # print(f"original min: {np.nanmin(original)}, original max: {np.nanmax(original)}")
    mask_low = (original < min_plot)
    # print(f"Number of low values (<= {min_plot}): {np.sum(mask_low)}")
    mask_high = (original >= max_plot)
    y_coords_low, x_coords_low = np.where(mask_low)
    y_coords_high, x_coords_high = np.where(mask_high)
    num_low = len(y_coords_low)
    num_high = len(y_coords_high)

    img1 = axes[1, 0].imshow(original, cmap='gray', extent=(0, width, height, 0))
    cbar1 = plt.colorbar(img1, ax=axes[1, 0])
    cbar1.set_ticks(np.linspace(np.nanmin(original), np.nanmax(original), 10))
    axes[1, 0].scatter(x_coords_high, y_coords_high, c='red', s=1, label=f'outliers: {num_high}', alpha=0.8)
    axes[1, 0].scatter(x_coords_low, y_coords_low, c='blue', s=1, label=f'negatives: {num_low}', alpha=0.8)
    circle = patches.Circle(
        (width // 2 + x_pos, height // 2 + y_pos),  # center
        radius,  # radius
        linewidth=1.5,
        edgecolor='white',
        facecolor='none',
        linestyle='dashed',
        label = f"border mask (r = {radius:.0f})",
        alpha=0.2
    )
    axes[1, 0].add_patch(circle)
    axes[1, 0].legend(loc='upper right', framealpha=0.5)
    axes[1, 0].set_title("Detected Outliers & Negative Values", fontsize=14, fontweight='bold')
    axes[1, 0].axis('on')

    # Histogram of outliers
    axes[1, 1].hist(original.ravel(), bins=256, color='gray')
    axes[1, 1].axvspan(min_plot, max_plot, color='green', alpha=0.3, label='valid range')
    axes[1, 1].axvspan(max_plot, np.nanmax(original), color='red', alpha=0.3, label='outliers')
    axes[1, 1].axvspan(np.nanmin(original), min_plot, color='blue', alpha=0.3, label='negatives')
    axes[1, 1].set_title("Outlier Detection Histogram", fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(np.linspace(int(original.min()), int(original.max()) + 1, 30))
    axes[1, 1].tick_params(axis='x', labelrotation=45)
    axes[1, 1].legend(loc='upper right', framealpha=0.5)

    fig.tight_layout()
    os.makedirs("TIE", exist_ok=True)  # <-- Make sure folder exists
    save_name = os.path.join("TIE/original-image-analysis.png")
    plt.savefig(save_name, dpi=300)
    return fig

def save_normalized_img(no_outliner_img, normalized_img, remap, object_peak, background_peak, filename):
    save_img = True
    show_img = False
    plot_normalized_image(no_outliner_img, normalized_img, remap, object_peak, background_peak, filename
                          , show_img, save_img, figsize=(14, 10))
    return

def plot_normalized_image(no_outliner_img, normalized_img, remap,object_peak,background_peak,filename, show_img = True, save_img = False, figsize=(14, 10)):
    # find peaks in histogram
    print(normalized_img.shape)
    hist, bin_edges = np.histogram(normalized_img.ravel(), bins=256, range=(0, 255))
    hist_smooth, peak_object, peak_bg = find_histogram_peaks(hist)
    nonzero_mask = hist > 100
    hist = hist[nonzero_mask]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])[nonzero_mask]

    if remap:
        normalized_img = remap_peaks(normalized_img, peak_object, peak_bg,target_obj=object_peak, target_bg=background_peak)
        hist_remap, bin_edges_remap = np.histogram(normalized_img.ravel(), bins=256, range=(0, 255))
        hist_smooth_remap, peak_object_remap,peak_bg_remap = find_histogram_peaks(hist_remap)
        nonzero_mask = hist_remap > 100
        hist_remap = hist_remap[nonzero_mask]
        bin_centers_remap = 0.5 * (bin_edges[:-1] + bin_edges[1:])[nonzero_mask]


    if show_img:
        height, width = no_outliner_img.shape
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # === [0, 0] Outlier-replaced image ===
        img0 = axes[0, 0].imshow(no_outliner_img, cmap='gray', extent=(0, width, height, 0))
        plt.colorbar(img0, ax=axes[0, 0])
        axes[0, 0].set_title("Replaced Outliers with Neighbors", fontsize=14, fontweight='bold')
        axes[0, 0].axis('on')

        # === [0, 1] Histogram of outlier-replaced image ===
        axes[0, 1].hist(no_outliner_img.ravel(), bins=256, color='gray')
        axes[0, 1].set_title("Outlier-Replaced Histogram", fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(np.linspace(int(np.nanmin(no_outliner_img)), int(np.nanmax(no_outliner_img)) + 1, 20))
        axes[0, 1].tick_params(axis='x', labelrotation=45)

        # === [1, 0] Normalized image ===
        img1 = axes[1, 0].imshow(normalized_img, cmap='gray', extent=(0, width, height, 0))
        plt.colorbar(img1, ax=axes[1, 0])
        axes[1, 0].set_title("Normalized Image", fontsize=14, fontweight='bold')
        axes[1, 0].axis('on')


        # === [1, 1] Histogram of normalized image ===
        # hist_smooth , peak_object, peak_background = find_histogram_peaks(normalized_img)
        if remap:
            axes[1, 1].hist(normalized_img.ravel(), bins=256, color='gray')
            axes[1, 1].plot(bin_centers_remap, hist_remap, color='gray')
        else :
            axes[1, 1].hist(normalized_img.ravel(), bins=256, color='gray')
            axes[1, 1].plot(bin_centers, hist, color='gray')

        # if remap:
        #     axes[1, 1].plot(hist_smooth, label='original histogram')
        #     axes[1, 1].plot(hist_smooth_remap, label='remap histogram')
        # else  :
        #     axes[1, 1].plot(hist_smooth, label='original histogram')
        # vertical lines for peaks


        if remap:
            axes[1, 1].axvline(x=peak_object, color='red', linestyle='--', label=f'Peak_object(original): {peak_object:.2f}',alpha=0.5)
            axes[1, 1].axvline(x=peak_bg, color='blue', linestyle='--', label=f'Peak_bg(original): {peak_bg:.2f}', alpha=0.5)
            axes[1, 1].axvline(x=peak_object_remap, color='red', linestyle='--', label=f'Peak_object: {peak_object_remap:.2f}')
            axes[1, 1].axvline(x=peak_bg_remap, color='blue', linestyle='--', label=f'Peak_bg: {peak_bg_remap:.2f}')
        else:
            axes[1, 1].axvline(x=peak_object, color='red', linestyle='--', label=f'Peak_object: {peak_object:.2f}')
            axes[1, 1].axvline(x=peak_bg, color='blue', linestyle='--', label=f'Peak_bg: {peak_bg:.2f}')
        axes[1, 1].set_title("Normalized Histogram", fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(np.linspace(int(np.nanmin(normalized_img)), int(np.nanmax(normalized_img)) + 1, 20))
        axes[1, 1].xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axes[1, 1].tick_params(axis='x', labelrotation=45)
        axes[1, 1].legend(loc='upper right', framealpha=0.5)
        # axes[1, 1].set_ylim(0, 1000)

        fig.tight_layout()
        return fig
    if save_img :
        save_name_01 = os.path.join("TIE/Fresnel imaging/Normalized/" + filename + "-normalized.png")
        save_name_02 = os.path.join("TIE/Fresnel imaging/Normalized/" + filename + "-tifforiginal.png")
        normalized_uint8 = np.clip(normalized_img, 0, 255).astype(np.uint8)
        plt.imsave(save_name_01, normalized_uint8, cmap='gray')



def plot_image_and_histograms(original,
                              no_outliner_img,
                              normalized,
                              defocus,
                              bias,
                              voltage,
                              min_plot,
                              max_plot,
                              radius,
                              x_pos,
                              y_pos,
                              filename,
                              save_path="Fresnel imaging/",
                              figsize=(32, 14)):
    height, width = original.shape
    fig, axes = plt.subplots(2, 4, figsize=figsize)

    # === [0, 0] Original image ===
    img0 = axes[0, 0].imshow(original, cmap='gray', extent=(0, width, height, 0))
    cbar0 = plt.colorbar(img0, ax=axes[0, 0])
    cbar0.set_ticks(np.linspace(np.nanmin(original), np.nanmax(original), 10))
    axes[0, 0].text(20, 20, f"Defocus: {defocus} mm - E_{bias} : {voltage}V", fontsize=12, ha='left', va='top',
                    color='white')
    axes[0, 0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0, 0].axis('on')

    # === [1, 0] Histogram of original ===
    axes[1, 0].hist(original.ravel(), bins=256, color='gray')
    axes[1, 0].set_title("Histogram", fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(np.linspace(int(original.min()), int(original.max()) + 1, 30))
    axes[1, 0].tick_params(axis='x', labelrotation=45)

    # === [0, 1] Detected outliers ===
    mask_low = (original <= min_plot)
    mask_high = (original >= max_plot)
    y_coords_low, x_coords_low = np.where(mask_low)
    y_coords_high, x_coords_high = np.where(mask_high)
    num_low = len(y_coords_low)
    num_high = len(y_coords_high)

    img1 = axes[0, 1].imshow(original, cmap='gray', extent=(0, width, height, 0))
    cbar1 = plt.colorbar(img1, ax=axes[0, 1])
    cbar1.set_ticks(np.linspace(np.nanmin(original), np.nanmax(original), 10))
    axes[0, 1].scatter(x_coords_high, y_coords_high, c='red', s=5, label=f'outliers: {num_high}', alpha=0.8)
    axes[0, 1].scatter(x_coords_low, y_coords_low, c='blue', s=5, label=f'negatives: {num_low}', alpha=0.8)
    circle = patches.Circle(
        (width // 2 + x_pos, height // 2 + y_pos),  # center
        radius,  # radius
        linewidth=1.5,
        edgecolor='white',
        facecolor='none',
        linestyle='dashed',
        label=f"border mask (r = {radius:.0f})"
    )
    axes[0, 1].add_patch(circle)
    axes[0, 1].legend(loc='upper right', framealpha=0.5)
    axes[0, 1].set_title("Detected Outliers & Negative Values", fontsize=14, fontweight='bold')
    axes[0, 1].axis('on')

    # Histogram of outliers
    axes[1, 1].hist(original.ravel(), bins=256, color='gray')
    axes[1, 1].axvspan(min_plot, max_plot, color='green', alpha=0.3, label='valid range')
    axes[1, 1].axvspan(max_plot, np.nanmax(original), color='red', alpha=0.3, label='outliers')
    axes[1, 1].axvspan(np.nanmin(original), min_plot, color='blue', alpha=0.3, label='negatives')
    axes[1, 1].set_title("Outlier Detection Histogram", fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(np.linspace(int(original.min()), int(original.max()) + 1, 30))
    axes[1, 1].tick_params(axis='x', labelrotation=45)
    axes[1, 1].legend(loc='upper right', framealpha=0.5)

    # === [0, 2] Replaced Image ===
    img0 = axes[0, 2].imshow(no_outliner_img, cmap='gray', extent=(0, width, height, 0))
    plt.colorbar(img0, ax=axes[0, 2])
    axes[0, 2].set_title("Replaced Outliers with Neighbors", fontsize=14, fontweight='bold')
    axes[0, 2].axis('on')

    # === [1, 2] Histogram of outlier-replaced image ===
    axes[1, 2].hist(no_outliner_img.ravel(), bins=256, color='gray')
    axes[1, 2].set_title("Outlier-Replaced Histogram", fontsize=14, fontweight='bold')
    axes[1, 2].set_xticks(np.linspace(int(np.nanmin(no_outliner_img)), int(np.nanmax(no_outliner_img)) + 1, 20))
    axes[1, 2].tick_params(axis='x', labelrotation=45)

    # === [0, 3] Normalized Image ===
    img1 = axes[0, 3].imshow(normalized, cmap='gray', extent=(0, width, height, 0))
    plt.colorbar(img1, ax=axes[0, 3])
    axes[0, 3].set_title("Normalized Image", fontsize=14, fontweight='bold')
    axes[0, 3].axis('on')

    # === [1, 1] Histogram of normalized image ===
    axes[1, 3].hist(normalized.ravel(), bins=256, color='gray')
    axes[1, 3].set_title("Normalized Histogram", fontsize=14, fontweight='bold')
    axes[1, 3].set_xticks(np.linspace(int(np.nanmin(normalized)), int(np.nanmax(normalized)) + 1, 20))
    axes[1, 3].xaxis.set_major_formatter(FormatStrFormatter('%d'))
    axes[1, 3].tick_params(axis='x', labelrotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join("TIE/Fresnel imaging/" + filename + "-analysis.png"), dpi=300)
    plt.close(fig)
    save_name = os.path.join("TIE/Fresnel imaging/" + filename + "-normalized.png")
    normalized_uint8 = np.clip(normalized, 0, 255).astype(np.uint8)
    plt.imsave(save_name, normalized_uint8, cmap='gray')







# def plot_image_and_histograms(original,
#                               image_data_no_outliner,
#                               normalized,
#                               min_plot,
#                               max_plot,
#                               titles=("Original", "Normalized")) :
#     defocus = 0
#     bias = 0
#
#     fig, axes = plt.subplots(2, 4, figsize=(32, 14))
#     height, width = original.shape
#     mask_low = (original <= min_plot)
#     mask_high = (original >= max_plot)
#     y_coords_low , x_coords_low = np.where(mask_low)
#     y_coords_high, x_coords_high = np.where(mask_high)
#     num_low = len(y_coords_low)
#     num_high = len(y_coords_high)
#
#     # Original image
#     img = axes[0, 0].imshow(original, cmap='gray', extent=(0, width, height, 0))
#     cbar = plt.colorbar(img, ax=axes[0, 0])
#     cbar.set_ticks(np.linspace(np.nanmin(original), np.nanmax(original), 10))
#     axes[0, 0].text(100, 100, f"Defocus: {defocus} mm - Bias: {bias} ", fontsize=16, ha='left', va='center', color='white')
#     # title bold
#     axes[0, 0].set_title("Original Image", fontsize=16, fontweight='bold')
#     axes[0, 0].axis('on')
#
#     # Histogram of original
#     axes[1, 0].hist(original.ravel(), bins=256, color='gray')
#     axes[1, 0].set_title(f"Original Image - Histogram", fontsize=16, fontweight='bold')
#     axes[1, 0].set_xticks(np.arange(int(original.min()), int(original.max()) + 1, 20))
#     axes[1, 0].tick_params(axis='x', labelrotation=45)
#
#
#
#     # plot detected outliers
#     img = axes[0, 1].imshow(original, cmap='gray', extent=(0, width, height, 0))
#     cbar = plt.colorbar(img, ax=axes[0, 1])
#     cbar.set_ticks(np.linspace(np.nanmin(original), np.nanmax(original), 10))
#     axes[0, 1].scatter(x_coords_high, y_coords_high, c='red', s=5, label=f'outliers : {num_high}', alpha=0.8)
#     axes[0, 1].scatter(x_coords_low, y_coords_low, c='blue', s=5, label=f'negative values : {num_low}', alpha=0.8)
#     axes[0, 1].legend(loc='upper right', framealpha=0.5)
#     axes[0, 1].set_title("Detected Outliers & Negative values", fontsize=16, fontweight='bold')
#     axes[0, 1].axis('on')
#     # Histogram of detected outliers
#     axes[1, 1].hist(original.ravel(), bins=256, color='gray')
#     axes[1, 1].axvspan(min_plot, max_plot, color='green', alpha=0.3, label='valid values')
#     axes[1, 1].axvspan(max_plot, np.nanmax(original), color='red', alpha=0.3, label='outliers')
#     axes[1, 1].axvspan(np.nanmin(original), min_plot, color='blue', alpha=0.3, label='negative values')
#     axes[1, 1].set_title("Detected Outliers & Negative values - Histogram", fontsize=16, fontweight='bold')
#     axes[1, 1].set_xticks(np.arange(int(original.min()), int(original.max()) + 1, 20))
#     axes[1, 1].tick_params(axis='x', labelrotation=45)
#     axes[1, 1].legend(loc='upper right', framealpha=0.5)
#     # axes[1, 1].set_ylim(0, 20)
#
#     # plot no otuliner image
#     img = axes[0, 2].imshow(image_data_no_outliner, cmap='gray', extent=(0, width, height, 0))
#     plt.colorbar(img, ax=axes[0, 2])
#     axes[0, 2].set_title("Replaced Outliers with Neighbors", fontsize=16, fontweight='bold')
#     axes[0, 2].axis('on')
#     # Histogram of no outliner
#     axes[1, 2].hist(image_data_no_outliner.ravel(), bins=256, color='gray')
#     axes[1, 2].set_title("Replaced Outliers with Neighbors - Histogram", fontsize=16, fontweight='bold')
#     axes[1, 2].set_xticks(
#         np.arange(int(np.nanmin(image_data_no_outliner)), int(np.nanmax(image_data_no_outliner)) + 1, 20))
#     axes[1, 2].tick_params(axis='x', labelrotation=45)
#
#
#
#     # Normalized image
#     img_norm = axes[0, 3].imshow(normalized*255, cmap='gray')
#     plt.colorbar(img_norm, ax=axes[0, 3])
#     axes[0, 3].set_title("Normalized Image (0-255)", fontsize=16, fontweight='bold')
#     axes[0, 3].axis('off')
#     # Histogram of normalized
#     axes[1, 3].hist((normalized * 255).ravel(), bins=256, color='gray')
#     axes[1, 3].set_title("Normalized Image - Histogram", fontsize=16, fontweight='bold')
#     axes[1, 3].tick_params(axis='x', labelrotation=45)
#
#     plt.tight_layout()
#     # save figure as png
#     plt.savefig("Fresnel imaging/" + filename + "-analysis.png", dpi=300)
#     # plt.show()

def z_score_normalization(img):
    mean, std = np.mean(img), np.std(img)
    return (img - mean) / std

def adaptive_histogram_equalization(img, clip_limit=10.0, tile_grid_size=(8, 8)):
    img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_8bit)

def subtract_background(img, sigma=10):
    from scipy.ndimage import gaussian_filter
    background = gaussian_filter(img, sigma=sigma)
    return img - background
def em_flat_field_normalization(img, sigma=200):
    from scipy.ndimage import gaussian_filter
    background = gaussian_filter(img, sigma=sigma)
    norm = (img - background) / (background + 1e-8)
    return norm

def em_tophat_normalization(img, radius=50):
    selem = disk(radius)
    return white_tophat(img, selem)

def em_contrast_stretch(img, lower_percentile=2, upper_percentile=98):
    p_low, p_high = np.percentile(img, (lower_percentile, upper_percentile))
    stretched = np.clip((img - p_low) / (p_high - p_low), 0, 1)
    return stretched

def em_contrast_stretch_masked(image, valid_mask=None):
    image = image.astype(np.float32)

    # Safe valid mask: exclude NaNs
    if valid_mask is None:
        valid_mask = (~np.isnan(image)) & (image >= 0)
    else:
        valid_mask = valid_mask & (~np.isnan(image))

    signal_pixels = image[valid_mask]

    # Handle empty mask case safely
    if signal_pixels.size == 0:
        print("Warning: No valid pixels found. Returning original image.")
        return image.copy()

    # Compute percentiles safely
    vmin, vmax = np.nanpercentile(signal_pixels, (0.01, 99.99))

    # Avoid divide-by-zero
    if np.isclose(vmax, vmin):
        print("Warning: vmin ≈ vmax. Returning zeros.")
        return np.zeros_like(image)

    # Apply contrast stretch
    stretched = (image - vmin) / (vmax - vmin)
    stretched = np.clip(stretched, 0, 1)
    stretched[np.isnan(image)] = np.nan  # preserve existing NaNs

    return stretched

def local_mean_filter(img, mask):
    mask_flat = mask.flatten()

    def replace_if_masked(values):
        center = values[len(values) // 2]
        if mask_flat[len(values) // 2]:
            neighbors = np.delete(values, len(values) // 2)
            return np.mean(neighbors)
        else:
            return center

    return generic_filter(img, replace_if_masked, size=3, mode='mirror')

import numpy as np

def mask_edges_where_negative(image):
    h, w = image.shape

    # Top → bottom
    top = 0
    for i in range(h):
        if np.any(image[i, :] < 0):
            top = i + 1
        else:
            break

    # Bottom → top
    bottom = h
    for i in range(h - 1, -1, -1):
        if np.any(image[i, :] < 0):
            bottom = i
        else:
            break

    # Left → right
    left = 0
    for j in range(w):
        if np.any(image[:, j] < 0):
            left = j + 1
        else:
            break

    # Right → left
    right = w
    for j in range(w - 1, -1, -1):
        if np.any(image[:, j] < 0):
            right = j
        else:
            break

    # Create mask for valid region
    mask = np.zeros_like(image, dtype=bool)
    mask[top:bottom, left:right] = True

    # Masked image: NaN outside valid region
    image_masked = image.astype(np.float32).copy()
    image_masked[~mask] = np.nan

    print("top, bottom, left, right", top, bottom, left, right)

    return mask

def circular_mask_excluding_outer_negatives(image, step=1):
    h, w = image.shape
    center = (h // 2, w // 2)
    max_r = min(center)

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((Y - center[0])**2 + (X - center[1])**2)

    image = image.astype(np.float32)
    valid_mask = image >= 0

    for r in range(max_r, 0, -step):
        # Define a narrow ring (edge of circle)
        edge_band = (dist_from_center >= r - 0.5) & (dist_from_center <= r + 0.5)

        # Check if the edge band contains only valid pixels
        if np.all(valid_mask[edge_band]):
            circle_mask = dist_from_center <= r
            image_masked = image.copy()
            image_masked[~circle_mask] = np.nan
            return image_masked, circle_mask, r

    # Fallback: only center pixel valid
    mask = np.zeros_like(image, dtype=bool)
    mask[center] = True
    image[~mask] = np.nan
    return image, mask, 1


def mask_negatives_outside_circle(image, radius,
                                  x_pos=0,
                                  y_pos=0,
                                  mask_circular=True,
                                  mask_smart=False):
    image = image.copy()
    h, w = image.shape
    # center = (h // 2, w // 2)
    center = (h // 2 + y_pos, w // 2 + x_pos)

    # 1. Build distance map from center
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - center[0])**2 + (X - center[1])**2)

    # 2. Create circular region mask
    inside_circle = dist <= radius

    # 3. Mask only the negative values outside the circle
    if mask_circular:
        mask = (~inside_circle)
    elif mask_smart:
        mask = (~inside_circle) & (image < 0)
    else:
        mask = (~inside_circle) & (image < 0)
    image[mask] = np.nan

    # plt.imshow(image, cmap='gray')
    # circle = patches.Circle(
    #     center,
    #     radius,
    #     color='red',
    #     fill=False,
    #     linewidth=2,
    #     linestyle='dashed'
    # )
    # plt.gca().add_patch(circle)
    # plt.title("Masked Image")
    # plt.colorbar()
    # plt.show()

    return image, mask, radius





def extract_defocus_bias_voltage(filename):
    # defocus: from def5mm
    defocus_match = re.search(r'def(-?\d+(?:\.\d+)?)mm', filename)
    defocus = int(defocus_match.group(1)) if defocus_match else None

    # bias: anything between "bias-" and the next underscore
    bias_match = re.search(r'bias-([^_]+)', filename)
    bias = bias_match.group(1) if bias_match else None

    # voltage: match signed or unsigned number before "v" or "V"
    voltage_match = re.search(r'_(-?\d+)[vV]', filename)
    voltage = int(voltage_match.group(1)) if voltage_match else None

    return defocus, bias, voltage

# def find_histogram_peaks(hist, min_distance=50):
#     # 1. Replace zero bins with small value or remove them
#     print("lenght of histogram:", len(hist))
#     nonzero_indices = np.where(hist > min_distance)[0]
#     hist_clean = hist[nonzero_indices]
#     print("lenght of histogram after :", len(hist_clean))
#
#     # Apply Gaussian smoothing
#     # hist_clean = uniform_filter1d(hist_clean, size=10)
#     hist_clean = gaussian_filter1d(hist_clean, sigma=5)
#
#     # fig = plt.plot(hist_clean)
#     # # save fig png
#     # plt.savefig( "TIE/Fresnel imaging/histogram.png", dpi=300)
#
#
#     # 2. Find peaks in the cleaned histogram
#     peaks_raw, _ = find_peaks(hist_clean)
#
#     # 3. Map back to original bin indices
#     peaks = nonzero_indices[peaks_raw]
#
#     # 4. Sort peaks by height
#     if len(peaks) < 2:
#         raise ValueError("Less than 2 peaks found")
#
#     top2 = sorted(peaks, key=lambda i: hist[i], reverse=True)[:2]
#     return hist_clean , tuple(sorted(top2))  # return as (object, background)

def find_histogram_peaks(hist, min_count=50, sigma=10, prominence=100):
    # print("Original histogram length:", len(hist))

    # Step 1: Copy histogram
    hist_copy = hist.copy()

    # Step 2: Mask low-count bins so they don't influence smoothing
    mask = hist_copy > min_count
    hist_masked = np.where(mask, hist_copy, 0)

    # Step 3: Apply Gaussian smoothing to masked histogram
    hist_smooth = gaussian_filter1d(hist_masked, sigma=sigma)

    # Step 4: Find peaks in smoothed version
    peaks, _ = find_peaks(hist_smooth, prominence=prominence)

    if len(peaks) < 2:
        raise ValueError("Less than two peaks found")

    # Step 5: Return the two most prominent peaks
    top2 = sorted(sorted(peaks, key=lambda i: hist[i], reverse=True)[:2])
    # print(top2[0])
    # print(top2[1])

    return hist_smooth, top2[0], top2[1]

# def remap_peaks(image, object_peak, background_peak, target_obj=40, target_bg=150):
#     print("object_peak, background_peak", object_peak, background_peak)
#     print("target_obj, target_bg", target_obj, target_bg)
#
#     # Linear mapping coefficients
#     A = np.array([[object_peak, 1], [background_peak, 1]])
#     B = np.array([target_obj, target_bg])
#     a, b = np.linalg.lstsq(A, B, rcond=None)[0]
#     print(f"Mapping: y = {a:.4f} * x + {b:.2f}")
#
#     # Create a safe copy
#     remapped = np.full_like(image, np.nan, dtype=np.float32)
#
#     # Only apply to finite (non-NaN) pixels
#     mask = np.isfinite(image)
#     remapped[mask] = a * image[mask] + b
#
#     # Clip valid range
#     remapped[mask] = np.clip(remapped[mask], 0, 255)
#
#     return remapped, (a, b)

# def remap_peaks(image, object_peak, background_peak, target_obj=40, target_bg=150, use_sigmoid_tail=True):
#     """
#     Remaps image intensities so that `object_peak` maps to `target_obj`
#     and `background_peak` maps to `target_bg`, with optional sigmoid smoothing
#     to suppress the tail at high intensity values.
#
#     Parameters:
#         image: np.ndarray
#         object_peak: float – the value in the histogram for the object
#         background_peak: float – the value in the histogram for the background
#         target_obj: float – new intensity value for the object
#         target_bg: float – new intensity value for the background
#         use_sigmoid_tail: bool – whether to apply sigmoid compression to suppress tail
#
#     Returns:
#         remapped (np.ndarray): final normalized image
#         (a, b): linear transformation parameters
#     """
#     assert np.isfinite(object_peak) and np.isfinite(background_peak), "Peaks must be finite"
#
#     # === Step 1: Solve linear system y = a*x + b
#     A = np.array([[object_peak, 1], [background_peak, 1]])
#     B = np.array([target_obj, target_bg])
#     a, b = np.linalg.lstsq(A, B, rcond=None)[0]
#     print(f"Linear Mapping: y = {a:.4f} * x + {b:.2f}")
#
#     # === Step 2: Remap only finite pixels
#     mask = np.isfinite(image)
#     remapped = np.full_like(image, np.nan, dtype=np.float32)
#     remapped[mask] = a * image[mask] + b
#
#     # === Step 3: Clip to 0–255
#     remapped[mask] = np.clip(remapped[mask], 0, 255)
#
#     # === Step 4: Optional sigmoid tail compression
#     if use_sigmoid_tail:
#         # Only compress finite region
#         r = remapped[mask]
#         midpoint = np.median([target_obj, target_bg])
#         steepness = 0.03  # Lower = smoother transition
#         compressed = 255 / (1 + np.exp(-steepness * (r - midpoint)))
#         remapped[mask] = compressed
#
#     # === Step 5: Return as uint8 for display
#     remapped_uint8 = np.zeros_like(image, dtype=np.uint8)
#
#     # Fill masked pixels only, leave rest as 255 (or any clean background value)
#     background_value = 255  # or 128, or any neutral background
#     remapped_uint8[:] = background_value
#     remapped_uint8[mask] = np.nan_to_num(remapped[mask], nan=background_value).astype(np.uint8)
#
#     return remapped_uint8, (a, b)


def remap_peaks(image,
                object_peak,
                background_peak,
                target_obj=40,
                target_bg=150):
    """
    Shift the two dominant histogram peaks of a grayscale image while
    preserving border pixels that are NaN.

    Parameters
    ----------
    image : ndarray
        2-D or 3-D grayscale image.  May be float with NaNs or uint8.
    object_peak : float or int
        Intensity of the darker (object) peak in the *source* histogram.
    background_peak : float or int
        Intensity of the brighter (background) peak in the *source* histogram.
    target_obj : int, default 40
        Desired location of the object peak in the *output* histogram.
    target_bg : int, default 150
        Desired location of the background peak in the *output* histogram.

    Returns
    -------
    ndarray
        Image with remapped intensities.
        • If the input had NaNs, the result is float32 with NaNs preserved.
        • Otherwise the result is uint8 (0…255).
    """

    # Ensure consistent ordering of “dark” vs “bright”
    if object_peak > background_peak:
        object_peak, background_peak = background_peak, object_peak
        target_obj,  target_bg       = target_bg,       target_obj

    src_knots = [0, object_peak, background_peak, 255]
    dst_knots = [0, target_obj,  target_bg,       255]

    # Work in float so NaNs are representable
    img_f   = image.astype(np.float32, copy=False)
    finite  = np.isfinite(img_f)        # True where pixel is *not* NaN

    if PchipInterpolator is not None:
        f = PchipInterpolator(src_knots, dst_knots)
        mapped = np.empty_like(img_f)
        mapped[:]      = np.nan         # initialise with NaN
        mapped[finite] = f(img_f[finite])
    else:
        # affine fallback (still anchors both peaks exactly)
        s = (target_bg - target_obj) / (background_peak - object_peak)
        b = target_obj - s * object_peak
        mapped = img_f * s + b          # NaNs stay NaN automatically

    # Clip finite values into displayable range
    mapped[finite] = np.clip(mapped[finite], 0, 255)

    # Decide return dtype: keep float if we need NaNs, else uint8
    if finite.all():                    # no NaNs at all → safe to cast
        return mapped.astype(np.uint8)
    else:                               # keep float32 so NaNs survive
        return mapped