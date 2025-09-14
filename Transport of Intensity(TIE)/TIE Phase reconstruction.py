import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq
from scipy.ndimage import gaussian_filter
from PIL import Image
import os

def load_image(path):
    # import the image as float32 grayscale
    img = Image.open(path).convert("F")
    return np.array(img, dtype=np.float32)

def poisson_solver(rhs, pixel_size):
    # Get the dimensions of the input array (source term)
    ny, nx = rhs.shape

    # Generate frequency grids for x and y directions using FFT frequency convention
    # 'd=pixel_size' sets the spatial sampling interval
    kx = fftfreq(nx, d=pixel_size)
    ky = fftfreq(ny, d=pixel_size)

    # Create 2D meshgrids of frequency components
    Kx, Ky = np.meshgrid(kx, ky)

    # Compute squared magnitude of the frequency vector: K^2 = kx^2 + ky^2
    Ksq = Kx**2 + Ky**2

    # Compute the Fourier transform of the source term (rhs)
    rhs_ft = fft2(rhs)

    # Solve the Poisson equation in the frequency domain:
    #   -K^2 * phi_ft = rhs_ft  -->  phi_ft = -rhs_ft / K^2
    # Handle divide-by-zero at K = 0 (DC component) carefully
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_laplace = np.zeros_like(Ksq)         # Initialize inverse Laplacian operator
        inv_laplace[Ksq > 0] = 1.0 / Ksq[Ksq > 0] # Avoid division by zero
        inv_laplace[Ksq == 0] = 0.0               # Set DC component to zero

    # Multiply by -rhs_ft to get phi in Fourier space
    phi_ft = -rhs_ft * inv_laplace

    # Inverse FFT to return the solution in real space (take only the real part)
    return np.real(ifft2(phi_ft))

def tie_accurate_phase_reconstruction(I_minus, I_0, I_plus, pixel_size, wavelength, defocus_distance):
    """
    TIE Accurate Phase Reconstruction:
        φ(r) = -k ∇⁻² { ∇ ⋅ [ (1 / I(r)) ∇ ∇⁻² ( ∂I(r) / ∂z ) ] }
    Where:
        φ(r)      → Reconstructed phase
        k         → Wavenumber (2π / wavelength)
        I(r)      → In-focus intensity
        ∂I/∂z     → Intensity derivative along z
        ∇⁻²       → Inverse Laplacian (solved via FFT Poisson solver)
    """
    # wavenumber
    k = 2.0 * np.pi / wavelength

    #avoid division by zero
    eps = 1e-15

    # Smooth the images by Gaussian filter
    sigma = 10.0
    I_minus = gaussian_filter(I_minus, sigma=sigma)
    I_0 = gaussian_filter(I_0, sigma=sigma)
    I_plus = gaussian_filter(I_plus, sigma=sigma)
    diff = I_plus - I_minus

    # Step 1: ∂I/∂z
    dIdz = (I_plus - I_minus) / (2.0 * defocus_distance)
    dIdz *= -1.0

    # Step 2: Poisson inverse of ∂I/∂z
    phi1 = poisson_solver(dIdz, pixel_size)

    # Step 3: Gradient of phi1
    grad_phi1_y, grad_phi1_x = np.gradient(phi1, pixel_size)

    # Step 4: Multiply by 1/I₀
    I_0_safe = np.clip(I_0, eps, None)
    vx = grad_phi1_x / I_0_safe
    vy = grad_phi1_y / I_0_safe

    # Step 5: Divergence
    div_v = np.gradient(vx, pixel_size, axis=1) + np.gradient(vy, pixel_size, axis=0)

    # Step 6: Final Poisson solve with full prefactor
    phase = -k * poisson_solver(div_v, pixel_size)

    return phase, dIdz

def tie_phase_reconstruction(I1, I0, I2, pixel_size, wavelength, defocus_step, tikhonov_qc=0.5):
    """
    TIE Phase Reconstruction:
        ∇² φ(r) = - (2π / λ) * (1 / I(r)) * (∂I(r) / ∂z)
    Where:
        φ(r)      → Reconstructed phase [radians]
        I(r)      → In-focus intensity
        ∂I/∂z     → Intensity derivative along optical axis (z)
        λ         → Wavelength of the beam
        ∇²        → 2D Laplacian operator

    The equation is solved in Fourier space using an inverse Laplacian.
    Optional Tikhonov regularization stabilizes the inversion.
    """

    dIdz = (I2 - I1) / (2 * defocus_step)

    eps = 1e-12
    rhs = -(2 * np.pi / wavelength) * dIdz / np.clip(I0, eps, None)
    # rhs *= soft_mask

    ny, nx = I0.shape
    fx = fftfreq(nx, d=pixel_size)
    fy = fftfreq(ny, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)
    K2 = FX**2 + FY**2

    if tikhonov_qc is not None:
        qc_squared = (tikhonov_qc * pixel_size * 1e9) ** 2
    else:
        qc_squared = 0.0

    inv_laplace = np.zeros_like(K2)
    inv_laplace[K2 > 0] = 1.0 / (K2[K2 > 0] + qc_squared)

    rhs_ft = fft2(rhs)
    phase_ft = -rhs_ft * inv_laplace
    phase = np.real(ifft2(phase_ft))

    return phase, dIdz

def plot(img_a, img_b, img_c, img_d, img_e, img_f,
             titles,
             cmap="gray",
             figsize=(15, 10)):

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # -------- helper to draw an image with nice colour-bar ------------
    def show(ax, data, title):
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=16)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_box_aspect(1)                    # <- square box
        # slim colour-bar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return im

    show(axes[0, 0], img_a, titles[0])
    show(axes[0, 1], img_b, titles[1])
    show(axes[0, 2], img_c, titles[2])
    show(axes[1, 0], img_d, titles[3])
    show(axes[1, 1], img_e, titles[4])

    # ------------- line-profile panel ---------------------------------
    line = img_f[img_f.shape[0] // 2, 300:-300]
    line_vertical = img_f[300:-300, (img_f.shape[0] // 2)+50]

    # save line profile to a text file
    np.savetxt("line_profile.txt", line, fmt='%.6f')
    x = np.arange(len(line))  # <- Add this line

    # Fit only the central part (exclude flat edges)
    start = 100
    line_fit = line[start: len(line) - start]
    x_fit = np.arange(start, len(line) - start)
    x_centered = x_fit - np.mean(x_fit)

    # Normalize x to [-1, 1]
    x_norm = x_centered / np.max(np.abs(x_centered))

    # Generate downward-opening quartic curve: -x^4
    y_quartic = -x_norm ** 4

    # Normalize and scale to match *line_fit* range
    y_scaled = y_quartic - np.min(y_quartic)
    y_scaled /= np.max(y_scaled)
    y_scaled *= (np.max(line_fit) - np.min(line_fit))
    y_scaled += np.min(line_fit)

    # Optional: create full-length array for plotting if needed
    quartic_full = np.full_like(line, np.nan)
    quartic_full[start: len(line) - start] = y_scaled


    # Plot the line profile with twin y-axes for vertical and horizontal
    ax_prof = axes[1, 2]
    # main axis (left, red)
    ax_prof.plot(x, line, color='tab:red')
    ax_prof.plot(x, quartic_full, label="Quartic Fit", color='red', linestyle='--')
    ax_prof.set_xlabel("x-pixel")
    ax_prof.set_ylabel("Vertical Phase (rad)", color='tab:red')
    ax_prof.tick_params(axis='y', labelcolor='tab:red')
    ax_prof.set_xlim(x[0], x[-1])
    ax_prof.set_title(titles[5], fontsize=16)
    ax_prof.grid(alpha=0.3)
    ax_prof.set_box_aspect(1)

    # hide the right spine on the main axis
    ax_prof.spines['right'].set_visible(False)
    ax_prof.spines['top'].set_visible(False)

    # twin axis (right, blue)
    ax2 = ax_prof.twinx()
    ax2.plot(x, line_vertical, color='tab:blue')
    ax2.set_ylabel("Horizontal Phase (rad)", color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # make twin axis clean: no left/top/bottom spines, transparent face
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.patch.set_alpha(0)  # transparent background
    ax2.grid(False)  # avoid a second grid

    plt.tight_layout()
    return fig, axes


def plot_3d_surface_central_circle(data, radius=1000, cmap='viridis'):
    """
    Plots a 3D surface for the central circular region of a 2D matrix.

    Parameters:
        data (2D array): Matrix of Z values.
        radius (int): Radius in pixels for the circular region.
        cmap (str): Colormap name.
    """
    data = np.array(data)
    # gaussian filter to smooth the data
    data = gaussian_filter(data, sigma=50)
    ny, nx = data.shape
    cy, cx = ny // 2, nx // 2  # center coordinates

    # Coordinate grids
    y, x = np.ogrid[:ny, :nx]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2

    # Apply mask: set outside region to NaN
    Z = np.where(mask, data, np.nan)

    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', linewidth=0, antialiased=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=0)
    fig.colorbar(surf, shrink=0.5, aspect=8)
    plt.show()

# === USAGE ===
if __name__ == "__main__":

    # Adjust the folder path to your images
    folder = ""

    # Load images directly using file paths
    path_under = os.path.join(folder, "over-pos.png")
    path_focus = os.path.join(folder, "focus.png")
    path_over = os.path.join(folder, "over-neg.png")

    I1 = load_image(path_under)
    I0 = load_image(path_focus)
    I2 = load_image(path_over)

    # Optional : crop the central part
    crop_size = 3000
    center = I0.shape[0] // 2
    I1 = I1[center - crop_size // 2:center + crop_size // 2, center - crop_size // 2:center + crop_size // 2]
    I0 = I0[center - crop_size // 2:center + crop_size // 2, center - crop_size // 2:center + crop_size // 2]
    I2 = I2[center - crop_size // 2:center + crop_size // 2, center - crop_size // 2:center + crop_size // 2]

    # Parameters
    pixel_size = 80e-6 / crop_size     # approximately 80 µm of the FOV divided by number of pixels
    wavelength = 2.5e-12           # 2.5 pm @ 200 kV based on the experimental setup - 1.96 pm for 300 kV
    defocus_distance = 400e-6       # 400 µm defocus - adjust based on your experimental setup

    # TIE phase reconstruction ( two options available )
    # phase, dIdz = tie_phase_reconstruction(I1, I0, I2, pixel_size=pixel_size, wavelength=wavelength, defocus_step=defocus_distance)
    phase, dIdz = tie_accurate_phase_reconstruction(I1, I0, I2, pixel_size=pixel_size, wavelength=wavelength, defocus_distance = defocus_distance)


    # Normalize the phase to 0-100 range (Optional)
    # TIE only recovers relative phase variations—it cannot determine an absolute constant phase offset
    phase = (phase - np.min(phase)) / (np.max(phase) - np.min(phase)) * 100

    # save the phase as txt file (Optional)
    np.savetxt("phase_reconstruction.txt", phase, fmt='%.6f')

    # plot the results
    fig, axes = plot( I1, I0, I2, dIdz, phase,phase,
                                    titles=["overfocus-positive", "focused", "overfocused-negative", r"$\dfrac{dI}{dz} = \dfrac{I_{+} - I_{-}}{2\Delta z}$", "Phase(Normalized 0-100)", "Phase(cut line)"],
                                    )
    plt.show()

    # plot_3d_surface_central_circle(phase, radius=1100)
