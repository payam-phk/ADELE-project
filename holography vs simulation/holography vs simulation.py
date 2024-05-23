import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, medfilt2d, wiener, medfilt
from scipy.optimize import least_squares
from matplotlib.ticker import FormatStrFormatter

def polynomial_surface(coefficients, x, y, degree):
    terms = np.zeros_like(x, dtype=float)
    idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms += coefficients[idx] * (x ** i) * (y ** j)
            idx += 1
    return terms

# Define the residual function for least squares optimization
def residuals(coefficients, x, y, data, degree):
    return data - polynomial_surface(coefficients, x, y, degree)

def fit_polynomial(data,degree):
    data_flat = data.flatten()
    X_flat, Y_flat = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    X_flat = X_flat.flatten()
    Y_flat = Y_flat.flatten()

    # Initial guess for coefficients
    initial_guess = np.ones((degree + 1) * (degree + 2) // 2)

    # Perform least squares fitting
    result = least_squares(residuals, initial_guess, args=(X_flat, Y_flat, data_flat, degree))

    # Reshape the optimized coefficients back to the 2D polynomial
    optimal_coefficients = result.x
    Z_fit = polynomial_surface(optimal_coefficients, X_flat, Y_flat, degree).reshape(data.shape)

    return Z_fit

def detect_noisy_pixels(data, sensitivity):
    noisy_pixels = np.abs(data - np.nanmean(data)) > sensitivity * np.std(data)
    noisy_pixels_indices = np.argwhere(noisy_pixels)
    return noisy_pixels , noisy_pixels_indices

def detect_nan_pixels(data):
    nan_pixels = np.isnan(data)
    nan_pixels_indices = np.argwhere(nan_pixels)
    return nan_pixels, nan_pixels_indices

def replace_noisy_pixels(data, noisy_pixels):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if noisy_pixels[i, j]:
                # check if they are located at the edge of the image
                if i == 0 or i == data.shape[0] - 1 or j == 0 or j == data.shape[1] - 1:
                    # Use nanmean for edges and corners
                    data[i, j] = np.nanmean(data[max(0, i - 1):min(data.shape[0], i + 2),
                                                 max(0, j - 1):min(data.shape[1], j + 2)])
                else:
                    data[i, j] = np.nanmean(data[i - 1:i + 2, j - 1:j + 2])
    return data

def replace_noisy_pixels_with_average(data, noisy_pixels):
    avg = np.nanmean(data)
    print("average: ",avg)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if noisy_pixels[i, j]:
                data[i, j] = avg
    return data

def replace_nan_pixels(data, nan_pixels):
    for i in range(0, data.shape[0] - 1):
        for j in range(0, data.shape[1] - 1):
            if nan_pixels[i, j]:
                # check if they are located at the edge of the image
                if nan_pixels[i, j] and i == 0:
                    # average of the 3x3 neighborhood
                    data[i, j] = np.nanmean(data[i:i + 2, j - 1:j + 2])
                elif nan_pixels[i, j] and i == data.shape[0] - 1:
                    # average of the 3x3 neighborhood
                    data[i, j] = np.nanmean(data[i - 1:i + 1, j - 1:j + 2])
                elif nan_pixels[i, j] and j == 0:
                    # average of the 3x3 neighborhood
                    data[i, j] = np.nanmean(data[i - 1:i + 2, j:j + 2])
                elif nan_pixels[i, j] and j == data.shape[1] - 1:
                    # average of the 3x3 neighborhood
                    data[i, j] = np.nanmean(data[i - 1:i + 2, j - 1:j + 1])
                else:
                    data[i, j] = np.nanmean(data[i - 1:i + 1, j - 1:j + 1])
    return data

# jump points in x and y axis
jumps_x = [0, 255, 511, 767, 1023, 1279, 1535, 1791, 2047, 2303]
jumps_y = [0, 235, 471, 707, 943, 1179, 1415, 1651, 1887]  # jump points in y

CE = 6526019.67538 # @300KV - lambda = 1.9687 pm

#open text file using np
E_first_quadrant_offset = np.loadtxt('holography data.txt')
central_phase_simulation = np.loadtxt('integ3.txt')
whole_phase_simulation = np.loadtxt('integ.txt')
print("Experiment data shape:  " + str(E_first_quadrant_offset.shape))
print("Simulation whole device data shape:  " + str(central_phase_simulation.shape))
print("Simulation data shape:  " + str(whole_phase_simulation.shape))

original_data = E_first_quadrant_offset.copy()
central = central_phase_simulation * CE
whole = whole_phase_simulation * CE
central = central * -1
whole = whole * -1
central = central - np.min(central)
whole = whole - np.min(whole)
central = central[0:999, 0:1000]
print(central.shape)

simulation_pixelized = np.zeros((central.shape[0],central.shape[1]))
big_pixel_size_x = 125
big_pixel_size_y = 111
for i in range(0, central.shape[0], big_pixel_size_y): # y axis
    for j in range(0, central.shape[1], big_pixel_size_x): # x axis
        n = int(i/111)
        m = int(j/125)
        # if (n == 4 and m >= 6) or (n == 5 and m >= 6) or (n == 6 and m >= 5) or (n == 7 and m >= 1):
        #     continue
        if (n == 1 and m >= 7) or (n == 2 and m >= 7) or (n == 3 and m >= 7) or (n == 4 and m >= 7) or (n == 5 and m >= 6) or (n == 6 and m >= 4) or (n == 7 and m >= 4) or (n == 8 and m >= 3):
            continue
        chunk = central[i:i+big_pixel_size_y, j:j+big_pixel_size_x]
        average = np.mean(chunk)
        simulation_pixelized[i:i+big_pixel_size_y, j:j+big_pixel_size_x] = average

j_y = [0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304]
j_x = [0, 236, 472, 708, 944, 1180, 1416, 1652, 1888]
final_image = np.zeros_like(E_first_quadrant_offset)

for i in range(0, 8): # x axis
    for j in range(0, 9): # y axis
        print(str(j_x[i]) + "-" + str(j_x[i+1]) + " , " + str(j_y[j]) + "-" + str(j_y[j+1]))
        pixel = E_first_quadrant_offset[j_y[j]:j_y[j+1], j_x[i]:j_x[i+1]]
        # print average of the pixel
        print("Average of the pixel: " + str(np.mean(pixel)))
        # First, detect nan pixels and replace them with the average of the 3x3 neighborhood
        print("----------------------------------REMOVE NAN-----------------------------------")
        nan_pixels , nan_pixels_indices = detect_nan_pixels(pixel)
        print("BEFORE nan-FILTERING : Number of nan pixels: " + str(nan_pixels_indices.shape[0]))
        if nan_pixels_indices.shape[0] > 0:
            pixel = replace_nan_pixels(pixel, nan_pixels)
            nan_pixels , nan_pixels_indices = detect_nan_pixels(pixel)
            print("AFTER nan-FILTERING : Number of nan pixels: " + str(nan_pixels_indices.shape[0]))
        original_pixel = pixel.copy()
        sensitivity = 4 # lower number means more pixels will be detected as noisy
        print("-----------------------------DETECT NOISY PIXELS---------------------------------")
        noisy_pixels01 , noisy_pixels_indices01 = detect_noisy_pixels(pixel, sensitivity)
        print("BEFORE noisy-FILTERING : Number of noisy pixels: " + str(noisy_pixels_indices01.shape[0]))
        pixel = replace_noisy_pixels_with_average(pixel, noisy_pixels01)
        noisy_pixels02 , noisy_pixels_indices02 = detect_noisy_pixels(pixel, sensitivity)
        print("AFTER noisy-FILTERING : Number of noisy pixels: " + str(noisy_pixels_indices01.shape[0]))
        print("-----------------------------MEDIAN FILTERING------------------------------------")
        median_pixel = medfilt2d(pixel, kernel_size=3)
        noisy_pixels03 , noisy_pixels_indices03 = detect_noisy_pixels(median_pixel, sensitivity)
        if noisy_pixels_indices03.shape[0] > 0:
            print("Noisy pixels detected after median filtering: " + str(noisy_pixels_indices03.shape[0]))
            print("Replacing noisy pixels with average value")
            m = 0
        while noisy_pixels_indices03.shape[0] > 0:
            median_pixel = replace_noisy_pixels_with_average(median_pixel, noisy_pixels03)
            noisy_pixels03 , noisy_pixels_indices03 = detect_noisy_pixels(median_pixel, sensitivity)
            m += 1
            print("try again : Noisy pixels detected after median filtering: " + str(noisy_pixels_indices03.shape[0]))
            if m == 10:
                print("Removing was not successful!!")
                break
            if noisy_pixels_indices03.shape[0] == 0:
                print("Removing was successful!!")
                break

        print("-----------------------------FITTING POLYNOMIAL----------------------------------")
        # fitted_pixel = fit_polynomial(median_pixel, 1)
        # final = median_pixel - fitted_pixel
        final_image[j_y[j]:j_y[j+1], j_x[i]:j_x[i+1]] = median_pixel


# change where the data is 0 to nan
final_image_surface = final_image.copy()
final_image_surface[final_image_surface == 0] = np.nan
simulation_pixelized_surface = simulation_pixelized.copy()
simulation_pixelized_surface[simulation_pixelized_surface == 0] = np.nan


X = E_first_quadrant_offset.shape[0]
X2 = central.shape[1]
Y = E_first_quadrant_offset.shape[1]
Y2 = central.shape[0]
x = np.arange(0, X, 1)
x2 = np.arange(0, X2, 1)
y = np.arange(0, Y, 1)
y2 = np.arange(0, Y2, 1)
X, Y = np.meshgrid(x, y)
X2, Y2 = np.meshgrid(x2, y2)


# plot four in one - two 2D plots and two 3D plots
fig, axs = plt.subplots(2, 2, figsize=(32, 30))
ax0 = axs[0, 0].imshow(final_image, cmap='magma')
axs[0, 0].set_title('Phase profile (Experimental)', fontsize=18, weight='bold', pad=20)
cbar0 = fig.colorbar(ax0, ax=axs[0, 0], fraction=0.046, pad=0.02)
#colorbar fontsize
cbar0.ax.tick_params(labelsize=14)
#colorbar ticks and format
cbar0.set_ticks(np.linspace(0, np.max(final_image), 7))
cbar0.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
cbar0.set_label('Phase (rad)', fontsize=16, rotation=270, labelpad=20)
axs[0, 0].set_xticks(j_x, minor=False)
x_ticks_labels = [str(0), str(6/8), str(12/8), str(18/8), str(24/8), str(30/8), str(36/8), str(42/8), str(48/8)]
axs[0, 0].set_xticklabels(x_ticks_labels, minor=False, rotation=0, fontsize=16)
axs[0, 0].set_yticks(j_y)

y_ticks_labels = [str(round(54/9,1)), str(round(48/9,1)), str(round(42/9,1)), str(round(36/9,1)), str(round(30/9,1)), str(round(24/9,1)), str(round(18/9,1)), str(round(12/9,1)), str(round(6/9,1)), str(0)]
axs[0, 0].set_yticklabels(y_ticks_labels, minor=False, rotation=0, fontsize=16)
# aspect ratio
axs[0, 0].set_aspect(1.0/axs[0, 0].get_data_ratio(), adjustable='box')
# x label is x(um) - y label is y(um) - micro symbol
axs[0, 0].set_xlabel('x(μm)', fontsize=16)
axs[0, 0].set_ylabel('y(μm)', fontsize=16)


ax1 = axs[0, 1].imshow(simulation_pixelized, cmap='magma', origin='lower')
axs[0, 1].set_title('Phase profile (Simulation)', fontsize=18, weight='bold', pad=20)
cbar1 = fig.colorbar(ax1, ax=axs[0, 1], fraction=0.046, pad=0.02)
# the labele of colorbar is phi(rad) in greek letter
cbar1.set_label('Phase (rad)', fontsize=16, rotation=270, labelpad=20)
#colorbar fontsize
cbar1.ax.tick_params(labelsize=14)
#colorbar ticks and format
cbar1.set_ticks(np.linspace(0, np.max(simulation_pixelized), 7))
cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
x_ticks = np.linspace(0, 1000, 9)
y_ticks = np.linspace(0, 999, 10)
axs[0, 1].set_xticks(x_ticks, minor=False)
axs[0, 1].set_xticklabels(x_ticks_labels, minor=False, rotation=0, fontsize=16)
axs[0, 1].set_yticks(x_ticks)
axs[0, 1].set_yticklabels(x_ticks_labels, minor=False, rotation=0, fontsize=16)
axs[0, 1].set_xlabel('x(μm)', fontsize=16)
axs[0, 1].set_ylabel('y(μm)', fontsize=16)


# 3D plot
ax2 = fig.add_subplot(2, 2, 3, projection='3d')
ax2.plot_surface(X, Y, final_image_surface.T, cmap='magma')
ax2.set_title('Phase profile (Experimental)', fontsize=18, weight='bold')
ax2.set_xlabel('x(μm)', fontsize=16, labelpad=20)
ax2.set_ylabel('y(μm)', fontsize=16, labelpad=20)
ax2.view_init(30, -30)
ax2.set_xticks(np.linspace(0, 2304, 10))
ax2.set_yticks(np.linspace(0, 1888, 9))


# 3D plot
ax3 = fig.add_subplot(2, 2, 4, projection='3d')
ax3.plot_surface(X2, Y2, simulation_pixelized_surface, cmap='magma')
ax3.set_title('Phase profile (Experimental)', fontsize=18, weight='bold')
ax3.set_xlabel('x(μm)', fontsize=16, labelpad=20)
ax3.set_ylabel('y(μm)', fontsize=16, labelpad=20)
# ax3.set_zlabel('Phase (rad)', fontsize=14)
ax3.view_init(30, -120)
ax3.set_xticks(np.linspace(0, 1000, 9))
ax3.set_yticks(np.linspace(0, 999, 10))
ax3.set_zticks(np.linspace(0, np.max(simulation_pixelized), 7))
ax3.set_zticklabels(np.linspace(0, np.max(simulation_pixelized), 7), minor=False, rotation=0, fontsize=16, verticalalignment='center_baseline', horizontalalignment='right')
# z ticks formatstyle
ax3.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.show()
