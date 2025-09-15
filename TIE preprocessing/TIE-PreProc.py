import streamlit as st
import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from TIE_func import *  # Includes threshold_where_pixel_count_rises, plot_image, plot_normalized_image, etc.

st.set_page_config(layout="wide")
st.title("EM Image Viewer (.dm3 / .emd)")

col1, col2 = st.columns([2, 3])

# === File uploader ===
with col1:
    uploaded_file = st.file_uploader("Choose a .dm3 or .emd file", type=["dm3", "emd", "tif"])
    if uploaded_file is not None:
        filename = uploaded_file.name

# === Main logic ===
if uploaded_file is not None:
    suffix = "." + uploaded_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load image
    s = hs.load(tmp_path)
    image_data = s.data.astype(np.float32)
    h = image_data.shape[0]
    defocus, bias, voltage = extract_defocus_bias_voltage(filename)

    # Auto thresholds
    auto_max_plot, FWQM = threshold_where_pixel_count_rises(image_data, 3)
    auto_min_plot = 0.0
    max_value = float(np.nanmax(image_data))
    min_value = float(np.nanmin(image_data))
    auto_max_plot = min(auto_max_plot, max_value)

    # === Reset session state if new file ===
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = None

    if st.session_state.uploaded_filename != filename:
        st.session_state.min_plot_slider = 0.0
        st.session_state.max_plot_slider = auto_max_plot
        st.session_state.min_plot_default = 0.0
        st.session_state.max_plot_default = auto_max_plot
        st.session_state.radius_slider = int(h // 3)
        st.session_state.x_pos_slider = 0
        st.session_state.y_pos_slider = 0
        st.session_state.uploaded_filename = filename
        st.success(f"New file uploaded: {filename}. Sliders reset.")

    # === Radius + Position sliders ===
    with col1:
        subcol1, subcol2, subcol3 = st.columns(3)

        with subcol1:
            radius_slider = st.slider(
                "Safe Radius", 0, int(h // 2),
                step=1,
                key="radius_slider"
            )
        with subcol2:
            x_pos_slider = st.slider(
                "X Position", -int(h // 5), int(h // 5),
                step=1,
                key="x_pos_slider"
            )
        with subcol3:
            y_pos_slider = st.slider(
                "Y Position", -int(h // 5), int(h // 5),
                step=1,
                key="y_pos_slider"
            )

    # === Threshold sliders + reset ===
    with col1:
        st.markdown("### Threshold Controls (auto-updates image)")
        if st.button("🔁 Reset Thresholds to Default"):
            st.session_state.min_plot_slider = st.session_state.min_plot_default
            st.session_state.max_plot_slider = st.session_state.max_plot_default

        st.slider(
            "Minimum Threshold (negatives)",
            min_value=min_value, max_value=1000.0,
            step=1.0, key="min_plot_slider",
            help="Minimum threshold for outliers.\n"
                    "Maximum value is 100.0 for better visualization.\n"
        )

        # st.slider(
        #     "Maximum Threshold (outliers)",
        #     min_value=float(FWQM), max_value=max_value,
        #     step=1.0, key="max_plot_slider"
        # )
        st.slider(
            "Maximum Threshold (outliers)",
            min_value=0.0, max_value=max_value,
            step=1.0, key="max_plot_slider"
        )

    # === Normalization options ===
    with col1:
        st.markdown("### -------------------------------Normalization-------------------------------")
        subcol1, subcol2, subcol3 = st.columns(3)

        with subcol1:
            mask_mode = st.radio(
                "Border Masking Mode",
                options=["Circular", "Smart"],
                index=0,
                horizontal=True
            )
            apply_border_mask_circular = mask_mode == "Circular"
            apply_border_mask_smart = mask_mode == "Smart"
        with subcol2:
            # checkbox for "apply remap peaks"
            apply_remap_peaks = st.checkbox(
                "Apply Remap Peaks",
                value=False,
                help="Remap the histogram peaks to the range of [0, 255].\n"
                     "This is useful for better visualization.",
                key="apply_remap_peaks"

            )
        with subcol3:
            plot_normalized_button = st.button("Plot No-Outliner Image")

    with col1:
        subcol1, subcol2 = st.columns(2)

        with subcol1:
            object_peak_input = st.text_input("Object peak pos", value="60")

        with subcol2:
            background_peak_input = st.text_input("Background peak pos", value="150")

        # Optional: Convert to float safely
        try:
            object_peak = float(object_peak_input)
            background_peak = float(background_peak_input)
            st.success(f"Using Object peak: {object_peak}, Background peak: {background_peak}")
        except ValueError:
            st.error("Please enter valid numbers for both peaks.")


    # === Plot original image ===
    with col2:
        masked_image, border_mask, radius = mask_negatives_outside_circle(
            image_data,
            radius=st.session_state.radius_slider,
            x_pos=st.session_state.x_pos_slider,
            y_pos=st.session_state.y_pos_slider,
            mask_circular=apply_border_mask_circular,
            mask_smart=apply_border_mask_smart,
        )

        fig = plot_image(
            image_data,
            defocus=defocus,
            bias=bias,
            voltage=voltage,
            min_plot=st.session_state.min_plot_slider,
            max_plot=st.session_state.max_plot_slider,
            radius=radius,
            x_pos=st.session_state.x_pos_slider,
            y_pos=st.session_state.y_pos_slider,
        )
        st.pyplot(fig)

# === Normalized image processing ===
if uploaded_file and plot_normalized_button:
    with st.spinner("Removing outlier pixels from the image..."):

        up_limit = st.session_state.max_plot_slider
        image_data_no_outliner = replace_outliers_with_neighbors(image_data, up_limit)

        if apply_border_mask_circular or apply_border_mask_smart:
            image_data_no_outliner[border_mask] = np.nan

        image_data_no_outliner = replace_negatives_with_neighbors(image_data_no_outliner)
        normalized_img = em_contrast_stretch_masked(image_data_no_outliner)
        normalized_img *= 255.0



        # save these into session_state
        st.session_state.image_data_no_outliner = image_data_no_outliner
        st.session_state.normalized_img = normalized_img

        # Plot normalized result
        with col2:
            st.markdown("### Normalized Image")
            fig = plot_normalized_image(image_data_no_outliner, normalized_img, apply_remap_peaks,object_peak,background_peak,
                                        filename)
            st.pyplot(fig)

# === Save button ===
with col1:
    st.markdown("### -----Save Result-----")
    save_button = st.button("Save Normalized Image as PNG")

    if save_button:
        if "image_data_no_outliner" in st.session_state and "normalized_img" in st.session_state:
            with st.spinner("💾 Saving image..."):
                # plot_image_and_histograms(
                #     image_data,
                #     st.session_state.image_data_no_outliner,
                #     st.session_state.normalized_img,
                #     defocus,
                #     bias,
                #     voltage,
                #     st.session_state.min_plot_slider,
                #     st.session_state.max_plot_slider,
                #     radius,
                #     st.session_state.x_pos_slider,
                #     st.session_state.y_pos_slider,
                #     filename
                # )
                save_normalized_img(st.session_state.image_data_no_outliner,
                                    st.session_state.normalized_img, apply_remap_peaks, object_peak, background_peak, filename)

            st.success("✅ Image saved successfully!")
        else:
            st.error("⚠️ You need to first click 'Plot No-Outliner Image' before saving.")