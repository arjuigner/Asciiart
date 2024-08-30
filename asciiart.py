import streamlit as st
from PIL import Image
import numpy as np

########### image manipulation

CHAR = np.array([
    # Character: "." (1 pixel)
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    # Character: "-" (3 pixels)
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    # Character: "I" (5 pixels, corrected)
    [
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
    ],
    # Character: "L" (7 pixels)
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    # Character: "T" (7 pixels)
    [
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    # Character: "H" (10 pixels)
    [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    # Character: "C" (10 pixels)
    [
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    # Character: "@" (22 pixels)
    [
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
], dtype=np.uint8)


def transform_img_gray(img: Image.Image) -> Image.Image:
    # convert image to grayscale and np array
    grey = img.convert('L')
    imga = np.array(grey)
    
    print(f"{imga.shape=}")
    H, W, = imga.shape

    # clip the image so that its size is a multiple of 8
    H, W = (H//8)*8, (W//8)*8

    new = np.zeros((H, W), dtype=np.uint8)
    print(f"New size : {new.shape}")

    # iterate over 8x8 blocks
    for r in range(0, H, 8):
        for c in range(0, W, 8):
            mean = imga[r:r+8, c:c+8].mean()
            new[r:r+8, c:c+8] = CHAR[int(mean) // 32] * 255
    
    return Image.fromarray(new)


def transform_img_color(img: Image.Image) -> Image.Image:
    # convert image to grayscale and np array
    imga = np.array(img.convert('RGB'))
    
    print(f"{imga.shape=}")
    H, W, _ = imga.shape

    # clip the image so that its size is a multiple of 8
    H, W = (H//8)*8, (W//8)*8

    new = np.zeros((H, W, 3), dtype=np.uint8)
    print(f"New size : {new.shape}")

    # iterate over 8x8 blocks
    for r in range(0, H, 8):
        for c in range(0, W, 8):
            mean = imga[r:r+8, c:c+8, :].mean(axis=(0,1))
            #char = CHAR[int(mean.dot([0.299, 0.587, 0.114])) // 32]
            char = CHAR[int(mean.mean()) // 32]
            scaling = 255 / mean.sum()
            new[r:r+8, c:c+8, 0] = char * mean[0] * scaling
            new[r:r+8, c:c+8, 1] = char * mean[1] * scaling
            new[r:r+8, c:c+8, 2] = char * mean[2] * scaling
    
    return Image.fromarray(new)


def transform_img(img: Image.Image, color: bool) -> Image.Image :
    if not color or img.mode == 'L' or img.mode == 'LA':
        return transform_img_gray(img)
    else:
        return transform_img_color(img)


########## app

if __name__ == '__main__':
    st.title("Ascii Art Generator")

    # Step 1: Upload image
    uploaded_file = st.file_uploader("Choose a PNG file", type="png")

    if uploaded_file is not None:
        # Step 2: Open the image and convert to NumPy array
        image = Image.open(uploaded_file)

        st.image(image, caption='Original Image', use_column_width=True)

        # Slider for selecting the new width
        original_width, original_height = image.size
        new_width = st.slider(
            "Select the new width",
            min_value=int(original_width * 0.1),
            max_value=int(original_width * 2),
            value=original_width,
        )

        # Calculate the new height while maintaining the aspect ratio
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
        
        # Calculate the percentage increase in size
        size_increase = (new_width * new_height) / (original_width * original_height) * 100
        
        # Display the selected width, corresponding height, and size increase
        st.write(f"Selected Width: {new_width} pixels")
        st.write(f"Corresponding Height: {new_height} pixels")
        st.write(f"Size Increase: {size_increase:.2f}%")

        color = st.checkbox("Color")

        if st.button("Generate"):
            # generate the new version 
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            modified_image = transform_img(resized, color)

            # display the modified image
            st.image(modified_image, caption='Modified Image', use_column_width=True)

            # step 5: Save the modified image and provide a download button
            modified_image.save("modified_image.png")
            with open("modified_image.png", "rb") as file:
                st.download_button(
                    label="Download Modified Image",
                    data=file,
                    file_name="modified_image.png",
                    mime="image/png"
                )
    else:
        st.warning("Please upload a PNG file.")