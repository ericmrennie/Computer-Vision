import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _():
    from io import BytesIO
    from urllib.request import Request, urlopen

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    return BytesIO, Image, Request, mo, np, plt, urlopen


@app.cell(hide_code=True)
def _(BytesIO, Image, Request, np, urlopen):
    def image_from_url(image_url):
        """Download image_url and return it as a grayscale NumPy array."""
        request = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            return np.asarray(Image.open(BytesIO(response.read())).convert("L"), dtype=float)

    return (image_from_url,)


@app.cell(hide_code=True)
def _(image_from_url):
    mare_image = image_from_url(
        "https://bigdatahealth.ucsb.edu/sites/default/files/styles/logo/"
        "public/logo/img_2289.jpeg"
    )
    mare_side = min(mare_image.shape)
    mare_top = (mare_image.shape[0] - mare_side) // 2
    mare_left = (mare_image.shape[1] - mare_side) // 2
    mare_image = mare_image[
        mare_top : mare_top + mare_side, mare_left : mare_left + mare_side
    ]
    storke_tower_image = image_from_url(
        "https://thumb.wikimedia.org/wikipedia/commons/thumb/4/48/"
        "Ucsb-storke-tower01.jpg/960px-Ucsb-storke-tower01.jpg"
    )
    kavli_image = image_from_url(
        "https://www.kitp.ucsb.edu/sites/default/files/kitp/kavlisb.jpeg"
    )
    return kavli_image, mare_image, storke_tower_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Homework 1

    ## Editing

    Marimo notebooks are just Python files, where each "cell" is a function. You can either edit here, in [Molab](https://molab.marimo.io/), or directly in Python.

    ### Option 1 - Molab [Recommended]

    This is the most recommended option. Take this notebook (.py file), upload it to Molab, and start working. Once you are finished making your edits, click "Export" and select "Python" to export it as a notebook source.

    ### Option 2 - Local Marimo

    To work locally, first create a python project and place this file `hw1.py` within it. Then, install marimo using your favorite package manager (`uv add marimo`). Finally, launch this notebook by typing in your terminal `uv run marimo edit` and a new tab should open in your browser.

    ### Option 3 - Regular Python [Not Recommended]

    It is theoretically possible to edit the `.py` file directly (although we don't recommend it). If you choose this route, make sure that your Python/Markdown renders properly in Marimo.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 1: Image Convolution and Edge Detection

    You are given the following two images $I_1$ and $I_2$, with three convolution kernels $S_x$, $S_y$ and $L$. [.] indicates the origin of each image.

    \[
    I_1 = \begin{bmatrix}
    [1] & 1 & 1 \\
    3 & 3 & 3 \\
    6 & 6 & 6
    \end{bmatrix}
    \qquad
    I_2 = \begin{bmatrix}
    [10] & 10 & 10 \\
    10 & 100 & 100 \\
    10 & 100 & 100
    \end{bmatrix}
    \]

    \[
    S_x = \begin{bmatrix}-1 & 0 & 1 \\ -2 & [0] & 2 \\ -1 & 0 & 1\end{bmatrix}
    \qquad
    S_y = \begin{bmatrix}-1 & -2 & -1 \\ 0 & [0] & 0 \\ 1 & 2 & 1\end{bmatrix}
    \qquad
    L = \begin{bmatrix}0 & 1 & 0 \\ 1 & [-4] & 1 \\ 0 & 1 & 0\end{bmatrix}
    \]
    """)
    return


@app.cell
def _(np):
    # EDIT BELOW: transcribe the matrices displayed above into NumPy arrays.
    # Replace the zero defaults with the instructor-provided entries for I_1 and I_2.
    image_1 = np.array([[1, 1, 1],
               [3, 3, 3],
               [6, 6, 6]])
    image_2 = np.array([[10, 10, 10],
               [10, 100, 100],
               [10, 100, 100]])

    sobel_x = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]])
    laplacian = np.array([[0, 1, 0],
                 [1, -4, 1],
                 [0, 1, 0]])
    return image_1, image_2, laplacian, sobel_x, sobel_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(a) Sobel X and Sobel Y By Hand

    Compute the full linear convolutions \(S_x * I_1\) and \(S_y * I_1\) **by hand**, where \(*\) denotes linear convolution. Mark the origin in each output matrix. Use the two responses to locate the edges, and explain your reasoning.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 1(a)

    S_x * I_1 = ([[-1, -1, 0, 1, 1],
                [-5, [-5], 0, 5, 5],
                [-13, -13, 0, 13, 13],
                [-15, -15, 0, 15, 15],
                [-6, -6, 0, 6, 6]])

    S_y * I_1 = ([[-1, -3, -4, -3, -1],
                [-3, [-9], -12, -9, -3],
                [-5, -15, -20, -15, -5],
                [3, 9, 12, 9, 3],
                [6, 18, 24, 18, 6

    I1 contains horizontal edges, with intensity increasing downward. It contains no vertical edges since Sx's interior response is zero. The edges are located between pixel rows 0-1 and 1-2 across the width of the image.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(b) Laplacian By Hand

    Compute the full linear convolution \(L * I_2\) **by hand**. Mark the origin in the output matrix. Use the response to locate any corner(s), and explain your reasoning.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 1(b)

    L * I_2 = ([[0,10,10,10,0],
                [10,[-20],80,70,10],
                [10,80,-180,-190,100],
                [10,70,-190,-200,100],
                [0,10,100,100,0]])

    The corner is at output matrix (1,1) = -180, matching I2 pixel (1,1), the 100 where the bright region starts in both directions. It's the only cell with no padding used in the Laplacian calculation. It's negative because the pixel is brighter than its neighbor average. It's a corner because the two neighbors above and left are dark. The cells beside it at (0,1) and (1,0) are +80 which marks a boundary due to the sign change. The larger values of -190 and -200 include the padding and therefore include some artifacts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(c) Verify in Python

    Complete the two code cells below. The function must perform full 2D linear convolution without calling a library convolution routine (you may not use any numpy functions).
    """)
    return


@app.cell
def _(image_1, image_2, laplacian, np, sobel_x, sobel_y):
    # image = np.array([[1, 2, 3],
    #                  [4, 5, 6],
    #                  [7, 8, 9]])
    # kernel = np.array([[1, 0, 1],
    #                 [-2, 0, 2],
    #                   [-1, 0, 1]])

    def convolve2d_full(image, kernel):
        """Return the full 2D linear convolution of image and kernel."""
        output_shape = (
            image.shape[0] + kernel.shape[0] - 1,
            image.shape[1] + kernel.shape[1] - 1,
        )
        output = np.empty(output_shape, dtype=float)

        # EDIT BELOW: flip kernel and accumulate every shifted product in output.

        # reverse the kernel
        reversed_kernel = kernel[::-1, ::-1];

        # pad
        padded_H = len(image) + (2*(len(kernel) - 1))
        padded_W = len(image[0]) + (2*(len(kernel[0]) - 1))

        padded_image = [[0 for _ in range(padded_W)] for _ in range(padded_H)]

        # place the image inside the padded image
        for i in range(len(image)):
            for j in range(len(image[i])):
                padded_image[(len(kernel) - 1) + i][(len(kernel[0]) - 1) + j] = image[i][j];

        # using the flipped kernel, iterate through the padded image
        kH = len(kernel)
        kW = len(kernel[0])

        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                total = 0.0
                for m in range(kH):
                    for n in range(kW):
                        total += padded_image[i+m][j + n] * reversed_kernel[m][n]
                output[i][j] = total;





        return output

    # print(convolve2d_full(image, kernel))
    print(convolve2d_full(image_1, sobel_x))
    print(convolve2d_full(image_1, sobel_y))
    print(convolve2d_full(image_2, laplacian))
    return (convolve2d_full,)


@app.cell(hide_code=True)
def _(convolve2d_full, image_1, image_2, laplacian, sobel_x, sobel_y):
    try:
        convolve2d_full(image_1, sobel_x)
        convolve2d_full(image_1, sobel_y)
        convolve2d_full(image_2, laplacian)
    except Exception as error:
        print(f"Problem 1(c) test is not ready: {error}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Test Your Convolution on a Real Image

    The next cell downloads a photograph, converts it to grayscale, and runs your convolution function with both Sobel kernels. It displays the input, the horizontal and vertical responses, and their combined edge magnitude. You can use this as a quick check after implementing `convolve2d_full`.
    """)
    return


@app.cell(hide_code=True)
def _(convolve2d_full, mare_image, np, sobel_x, sobel_y):
    try:
        mare_sobel_x = convolve2d_full(mare_image, sobel_x)
        mare_sobel_y = convolve2d_full(mare_image, sobel_y)
        mare_edges = np.hypot(mare_sobel_x, mare_sobel_y)
        mare_error = None
    except Exception as error:
        mare_sobel_x = mare_sobel_y = mare_edges = None
        mare_error = str(error)
    return mare_edges, mare_error, mare_sobel_x, mare_sobel_y


@app.cell(hide_code=True)
def _(mare_edges, mare_error, mare_image, mare_sobel_x, mare_sobel_y, mo):
    try:
        if mare_error:
            mo.output.replace(mo.md(f"Sobel test on mare is not ready: {mare_error}"))
        else:
            mo.output.replace(mo.vstack(
                [
                    mo.hstack(
                        [mo.image(mare_image, width=220), mo.image(mare_sobel_x, width=220)]
                    ),
                    mo.hstack(
                        [mo.image(mare_sobel_y, width=220), mo.image(mare_edges, width=220)]
                    ),
                ]
            ))
    except Exception as error:
        mo.output.replace(mo.md(f"Could not display the Sobel test: {error}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 2: Gaussian and Laplacian Pyramids

    You are given a color image. Convert it to grayscale, then crop the 512×512 region beginning at the upper-left corner. Use this crop as the input image for both parts.

    You will build pyramid levels at 512×512, 256×256, 128×128, and 32×32. To reach 32×32 from 128×128, blur and downsample twice. Do not use a prebuilt pyramid implementation.

    A normalized 5×5 Gaussian kernel is provided in the starter code. In your report, state your boundary convention, show the requested pyramid images at their corresponding resolutions, and include your code. If you use any non-standard-library helper, include it with a brief explanation.
    """)
    return


@app.cell
def _(mo, storke_tower_image):
    # EDIT BELOW: take the upper-left crop from storke_tower_image.
    input_image = storke_tower_image

    cropped_image = input_image[0:512, 0:512]

    mo.image(cropped_image, width=220)
    return cropped_image, input_image


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(a) Gaussian Pyramid

    Create a Gaussian pyramid from the 512×512 grayscale crop using Gaussian convolution and the provided normalized 5×5 kernel. Apply blur and downsampling to produce four levels, in this order: 512×512, 256×256, 128×128, and 32×32. Document your boundary convention, and include the four images and your code in the report. Do not use a prebuilt pyramid implementation.
    """)
    return


@app.cell
def _(convolve2d_full, cropped_image, np):
    kernel = np.array(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ],
            dtype=float,
        ) / 256

    def blur_same_size(image, kernel):
        full = convolve2d_full(image,kernel)

        H, W = image.shape
        kH, kW = kernel.shape
        start_r = (kH - 1) // 2
        start_c = (kW - 1) // 2
        blurred = full[start_r: start_r + H, start_c : start_c + W]

        return blurred

    def blur_and_downsample(image, kernel):
        """Blur image with kernel, then reduce each spatial dimension by two."""
        # EDIT BELOW: call convolve2d_full, crop to image.shape, then subsample.
        # Document your boundary convention.


        # boundary convention: zero padding. pixels outside the image are treated as 0, which darkens the edges of the blurred image

        blurred = blur_same_size(image, kernel)

        return blurred[::2, ::2]

    print(blur_and_downsample(cropped_image, kernel))
    return blur_and_downsample, blur_same_size


@app.cell
def _(blur_and_downsample, np):
    def build_gaussian_pyramid(
        image, level_shapes=((512, 512), (256, 256), (128, 128), (32, 32))
    ):
        """Return Gaussian-pyramid levels with exactly the requested shapes."""
        kernel = np.array(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ],
            dtype=float,
        ) / 256
        pyramid = [image]

        # EDIT BELOW: call blur_and_downsample until each requested level is reached.
        for target in level_shapes[1:]: # target is 256
            current = pyramid[-1] # current is 512
            while current.shape != target:
                current = blur_and_downsample(current, kernel)
            pyramid.append(current)

        # Keep levels in the same order as level_shapes.
        return pyramid

    return (build_gaussian_pyramid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(b) Laplacian Pyramid

    Using your Gaussian pyramid, construct the corresponding Laplacian pyramid. Construct the Laplacian pyramid for each of the four resolutions given above. For example, to construct the Laplacian pyramid at resolution 512x512, you take the Gaussian smoothed image at resolution 512x512 and subtract it from the original 512x512 image.
    """)
    return


@app.cell
def _(blur_same_size, np):
    def build_laplacian_pyramid(gaussian_pyramid):

        kernel = np.array(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ],
            dtype=float,
        ) / 256

        """Return Laplacian levels followed by the coarsest Gaussian residual."""
        pyramid = []

        # EDIT BELOW: construct the Laplacian level at each requested resolution,
        # then append the final coarsest Gaussian level.

        g = None

        for level in gaussian_pyramid:
            g = level
            g_blurred = blur_same_size(g, kernel)

            pyramid.append(g - g_blurred)

        pyramid.append(g)

        return pyramid


    return (build_laplacian_pyramid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pyramid Visualizations

    After completing your functions, the hidden cells below display the Gaussian pyramid in the first row and the Laplacian pyramid in the second row.
    """)
    return


@app.cell(hide_code=True)
def _(build_gaussian_pyramid, build_laplacian_pyramid, input_image):
    try:
        gaussian_pyramid = build_gaussian_pyramid(input_image)
        laplacian_pyramid = build_laplacian_pyramid(gaussian_pyramid)
        pyramid_error = None
    except Exception as error:
        gaussian_pyramid = laplacian_pyramid = None
        pyramid_error = str(error)
    return gaussian_pyramid, laplacian_pyramid, pyramid_error


@app.cell(hide_code=True)
def _(gaussian_pyramid, laplacian_pyramid, mo, pyramid_error):
    try:
        if pyramid_error:
            mo.output.replace(mo.md(f"Pyramid visualization is not ready: {pyramid_error}"))
        else:
            mo.output.replace(mo.vstack(
                [
                    mo.hstack([mo.image(level, width=150) for level in gaussian_pyramid]),
                    mo.hstack([mo.image(level, width=150) for level in laplacian_pyramid]),
                ]
            ))
    except Exception as error:
        mo.output.replace(mo.md(f"Could not display the pyramids: {error}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 3: Harris Corner Detector

    For this part, implement a Harris corner detector. At each image location, compute the corner response function (CRF)

    \[
    R(x, y) = \det(A) - k\bigl(\operatorname{trace}(A)\bigr)^2,
    \]

    where \(A\) is the local structure matrix

    \[
    A =
    \begin{bmatrix}
    \langle I_x^2 \rangle & \langle I_x I_y \rangle \\
    \langle I_x I_y \rangle & \langle I_y^2 \rangle
    \end{bmatrix}.
    \]

    Use a simple constant 3×3 averaging window for the local averages. You may use any discrete derivative approximation, but write down the equations you use. You may ignore image-border effects by zero-padding or by limiting calculations to the image interior.

    Use the CRF to identify the top \(N\) candidate corners, where \(N\) is around 20–50. Threshold and sort the candidates before selecting the final locations. Your function should return `(row, column)` locations.

    Put your complete implementation, including any helper functions, in the cell below. You may use functions defined elsewhere in this notebook. Use the KAVLI image to demonstrate your detector: include your code in the report, overlay the detected corners on the image, list their coordinates, state your parameter values, and comment on the results.
    """)
    return


@app.cell
def _(blur_same_size, np):
    # ran out of time - this problem was solved with AI. Can you let me know if this is the correct approach?

    def harris_corners(image, threshold, max_corners=40, k=0.04,
                       min_distance=7, margin=5):
        """Return up to max_corners Harris corners as (row, column) locations.

        threshold    : fraction of R.max(); only pixels with R > threshold * R.max()
                       are candidates.
        min_distance : minimum spacing (in pixels) between returned corners.
        margin       : border width ignored, since zero padding creates fake
                       responses at the image edge.
        """
        img = image.astype(float)

        # Derivative kernels (Sobel). Defined locally so the function is self-contained.
        sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=float)
        sy = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=float)

        # Constant 3x3 averaging window
        avg = np.ones((3, 3)) / 9.0

        # 1. Image derivatives (same-size convolution, zero padding)
        Ix = blur_same_size(img, sx)
        Iy = blur_same_size(img, sy)

        # 2. Products
        Ix2 = Ix * Ix
        Iy2 = Iy * Iy
        Ixy = Ix * Iy

        # 3. Local averages: entries of the structure matrix A
        Sxx = blur_same_size(Ix2, avg)
        Syy = blur_same_size(Iy2, avg)
        Sxy = blur_same_size(Ixy, avg)

        # 4. Corner response R = det(A) - k * trace(A)^2
        R = (Sxx * Syy - Sxy * Sxy) - k * (Sxx + Syy) ** 2

        # 5. Ignore the border
        R[:margin, :] = 0
        R[-margin:, :] = 0
        R[:, :margin] = 0
        R[:, -margin:] = 0

        # 6. Threshold, then sort candidates strongest to weakest
        cutoff = threshold * R.max()
        candidates = np.argwhere(R > cutoff)            # rows of (row, col)
        if len(candidates) == 0:
            return np.empty((0, 2), dtype=int)
        scores = R[candidates[:, 0], candidates[:, 1]]
        candidates = candidates[np.argsort(-scores)]

        # 7. Greedy non-maximum suppression, stop at max_corners
        kept = []
        for r, c in candidates:
            if all(max(abs(r - kr), abs(c - kc)) >= min_distance for kr, kc in kept):
                kept.append((r, c))
                if len(kept) == max_corners:
                    break

        return np.array(kept, dtype=int).reshape(-1, 2)




    return (harris_corners,)


@app.cell
def _(harris_corners, kavli_image, plt):
    corners = harris_corners(kavli_image, threshold=0.01)
    print(len(corners))
    print(corners)                      # (row, column) list for the report

    plt.imshow(kavli_image, cmap="gray")
    plt.axis("off")
    plt.scatter(corners[:, 1], corners[:, 0], c="r", s=10)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Question 4: Philosophy of Computer Vision

    ## 4(a) Biological Facial Recognition

    How do humans recognize faces? Describe the features the visual system could use and the processing that might occur at each stage, from seeing a face to identifying the person.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 4(a)

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4(b) Computer Vision in the Age of AI

    As AI becomes better at writing code and optimizing pipelines, what should students learn in a computer vision course?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 4(b)

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4(c) Course Milestone

    What would you like to accomplish by the end of this course? Describe a project, a skill you want to develop, or another goal you hope to reach.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Your response to 4(c)

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Submitting Your Work

    If you used Molab, click **Export** and choose **Python** to download your completed notebook as a `.py` file. Submit that `.py` notebook file to Gradescope.
    """)
    return


if __name__ == "__main__":
    app.run()
