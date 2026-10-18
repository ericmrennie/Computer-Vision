# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib>=3.11.2",
#     "numpy>=2.5.3",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Homework 2: Corner and Blob Detection

    In this assignment, you will generate simple images containing black squares on a white background and apply different operators for corner and blob detection. You will compare Harris, Laplacian of Gaussian (LoG), and Difference of Gaussian (DoG) detections across smoothing scales and image resolutions.

    **Submission:** upload `hw2.zip` containing `hw2.py`, `hw1.py`, and `hw2.pdf` if you submit handwritten answers. Keep these files in the same folder before creating the ZIP.

    **For every short-answer question, show all steps and explain your reasoning. You may write your answer in the notebook's Markdown cell using LaTeX, or, if you prefer handwriting, include your written solutions in a PDF named exactly `hw2.pdf`. Clearly label each problem in the PDF and write "See hw2.pdf, Problem [number]" in the corresponding Markdown answer cell. Illegible work will receive a zero.**

    **Rules:** use your Homework 1 functions `laplacian`, `convolve2d_full`, and `harris_corners`. Do not use `np.convolve`, `scipy`, or `cv2`.
    """)
    return


@app.function(hide_code=True)
def homework1_definitions():
    """Load and cache Homework 1 helpers for importable solution functions."""
    from contextlib import redirect_stdout
    from io import StringIO
    from unittest.mock import patch
    import marimo
    import matplotlib.pyplot as plt
    import hw1

    if not hasattr(hw1, "_hw2_definitions"):
        with patch.object(marimo.output, "replace"), patch.object(marimo.output, "append"), patch.object(plt, "show"), redirect_stdout(StringIO()):
            _, hw1._hw2_definitions = hw1.app.run()
    return hw1._hw2_definitions


@app.cell
def _():
    hw1_definitions = homework1_definitions()
    laplacian = hw1_definitions["laplacian"]
    convolve2d_full = hw1_definitions["convolve2d_full"]
    harris_corners = hw1_definitions["harris_corners"]
    return (convolve2d_full,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 1: Eigenvalues of the Structure Matrix

    In Homework 1, the Harris detector used $R = \det(A) - k\,\bigl(\operatorname{trace}(A)\bigr)^2$ without computing the eigenvalues of the structure matrix $A$. Here you compute them directly.

    $$
    A_1 = \begin{bmatrix} 3 & 3 \\ 3 & 3 \end{bmatrix}
    \qquad
    A_2 = \begin{bmatrix} 8 & 0 \\ 0 & 7 \end{bmatrix}
    \qquad
    A_3 = \frac{1}{1000}\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
    \qquad
    A_4 = \begin{bmatrix} 1 & 3 \\ 3 & 1 \end{bmatrix}
    $$

    ## 1(a) Structure Matrix
    In the `structure_matrices` cell below, assign the four matrices given above as nested lists.
    """)
    return


@app.cell
def structure_matrices():
    # EDIT BELOW: replace the zero defaults with the displayed matrix entries.
    structure_1 = [[3, 3], [3, 3]]
    structure_2 = [[8, 0], [0, 7]]
    structure_3 = [(1/1000)*[1, 0], [0, 1]]
    structure_4 = [[1, 3], [3, 1]]
    return structure_1, structure_2, structure_3, structure_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(b) Eigenvalues and Feature Types
    Solve $\det(A - \lambda I) = 0$ by hand for each matrix and show your derivation in the Markdown answer cell below. Report both eigenvalues and use them to justify whether the matrix represents a corner, edge, flat region, or an invalid structure matrix. Use an eigenvalue threshold of 1 and state the assumptions behind your classification, including what makes a structure matrix valid. Do not use code or `np.linalg` for this part.

    **Show all steps. Answer in the Markdown cell below using LaTeX, or handwrite your solution in `hw2.pdf`, clearly labeled with the problem number. If you use the PDF, write "See hw2.pdf, Problem [number]" in the Markdown answer cell.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 1(b).**

    See hw2.pdf, Problem 1b.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(c) Classify Feature
    Complete `classify_feature(structure)` using your eigenvalue derivation from 1(b). Compute the eigenvalues directly (no `np.linalg`), then return `"corner"`, `"edge"`, `"flat"`, or `"invalid"`.
    """)
    return


@app.cell
def _(x):
    test = structure_2 = [[8, 0], [0, 7]]

    def classify_feature(structure, threshold=1.0):
        """Return 'corner', 'edge', 'flat' or 'invalid' for a 2x2 structure matrix."""

        # EDIT BELOW: compute both eigenvalues of [[a, b], [b, c]], then use
        # them to decide and return the feature type or invalid.
        # going to represent lambda with x
        lambda_matrix = structure - [[x, 0], [0, x]]

        return lambda_matrix

        # return "flat"

    print(classify_feature(test))
    return classify_feature, structure_2


@app.cell(hide_code=True)
def _(classify_feature, structure_1, structure_2, structure_3, structure_4):
    try:
        print("A1:", classify_feature(structure_1))
        print("A2:", classify_feature(structure_2))
        print("A3:", classify_feature(structure_3))
        print("A4:", classify_feature(structure_4))
    except Exception as error:
        print(f"Problem 1 test is not ready: {error}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 2: Single-Square Image (Image 1)

    ## 2(a) Image generation

    You will generate the following grayscale images using code (no images are provided). Use a white background (255) with black squares (0). Coordinates are $(x, y) = (\text{column}, \text{row})$.

    - **Image 1:** 100×100, one 32×32 square with top-left corner $(30, 40)$.
    - **Image 2:** 512×512, four squares:

    | Square | Top-left $(x, y)$ | Side |
    |---|---|---|
    | 1 | $(60, 60)$ | 40 |
    | 2 | $(180, 300)$ | 80 |
    | 3 | $(300, 100)$ | 120 |
    | 4 | $(340, 340)$ | 140 |

    Complete `make_square_image` below and fill in the square tuples for Image 1 and Image 2 as given above.
    """)
    return


@app.function
def make_square_image(size, squares):
    """Return a size-by-size white image with black squares.

    squares is a list of (x, y, side) tuples, where (x, y) = (column, row)
    is the top-left pixel of a square and side is its side length.
    """
    import numpy as np

    # EDIT BELOW: set the pixels of image.
    image = np.zeros((size, size))

    return image


@app.cell
def homework_squares():
    # EDIT BELOW: replace each template with (x, y, side) tuples.
    image_1_squares = [
        # EDIT HERE
    ]
    image_2_squares = [
        # EDIT HERE
    ]
    image_1 = make_square_image(100, image_1_squares)
    image_2 = make_square_image(512, image_2_squares)
    return image_1, image_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Display your generated images side by side below. These images will serve as the inputs for the subsequent detection tasks.
    """)
    return


@app.cell(hide_code=True)
def _(image_1, image_2, mo, plt):
    try:
        _figure, _axes = plt.subplots(1, 2, figsize=(8, 4))
        for _axis, _image, _title in (
            (_axes[0], image_1, "Image 1: single square"),
            (_axes[1], image_2, "Image 2: four squares"),
        ):
            _axis.imshow(_image, cmap="gray", vmin=0, vmax=255)
            _axis.set_title(_title)
            _axis.set_xticks([])
            _axis.set_yticks([])
        plt.tight_layout()
        _display = plt.gcf()
    except Exception as error:
        _display = mo.md(f"2(a) display is not ready: {error}")
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(b) Gaussian smoothing

    Using the generated 100×100 single-square image, apply Gaussian smoothing at four different scales $\sigma \in \{1, 2, 4, 8\}$. These smoothed images will be used to compute the Harris Corner Response Function (CRF) and compare detected corners across scales.

    Given $\sigma$, complete `gaussian_kernel_1d(sigma)` to return a vector whose entries correspond to the integer offsets $-r, -r+1, \dots, r$, where $r = \lceil 3\sigma \rceil$. At each offset $x$, the value should be the Gaussian $G_\sigma(x)$. Normalize the vector so its entries sum to 1.

    For example, when $\sigma = 1$, $r = 3$, so the kernel has 7 entries corresponding to offsets $-3, -2, -1, 0, 1, 2, 3$.
    """)
    return


@app.function
def gaussian_kernel_1d(sigma):
    """Return the normalized 1D Gaussian kernel of radius ceil(3 * sigma)."""
    import numpy as np

    # EDIT BELOW: compute the radius, sample g at the integer offsets, normalize.
    kernel = ...

    return kernel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using your `gaussian_kernel_1d(sigma)` function, complete `gaussian_kernel_2d(sigma)` to compute the 2D Gaussian kernel. Return a $(2r+1) \times (2r+1)$ array whose entries sum to 1.
    """)
    return


@app.function
def gaussian_kernel_2d(sigma):
    """Return the normalized 2D Gaussian kernel using gaussian_kernel_1d."""
    import numpy as np

    # EDIT BELOW: construct the 2D kernel from your 1D Gaussian kernel.
    kernel = ...

    return kernel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's write the function for applying your 2D Gaussian filter! Complete `convolve_same(image, kernel)`: pad the image using edge replication (`np.pad(..., mode="edge")`), then use `convolve2d_full` and crop the result to the original image size, with the kernel centred on each pixel. Edge replication repeats the nearest boundary pixels, so smoothing a constant image should return the same image.
    """)
    return


@app.function
def convolve_same(image, kernel):
    """Return the same-size convolution of image with an odd-sized 2D kernel."""
    import numpy as np
    convolve2d_full = homework1_definitions()["convolve2d_full"]

    # EDIT BELOW: use convolve2d_full, then crop to image.shape.
    same = ...

    return same


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now apply it! Complete `gaussian_smooth(image, sigma)` for 2D Gaussian smoothing.
    """)
    return


@app.function
def gaussian_smooth(image, sigma):
    """Return image smoothed by a 2D Gaussian of standard deviation sigma."""
    # EDIT BELOW: smooth with gaussian_kernel_1d and convolve_same.
    smoothed = ...

    return smoothed


@app.cell(hide_code=True)
def _(image_1, mo, plt):
    sigmas = (1, 2, 4, 8)
    try:
        smoothed_by_sigma = {sigma: gaussian_smooth(image_1, sigma) for sigma in sigmas}
        smoothing_error = None
        _figure, _axes = plt.subplots(1, 4, figsize=(13, 3.6))
        for _axis, _sigma in zip(_axes, sigmas):
            _axis.imshow(smoothed_by_sigma[_sigma], cmap="gray", vmin=0, vmax=255)
            _axis.set_title(f"Smoothed, σ = {_sigma}")
            _axis.axis("off")
        plt.tight_layout()
        _display = plt.gcf()
    except Exception as error:
        smoothed_by_sigma = None
        smoothing_error = str(error)
        _display = mo.md(f"Gaussian smoothing is not ready: {error}")
    _display
    return sigmas, smoothed_by_sigma, smoothing_error


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Short Answer Question** Which is faster: convolving with a 2D kernel, or convolving with a 1D kernel twice? Justify your response.

    **Show all steps. Answer in the Markdown cell below using LaTeX, or handwrite your solution in `hw2.pdf`, clearly labeled with the problem number. If you use the PDF, write "See hw2.pdf, Problem [number]" in the Markdown answer cell.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to 2(b).**

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(c) Harris corners

    Let's explore how corner detection works at each scale. Complete `corners_at_scale`: run `harris_corners` on each smoothed image, Choose default values so that **exactly the 4 corners** of the square are found at every $\sigma$.
    """)
    return


@app.function
def corners_at_scale(smoothed, threshold=..., max_corners=...):
    """Return the Harris corners of a smoothed image as (row, column)."""
    import numpy as np
    harris_corners = homework1_definitions()["harris_corners"]

    # EDIT BELOW: call your hw1 harris_corners function using threshold and max_corners.
    corners = ...

    return corners


@app.cell(hide_code=True)
def _(
    harris_max_corners,
    harris_threshold,
    mo,
    plt,
    sigmas,
    smoothed_by_sigma,
    smoothing_error,
):
    try:
        if smoothing_error:
            raise RuntimeError(smoothing_error)
        _figure, _axes = plt.subplots(1, 4, figsize=(13, 3.6))
        for _axis, _sigma in zip(_axes, sigmas):
            _smoothed = smoothed_by_sigma[_sigma]
            _corners = corners_at_scale(
                _smoothed,
                threshold=harris_threshold.value,
                max_corners=int(harris_max_corners.value),
            ).reshape(-1, 2)
            _axis.imshow(_smoothed, cmap="gray", vmin=0, vmax=255)
            _axis.scatter(_corners[:, 1], _corners[:, 0], c="r", s=18)
            _axis.set_title(f"Harris, σ = {_sigma} ({len(_corners)} found)")
            _axis.axis("off")
        plt.tight_layout()
        _display = plt.gcf()
    except Exception as error:
        _display = mo.md(f"Problem 2(c) is not ready: {error}")
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Short Answer Question: Shi–Tomasi corners.** Using the Shi–Tomasi detector covered in class, what would you expect to change if you replaced Harris in this experiment? Compare the two corner scores, explain whether you would expect the same four square corners to be detected, and discuss whether the Harris threshold could be reused unchanged. No implementation is required.

    **Show all steps. Answer in the Markdown cell below using LaTeX, or handwrite your solution in `hw2.pdf`, clearly labeled with the problem number. If you use the PDF, write "See hw2.pdf, Problem [number]" in the Markdown answer cell.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to 2(c): Shi–Tomasi comparison.**

    *Double-click this cell to write your answer.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(d) Laplacian of Gaussian (LoG)

    The Laplacian of Gaussians can be used for detecting the location of "blobs" in images. In this problem, we will apply the Laplacian of Gaussian (LoG) operator at the same $\sigma \in \{1, 2, 4, 8\}$ values used for Harris.

    Complete `log_response(image, sigma)` to compute the Laplacian of Gaussian response. First smooth `image` using your `gaussian_smooth(image, sigma)` function, then apply the Homework 1 Laplacian kernel using `convolve_same`. Return a 2D response array with the same shape as the input image.
    """)
    return


@app.function
def log_response(image, sigma):
    """Return the Laplacian of Gaussian response of image at scale sigma."""
    laplacian = homework1_definitions()["laplacian"]

    # EDIT BELOW
    response = ...

    return response


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Complete `local_extrema(response, fraction)` to find strong local maxima and minima. Compare absolute response values, skipping the outermost row and column. Include a pixel when both conditions hold:

    - Its magnitude is at least that of all 8 neighbours. Include ties, allowing a tolerance of $10^{-9}\max|\text{response}|$.
    - Its magnitude is at least `fraction` times the largest magnitude anywhere in the response.

    Return an integer array of `(row, column)` coordinates with shape `(N, 2)`, including shape `(0, 2)` when no pixels qualify. Choose a default `fraction` between 0 and 1.
    """)
    return


@app.function
# EDIT BELOW: choose a default fraction between 0 and 1.
def local_extrema(response, fraction=...):
    """Return (row, column) locations of the strong local extrema of response."""
    import numpy as np

    magnitude = np.abs(np.asarray(response, dtype=float))
    extrema = []

    # EDIT BELOW: append (row, column) for every pixel meeting the conditions above.

    return np.asarray(extrema, dtype=int).reshape(-1, 2)


@app.cell(hide_code=True)
def _(image_1, mo, np, plt, sigmas, smoothing_error):
    try:
        if smoothing_error:
            raise RuntimeError(smoothing_error)
        log_by_sigma = {sigma: log_response(image_1, sigma) for sigma in sigmas}
        _figure, _axes = plt.subplots(1, 4, figsize=(13, 3.6))
        for _axis, _sigma in zip(_axes, sigmas):
            _response = log_by_sigma[_sigma]
            _extrema = local_extrema(_response)
            _limit = float(np.abs(_response).max()) or 1.0
            _axis.imshow(_response, cmap="RdBu", vmin=-_limit, vmax=_limit)
            _axis.scatter(_extrema[:, 1], _extrema[:, 0], c="k", marker="x", s=30)
            _axis.set_title(f"LoG, σ = {_sigma}")
            _axis.axis("off")
        plt.tight_layout()
        _display = plt.gcf()
    except Exception as error:
        log_by_sigma = None
        _display = mo.md(f"Problem 2(d) is not ready: {error}")
    _display
    return (log_by_sigma,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(e) Difference of Gaussian (DoG)

    We can approximate the LoG using a DoG.

    To do this, we apply the Difference of Gaussian (DoG) operator using the same $\sigma$ values, with $k \approx 1.4$ between the two smoothing scales. Complete `dog_response(image, sigma, k=1.4)` using your `gaussian_smooth` function. Smooth the image at scales $k\sigma$ and $\sigma$, then subtract the smaller-scale result from the larger-scale result:

    $$D_{\sigma,k} = G_{k\sigma} * I - G_\sigma * I.$$

    Return a 2D response array with the same shape as the input image.
    """)
    return


@app.function
def dog_response(image, sigma, k=1.4):
    """Return the Difference of Gaussian response of image at scale sigma."""
    # EDIT BELOW
    response = ...

    return response


@app.cell(hide_code=True)
def _(image_1, log_by_sigma, mo, np, plt, sigmas, smoothing_error):
    try:
        if smoothing_error:
            raise RuntimeError(smoothing_error)
        if log_by_sigma is None:
            raise RuntimeError("Complete the LoG response and extrema functions first.")
        _dog_by_sigma = {sigma: dog_response(image_1, sigma) for sigma in sigmas}
        _figure, _axes = plt.subplots(2, len(sigmas), figsize=(13, 7.2), squeeze=False)
        for _row, (_method, _responses) in enumerate((
            ("LoG", log_by_sigma), ("DoG", _dog_by_sigma)
        )):
            for _axis, _sigma in zip(_axes[_row], sigmas):
                _response = _responses[_sigma]
                _extrema = local_extrema(_response)
                _limit = float(np.abs(_response).max()) or 1.0
                _axis.imshow(_response, cmap="RdBu", vmin=-_limit, vmax=_limit)
                _axis.scatter(_extrema[:, 1], _extrema[:, 0], c="k", marker="x", s=30)
                _axis.set_title(f"{_method}, σ = {_sigma} ({len(_extrema)} extrema)")
                _axis.axis("off")
        _figure.tight_layout()
        _display = _figure
    except Exception as error:
        _display = mo.md(f"Problem 2(e) is not ready: {error}")
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **In the answer cell:**

    Compare and give explaination for the locations and counts of the DoG and LoG extrema at each scale using the plots above. Both rows use your `local_extrema` function with the same default `fraction`.

    **Show all steps. Answer in the Markdown cell below using LaTeX, or handwrite your solution in `hw2.pdf`, clearly labeled with the problem number. If you use the PDF, write "See hw2.pdf, Problem [number]" in the Markdown answer cell.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to 2(e).**

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 3: Multi-Scale Blob and Corner Detection (Image 2)

    Consider the 512×512 Image 2 from 2(a), with four black squares of different fixed sizes. This problem will help you understand the relationship between image pyramids, the Gaussian scale-space, and blob or corner detection using the Laplacian of Gaussian (LoG) and the Difference of Gaussian (DoG). Reuse your functions from Problems 1 and 2.

    ## 3(a) Gaussian Smoothing and Pyramid Construction

    First, complete `downsample(image, factor)` to keep every `factor`-th row and column, starting at row 0 and column 0. For example, factor 2 turns a 512×512 image into a 256×256 image.
    """)
    return


@app.function
def downsample(image, factor):
    """Keep every factor-th row and column of image, starting at [0, 0]."""
    import numpy as np

    # EDIT BELOW
    sampled = ...

    return sampled


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, implement `build_scale_pyramid(image, sigmas)` to return `(smoothed, levels)`, two lists in the same order as `sigmas`: `smoothed[i]` is `gaussian_smooth(image, sigmas[i])`, and `levels[i]` is `smoothed[i]` downsampled by `int(sigmas[i])`. The preview below displays the smoothed images (top) and the pyramid levels (bottom).
    """)
    return


@app.function
def build_scale_pyramid(image, sigmas):
    """Return (smoothed, levels): full-resolution and downsampled images per sigma."""
    # EDIT BELOW
    smoothed = ...
    levels = ...

    return smoothed, levels


@app.cell(hide_code=True)
def _(image_2, mo, plt, sigmas):
    try:
        image_2_smoothed, image_2_levels = build_scale_pyramid(image_2, sigmas)
        pyramid_error = None
        _figure, _axes = plt.subplots(2, len(sigmas), figsize=(13, 6.2))
        for _column, _sigma in enumerate(sigmas):
            _axes[0, _column].imshow(image_2_smoothed[_column], cmap="gray", vmin=0, vmax=255)
            _axes[0, _column].set_title(f"Smoothed, σ = {_sigma} (512×512)")
            _level = image_2_levels[_column]
            _axes[1, _column].imshow(
                _level, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
            )
            _axes[1, _column].set_title(f"Level {_column}: ÷{_sigma} ({_level.shape[0]}×{_level.shape[1]})")
        for _axis in _axes.ravel():
            _axis.axis("off")
        plt.tight_layout()
        _display = plt.gcf()
    except Exception as error:
        image_2_smoothed = image_2_levels = None
        pyramid_error = str(error)
        _display = mo.md(f"Problem 3(a) is not ready: {error}")
    _display
    return image_2_levels, image_2_smoothed, pyramid_error


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **In the answer cell:** comment briefly on the visual differences between levels (e.g., how edges and corners blur or disappear).

    **Show all steps. Answer in the Markdown cell below using LaTeX, or handwrite your solution in `hw2.pdf`, clearly labeled with the problem number. If you use the PDF, write "See hw2.pdf, Problem [number]" in the Markdown answer cell.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to 3(a).**

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3(b) Harris Corners at Multiple Scales

    Using the images generated in 3(a), compute the Harris corner response at each smoothing scale $\sigma \in \{1, 2, 4, 8\}$. We'll compare two approaches:

    1. Harris detector on the full-resolution, smoothed images (pure scale-space analysis).
    2. Harris detector on the corresponding downsampled images (pyramid representation).

    Complete `image_2_corners(image, threshold, max_corners)` by calling your Homework 1 `harris_corners`, handling the white image border as in 2(c). Choose default values and use them for both approaches. The preview below runs both approaches and lists the number of corners and their median Harris response $R$.
    """)
    return


@app.function
def image_2_corners(image, threshold=..., max_corners=...):
    """Return the Harris corners of a smoothed Image 2 or pyramid level as (row, column)."""
    # EDIT BELOW: call your hw1 harris_corners function using threshold and max_corners.
    corners = ...

    return corners


@app.cell(hide_code=True)
def _(
    convolve2d_full,
    image_2_levels,
    image_2_smoothed,
    mo,
    np,
    plt,
    pyramid_error,
    sigmas,
):
    def _harris_response(image, k=0.04):
        # The CRF of Homework 1, Problem 3, used to report corner strengths.
        image = np.asarray(image, dtype=float)
        kernel = np.array([[0.5, 0.0, -0.5]])
        gradient_x = convolve2d_full(image, kernel)[:, 1 : 1 + image.shape[1]]
        gradient_y = convolve2d_full(image, kernel.T)[1 : 1 + image.shape[0], :]
        box = np.ones((3, 3)) / 9

        def _average(values):
            return convolve2d_full(values, box)[1 : 1 + image.shape[0], 1 : 1 + image.shape[1]]

        a_xx, a_xy, a_yy = _average(gradient_x**2), _average(gradient_x * gradient_y), _average(gradient_y**2)
        return a_xx * a_yy - a_xy**2 - k * (a_xx + a_yy) ** 2

    try:
        if pyramid_error:
            raise RuntimeError(pyramid_error)
        # (i) corners are in full-resolution pixels; (ii) corners are in level pixels.
        image_2_corners_full = {}
        image_2_corners_pyramid = {}
        _lines = [
            "| σ | Corners (i) | Corners (ii) | Median R (i) | Median R (ii) |",
            "|---|---|---|---|---|",
        ]
        _figure, _axes = plt.subplots(2, len(sigmas), figsize=(14, 7.4), squeeze=False)
        for _index, _sigma in enumerate(sigmas):
            _strengths = []
            for _row, (_image, _store, _name) in enumerate((
                (image_2_smoothed[_index], image_2_corners_full, "(i) full resolution"),
                (image_2_levels[_index], image_2_corners_pyramid, "(ii) pyramid"),
            )):
                _points = np.asarray(image_2_corners(_image), dtype=int).reshape(-1, 2)
                _store[_sigma] = _points
                _values = _harris_response(_image)[_points[:, 0], _points[:, 1]]
                _strengths.append(float(np.median(_values)) if len(_values) else float("nan"))
                _axis = _axes[_row, _index]
                _axis.imshow(_image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                _axis.scatter(_points[:, 1], _points[:, 0], s=18, c="r")
                _axis.set_title(f"{_name}, σ = {_sigma} ({len(_points)} found)", fontsize=9)
                _axis.axis("off")
            _lines.append(
                f"| {_sigma} | {len(image_2_corners_full[_sigma])} | {len(image_2_corners_pyramid[_sigma])} "
                f"| {_strengths[0]:.3g} | {_strengths[1]:.3g} |"
            )
        plt.tight_layout()
        _display = mo.vstack([plt.gcf(), mo.md("\n".join(_lines))])
    except Exception as error:
        image_2_corners_full = image_2_corners_pyramid = None
        _display = mo.md(f"Problem 3(b) is not ready: {error}")
    _display
    return image_2_corners_full, image_2_corners_pyramid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **In the answer cell:** discuss how corner locations, counts, and strengths vary between these two approaches.

    **Show all steps. Answer in the Markdown cell below using LaTeX, or handwrite your solution in `hw2.pdf`, clearly labeled with the problem number. If you use the PDF, write "See hw2.pdf, Problem [number]" in the Markdown answer cell.**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to 3(b).**

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3(c) Laplacian-of-Gaussian (LoG) on the Gaussian Pyramid

    Using the Gaussian pyramid constructed in 3(a), compute the Laplacian-of-Gaussian response for each pyramid level. Each level corresponds to a downsampled version of the image that has been smoothed with a Gaussian of increasing $\sigma$. For each level, evaluate the LoG at the same nominal smoothing scales $\sigma \in \{1, 2, 4, 8\}$, relative to that level's resolution. Detect local extrema (both maxima and minima) that correspond to blob centres. Visualize the strongest responses as circles centred at the detected locations, with radii proportional to the effective $\sigma$ in the original image coordinates.

    > **Note:** The Laplacian-of-Gaussian (LoG) response can be computed in two conceptually different ways:
    >
    > 1. On the original full-resolution image at multiple smoothing scales $\sigma$ (pure scale-space analysis), where only the filter scale changes, not the image sampling.
    > 2. On the Gaussian pyramid, where each level is both smoothed and downsampled. This approach, used in practice by methods such as SIFT, provides a computationally efficient approximation to the continuous scale-space while capturing the effects of both scale and resolution.
    >
    > In this exercise, the LoG is applied to each downsampled pyramid level to explore how feature detection behaves under changes in both image scale and sampling resolution.

    First, compute `effective_sigma(factor, sigma)`. A level downsampled by $f$ was already smoothed at $\sigma = f$ in original pixels, and filtering it at $\sigma$ level pixels adds a blur of $f\sigma$ original pixels. Gaussian variances add, so the effective scale in original-image pixels is

    $$
    \sigma_{\mathrm{eff}} = \sqrt{f^2 + (f\sigma)^2}.
    $$
    """)
    return


@app.function
def effective_sigma(factor, sigma):
    """Return the scale, in original pixels, of sigma applied to the level downsampled by factor."""
    import numpy as np

    # EDIT BELOW
    sigma_eff = ...

    return sigma_eff


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, complete `pyramid_extrema(levels, factors, sigmas, method="log")`. For every level (with downsampling factor `f`) and every $\sigma$ in `sigmas`, compute the response with `log_response(level, sigma)`, or with `dog_response(level, sigma)` when `method="dog"` (used in 3(d)). Find its strongest local extrema, both maxima and minima, with your `local_extrema` from 2(d). For every extremum, append the row `(row, column, f, sigma_eff)`, where `(row, column)` is in level pixels. Return a floating-point array of shape `(N, 4)`.

    The preview cell below draws each LoG extremum on Image 2 as a circle of radius $\sigma_{\mathrm{eff}}$ centred at $(f \cdot \text{row}, f \cdot \text{column})$.
    """)
    return


@app.function
def pyramid_extrema(levels, factors, sigmas, method="log"):
    """Return (row, column, factor, sigma_eff) for the extrema of every level and scale."""
    import numpy as np

    extrema = []
    # EDIT BELOW

    return np.asarray(extrema, dtype=float).reshape(-1, 4)


@app.cell(hide_code=True)
def _(image_2, image_2_levels, mo, plt, pyramid_error, sigmas):
    try:
        if pyramid_error:
            raise RuntimeError(pyramid_error)
        image_2_blobs = {
            method: pyramid_extrema(image_2_levels, sigmas, sigmas, method=method)
            for method in ("log", "dog")
        }
        _figure, _axes = plt.subplots(2, 2, figsize=(10, 10))
        for _index, (_axis, _factor) in enumerate(zip(_axes.flat, sigmas)):
            _axis.imshow(image_2, cmap="gray", vmin=0, vmax=255)
            _count = 0
            for _row, _column, _level_factor, _sigma_eff in image_2_blobs["log"]:
                if _level_factor != _factor:
                    continue
                _axis.add_patch(plt.Circle(
                    (_factor * _column, _factor * _row), _sigma_eff,
                    fill=False, color=f"C{_index}", linewidth=1.2,
                ))
                _count += 1
            _axis.set_title(f"Level f = {_factor}: {_count} LoG extrema", color=f"C{_index}")
            _axis.axis("off")
        _figure.tight_layout()
        _display = _figure
    except Exception as error:
        image_2_blobs = None
        _display = mo.md(f"Problem 3(c) is not ready: {error}")
    _display
    return (image_2_blobs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3(d) Mapping and Visualization

    Map the detected feature points back to the coordinate system of the original image. The visualization code below is provided for reference and calls your `to_original` function to overlay the detections.

    Complete `to_original(points, factor)` to map `(N, 2)` `(row, column)` points in a level downsampled by `factor` to original-image pixels. For example, `(10, 15)` at factor 2 maps to `(20, 30)`.
    """)
    return


@app.function
def to_original(points, factor):
    """Map (row, column) points in a level downsampled by factor to the original image."""
    import numpy as np

    # EDIT BELOW
    mapped = ...

    return mapped


@app.cell
def _(plt):
    # For reference only: this visualization is provided for you.
    def plot_detections(image, corners_full, corners_pyramid, blobs, c=1.0):
        """Show corners and blobs on the original image."""
        import numpy as np

        figure, axis = plt.subplots()
        axis.imshow(image, cmap="gray")

        # Full-resolution corners already use original-image coordinates (red dots).
        for points in corners_full.values():
            points = np.asarray(points).reshape(-1, 2)
            axis.scatter(points[:, 1], points[:, 0], color="red")

        # Map pyramid corners back to the original image (blue crosses).
        for factor, points in corners_pyramid.items():
            points = to_original(points, factor)
            axis.scatter(points[:, 1], points[:, 0], color="blue", marker="x")

        # Draw each blob as a circle: green for LoG, orange for DoG.
        for method, color in (("log", "green"), ("dog", "orange")):
            for row, column, factor, scale in blobs[method]:
                row, column = to_original([(row, column)], factor)[0]
                circle = plt.Circle((column, row), c * scale, fill=False, color=color)
                axis.add_patch(circle)

        return figure

    return (plot_detections,)


@app.cell(hide_code=True)
def _(
    image_2,
    image_2_blobs,
    image_2_corners_full,
    image_2_corners_pyramid,
    mo,
    plot_detections,
):
    try:
        if image_2_corners_full is None or image_2_blobs is None:
            raise RuntimeError("Complete 3(b) and 3(c) first.")
        _display = plot_detections(image_2, image_2_corners_full, image_2_corners_pyramid, image_2_blobs)
    except Exception as error:
        _display = mo.md(f"Problem 3(d) is not ready: {error}")
    _display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Submission

    **Grading disclaimer:** A score of **70/70 on the autograder does not guarantee 100% on the assignment**. The assignment also includes manually graded components that contribute to your final grade.

    1. Export this completed notebook as a Python file named exactly **`hw2.py`**.
    2. Put your Homework 1 Python file, named exactly **`hw1.py`**, in the same folder.
    3. If you handwrote any answers, include a PDF named exactly **`hw2.pdf`** in that folder. **Clearly label every problem and subpart in the PDF, and write "See hw2.pdf, Problem [number]" in each corresponding Markdown answer cell. Illegible work will receive a zero.**
    4. Put these files together in **`hw2.zip`** and upload it to **Gradescope**.
    """)
    return


if __name__ == "__main__":
    app.run()
