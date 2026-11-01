import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    # Everything imported here is available to every @app.function below.
    from io import BytesIO
    from pathlib import Path
    from urllib.request import Request, urlopen

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

    # Fixed SIFT parameters used throughout this homework.
    SIGMA_0 = 1.6  # blur of the first level of every octave
    S = 3  # scales per octave  ->  S + 3 Gaussian images, S + 2 DoG images
    NUM_OCTAVES = 4
    K = 2 ** (1 / S)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Homework 3: SIFT

    ## Editing

    Marimo notebooks are just Python files, where each "cell" is a function. You can either edit here, in [Molab](https://molab.marimo.io/), or directly in Python.

    ### Option 1 - Molab [Recommended]

    This is the most recommended option. Take this notebook (.py file), upload it to Molab, and start working (the images are downloaded automatically). Once you are finished making your edits, click "Export" and select "Python" to export it as a notebook source.

    ### Option 2 - Local Marimo

    To work locally, first create a python project and place this file `hw3.py` within it. Then, install marimo using your favorite package manager (`uv add marimo numpy scipy matplotlib pillow`). Finally, launch this notebook by typing in your terminal `uv run marimo edit hw3.py` and a new tab should open in your browser.

    ### Option 3 - Regular Python [Not Recommended]

    It is theoretically possible to edit the `.py` file directly (although we don't recommend it). If you choose this route, make sure that your Python/Markdown renders properly in Marimo.

    ## Rules

    * Implement every function marked **EDIT BELOW**. Keep the function names and signatures exactly as given: the autograder calls them one by one with its own inputs.
    * **Do not use OpenCV** (`cv2`) or any ready-made SIFT/feature-detector implementation. You may use `numpy` and `scipy.ndimage` (`gaussian_filter`, `maximum_filter`, `minimum_filter`, `convolve`, `map_coordinates`, ...). These are already imported in the setup cell at the top.
    * Coordinates are always `(row, col)` and angles are always in **degrees in [0, 360)**.
    * Images are grayscale floats in the range **[0, 1]**.
    """)
    return


@app.function(hide_code=True)
def load_grayscale(filename, fallback_url=None):
    """Return the image as a grayscale float array in [0, 1].

    Looks for `filename` next to this notebook first; if it is not there and a
    fallback URL is given, downloads it from there instead.
    """
    path = Path(filename)
    if path.exists():
        image = Image.open(path)
    elif fallback_url:
        request = Request(fallback_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            image = Image.open(BytesIO(response.read()))
    else:
        raise FileNotFoundError(f"{filename} not found next to this notebook.")
    return np.asarray(image.convert("L"), dtype=float) / 255.0


@app.cell(hide_code=True)
def _():
    # The images are downloaded from the course data repository; a local copy
    # named imageA.png / imageB.png next to the notebook is used instead if present.
    IMAGE_A_URL = "https://github.com/hp6318/ece181-hw3-data/blob/main/imgA.png?raw=true"
    IMAGE_B_URL = "https://github.com/hp6318/ece181-hw3-data/blob/main/imgB.png?raw=true"
    image_a = load_grayscale("imageA.png", IMAGE_A_URL)
    image_b = load_grayscale("imageB.png", IMAGE_B_URL)
    return image_a, image_b


@app.cell(hide_code=True)
def _(image_a, image_b):
    mo.hstack([mo.image(image_a, width=360), mo.image(image_b, width=360)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Overview

    SIFT (Scale-Invariant Feature Transform, Lowe 2004) finds *keypoints* that can be re-found when the image is scaled, rotated, or seen from a slightly different viewpoint, and describes each one with a 128-number vector. You will build it step by step on two photos of the same graffiti wall, **Image A** and **Image B**, and finish by drawing lines between the matched keypoints.

    | Problem | Function(s) you write | What it does |
    |---|---|---|
    | 1 | `build_gaussian_octaves`, `build_dog_octaves` | Gaussian scale space and Difference-of-Gaussians |
    | 2 | `find_extrema` | 3×3×3 extrema in the DoG |
    | 3 | `filter_keypoints` | Remove low-contrast and edge-like candidates |
    | 4 | `image_gradients`, `assign_orientations` | Gradient magnitude/direction and the dominant orientation |
    | 5 | `compute_descriptor` | The 128-D descriptor |
    | 6 | `match_descriptors` | Nearest-neighbour matching with the ratio test |
    | 7 | `sift` | Put everything together |
    | 8 | (written) | Show and discuss your matches |
    | 9 | (written) | Pen-and-paper questions |

    Every problem ends with a hidden cell that runs your function and shows a picture, so you can check each stage before moving on. The handout PDF has the full description of every step; this notebook only repeats the exact conventions you must follow.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Problem 1: Gaussian Scale Space and DoG

    Build `NUM_OCTAVES = 4` octaves. Within one octave every level has the **same size**; level $i$ ($i = 0, \dots, S+2$) is the input blurred to an absolute scale

    $$\sigma_i = \sigma_0\, k^{i}, \qquad k = 2^{1/S}, \quad \sigma_0 = 1.6,\quad S = 3 .$$

    * **Level 0** of the first octave is `gaussian_filter(image, SIGMA_0)`.
    * **Level $i$** is obtained from level $i-1$ by blurring with the *extra* amount $\sigma_{\text{inc}} = \sqrt{\sigma_i^2 - \sigma_{i-1}^2}$ (blurring twice adds variances).
    * **Next octave**: take level $S$ of the current octave (its blur is $2\sigma_0$), keep every second row and column (`[::2, ::2]`), and treat it as the already-blurred level 0 of the next octave. Then continue with the same $\sigma_{\text{inc}}$ steps.

    Return a list of `NUM_OCTAVES` arrays, octave $o$ having shape `(S + 3, H / 2**o, W / 2**o)`.

    The Difference-of-Gaussians of an octave is simply $D_i = L_{i+1} - L_i$ for $i = 0, \dots, S+1$, giving an array of shape `(S + 2, H_o, W_o)` per octave.
    """)
    return


@app.function
def build_gaussian_octaves(image, num_octaves=NUM_OCTAVES, s=S, sigma0=SIGMA_0):
    """Return a list of num_octaves arrays; octave o has shape (s + 3, H / 2**o, W / 2**o)."""
    image = np.asarray(image, dtype=float)
    k = 2 ** (1 / s)
    sigmas = sigma0 * k ** np.arange(s + 3)  # absolute blur of every level
    sig_inc = np.sqrt(sigmas[1:]**2 - sigmas[:-1]**2)
    octaves = []
    # EDIT BELOW: blur `image` with sigmas[0] to get level 0, then blur each
    # level by sigma_inc = sqrt(sigmas[i]**2 - sigmas[i-1]**2) to get the next one.
    base = gaussian_filter(image, SIGMA_0)
    for o in range(NUM_OCTAVES):
        levels = [base]
        for i in range(1, S + 3):
            levels.append(gaussian_filter(levels[-1], sig_inc[i-1]))
        octaves.append(np.stack(levels))
        # Downsample level `s` with [::2, ::2] to start the next octave. Append every
        base = levels[S][::2, ::2]
    # octave (an array of shape (s + 3, H_o, W_o)) to `octaves`.
    ...
    return octaves


@app.function
def build_dog_octaves(gaussian_octaves):
    """Return a list of arrays; octave o has shape (s + 2, H_o, W_o) with D[i] = L[i+1] - L[i]."""
    # EDIT BELOW: for every octave return L[1:] - L[:-1].
    ...
    return [oct[1:] - oct[:-1] for oct in gaussian_octaves]


@app.cell(hide_code=True)
def _(image_a):
    try:
        gaussian_octaves_a = build_gaussian_octaves(image_a)
        dog_octaves_a = build_dog_octaves(gaussian_octaves_a)
        if len(gaussian_octaves_a) != NUM_OCTAVES or len(dog_octaves_a) != NUM_OCTAVES:
            raise ValueError(f"expected {NUM_OCTAVES} octaves, got {len(gaussian_octaves_a)} Gaussian and {len(dog_octaves_a)} DoG")
        scale_space_error = None
    except Exception as error:
        gaussian_octaves_a = dog_octaves_a = None
        scale_space_error = str(error)
    return dog_octaves_a, gaussian_octaves_a, scale_space_error


@app.cell(hide_code=True)
def _(dog_octaves_a, gaussian_octaves_a, scale_space_error):
    try:
        if scale_space_error:
            mo.output.replace(mo.md(f"Problem 1 is not ready: {scale_space_error}"))
        else:
            _shapes = ", ".join(str(o.shape) for o in gaussian_octaves_a)
            _crop = (slice(150, 350), slice(250, 450))
            _gauss = [mo.image(level[_crop], width=120) for level in gaussian_octaves_a[0]]
            _lim = max(abs(dog_octaves_a[0][:, _crop[0], _crop[1]]).max(), 1e-6)
            _dogs = [
                mo.image((level[_crop] + _lim) / (2 * _lim), width=120)
                for level in dog_octaves_a[0]
            ]
            mo.output.replace(mo.vstack([
                mo.md(f"Octave shapes: {_shapes}  \nTop row: the 6 Gaussian levels of octave 0 (crop). Bottom row: the 5 DoG levels (gray = 0)."),
                mo.hstack(_gauss),
                mo.hstack(_dogs),
            ]))
    except Exception as error:
        mo.output.replace(mo.md(f"Could not display Problem 1: {error}"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Problem 2: Scale-Space Extrema

    A pixel of the DoG stack is a candidate keypoint if its value is **strictly larger than all 26 neighbours** (8 in its own level, 9 in the level below, 9 in the level above) or **strictly smaller than all 26**.

    Conventions:

    * Only levels $1, \dots, S$ can hold extrema (levels $0$ and $S+1$ have no neighbour on one side).
    * Skip the one-pixel image border (no full 3×3 neighbourhood there).
    * Return an integer array of shape `(N, 3)` with rows `(level, row, col)`.

    Hint: a double `for` loop over every pixel is far too slow in Python. Use `maximum_filter` / `minimum_filter` from `scipy.ndimage` with a 3×3×3 footprint whose centre is `False`, or compare 26 shifted copies of the array with NumPy slicing. `np.argwhere` turns a boolean array into a list of indices.
    """)
    return


@app.function
def find_extrema(dog_octave):
    """Return an (N, 3) integer array of (level, row, col) strict 26-neighbour extrema."""
    dog = np.asarray(dog_octave, dtype=float)
    # EDIT BELOW: compare each DoG value with its 26 neighbours (strict > or <).
    # Ignore levels 0 and s + 1 and the one-pixel image border.
    # Return np.argwhere(is_extremum), an (N, 3) array of (level, row, col).
    ...

    # 3x3x3 footprint, all True except the center
    footprint = np.ones((3,3,3), dtype=bool)
    footprint[1, 1, 1] = False

    # Step 1-2: max / min over the 26 neighbours of every pixel
    neigh_max = maximum_filter(dog, footprint=footprint)
    neigh_min = minimum_filter(dog, footprint=footprint)

    # Step 3: strict comparison
    candidate = (dog > neigh_max) | (dog < neigh_min)

    # Step 4: keep only valid levels and drop the 1-pixel border
    is_extremum = np.zeros(dog.shape, dtype=bool)
    is_extremum[1:-1, 1:-1, 1:-1] = candidate[1:-1, 1:-1, 1:-1]

    # Step 5: indices as (level, row, col)
    return np.argwhere(is_extremum)


@app.cell(hide_code=True)
def _(dog_octaves_a, image_a, scale_space_error):
    try:
        if scale_space_error:
            raise RuntimeError("finish Problem 1 first")
        candidates_a = find_extrema(dog_octaves_a[0])
        _fig, _ax = plt.subplots(figsize=(8, 6.4))
        _ax.imshow(image_a, cmap="gray")
        _ax.scatter(candidates_a[:, 2], candidates_a[:, 1], s=4, c="r")
        _ax.set_title(f"{len(candidates_a)} candidate extrema in octave 0")
        _ax.axis("off")
        mo.output.replace(_fig)
    except Exception as error:
        candidates_a = None
        mo.output.replace(mo.md(f"Problem 2 is not ready: {error}"))
    return (candidates_a,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Problem 3: Removing Unstable Keypoints

    Two tests, applied to every candidate `(level, row, col)` using the DoG level $D$ it was found in:

    **(a) Contrast.** Discard the candidate if $|D(\text{row}, \text{col})| < 0.03$ (images are in $[0,1]$).

    **(b) Edges.** Estimate the 2×2 Hessian of $D$ at the candidate with finite differences:

    $$D_{xx} = D(r, c{+}1) + D(r, c{-}1) - 2D(r,c), \qquad D_{yy} = D(r{+}1, c) + D(r{-}1, c) - 2D(r,c),$$
    $$D_{xy} = \tfrac{1}{4}\bigl[D(r{+}1, c{+}1) - D(r{+}1, c{-}1) - D(r{-}1, c{+}1) + D(r{-}1, c{-}1)\bigr].$$

    With $\operatorname{Tr} = D_{xx} + D_{yy}$ and $\operatorname{Det} = D_{xx}D_{yy} - D_{xy}^2$, discard the candidate if

    $$\operatorname{Det} \le 0 \qquad\text{or}\qquad \frac{\operatorname{Tr}^2}{\operatorname{Det}} > \frac{(r+1)^2}{r}, \quad r = 10 .$$

    Return the surviving candidates, as an `(M, 3)` integer array in the same `(level, row, col)` format and the same order.
    """)
    return


@app.function
def filter_keypoints(dog_octave, candidates, contrast_threshold=0.03, edge_ratio=10):
    """Return the (M, 3) subset of candidates that pass the contrast and edge tests."""
    dog = np.asarray(dog_octave, dtype=float)
    candidates = np.asarray(candidates, dtype=int).reshape(-1, 3)
    kept = []
    # EDIT BELOW: for each (level, r, c) keep it only if |D| >= contrast_threshold
    # and the Hessian test passes; append the survivors to `kept`.
    ...
    limit = (edge_ratio + 1) ** 2 / edge_ratio

    for level, r, c, in candidates:
        D = dog[level]

        # contast test
        if abs(D[r, c]) < contrast_threshold:
            continue

        # (b) Hessian via finite differences
        dxx = D[r, c + 1] + D[r, c - 1] - 2 * D[r, c]
        dyy = D[r + 1, c] + D[r - 1, c] - 2 * D[r, c]
        dxy = 0.25 * (D[r + 1, c + 1] - D[r + 1, c - 1]
                      - D[r - 1, c + 1] + D[r - 1, c - 1])

        tr = dxx + dyy
        det = dxx * dyy - dxy ** 2

        if det <= 0:
            continue
        if tr ** 2 / det > limit:
            continue

        kept.append((level, r, c))
        
    return np.array(kept, dtype=int).reshape(-1, 3)


@app.cell(hide_code=True)
def _(candidates_a, dog_octaves_a, image_a):
    try:
        if candidates_a is None:
            raise RuntimeError("finish Problem 2 first")
        keypoints_a0 = filter_keypoints(dog_octaves_a[0], candidates_a)
        _fig, _ax = plt.subplots(figsize=(8, 6.4))
        _ax.imshow(image_a, cmap="gray")
        _ax.scatter(keypoints_a0[:, 2], keypoints_a0[:, 1], s=6, c="lime")
        _ax.set_title(f"{len(keypoints_a0)} of {len(candidates_a)} candidates kept in octave 0")
        _ax.axis("off")
        mo.output.replace(_fig)
    except Exception as error:
        keypoints_a0 = None
        mo.output.replace(mo.md(f"Problem 3 is not ready: {error}"))
    return (keypoints_a0,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Problem 4: Orientation Assignment

    **(a) Gradients.** For a Gaussian image $L$, use central differences

    $$d_x = L(r, c{+}1) - L(r, c{-}1), \qquad d_y = L(r{+}1, c) - L(r{-}1, c),$$
    $$m(r,c) = \sqrt{d_x^2 + d_y^2}, \qquad \theta(r,c) = \operatorname{atan2}(d_y, d_x) \text{ in degrees, wrapped to } [0, 360).$$

    `image_gradients` returns the two arrays `(magnitude, orientation)`, the same shape as $L$. Set the magnitude of the one-pixel border (where a central difference does not exist) to 0. The orientation on that border can be any value in $[0, 360)$ (0 is fine): its weight is always 0, so it never affects a histogram.

    **(b) Orientation histogram.** For a keypoint at `(r, c)` with scale $\sigma$ (the scale of its DoG level, $\sigma = \sigma_0 k^{\text{level}}$) use $\sigma_w = 1.5\sigma$ and the square window of pixels with $|r' - r| \le R$ and $|c' - c| \le R$, $R = \operatorname{round}(3\sigma_w)$, clipped to the image. Each pixel adds the weight

    $$w = m(r', c')\, \exp\!\Bigl(-\frac{(r'-r)^2 + (c'-c)^2}{2\sigma_w^2}\Bigr)$$

    to bin $\lfloor \theta(r',c') / 10^\circ \rfloor$ of a 36-bin histogram.

    **(c) Peaks.** Every bin $b$ whose value is at least **80 %** of the largest bin produces a keypoint with orientation $\theta_k = (b + 0.5)\cdot 10^\circ$ (the bin centre). So one input keypoint may give several output keypoints.

    `assign_orientations` takes the gradient arrays of the Gaussian image, an `(N, 2)` array of `(row, col)` keypoints and the `(N,)` array of their scales, and returns an `(M, 4)` float array with rows `(row, col, sigma, theta)`.
    """)
    return


@app.function
def image_gradients(image):
    image = np.asarray(image, dtype=float)
    dx = np.zeros_like(image)
    dy = np.zeros_like(image)

    dx[1:-1, 1:-1] = image[1:-1, 2:] - image[1:-1, :-2]   # right neighbour minus left
    dy[1:-1, 1:-1] = image[2:, 1:-1] - image[:-2, 1:-1]   # below neighbour minus above

    magnitude = np.hypot(dx, dy)
    orientation = np.degrees(np.arctan2(dy, dx)) % 360
    return magnitude, orientation


@app.function
def assign_orientations(magnitude, orientation, keypoints, sigmas, num_bins=36, peak_ratio=0.8):
    magnitude = np.asarray(magnitude, dtype=float)
    orientation = np.asarray(orientation, dtype=float)
    keypoints = np.asarray(keypoints, dtype=int).reshape(-1, 2)
    sigmas = np.asarray(sigmas, dtype=float).reshape(-1)
    height, width = magnitude.shape
    bin_width = 360 / num_bins
    results = []

    for (r, c), sigma in zip(keypoints, sigmas):
        sigma_w = 1.5 * sigma
        R = int(round(3 * sigma_w))

        # 1-2. window, clipped to the image
        r0, r1 = max(r - R, 0), min(r + R, height - 1)
        c0, c1 = max(c - R, 0), min(c + R, width - 1)

        # 3. weights
        hist = np.zeros(num_bins)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                dist2 = (rr - r) ** 2 + (cc - c) ** 2
                w = magnitude[rr, cc] * np.exp(-dist2 / (2 * sigma_w ** 2))

                # 4. vote
                b = int(orientation[rr, cc] // bin_width) % num_bins
                hist[b] += w

        # 5-7. peaks
        peak = hist.max()
        if peak <= 0:
            continue
        for b in range(num_bins):
            if hist[b] >= peak_ratio * peak:
                results.append((r, c, sigma, (b + 0.5) * bin_width))

    return np.array(results, dtype=float).reshape(-1, 4)


@app.cell(hide_code=True)
def _(gaussian_octaves_a, image_a, keypoints_a0):
    try:
        if keypoints_a0 is None:
            raise RuntimeError("finish Problem 3 first")
        _level = 1
        _at_level = keypoints_a0[keypoints_a0[:, 0] == _level][:, 1:]
        _magnitude, _orientation = image_gradients(gaussian_octaves_a[0][_level])
        oriented_a0 = assign_orientations(
            _magnitude, _orientation, _at_level, np.full(len(_at_level), SIGMA_0 * K**_level)
        )
        _fig, _ax = plt.subplots(figsize=(8, 6.4))
        _ax.imshow(image_a, cmap="gray")
        for _r, _c, _sigma, _theta in oriented_a0:
            _ax.add_patch(plt.Circle((_c, _r), 3 * _sigma, fill=False, color="lime", lw=0.8))
            _ax.plot(
                [_c, _c + 3 * _sigma * np.cos(np.radians(_theta))],
                [_r, _r + 3 * _sigma * np.sin(np.radians(_theta))],
                color="yellow", lw=0.8,
            )
        _ax.set_title(f"{len(oriented_a0)} oriented keypoints from {len(_at_level)} at octave 0, level {_level}")
        _ax.axis("off")
        mo.output.replace(_fig)
    except Exception as error:
        oriented_a0 = None
        mo.output.replace(mo.md(f"Problem 4 is not ready: {error}"))
    return (oriented_a0,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Problem 5: The 128-D Descriptor

    Work on the gradient arrays of the Gaussian image of the keypoint's level (never on the original image). The descriptor is a 4×4 grid of cells, each holding an 8-bin orientation histogram, over a 16×16 window that is **rotated to the keypoint orientation** $\theta_k$. Instead of resampling the image, rotate the *coordinates* of each pixel (the method used by OpenCV/VLFeat):

    1. Look at every pixel $(r', c')$ with $|r' - r| \le 12$ and $|c' - c| \le 12$ (a 25×25 square; the rotated 16×16 window always fits inside it). Skip pixels outside the image.
    2. Rotate its offset $(d_c, d_r) = (c' - c,\; r' - r)$ by $-\theta_k$:
       $$u = \cos\theta_k\, d_c + \sin\theta_k\, d_r, \qquad v = -\sin\theta_k\, d_c + \cos\theta_k\, d_r .$$
       Keep the pixel only if $-8 \le u < 8$ and $-8 \le v < 8$.
    3. Its cell is row $i = \lfloor (v + 8)/4 \rfloor$, column $j = \lfloor (u + 8)/4 \rfloor$ (both in $0..3$); its bin is $b = \lfloor \theta_{\text{rel}} / 45^\circ \rfloor$ with $\theta_{\text{rel}} = (\theta(r',c') - \theta_k) \bmod 360^\circ$.
    4. Add $w = m(r', c')\,\exp\!\bigl(-(u^2 + v^2) / (2 \cdot 8^2)\bigr)$ to `descriptor[i, j, b]`.

    Flatten the `(4, 4, 8)` array to 128 numbers in that index order (`.ravel()`), then normalise to unit length, clip every entry to at most **0.2**, and normalise again. If the vector is all zeros, return it as is.
    """)
    return


@app.function
def compute_descriptor(magnitude, orientation, row, col, theta, window=16, cells=4, num_bins=8, search_radius=12, clip=0.2):
    magnitude = np.asarray(magnitude, dtype=float)
    orientation = np.asarray(orientation, dtype=float)
    height, width = magnitude.shape
    half = window / 2  # 8
    cell_size = window / cells  # 4
    bin_width = 360 / num_bins  # 45
    sigma_w = half
    cos_t, sin_t = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    descriptor = np.zeros((cells, cells, num_bins))

    # Step 1: every pixel in the square around the keypoint, clipped to the image
    r_lo, r_hi = max(row - search_radius, 0), min(row + search_radius, height - 1)
    c_lo, c_hi = max(col - search_radius, 0), min(col + search_radius, width - 1)

    for rr in range(r_lo, r_hi + 1):
        for cc in range(c_lo, c_hi + 1):
            # Step 2: offset from the keypoint, then rotate by -theta
            dr = rr - row
            dc = cc - col
            u = cos_t * dc + sin_t * dr
            v = -sin_t * dc + cos_t * dr

            # keep only pixels inside the rotated 16x16 window
            if not (-half <= u < half and -half <= v < half):
                continue

            # Step 3: which cell and which orientation bin
            i = int((v + half) // cell_size)   # cell row, from v
            j = int((u + half) // cell_size)   # cell column, from u
            theta_rel = (orientation[rr, cc] - theta) % 360
            b = int(theta_rel // bin_width) % num_bins

            # Step 4: weighted vote
            w = magnitude[rr, cc] * np.exp(-(u * u + v * v) / (2 * sigma_w ** 2))
            descriptor[i, j, b] += w

    vector = descriptor.ravel()
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    vector = np.minimum(vector / norm, clip)
    return vector / np.linalg.norm(vector)


@app.cell(hide_code=True)
def _(gaussian_octaves_a, oriented_a0):
    try:
        if oriented_a0 is None:
            raise RuntimeError("finish Problem 4 first")
        _level_image = gaussian_octaves_a[0][1]
        _magnitude, _orientation = image_gradients(_level_image)
        _r, _c, _sigma, _theta = oriented_a0[0]
        _descriptor = compute_descriptor(_magnitude, _orientation, int(_r), int(_c), _theta)
        # Rotation check: rotate the image by 90 degrees and the keypoint with it.
        _rot_magnitude, _rot_orientation = image_gradients(np.rot90(_level_image))
        _width = _level_image.shape[1]
        _rotated = compute_descriptor(
            _rot_magnitude, _rot_orientation, int(_width - 1 - _c), int(_r), (_theta - 90) % 360
        )
        _fig, _ax = plt.subplots(figsize=(9, 2.5))
        _ax.bar(np.arange(128), _descriptor, width=1.0, label="original")
        _ax.bar(np.arange(128), -_rotated, width=1.0, label="image rotated 90°")
        _ax.set_xlim(0, 128)
        _ax.set_title(
            f"Keypoint ({int(_r)}, {int(_c)}), θ = {_theta:.0f}°.  Length = {np.linalg.norm(_descriptor):.3f} "
            f"(want 1).  Similarity to the rotated-image descriptor = {_descriptor @ _rotated:.3f} (want ≈ 1)",
            fontsize=9,
        )
        _ax.legend(loc="upper right")
        mo.output.replace(_fig)
    except Exception as error:
        mo.output.replace(mo.md(f"Problem 5 is not ready: {error}"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Problem 6: Matching

    Given descriptors `A` of shape `(N, 128)` and `B` of shape `(M, 128)`, compare every row of `A` with every row of `B` using the $\ell_1$ (Manhattan) distance $d(f_1, f_2) = \sum_i |f_{1,i} - f_{2,i}|$.

    For each descriptor in `A`, find its nearest ($d_1$) and second-nearest ($d_2$) descriptor in `B`. Keep the match only if it passes **Lowe's ratio test** $d_1 < 0.8\, d_2$ (a match is only trustworthy if it is clearly better than the runner-up).

    Return a float array of shape `(P, 3)` with rows `(index_in_A, index_in_B, d_1)`, sorted by increasing distance.
    """)
    return


@app.function
def match_descriptors(descriptors_a, descriptors_b, ratio=0.8):
    a = np.asarray(descriptors_a, dtype=float).reshape(-1, 128)
    b = np.asarray(descriptors_b, dtype=float).reshape(-1, 128)
    matches = []

    if len(b) >= 2:                                  # need a runner-up to compare against
        for i in range(len(a)):
            # step 1: L1 distance from a[i] to every row of b, shape (M,)
            dists = np.abs(b - a[i]).sum(axis=1)

            # step 2: the two smallest distances
            order = np.argsort(dists)
            j = order[0]                             # index of nearest
            d1 = dists[j]
            d2 = dists[order[1]]                     # second nearest

            # step 3-4: ratio test, then keep
            if d1 < ratio * d2:
                matches.append((i, j, d1))

    matches = np.array(matches, dtype=float).reshape(-1, 3)
    return matches[np.argsort(matches[:, 2])] if len(matches) else matches


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Problem 7: The Full Pipeline

    Chain everything together in `sift(image)`. For every octave $o$ and every DoG level $\ell = 1, \dots, S$:

    1. find and filter the extrema of the octave (Problems 2 and 3),
    2. compute the gradients of Gaussian level $\ell$ of that octave (not the DoG!), and assign orientations with $\sigma_\ell = \sigma_0 k^{\ell}$ (Problem 4),
    3. compute one descriptor per oriented keypoint (Problem 5),
    4. map the keypoint back to the **original image**: multiply `row`, `col` and `sigma` by $2^o$; `theta` is unchanged.

    Return `(keypoints, descriptors)`: a float array of shape `(M, 4)` with rows `(row, col, sigma, theta)` in original-image coordinates, and a float array of shape `(M, 128)`, row $i$ describing keypoint $i$.
    """)
    return


@app.function
def sift(image, num_octaves=NUM_OCTAVES, s=S, sigma0=SIGMA_0, contrast_threshold=0.03, edge_ratio=10):
    k = 2 ** (1 / s)
    gaussian_octaves = build_gaussian_octaves(image, num_octaves, s, sigma0)
    dog_octaves = build_dog_octaves(gaussian_octaves)
    keypoints, descriptors = [], []

    for o in range(num_octaves):
        gauss = gaussian_octaves[o]      # shape (s + 3, H_o, W_o)
        dog = dog_octaves[o]             # shape (s + 2, H_o, W_o)
        scale = 2 ** o

        # Problems 2 and 3: candidates, then filter
        candidates = find_extrema(dog)
        kept = filter_keypoints(dog, candidates, contrast_threshold, edge_ratio)
        if len(kept) == 0:
            continue

        for level in range(1, s + 1):
            # keypoints that were found at this level, as (row, col) pairs
            pts = kept[kept[:, 0] == level][:, 1:]
            if len(pts) == 0:
                continue

            sigma = sigma0 * k ** level
            sigmas = np.full(len(pts), sigma)

            # Problem 4: gradients of the Gaussian level (not the DoG), then orientations
            magnitude, orientation = image_gradients(gauss[level])
            oriented = assign_orientations(magnitude, orientation, pts, sigmas)

            # Problem 5: one descriptor per oriented keypoint
            for r, c, sig, theta in oriented:
                desc = compute_descriptor(magnitude, orientation, int(r), int(c), theta)

                # map back to the original image
                keypoints.append((r * scale, c * scale, sig * scale, theta))
                descriptors.append(desc)

    return (
        np.array(keypoints, dtype=float).reshape(-1, 4),
        np.array(descriptors, dtype=float).reshape(-1, 128),
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Keypoints and Matches

    The hidden cells below run `sift` on both images, match the descriptors, and draw the **15 best matches** (smallest distance). Each keypoint is drawn as a circle of radius $3\sigma$ with a line showing its orientation.
    """)
    return


@app.cell(hide_code=True)
def _(image_a, image_b):
    try:
        keypoints_a, descriptors_a = sift(image_a)
        keypoints_b, descriptors_b = sift(image_b)
        if len(keypoints_a) == 0 or len(keypoints_b) == 0:
            raise ValueError("sift returned no keypoints")
        matches_ab = match_descriptors(descriptors_a, descriptors_b)
        pipeline_error = None
    except Exception as error:
        keypoints_a = descriptors_a = keypoints_b = descriptors_b = matches_ab = None
        pipeline_error = str(error)
    return keypoints_a, keypoints_b, matches_ab, pipeline_error


@app.cell(hide_code=True)
def _(image_a, image_b, keypoints_a, keypoints_b, pipeline_error):
    try:
        if pipeline_error:
            raise RuntimeError(pipeline_error)
        _fig, _axes = plt.subplots(1, 2, figsize=(14, 5.6))
        for _ax, _image, _keypoints, _name in zip(_axes, (image_a, image_b), (keypoints_a, keypoints_b), "AB"):
            _ax.imshow(_image, cmap="gray")
            for _r, _c, _sigma, _theta in _keypoints:
                _ax.add_patch(plt.Circle((_c, _r), 3 * _sigma, fill=False, color="lime", lw=0.5))
                _ax.plot(
                    [_c, _c + 3 * _sigma * np.cos(np.radians(_theta))],
                    [_r, _r + 3 * _sigma * np.sin(np.radians(_theta))],
                    color="yellow", lw=0.5,
                )
            _ax.set_title(f"Image {_name}: {len(_keypoints)} keypoints")
            _ax.axis("off")
        mo.output.replace(_fig)
    except Exception as error:
        mo.output.replace(mo.md(f"Problem 7 is not ready: {error}"))
    return


@app.cell(hide_code=True)
def _(image_a, image_b, keypoints_a, keypoints_b, matches_ab, pipeline_error):
    try:
        if pipeline_error:
            raise RuntimeError(pipeline_error)
        _num_shown = 15
        _canvas = np.hstack([image_a, image_b])
        _offset = image_a.shape[1]
        _fig, _ax = plt.subplots(figsize=(14, 5.6))
        _ax.imshow(_canvas, cmap="gray")
        for _i, _j, _distance in matches_ab[:_num_shown]:
            _ra, _ca = keypoints_a[int(_i), :2]
            _rb, _cb = keypoints_b[int(_j), :2]
            _ax.plot([_ca, _cb + _offset], [_ra, _rb], "-o", ms=3, lw=1)
        _ax.set_title(f"{min(_num_shown, len(matches_ab))} best of {len(matches_ab)} matches")
        _ax.axis("off")
        mo.output.replace(_fig)
    except Exception as error:
        mo.output.replace(mo.md(f"Matches are not ready: {error}"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Problem 8: Discussion (written)

    Look at your final figure and answer briefly (a few sentences each):

    **8(a)** How many keypoints did you find in each image, and how many matches passed the ratio test? Are the 15 drawn matches correct? If some are wrong, what do the wrong ones have in common?

    **8(b)** Why does SIFT compute the descriptor on the blurred image of the keypoint's own octave and level, rather than on the original full-resolution image?

    **8(c)** Explain in one or two sentences why expressing gradient directions *relative to* $\theta_k$ makes the descriptor invariant to rotating the image.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Your response to 8(a)

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Your response to 8(b)

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Your response to 8(c)

    *Double-click this cell to write your answer. You may replace everything here.*
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Problem 9: Pen-and-Paper Questions (written)

    No code here. Answer in the cells below; these are graded by the course staff. Be brief: one or two sentences per item is enough when a reason is asked for.

    ## 9.1 Good corners and bad corners (6 points)

    The figure below marks five corner points on a photo of a mug on a table, with a 64×64 close-up of each. To the eye, and to the Harris detector of Lecture 3, every one of them is a corner: the intensity changes in two directions around it. But a corner is a good *keypoint* only if a second photo shows it again at the *same physical point*, and if its descriptor picks out that point and no other.

    Imagine a second photo of the same scene, taken after a small step to the left, with the same lighting. Reason from the image content; no computation is needed.

    **(a)** For each of **A–E**, answer in one or two sentences (one point per location):
    (i) What physical thing makes the corner? Is it a fixed point on the surface of the mug or the table, or will it be at a different physical point in the second photo?
    (ii) Does the patch around it look like other places in the photo?
    (iii) Is it a good keypoint or a poor one?

    **(b)** (1 point) SIFT has three tests of its own: the contrast and edge tests (Problem 3) and the ratio test (Problem 6). Which of your poor keypoints can these tests be counted on to reject? Which can they not reject, and what do those points have in common? Which step after matching (Lecture 4) can still remove the wrong matches they cause?
    """)
    return


@app.cell(hide_code=True)
def _():
    try:
        _mug_url = "https://github.com/hp6318/ece181-hw3-data/blob/main/mug_candidates.png?raw=true"
        if Path("mug_candidates.png").exists():
            _candidates = Image.open("mug_candidates.png")
        else:
            _request = Request(_mug_url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(_request, timeout=30) as _response:
                _candidates = Image.open(BytesIO(_response.read()))
        mo.output.replace(mo.image(np.asarray(_candidates.convert("RGB")), width=820))
    except Exception as error:
        mo.output.replace(mo.md(f"*Could not load mug_candidates.png ({error}); see the handout, Figure 4.*"))
    return


@app.cell
def _():
    mo.md(r"""
    Your response to 9.1

    * **A**:
    * **B**:
    * **C**:
    * **D**:
    * **E**:
    * **(b)**:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 9.2 Short reasoning questions (6 points, 1 each)

    1. Why does an octave with $s$ scales need $s + 3$ Gaussian images?
    2. A strong extremum of $D$ lies on the long straight edge of a table. Which test removes it? Why would keeping it hurt matching?
    3. A brick wall has many identical bricks. Why does a threshold on the nearest distance $d_1$ alone accept many wrong matches there, while the ratio test $d_1 / d_2 < 0.8$ rejects them?
    4. *A peak in space but not in scale.* A pixel of the DoG level $D_\ell$ is larger than its 8 neighbours in $D_\ell$, but the pixel at the same place in $D_{\ell+1}$ is larger still. What does this tell you about the size of the structure there? Why does SIFT not make it a keypoint at level $\ell$?
    5. *The smallest possible ratio.* Lecture 4 showed that $\operatorname{Tr}^2/\operatorname{Det} = (r+1)^2/r$, where $r \ge 1$ is the ratio of the two curvatures of $D$. Show that this is never less than 4, and say when it equals 4. What would the edge test do if someone set $r_e = 1$?
    6. *Inliers before and after the ratio test.* Of 300 nearest neighbours between two photos, 100 are correct and 200 are wrong. Lowe found that the threshold 0.8 removes 90 % of the wrong matches and fewer than 5 % of the correct ones. Estimate the fraction $w$ of correct matches before and after the ratio test. RANSAC (Lecture 4) then fits the $3 \times 3$ mapping between the two photos. Each trial picks 4 matches at random and fits the mapping to them. It succeeds only if all 4 are correct. (Take the number 4 as given.) What is the chance that one trial succeeds, before and after the ratio test?
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Your response to 9.2

    1.
    2.
    3.
    4.
    5.
    6.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 9.3 How noise reaches the gradients and the Hessian (5 points, extra credit)

    *These 5 points can replace one of the quiz grades if you missed one.*

    Every photo has some noise. This question asks how much of it gets into the gradients of Problem 4 and the Hessian of Problem 3, and how SIFT copes with it.

    Write an image as $L(r,c) = s(r,c) + n(r,c)$, where $s$ is the noise-free image and $n$ is the noise. Assume that the noise values at different pixels are independent, and that each has mean 0 and variance $\sigma_n^2$, so its standard deviation is $\sigma_n$. (Blurring makes the noise at adjacent pixels similar. We ignore that here; it changes the numbers but not the conclusions.) Part (c) uses the same model for the DoG image $D$.

    **Facts you may use.** (1) Every formula in this question is a weighted sum of pixel values, so it is linear: the output for $s + n$ is the output for $s$ plus the output for $n$. (2) If $n_1$ and $n_2$ are independent, the variance of $n_1 + n_2$ is the sum of their variances. (3) Multiplying by a constant $a$ multiplies the variance by $a^2$.
    *Example.* The forward difference $L(r,c{+}1) - L(r,c)$ carries the noise $n(r,c{+}1) - n(r,c)$; by facts 2 and 3 its variance is $\sigma_n^2 + (-1)^2 \sigma_n^2 = 2\sigma_n^2$, so its standard deviation is $\sqrt{2}\,\sigma_n$.
    *Useful numbers.* $\sqrt{2} \approx 1.41$, $\sqrt{6} \approx 2.45$, 1 radian $\approx 57.3^\circ$.

    **(a)** (2 points) Show that a weighted sum $\sum_i a_i L(r_i, c_i)$ over different pixels has output noise with variance $\sigma_n^2 \sum_i a_i^2$. Use this to find the standard deviation of the noise in (i) the gradient component $d_x$ of Problem 4(a), (ii) the second derivative $D_{xx}$ of Problem 3(b), (iii) the mixed derivative $D_{xy}$ of Problem 3(b). Which of the three is most affected by noise, and which least? Why?

    **(b)** (2 points) *Noise in the orientation.* At one pixel the noise-free gradient points along $+x$ with magnitude $m$, so $(d_x, d_y) = (m, 0)$. Noise adds $e_x$ to $d_x$ and $e_y$ to $d_y$. Each has the standard deviation you found in (a)(i), and they are independent because they come from different pixels. When the noise is small, the measured angle is $\theta \approx e_y / m$ radians.
    (i) Find the standard deviation of $\theta$ in degrees, in terms of $m / \sigma_n$.
    (ii) How large must $m$ be, compared with $\sigma_n$, for this to be below $5^\circ$ (half a bin of the 36-bin orientation histogram)? And below $22.5^\circ$ (half a bin of the descriptor)?
    (iii) Use your answer to explain why each vote in the orientation histogram and in the descriptor is weighted by the gradient magnitude $m$.

    **(c)** (1 point) *Noise and the edge test.* Suppose the noise in $D$ has standard deviation $0.008$ at each pixel. By (a), the noise in $D_{xx}$ and in $D_{yy}$ then has a standard deviation of about $0.02$. Take two round blobs, both with $D_{xy} = 0$: a weak one with $D_{xx} = D_{yy} = -0.02$, and a strong one with $D_{xx} = D_{yy} = -0.2$. Without noise both have $\operatorname{Tr}^2/\operatorname{Det} = 4$ and pass the edge test. Now let the noise change $D_{xx}$ by $+0.02$ and $D_{yy}$ by $-0.02$, a typical amount. Compute $\operatorname{Tr}^2/\operatorname{Det}$ for each blob and say whether it passes the edge test with $r_e = 10$. What does this tell you about trusting the edge test, and the position of a keypoint, at weak extrema?
    """)
    return


@app.cell
def _():
    mo.md(r"""
    Your response to 9.3 (extra credit)

    **(a)**

    **(b)**

    **(c)**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Submitting Your Work

    If you used Molab, click **Export** and choose **Python** to download your completed notebook as a `.py` file. Submit that `.py` notebook file to Gradescope. Problems 1–7 are graded automatically; Problems 8 and 9 (and the extra-credit 9.3) and your final figures are graded by the course staff.
    """)
    return


if __name__ == "__main__":
    app.run()
