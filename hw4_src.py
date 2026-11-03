# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.25.0",
#     "matplotlib>=3.10.0",
#     "numpy>=2.5.3",
#     "scipy>=1.14",
#     "pillow>=11.0.0",
#     "opencv-python-headless>=4.10",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Homework 4: Optical Flow
    <!--
    ## Important notes

    - This homework covers Lecture 5, Motion and Optical Flow, Sections 5.4 to 5.13 of the notes. The appendices are not needed.
    - Do Problems 1 to 3 by hand, as you would on the exam. The numbers are chosen so that
    no calculator is needed. Show your steps.
    - In Problem 4 you write Lucas–Kanade yourself, with NumPy, `scipy.ndimage` and
    matplotlib. Do not use a library routine that computes optical flow or selects
    features for you, such as OpenCV's `calcOpticalFlowPyrLK`,
    `calcOpticalFlowFarneback` or `goodFeaturesToTrack`.
    - Submit one PDF to Homework 4 on Gradescope, with pages assigned to problems, and
    your code to the Homework 4 programming assignment. You may work with your study
    group, but write up your own solutions and list the group's members on the first
    page. AI tools may be used to clarify ideas and to debug, not to produce the solutions
    you submit (see the syllabus). -->

    **Submission Instructions.** Your submission should be a zip file, containing at least this Marimo notebook (just like previous weeks) as well as handwritten work attached as a PDF (hw4.pdf) for the written questions.

    **Notation.**  As in the notes, $x$ runs to the right and $y$ runs down. $(u,v)$ is
    the motion from frame $t$ to frame $t+1$ in pixels, with $u$ along $x$ and $v$ along $y$.
    $I_t = I (x,y,t+1) - I (x,y,t)$ is the frame difference. Over a window, Lucas–Kanade
    solves $\mathtt{A} \,\mathbf{u} = -\mathbf{b}$ with:

    $$
    \mathtt{A}  = \begin{bmatrix} a_{11} & a_{12} \\ a_{12} & a_{22} \end{bmatrix}
          = \begin{bmatrix} \left\langle I_x ^{2} \right\rangle & \left\langle I_x I_y  \right\rangle \\ \left\langle I_x I_y  \right\rangle & \left\langle I_y ^{2} \right\rangle \end{bmatrix},
    $$

    $$
    \mathbf{b} = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}
            = \begin{bmatrix} \left\langle I_x  I_t \right\rangle \\ \left\langle I_y  I_t \right\rangle \end{bmatrix},
    $$

    $$
    \begin{bmatrix} u \\ v \end{bmatrix}
     = \frac{1}{\det\mathtt{A} }
       \begin{bmatrix} -a_{22}\, b_1 + a_{12}\, b_2 \\ \phantom{-}a_{12}\, b_1 - a_{11}\, b_2 \end{bmatrix},
    $$

    where the angle brackets are sums over the window (here with $w = 1$). The eigenvalues of
    $\mathtt{A}$ are:

    $$
    \lambda_{1,2} = \tfrac12 \operatorname{tr}\mathtt{A} \pm
    \sqrt{(\tfrac12 \operatorname{tr}\mathtt{A})^2 - \det\mathtt{A}},
    $$

    with $\lambda_1 \ge \lambda_2$ (Lecture 3).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 1: One pixel at a time
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(a)

    Start from brightness constancy,

    $$
    I(x+u,\, y+v,\, t+1) = I(x,\, y,\, t),
    $$

    and derive the optical flow constraint equation

    $$
    I_x u + I_y v + I_t = 0,
    $$

    with $I_t$ the frame difference. Say which step is an approximation, and when that approximation is good.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 1(a).**

    Brightness constancy says a point keeps its brightness as it moves:

    I(x+u, y+v, t+1) = I(x, y, t)

    The unknowns u and v sit inside the image function, so we linearize with a first-order Taylor expansion. Starting from the brightness at (x, y) in frame t+1, we add the slope times the step in each direction, where I_x and I_y are the spatial derivatives of the image:

    I(x+u, y+v, t+1) ≈ I(x, y, t+1) + I_x u + I_y v

    Frame t+1 equals frame t plus the frame difference, I_t = I(x, y, t+1) − I(x, y, t). This is exact:

    I(x, y, t+1) = I(x, y, t) + I_t

    Substituting:

    I(x+u, y+v, t+1) ≈ I(x, y, t) + I_x u + I_y v + I_t

    By brightness constancy the left side equals I(x, y, t), so:

    I(x, y, t) ≈ I(x, y, t) + I_x u + I_y v + I_t

    The I(x, y, t) terms cancel, leaving the optical flow constraint equation:

    I_x u + I_y v + I_t = 0

    Which step is the approximation? Only the first-order Taylor expansion. Rewriting frame t+1 as frame t plus I_t is exact.

    When is it good? When the motion is small (about a pixel or less in a sharp image). For larger motions the brightness no longer varies linearly over the step, so the linear approximation breaks down.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(b)

    At one pixel, $I_x  = 8$, $I_y  = 6$ and $I_t = -20$.

    1. Write the constraint line and sketch it in the $(u,v)$ plane. Mark where it crosses
       the two axes.
    2. Show that the motions $(1, 2)$ and $(4, -2)$ both lie on the line, and find a third.
    3. Find the normal flow and its length.
    4. In which direction does the edge through this pixel run? The true motion is
       $(1, 2)$. Split it into a part across the edge and a part along the edge. Which part
       does this pixel measure?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 1(b).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.

    1. Your answer.

    2. Your answer.

    3. Your answer.

    4. Your answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1(c)

    *The barber pole again.* A pattern of straight stripes moves to the right
    by $(u, v) = (2, 0)$ pixels per frame. At a pixel on the side of one stripe the gradient is
    $\nabla I  = (3, -3)$.

    1. Use the constraint equation to find the frame difference $I_t$ at this pixel.
    2. What normal flow does a small window around this pixel measure? Give its direction
       and its length.
    3. In which direction do the stripes run? Explain why the pattern, seen through a small
       window, seems to move in a different direction from its true motion. How does this
       explain the barber pole of Lecture 5?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 1(c).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.

    1. Your answer.

    2. Your answer.

    3. Your answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 2: Lucas–Kanade by hand
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(a)

    A $3\times3$ window with $w = 1$ has these derivatives at its nine pixels
    (rows from top to bottom):

    $$
    I_x  = \begin{bmatrix} 0&0&0\\ 1&2&0\\ 1&2&0 \end{bmatrix},
    $$

    $$
    I_y  = \begin{bmatrix} 0&1&1\\ 0&2&2\\ 0&0&0 \end{bmatrix},
    $$

    $$
    I_t = \begin{bmatrix} \phantom{-}0&\phantom{-}1&1\\ -1&\phantom{-}0&2\\ -1&-2&0 \end{bmatrix}.
    $$

    1. Compute $a_{11}$, $a_{12}$, $a_{22}$, $b_1$ and $b_2$.
    2. Solve for $(u, v)$.
    3. Find the eigenvalues of $\mathtt{A}$. Is the window flat, edge-like or corner-like?
    4. Check that your $(u,v)$ satisfies the constraint equation at every pixel of the window.
    5. At the center pixel $I_t = 0$. Does this mean that the center pixel did not move?
       Explain with the constraint equation at that pixel.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 2(a).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.

    1. Your answer.

    2. Your answer.

    3. Your answer.

    4. Your answer.

    5. Your answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(b)

    A second $3\times3$ window, with $w = 1$, lies on a straight edge. Its
    derivatives are

    $$
    I_x  = I_y  = \begin{bmatrix} 0&1&2\\ 1&2&1\\ 2&1&0 \end{bmatrix},
    $$

    $$
    I_t = -\begin{bmatrix} 0&1&2\\ 1&2&1\\ 2&1&0 \end{bmatrix}.
    $$

    1. Compute $\mathtt{A}$, $\mathbf{b}$, $\det\mathtt{A}$ and the eigenvalues of $\mathtt{A}$.
    2. Show that the two equations of $\mathtt{A} \,\mathbf{u} = -\mathbf{b}$ are the same equation.
       Describe all the motions that fit this window equally well.
    3. Find the motion of part (ii) closest to $(0, 0)$. Show that it equals the normal flow
       at the center pixel.
    4. The true motion was $(1.5, -0.5)$. In which direction does the edge run? Which part
       of the true motion did the window recover, and which part was lost?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 2(b).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.

    1. Your answer.

    2. Your answer.

    3. Your answer.

    4. Your answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2(c)

    Two windows have

    $$
    \mathtt{A} _P = \begin{bmatrix} 10 & 0 \\ 0 & 10 \end{bmatrix}
    $$

    and

    $$
    \mathtt{A} _Q = \begin{bmatrix} 100 & 0 \\ 0 & 1 \end{bmatrix}.
    $$

    1. Give the eigenvalues and the determinant of each matrix. Which window is on a corner
       and which is on an edge? In which direction does that edge run?
    2. Noise changes $\mathbf{b}$ by $\delta\mathbf{b} = (0.5, 0.5)^{\mathsf{T}}$ in both windows. By how much,
       and in which direction, does the solution $(u,v)$ change in each window?
    3. One tracker accepts a window when $\det\mathtt{A}  > 50$. Another accepts it when
       $\lambda_2 > 5$. Which windows does each tracker accept? Which test is better, and why?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 2(c).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.

    1. Your answer.

    2. Your answer.

    3. Your answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 3: Time to collision and the focus of expansion
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3(a)

    An object of size $L$ at distance $D(t)$ from a pinhole camera with focal
    length $f$ has an image of size $l(t) = fL/D(t)$. Show that $l/\dot l = -D/\dot D$. Which
    quantities cancel? Under what conditions is $\tau = -D/\dot D$ the time until the object
    reaches the camera's plane?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 3(a).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3(b)

    You follow a truck on a straight road. Its image width grows from 60 to 62
    pixels between two consecutive frames, at 30 frames per second, and the closing speed is
    constant.

    1. What is the time to collision at the second frame, in frames and in seconds? What is
       it at the first frame?
    2. The closing speed is 4 m/s. How far away is the truck at the second frame? Could you
       have found this distance from the two images alone?
    3. Suppose the camera zoomed in slightly between the two frames. Could you still use
       your estimate? Explain.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 3(b).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.

    1. Your answer.

    2. Your answer.

    3. Your answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3(c)

    A camera moves straight toward a flat wall that faces it, without turning.
    Three points on the wall have these positions in frame $t$ and these flows to frame $t+1$
    (in pixels and pixels per frame):

    | Point | Position $(x, y)$ | Flow $(u, v)$ |
    | --- | --- | --- |
    | A | $(420, 290)$ | $(2, 1)$ |
    | B | $(170, 290)$ | $(-3, 1)$ |
    | C | $(300, 90)$ | $(-0.4, -3)$ |

    1. Use A and B with $u = (x - x_0)/\tau$ and $v = (y - y_0)/\tau$ to find the focus of
       expansion $(x_0, y_0)$ and the time to collision $\tau$. Check your answer with C.
    2. The video runs at 25 frames per second. How many seconds remain until the camera
       reaches the wall?
    3. A lamp post stands between the camera and the wall. A point on it, at $(370, 340)$,
       moves by $(2, 4)$. What flow would a point of the wall at the same position have?
       How does the depth of the post compare with the depth of the wall?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 3(c).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.

    1. Your answer.

    2. Your answer.

    3. Your answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 4: Lucas–Kanade from scratch

    In this problem you write Lucas–Kanade, test it on frames whose motion is known, and then
    use it on two frames of the drive from Lecture 5. Verify that you have the following five images in a directory `data/hw4/`.

    | File | Contents |
    | --- | --- |
    | `board_t0.png` | frame $t$: the lab board of Lectures 4 and 5, $640\times480$, gray |
    | `board_t1_small.png` | frame $t+1$: the board moved by $(u,v) = (0.6,\ 0.45)$ pixels |
    | `board_t1_large.png` | frame $t+1$: the board moved by $(u,v) = (6.0,\ 4.5)$ pixels |
    | `jeep_t0.png` | frame 170 of the drive, cropped to the scene ahead, $500\times340$, color |
    | `jeep_t1.png` | frame 171, cropped the same way |

    The board frames were made by moving the photograph by an exact amount, which need not be a
    whole number of pixels, and adding a small amount of noise to each frame. The true flow is
    therefore known at every pixel.

    **Settings.**  Use the settings of the notes throughout. Blur each frame lightly with
    a Gaussian before taking derivatives (the notebook gives $\sigma$). For $I_x$ and $I_y$, take
    central differences of frame $t$ divided by 2, as in the notes:

    $$
    I_x = \big(I(x{+}1, y) - I(x{-}1, y)\big)/2.
    $$

    Use the frame difference for $I_t$ and a
    $7\times7$ window with $w = 1$. Intensities are in $[0, 1]$. Positions are $(x, y)$, with $x$
    along a row and $y$ down, so pixel $(x, y)$ is `I[y, x]`. (Homework 3 used
    `(row, col)`.) Implement `lk_point` for the one-step estimate at a single point in
    part (a), `grid_optical_flow` for the grid of one-step estimates in part (b), and
    `optical_flow` for the iterative, pyramidal flow field in part (c). You may reuse
    your code from Homework 3 where useful. In `optical_flow`, follow Section 5.10:
    at each pyramid level, from the top down, sample frame $t+1$ at the window's pixels moved by
    the current estimate, form $I_t$ and $\mathbf{b}$, solve for the correction, and repeat. Going
    down one level, double the estimate.

    **Implementation conventions.** For a multichannel image in $[0, 1]$, convert to
    grayscale with `Image.fromarray((image * 255).astype(np.uint8)).convert("L")`,
    then convert the result to floating point and divide by 255. For smoothing, use
    `scipy.ndimage.gaussian_filter(image, sigma, mode="reflect", truncate=4.0)`.
    You may ignore all boundary effects.

    We have provided the image loading, function calls, and results table for part (a),
    the motion-comparison table for part (c), and the function calls, visualizations,
    and focus-of-expansion fitting for part (d).

    **What to submit.** The functions are checked by the autograder. Submit the completed notebook to the
    Homework 4 programming assignment on Gradescope. The tables, plots and answers of parts (a)
    to (d) go in the PDF, part by part.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(
        src=mo.notebook_dir() / "data" / "hw4" / "hw4_inputs.png",
        alt="Homework 4 inputs",
        width="100%",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Figure 1: Left: frame $t$ of the board, with the three windows of part (a), centered at the
    pixels given as $(x, y)$. Right: frame $t$ of the drive, for part (d). The next frame is
    $1/24$ second later.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4(a)

    *One window at a time.* Run one Lucas–Kanade step (one iteration, no
    pyramid) from `board_t0.png` to `board_t1_small.png` at the three windows of
    Figure 1. We have provided code to display a table with $a_{11}$, $a_{12}$, $a_{22}$, $\lambda_1$,
    $\lambda_2$, $\lambda_2/\lambda_1$, the estimate $(u,v)$ and its error
    $\left\lVert (u,v) - (0.6,\, 0.45) \right\rVert$, one row per window. Give $a_{11}$, $a_{12}$, $a_{22}$ and the
    eigenvalues in units of $10^{-3}$. Explain each result with the eigenvalues, in two or three
    sentences. For the edge window, which component of the estimate is wrong, and why?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 4(a).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import numpy as flow_np
    from PIL import Image as _Image

    _folder = mo.notebook_dir() / "data/hw4"
    _images = []
    for _name in ("board_t0.png", "board_t1_small.png", "board_t1_large.png", "jeep_t0.png", "jeep_t1.png"):
        with _Image.open(_folder / _name) as _image:
            _images.append(flow_np.asarray(_image, dtype=flow_np.float32) / 255.0)
    board_t0, board_t1_small, board_t1_large, jeep_t0, jeep_t1 = _images
    return board_t0, board_t1_large, board_t1_small, flow_np, jeep_t0, jeep_t1


@app.function
def lk_point(image_t, image_t1, x, y, *, sigma=1.0, window_size=7):
    """Compute one Lucas–Kanade step at (x, y), with no pyramid.

    Both images have shape (H, W) or (H, W, C), with values in [0, 1].
    Handle multichannel inputs in your implementation. Coordinates use
    (x, y) order; array indexing uses [y, x].

    Return a dictionary with:
      flow: shape (2,), containing (u, v).
      structure: shape (2, 2), containing the window's structure matrix.
      eigenvalues: shape (2,), ordered from largest to smallest.
    """

    import numpy as np
    from scipy.ndimage import gaussian_filter
    from PIL import Image

    # TODO: preprocess the images, compute derivatives and window sums,
    # then compute the eigenvalues and solve for the one-step displacement.
    # Replace these dummy outputs with your computed results.
    return {
        "flow": np.zeros(2, dtype=float),
        "structure": np.zeros((2, 2), dtype=float),
        "eigenvalues": np.zeros(2, dtype=float),
    }


@app.cell
def _(board_t0, board_t1_small, flow_np, mo):
    board_windows = [("corner", 426, 374), ("edge", 106, 98), ("flat", 470, 247)]
    board_step_results = []
    _rows = []
    for _name, _x, _y in board_windows:
        _result = lk_point(board_t0, board_t1_small, _x, _y)
        board_step_results.append(_result)
        _u, _v = _result["flow"]
        _A = _result["structure"]
        _lambda1, _lambda2 = _result["eigenvalues"]
        _rows.append({
            "Window": _name, "x": _x, "y": _y,
            "a11 (1e-3)": 1000 * _A[0, 0],
            "a12 (1e-3)": 1000 * _A[0, 1],
            "a22 (1e-3)": 1000 * _A[1, 1],
            "lambda1 (1e-3)": 1000 * _lambda1,
            "lambda2 (1e-3)": 1000 * _lambda2,
            "lambda2 / lambda1": _lambda2 / _lambda1 if _lambda1 else None,
            "u": _u, "v": _v,
            "Error (pixels)": flow_np.linalg.norm([_u - 0.6, _v - 0.45]),
        })
    mo.ui.table(
        _rows, selection=None, pagination=False,
        format_mapping={
            column: lambda value: f"{value:.3f}" if value is not None else ""
            for column in ("a11 (1e-3)", "a12 (1e-3)", "a22 (1e-3)",
                           "lambda1 (1e-3)", "lambda2 (1e-3)", "lambda2 / lambda1",
                           "u", "v", "Error (pixels)")
        },
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4(b)

    *Which windows can be trusted.* Run one step at the windows centered on a
    grid of points 16 pixels apart, starting 16 pixels from the border, which gives 1131 windows.
    Classify the windows by their eigenvalues:
    - *corner-like* if $\lambda_2$ is in the top 15% of the grid windows (at or above
       the 85th percentile of $\lambda_2$);
    - *flat* if $\lambda_1$ is in the bottom 40% (below the 40th percentile of
       $\lambda_1$);
    - *edge-like* if it is neither of these and $\lambda_2 < 0.05\,\lambda_1$.

    Windows in none of the three classes are not used. Report:

    1. the number of windows in each class and the median error of each class;
    2. `board_t0.png` with the estimate drawn as an arrow at every corner-like and
       edge-like window, in two colors, about 10 times longer than the motion, and a dot at
       every flat window;
    3. every estimate as a point in the $(u,v)$ plane, colored by class, with the true motion
       marked (draw $v$ increasing downward, as in the image);
    4. in two or three sentences, where the edge-like estimates lie in the $(u,v)$ plane, and
       why.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 4(b).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.

    1. Your answer.

    2. Your answer.

    3. Your answer.

    4. Your answer.
    """)
    return


@app.function
def grid_optical_flow(image_t, image_t1, *, spacing=16, border=16, sigma=1.0, window_size=7):
    """Run one Lucas–Kanade step at every grid point for Problem 4(b).

    Accept grayscale (H, W) or multichannel (H, W, C) images in [0, 1].
    Flatten any multichannel inputs into greyscale before any further computations.

    Return a dictionary with points (N, 2) in (x, y) order, flow (N, 2),
    structure (N, 2, 2), and eigenvalues (N, 2), largest eigenvalue first.
    Use the same point order in every output.
    """

    import numpy as np

    height, width = np.asarray(image_t).shape[:2]
    xs = np.arange(border, width - border + 1, spacing)
    ys = np.arange(border, height - border + 1, spacing)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    # TODO: call lk_point at each grid center and collect its results.
    # Replace the dummy flow, structure matrices, and eigenvalues below.
    return {
        "points": points,
        "flow": np.zeros((len(points), 2), dtype=float),
        "structure": np.zeros((len(points), 2, 2), dtype=float),
        "eigenvalues": np.zeros((len(points), 2), dtype=float),
    }


@app.cell
def _(board_t0, board_t1_small):
    # Use these results for the classification and analysis in 4(b).
    board_grid_results = grid_optical_flow(board_t0, board_t1_small)
    return (board_grid_results,)


@app.cell(hide_code=True)
def _(board_grid_results, board_t0, flow_np, mo):
    import matplotlib.pyplot as _grid_plt

    _points = board_grid_results["points"]
    _flow = board_grid_results["flow"]
    _lambda1, _lambda2 = board_grid_results["eigenvalues"].T

    # Apply the classifications in the order given in the problem.
    _corners = _lambda2 >= flow_np.percentile(_lambda2, 85)
    _flat = (~_corners) & (_lambda1 < flow_np.percentile(_lambda1, 40))
    _edges = (~_corners) & (~_flat) & (_lambda2 < 0.05 * _lambda1)
    # Colorblind-friendly colors; scatter markers also distinguish classes.
    _classes = [("Corner-like", _corners, "#0072B2"),
                ("Edge-like", _edges, "#D55E00"),
                ("Flat", _flat, "#009E73")]
    _finite = flow_np.isfinite(_flow).all(axis=1)
    _errors = flow_np.linalg.norm(_flow - [0.6, 0.45], axis=1)

    # Summarize the number of windows and their estimation errors.
    _rows = []
    for _name, _mask, _color in _classes:
        _valid = _mask & _finite
        _rows.append({
            "Class": _name,
            "Windows": int(_mask.sum()),
            "Median error (pixels)": float(flow_np.median(_errors[_valid])) if _valid.any() else None,
        })
    _table = mo.ui.table(
        _rows, selection=None, pagination=False,
        format_mapping={"Median error (pixels)": lambda value: f"{value:.3f}" if value is not None else ""},
    )

    # Draw the flow on the board and in the displacement plane.
    _flow_figure, _flow_axis = _grid_plt.subplots(figsize=(12, 9))
    _scatter_figure, _scatter_axis = _grid_plt.subplots(figsize=(8, 6))
    _flow_axis.imshow(board_t0, cmap="gray")
    for _name, _mask, _color in _classes:
        _valid = _mask & _finite
        _zorder = {"Flat": 2, "Edge-like": 3, "Corner-like": 4}[_name]
        if _name == "Flat":
            _flow_axis.scatter(_points[_mask, 0], _points[_mask, 1],
                             s=5, color=_color, label=_name, zorder=_zorder)
        else:
            _flow_axis.quiver(
                _points[_valid, 0], _points[_valid, 1],
                _flow[_valid, 0], _flow[_valid, 1],
                angles="xy", scale_units="xy", scale=0.1,
                color=_color, label=_name, zorder=_zorder,
            )
        _marker = {"Corner-like": "o", "Edge-like": "^", "Flat": "s"}[_name]
        _scatter_axis.scatter(_flow[_valid, 0], _flow[_valid, 1],
                         s=14, marker=_marker, color=_color, label=_name, alpha=0.7, zorder=_zorder)

    # Show unclassified estimates too, so every finite estimate is included.
    _unused = ~(_corners | _edges | _flat) & _finite
    _scatter_axis.scatter(_flow[_unused, 0], _flow[_unused, 1],
                     s=8, color="gray", label="Unclassified", alpha=0.4, zorder=1)
    _scatter_axis.scatter(0.6, 0.45, marker="x", s=90, color="black", label="True motion", zorder=5)
    _flow_axis.set_title("One-step flow (arrows at 10× scale)")
    _flow_axis.set_axis_off()
    _flow_axis.legend()
    _scatter_axis.set(title="Estimated displacements", xlabel="u (pixels)", ylabel="v (pixels)")
    _scatter_axis.invert_yaxis()
    _scatter_axis.legend()
    _flow_figure.tight_layout()
    _scatter_figure.tight_layout()
    _grid_plt.close(_flow_figure)
    _grid_plt.close(_scatter_figure)

    mo.vstack([_table, _flow_figure, _scatter_figure])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4(c)

    *Large motions.* Use the corner-like windows of part (b). For each of the
    two motions, find the median error of one step, and of 5 iterations per level with 0, 1, 2
    and 3 halvings (0 halvings is full resolution only). We have provided code to display the ten numbers in one table, with
    a row for each motion. Then answer, briefly:

    1. Why does one step fail on the large motion, even at corners? Compare its median error
       with the size of the motion, 7.5 pixels.
    2. Why do iterations at full resolution not fix it?
    3. How many halvings does the large motion need? Compare with the rule of thumb of
       Section 5.9, that a few iterations handle about 2 pixels.
    4. Do the iterations help the small motion too? Why?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 4(c).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.

    1. Your answer.

    2. Your answer.

    3. Your answer.

    4. Your answer.
    """)
    return


@app.function
def optical_flow(image_t, image_t1, *, iterations=5, halvings=3, sigma=1.0, window_size=7):
    """Return the optical flow image from image_t to image_t1 for Problem 4(c).

    Inputs have equal spatial dimensions and shapes (H, W) or (H, W, C),
    with intensities in [0, 1].

    Return a floating-point (H, W, 2) array in original-image pixels:
    flow[y, x, 0] is horizontal motion u; flow[y, x, 1] is vertical motion v.
    Use NaN for pixels where a valid displacement cannot be estimated.

    iterations specifies corrections per pyramid level; halvings=0 uses
    only full resolution.
    """

    import numpy as np
    from scipy.ndimage import gaussian_filter, map_coordinates
    from PIL import Image

    # TODO: build your pyramids and iteratively estimate the flow image.
    # Replace this dummy field with your computed displacement at each pixel.
    return np.zeros((*np.asarray(image_t).shape[:2], 2), dtype=float)


@app.cell(hide_code=True)
def _(mo):
    flow_comparison_run = mo.ui.run_button(label="Run 4(c) motion comparison")
    flow_comparison_run
    return (flow_comparison_run,)


@app.cell
def _(
    board_grid_results,
    board_t0,
    board_t1_large,
    board_t1_small,
    flow_comparison_run,
    flow_np,
    mo,
):
    # You do not need to edit this cell. It has been provided for you.
    _table = None
    if flow_comparison_run.value:
        _points = board_grid_results["points"]
        _lambda2 = board_grid_results["eigenvalues"][:, 1]
        _corners = flow_np.rint(_points[_lambda2 >= flow_np.percentile(_lambda2, 85)]).astype(int)
        _settings = [("One step", 1, 0), ("0 halvings", 5, 0), ("1 halving", 5, 1),
                     ("2 halvings", 5, 2), ("3 halvings", 5, 3)]
        _rows = []
        for _name, _image, _truth in [("Small", board_t1_small, [0.6, 0.45]),
                                      ("Large", board_t1_large, [6.0, 4.5])]:
            _row = {"Motion": _name}
            for _label, _iterations, _halvings in _settings:
                _field = optical_flow(board_t0, _image, iterations=_iterations, halvings=_halvings)
                _estimates = _field[_corners[:, 1], _corners[:, 0]]
                _errors = flow_np.linalg.norm(_estimates - _truth, axis=1)
                _finite_errors = _errors[flow_np.isfinite(_errors)]
                _row[_label] = float(flow_np.median(_finite_errors)) if len(_finite_errors) else None
            _rows.append(_row)
        _table = mo.ui.table(
            _rows, selection=None, pagination=False,
            format_mapping={
                label: lambda value: f"{value:.3f}" if value is not None else ""
                for label, _, _ in _settings
            },
        )
    _table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4(d)

    *Flow on a real video.* `jeep_t0.png` and `jeep_t1.png` are
    frames 170 and 171 of my video of a drive toward an airstrip (Section 5.12), cropped to the
    scene ahead of the jeep (Figure 1, right). Here the motion is not known.

    1. Compute $\lambda_2$ for the windows of the same kind of grid on the gray frame $t$
       (600 windows), and keep the textured half: the windows with $\lambda_2$ at or above
       the median.
    2. Track these points with 5 iterations per level and 3 halvings.
    3. Plot `jeep_t0.png` with the flow drawn as arrows, twice as long as the motion.
    4. We have provided code to fit the focus of expansion and refit twice, each time without
       the 20% of the vectors with the largest angle between the vector and the direction
       away from the current estimate. The provided plot marks the final focus of expansion.
       Report its $(x, y)$ location together with the median of that angle over all the vectors.
    5. Answer in a few sentences. Why are there no arrows in the sky? Which arrows are the
       longest, and why? Where is the jeep heading?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 4(d).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.

    1. Your answer.

    2. Your answer.

    3. Your answer.

    4. Your answer.

    5. Your answer.
    """)
    return


@app.function(hide_code=True)
def display_focus_of_expansion(points, displacements):
    """Provided focus fit for the 4(d) display, with two angular trimming passes."""
    import numpy as np

    points = np.asarray(points, dtype=float)
    displacements = np.asarray(displacements, dtype=float)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(displacements).all(axis=1)
    valid &= np.linalg.norm(displacements, axis=1) > 1e-9
    points, displacements = points[valid], displacements[valid]
    if len(points) < 2:
        return None, None
    normals = np.column_stack((-displacements[:, 1], displacements[:, 0]))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    rhs = np.sum(normals * points, axis=1)
    keep = np.arange(len(points))
    for iteration in range(3):
        focus, _, rank, _ = np.linalg.lstsq(normals[keep], rhs[keep], rcond=None)
        if rank < 2:
            return None, None
        radial = points - focus
        denominator = np.linalg.norm(radial, axis=1) * np.linalg.norm(displacements, axis=1)
        cosine = np.divide(np.sum(radial * displacements, axis=1), denominator,
                           out=np.full(len(points), np.nan), where=denominator > 1e-9)
        angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        if iteration < 2:
            candidates = keep[np.isfinite(angles[keep])]
            if len(candidates) < 2:
                return None, None
            order = np.argsort(angles[candidates])
            keep = candidates[order[:max(2, int(0.8 * len(candidates)))]]
    finite_angles = angles[np.isfinite(angles)]
    return focus, float(np.median(finite_angles)) if len(finite_angles) else None


@app.cell(hide_code=True)
def _(mo):
    drive_run = mo.ui.run_button(label="Run 4(d) drive analysis")
    return (drive_run,)


@app.cell(hide_code=True)
def _(drive_run, flow_np, jeep_t0, jeep_t1, mo):
    drive_points = flow_np.empty((0, 2))
    drive_displacements = flow_np.empty((0, 2))
    drive_focus, drive_angle = None, None
    if drive_run.value:
        _grid = grid_optical_flow(jeep_t0, jeep_t1)
        _lambda2 = _grid["eigenvalues"][:, 1]
        drive_points = _grid["points"][_lambda2 >= flow_np.median(_lambda2)]
        _field = optical_flow(jeep_t0, jeep_t1, iterations=5, halvings=3)
        _indices = flow_np.rint(drive_points).astype(int)
        drive_displacements = _field[_indices[:, 1], _indices[:, 0]]
        drive_focus, drive_angle = display_focus_of_expansion(drive_points, drive_displacements)
    mo.vstack([drive_run, drive_results_view(
        drive_angle, drive_displacements, drive_focus, drive_points, flow_np, jeep_t0, mo,
    )])
    return


@app.function(hide_code=True)
def drive_results_view(
    drive_angle,
    drive_displacements,
    drive_focus,
    drive_points,
    flow_np,
    jeep_t0,
    mo,
):
    import matplotlib.pyplot as _drive_plt

    _figure, _axis = _drive_plt.subplots(figsize=(10, 7))
    _axis.imshow(jeep_t0)
    _axis.set_title("Frame t: flow at 2× scale")
    _axis.set_axis_off()
    _valid = flow_np.isfinite(drive_displacements).all(axis=1)
    _points, _motion = drive_points[_valid], drive_displacements[_valid]
    if len(_points):
        _axis.quiver(_points[:, 0], _points[:, 1], _motion[:, 0], _motion[:, 1],
                        angles="xy", scale_units="xy", scale=0.5, color="#F0E442", width=0.003)
    if drive_focus is not None:
        _axis.plot(*drive_focus, marker="x", color="#D55E00", markersize=12, markeredgewidth=2)
    _figure.tight_layout()
    _drive_plt.close(_figure)
    _summary = mo.ui.table([{
        "Textured windows": len(drive_points),
        "Finite flow estimates": int(_valid.sum()),
        "Focus x": float(drive_focus[0]) if drive_focus is not None else None,
        "Focus y": float(drive_focus[1]) if drive_focus is not None else None,
        "Median angle (degrees)": drive_angle,
    }], selection=None, pagination=False, show_column_summaries=False, show_data_types=False,
        format_mapping={
            column: lambda value: f"{value:.3f}" if value is not None else ""
            for column in ("Focus x", "Focus y", "Median angle (degrees)")
        })
    return mo.vstack([_figure, _summary])


@app.function(hide_code=True)
def tracking_gray_frames(frames):
    """Convert RGB uint8 frames to grayscale float32 images in [0, 1]."""
    import numpy as np
    from PIL import Image

    return [
        np.asarray(Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("L"),
                   dtype=np.float32) / 255.0
        for frame in frames
    ]


@app.function(hide_code=True)
def tracking_animation(frames, tracks=None, fps=8):
    """Create an embedded GIF; colors identify the same point across frames."""
    from io import BytesIO
    from PIL import Image, ImageDraw
    import numpy as np

    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
    rendered = []
    for index, frame in enumerate(frames):
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(image)
        if tracks is not None:
            for point_id, point in enumerate(tracks[index]):
                if not np.isfinite(point).all():
                    continue
                x, y = point
                if not (0 <= x < image.width and 0 <= y < image.height):
                    continue
                color = colors[point_id % len(colors)]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline="black")
        draw.rectangle((0, 0, 130, 22), fill="black")
        draw.text((5, 5), f"Frame {index + 1}/{len(frames)}", fill="white")
        image.thumbnail((540, 432))
        rendered.append(image)
    output = BytesIO()
    rendered[0].save(output, format="GIF", save_all=True, append_images=rendered[1:],
                     duration=max(20, round(1000 / fps)), loop=0)
    return output.getvalue()


@app.function(hide_code=True)
def tracking_correspondences(frames, tracks, index, show_trails=True):
    """Show adjacent frames with identity-preserving correspondence lines."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    import numpy as np

    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
    previous = max(0, index - 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, frame_index in zip(axes, (previous, index)):
        axis.imshow(frames[frame_index])
        axis.set_title(f"Frame {frame_index + 1}")
        axis.set_axis_off()
    for point_id in range(len(tracks[0])):
        color = colors[point_id % len(colors)]
        endpoints = []
        for axis, frame_index in zip(axes, (previous, index)):
            point = tracks[frame_index][point_id]
            height, width = frames[frame_index].shape[:2]
            valid = np.isfinite(point).all() and 0 <= point[0] < width and 0 <= point[1] < height
            endpoints.append(valid)
            if valid:
                axis.plot(*point, "o", color=color, markersize=3)
            if show_trails:
                trail = np.array([points[point_id] for points in tracks[:frame_index + 1]], copy=True)
                trail[(trail[:, 0] < 0) | (trail[:, 0] >= width)
                      | (trail[:, 1] < 0) | (trail[:, 1] >= height)] = np.nan
                axis.plot(trail[:, 0], trail[:, 1], color=color, linewidth=0.8)
        if all(endpoints) and index != previous:
            figure.add_artist(ConnectionPatch(
                xyA=tracks[previous][point_id], xyB=tracks[index][point_id],
                coordsA="data", coordsB="data", axesA=axes[0], axesB=axes[1],
                color=color, linewidth=0.6, alpha=0.5,
            ))
    figure.tight_layout()
    plt.close(figure)
    return figure


@app.cell(hide_code=True)
def _(mo):
    import numpy as tracking_np
    from PIL import Image as TrackingImage

    _paths = sorted((mo.notebook_dir() / "data/hw4/dino").glob("*.ppm"))
    dino_frames = []
    for _path in _paths:
        with TrackingImage.open(_path) as _image:
            dino_frames.append(tracking_np.array(_image.convert("RGB")))
    dino_gray = tracking_gray_frames(dino_frames)
    return dino_frames, dino_gray, tracking_np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem 5: Optical Flow for Tracking

    One application of optical flow is for tracking points in video sequences. In this exercise, you will use your optical flow function to track keypoints in the video sequence displayed below.

    Round keypoint coordinates to the nearest pixel when sampling the flow field, and convert multichannel images to grayscale using PIL's `convert("L")`.
    """)
    return


@app.cell(hide_code=True)
def _(dino_frames, mo):
    mo.vstack([
        mo.image(tracking_animation(dino_frames), alt="Dinosaur image sequence"),
        mo.md("""
        **Source:** Wolfgang Niem, University of Hannover. *Dinosaur image sequence*.
        Distributed by the Visual Geometry Group, University of Oxford, in its
        [Multi-view reconstruction data collection](https://www.robots.ox.ac.uk/~vgg/data/mview/).
        Accessed October 26, 2026.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5(a)

    Start by extracting keypoint proposals. We'll use OpenCV's implementation of SIFT to do this using `cv2.SIFT_create`. Read the documentation for the function, then implement the `extract_keypoints` function below, which should return a numpy array of shape (N, 2), where N is the maximum number of keypoints retained. We provide the visualization code for the SIFT keypoints in the cells below.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 5(a).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.
    """)
    return


@app.function
def extract_keypoints(image, n=100):
    """Return at most n SIFT keypoint positions as an (N, 2) array, in (x, y) order.

    image is a grayscale float array in [0, 1]. You may need to convert to uint8 for OpenCV.
    If there are no keypoints, return an empty array with shape (0, 2).
    """

    import cv2
    import numpy as np
    # TODO: create the SIFT detector, detect keypoints, and retain at most n.
    # Return coordinates, rather than cv2.KeyPoint objects or SIFT descriptors.

    return np.empty((0, 2), dtype=float)


@app.cell(hide_code=True)
def _(dino_frames, mo):
    sift_count = mo.ui.slider(1, 100, value=60, label="Maximum keypoints", show_value=True, debounce=True)
    sift_frame = mo.ui.slider(1, len(dino_frames), value=1, label="Frame", show_value=True, debounce=True)
    return sift_count, sift_frame


@app.cell(hide_code=True)
def _(dino_frames, dino_gray, mo, sift_count, sift_frame, tracking_np):
    import matplotlib.pyplot as _plt

    _index = sift_frame.value - 1
    _points = tracking_np.asarray(extract_keypoints(dino_gray[_index], sift_count.value), dtype=float)
    _figure, _axis = _plt.subplots(figsize=(8, 6))
    _axis.imshow(dino_frames[_index])
    _axis.scatter(_points[:, 0], _points[:, 1], s=25, facecolors="none", edgecolors="#E69F00")
    _axis.set_title(f"Frame {_index + 1}: {len(_points)} SIFT keypoints")
    _axis.set_axis_off()
    _plt.close(_figure)
    _display = _figure
    mo.vstack([mo.hstack([sift_count, sift_frame]), _display])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5(b)

    Given a sequence of $K$ images and a maximum keypoint count `n`, implement `track_forward` using the following algorithm:

    1. Call your `extract_keypoints` function from part (a) to extract at most `n` keypoints on the first image $I_0$ of the sequence. We will refer to this as $P_0$ which has a shape of $(N, 2)$ for some $N \le n$.

    2. Use the optical flow function you defined in Problem 4 to compute the $(u, v)$ flow vector at each keypoint $(x_i, y_i)$ in $I_0$. Then, estimate the keypoint's new position $(x_f, y_f) \approx (x_i + u, y_i + v)$. We will refer to these tracked keypoints at frame 1 as $P_1$ which also has a shape of $(N, 2)$.

    3. Using the tracked keypoints at frame 1, $P_1$, compute the tracked keypoints at frame 2 $P_2$. Repeat this to compute $P_3, P_4, ..., P_{K - 1}$.

    Your function should return $K$ numpy arrays, each of shape $(N, 2)$ which store the locations of your tracked keypoints in each frame.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 5(b).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.

    1. Your answer.

    2. Your answer.

    3. Your answer.
    """)
    return


@app.function
def track_forward(images, n=100):
    """Return K arrays of shape (N, 2), keeping point order fixed across frames.

    images contains grayscale or multichannel images with values in [0, 1].
    Use extract_keypoints and optical_flow from Problem 4(c).
    """

    import numpy as np
    from PIL import Image
    # TODO: extract points once, then propagate them through consecutive frames.
    # Call optical_flow(images[i], images[i + 1]) for each pair and sample
    # its (H, W, 2) output at the current keypoint positions.

    return [np.empty((0, 2), dtype=float) for _ in images]


@app.cell(hide_code=True)
def _(dino_frames, mo):
    forward_count = mo.ui.slider(1, 100, value=60, label="Maximum keypoints", show_value=True, debounce=True)
    forward_length = mo.ui.slider(2, len(dino_frames), value=min(12, len(dino_frames)), label="Frames to track", show_value=True, debounce=True)
    forward_run = mo.ui.run_button(label="Run forward tracking")
    return forward_count, forward_length, forward_run


@app.cell(hide_code=True)
def _(dino_frames, forward_count, forward_length, forward_run, mo):
    forward_tracks, forward_animation = None, None
    forward_message = "Choose your settings, then run tracking."
    if forward_run.value:
        with mo.status.spinner(title="Tracking dinosaur points..."):
            _images = [frame.astype(float) / 255.0 for frame in dino_frames[:forward_length.value]]
            forward_tracks = track_forward(_images, n=forward_count.value)
            forward_animation = tracking_animation(dino_frames[:len(forward_tracks)], forward_tracks)
        forward_message = f"{len(forward_tracks[0])} point identities across {len(forward_tracks)} frames."
    forward_frame = mo.ui.slider(1, max(2, len(forward_tracks) if forward_tracks is not None else 2), value=1,
                                  label="Inspect frame", show_value=True, disabled=forward_tracks is None, debounce=True)
    forward_trails = mo.ui.checkbox(value=True, label="Show track history")
    return (
        forward_animation,
        forward_frame,
        forward_message,
        forward_tracks,
        forward_trails,
    )


@app.cell(hide_code=True)
def _(
    dino_frames,
    forward_animation,
    forward_count,
    forward_frame,
    forward_length,
    forward_message,
    forward_run,
    forward_tracks,
    forward_trails,
    mo,
):
    _parts = [mo.hstack([forward_count, forward_length]), forward_run,
              mo.md(forward_message)]
    if forward_tracks is not None:
        _index = min(forward_frame.value - 1, len(forward_tracks) - 1)
        _parts += [mo.hstack([forward_frame, forward_trails]), mo.ui.tabs({
            "Animation": mo.image(forward_animation, alt="Tracked dinosaur keypoints"),
            "Correspondences": tracking_correspondences(dino_frames, forward_tracks, _index, forward_trails.value),
        })]
    mo.vstack(_parts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5(c)

    Identify at least one **major** tracking error. Why does this happen using our algorithm?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 5(c).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5(d)

    Let's refine our tracking algorithm. Implement the `track` function below, using the following algorithm:

    1. Compute `n` keypoints using OpenCV's SIFT implementation.

    2. Compute the "forward" optical flow from $I_0$ to $I_1$, just like in your previous function. Also compute the "backward" optical flow, from $I_1$ to $I_0$ (you shouldn't need to re-implement your optical flow function to do this).

    3. Just like in `track_forward`, use the forward map to track the keypoints and compute $P_1$. Then, use the backward map to re-track the points in $P_1$ back into frame $I_0$. We'll call this quantity ${P_0}'$.

    4. Compute the error between $P_0$ and ${P_0}'$ by taking the difference of the two matrices and computing the magnitude of each keypoint's error using `np.linalg.norm(..., axis=1)`. Then, remove all keypoints which have an error exceeding `threshold` (which is provided to you as an input to your function) *from both $P_0$ and $P_1$.

    5. Repeat these steps to compute tracked keypoints for all of the frames. When you compute the error and threshold, you should remove all points which exceed the threshold *from all frames*. This means that as you add more frames, the number of tracked points will be non-increasing.

    Just like before, you should return $K$ numpy arrays, each with the same shape $(_, 2)$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 5(d).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.

    1. Your answer.

    2. Your answer.

    3. Your answer.

    4. Your answer.

    5. Your answer.
    """)
    return


@app.function
def track(images, n=100, threshold=1.0):
    """Return K (M, 2) arrays containing the same surviving point identities.

    images contains grayscale or multichannel images in [0, 1]. Reuse
    optical_flow from Problem 4(c) in both directions. threshold is the maximum
    per-point forward–backward Euclidean error in pixels.
    """

    import numpy as np
    from PIL import Image
    # TODO: call optical_flow(images[i], images[i + 1]) and
    # optical_flow(images[i + 1], images[i]); sample the respective flow fields
    # at the starting points and their forward matches, then filter.
    # Reject failed/nonfinite tracks as well as errors exceeding threshold.

    return [np.empty((0, 2), dtype=float) for _ in images]


@app.cell(hide_code=True)
def _(dino_frames, mo):
    checked_count = mo.ui.slider(1, 100, value=60, label="Maximum keypoints", show_value=True, debounce=True)
    checked_length = mo.ui.slider(2, len(dino_frames), value=min(12, len(dino_frames)), label="Frames to track", show_value=True, debounce=True)
    checked_threshold = mo.ui.slider(0.1, 5.0, step=0.1, value=1.0, label="Forward–backward threshold (pixels)", show_value=True, debounce=True)
    checked_run = mo.ui.run_button(label="Run checked tracking")
    return checked_count, checked_length, checked_run, checked_threshold


@app.cell(hide_code=True)
def _(
    checked_count,
    checked_length,
    checked_run,
    checked_threshold,
    dino_frames,
    mo,
):
    checked_tracks, checked_animation = None, None
    checked_message = "Choose your settings, then run tracking."
    if checked_run.value:
        with mo.status.spinner(title="Tracking dinosaur points..."):
            _images = [frame.astype(float) / 255.0 for frame in dino_frames[:checked_length.value]]
            checked_tracks = track(_images, n=checked_count.value, threshold=checked_threshold.value)
            checked_animation = tracking_animation(dino_frames[:len(checked_tracks)], checked_tracks)
        checked_message = f"{len(checked_tracks[0])} point identities across {len(checked_tracks)} frames."
    checked_frame = mo.ui.slider(1, max(2, len(checked_tracks) if checked_tracks is not None else 2), value=1,
                                  label="Inspect frame", show_value=True, disabled=checked_tracks is None, debounce=True)
    checked_trails = mo.ui.checkbox(value=True, label="Show track history")
    return (
        checked_animation,
        checked_frame,
        checked_message,
        checked_tracks,
        checked_trails,
    )


@app.cell(hide_code=True)
def _(
    checked_animation,
    checked_count,
    checked_frame,
    checked_length,
    checked_message,
    checked_run,
    checked_threshold,
    checked_tracks,
    checked_trails,
    dino_frames,
    mo,
):
    _parts = [mo.hstack([checked_count, checked_length, checked_threshold]), checked_run,
              mo.md(checked_message)]
    if checked_tracks is not None:
        _index = min(checked_frame.value - 1, len(checked_tracks) - 1)
        _parts += [mo.hstack([checked_frame, checked_trails]), mo.ui.tabs({
            "Animation": mo.image(checked_animation, alt="Tracked dinosaur keypoints"),
            "Correspondences": tracking_correspondences(dino_frames, checked_tracks, _index, checked_trails.value),
        })]
    mo.vstack(_parts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5(e)

    Answer the following written questions either on a sheet of paper (with attached PDF), or in the cell below.

    1. In your own words, explain why using the forward and backward optical flow estimates can help reduce tracking errors. Be sure to include a discussion of what kinds of errors this approach aims to solve.

    2. What type of errors might still be present? Describe a video sequence which might fail to be tracked correctly.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Your answer to Problem 5(e).**

    You may either write your response here in LaTeX (if you are concerned your handwriting is illegible), or include your written response in the attached PDF. If your response is in the attached PDF, please indicate so here. You may (and should) clear all of the content in this cell.

    1. Your answer.

    2. Your answer.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Optional - your own video

    This is strictly optional and will not be graded. Using the UI below, you can upload a video file of your own and see the keypoints tracked using your implementation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    video_upload = mo.ui.file(filetypes=[".mp4", ".mov", ".avi", ".webm", ".mkv"], label="Upload a video")
    video_method = mo.ui.dropdown(["Forward only", "Forward–backward check"], value="Forward–backward check", label="Tracking method")
    video_count = mo.ui.slider(1, 100, value=60, label="Maximum keypoints", show_value=True, debounce=True)
    video_limit = mo.ui.slider(2, 120, value=30, label="Maximum sampled frames", show_value=True, debounce=True)
    video_stride = mo.ui.slider(1, 10, value=1, label="Keep every nth frame", show_value=True, debounce=True)
    video_threshold = mo.ui.slider(0.1, 5.0, step=0.1, value=1.0, label="Forward–backward threshold (pixels)", show_value=True, debounce=True)
    video_run = mo.ui.run_button(label="Track uploaded video")
    return (
        video_count,
        video_limit,
        video_method,
        video_run,
        video_stride,
        video_threshold,
        video_upload,
    )


@app.cell(hide_code=True)
def _(
    mo,
    video_count,
    video_limit,
    video_method,
    video_run,
    video_stride,
    video_threshold,
    video_upload,
):
    video_frames = None
    video_tracks = None
    if not video_run.value:
        _display = mo.md("Upload a video, choose your settings, and run tracking.")
    elif not video_upload.value:
        _display = mo.callout("Choose a video file first.", kind="info")
    else:
        import cv2 as _cv2
        import tempfile as _tempfile
        from pathlib import Path as _Path

        with mo.status.spinner(title="Decoding and tracking your video..."):
            _upload = video_upload.value[0]
            with _tempfile.TemporaryDirectory() as _directory:
                _path = _Path(_directory) / ("upload" + _Path(_upload.name).suffix)
                _path.write_bytes(_upload.contents)
                _capture = _cv2.VideoCapture(str(_path))
                _frames = []
                _fps = _capture.get(_cv2.CAP_PROP_FPS)
                try:
                    _source_index = 0
                    while len(_frames) < video_limit.value:
                        _ok, _frame = _capture.read()
                        if not _ok:
                            break
                        if _source_index % video_stride.value == 0:
                            # Bound the image size so custom videos stay practical to track.
                            _height, _width = _frame.shape[:2]
                            if max(_height, _width) > 720:
                                _scale = 720 / max(_height, _width)
                                _frame = _cv2.resize(_frame, (round(_width * _scale), round(_height * _scale)), interpolation=_cv2.INTER_AREA)
                            _frames.append(_cv2.cvtColor(_frame, _cv2.COLOR_BGR2RGB))
                        _source_index += 1
                finally:
                    _capture.release()
            if video_method.value == "Forward only":
                _result = track_forward([frame.astype(float) / 255.0 for frame in _frames], n=video_count.value)
            else:
                _result = track([frame.astype(float) / 255.0 for frame in _frames], n=video_count.value, threshold=video_threshold.value)
            _tracks = _result
            video_frames, video_tracks = _frames, _tracks
            _playback_fps = _fps / video_stride.value if 0 < _fps <= 240 else 8
            _display = mo.vstack([
                mo.md(f"**{len(_frames)} frames, {len(_tracks[0])} point identities.**"),
                mo.image(tracking_animation(_frames, _tracks, fps=_playback_fps), alt="Your video with tracked points"),
            ])
    video_preview = _display
    video_frame = mo.ui.slider(1, max(2, len(video_tracks) if video_tracks is not None else 2), value=1,
                               label="Inspect frame", show_value=True, disabled=video_tracks is None, debounce=True)
    video_trails = mo.ui.checkbox(value=True, label="Show track history")
    return video_frame, video_frames, video_preview, video_tracks, video_trails


@app.cell(hide_code=True)
def _(
    mo,
    video_count,
    video_frame,
    video_frames,
    video_limit,
    video_method,
    video_preview,
    video_run,
    video_stride,
    video_threshold,
    video_tracks,
    video_trails,
    video_upload,
):
    _parts = [video_upload, video_method, mo.hstack([video_count, video_limit]),
              mo.hstack([video_stride, video_threshold]), video_run]
    if video_tracks is None:
        _parts.append(video_preview)
    else:
        _index = min(video_frame.value - 1, len(video_tracks) - 1)
        _parts += [mo.hstack([video_frame, video_trails]), mo.ui.tabs({
            "Animation": video_preview,
            "Correspondences": tracking_correspondences(video_frames, video_tracks, _index, video_trails.value),
        })]
    mo.vstack(_parts)
    return


if __name__ == "__main__":
    app.run()
