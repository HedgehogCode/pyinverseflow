# distutils: language = c++
import numpy as np
cimport numpy as np

cdef extern from "inverse_flow/backward_flow.h":
    int backward_flow(
        const float *I1r,
        const float *I1g,
        const float *I1b,
        const float *I2r,
        const float *I2g,
        const float *I2b,
        const float *u,
        const float *v,
        float *u_,
        float *v_,
        float *mask,
        int nx,
        int ny,
        int strategy,
        int fill,
    );

def inverse_flow(
    np.ndarray[float, ndim=3, mode="c"] flow not None,
    np.ndarray[float, ndim=3, mode="c"] img1=None,
    np.ndarray[float, ndim=3, mode="c"] img2=None,
    str strategy="max_flow",
    str fill="oriented",
):
    """Compute the backward flow from the forward flow.

    This function uses the algorithms and implementation described by
    Javier Sánchez, Agustín Salgado and Nelson Monzón.

    Args:
        flow (ndarray): Forward flow with shape [H, W, 2].
            Note that flow[:,:,0] is the flow in y direction
            and flow[:,:,1] is the flow in x direction.
        img1 (ndarray, optional): First image with shape [H, W, 3].
        img2 (ndarray, optional): Second image with shape [H, W, 3].
        strategy (string): One of ["max_flow", "max_image", "avg_flow", "avg_image"].
            If "max_image" or "avg_image" are used, the images must be given. Default "max_flow".
        fill (string): One of ["min", "avg", "oriented", "none"]. Default "oriented".

    Returns:
        The tuple (backwards_flow, mask). The backwards_flow has the same shape as flow.
        The mask has the shape [H, W].

    See:
        Details of the algorithms:
            https://ctim.ulpgc.es/research_works/computing_inverse_optical_flow/
    """

    # Get height and width
    cdef int h = flow.shape[0]
    cdef int w = flow.shape[1]

    # Get the strategy
    cdef int strategy_i
    if strategy == "max_flow":
        strategy_i = 1
    elif strategy == "max_image":
        strategy_i = 2
    elif strategy == "avg_flow":
        strategy_i = 3
    elif strategy == "avg_image":
        strategy_i = 4
    else:
        raise ValueError("Unknown strategy.")

    # Get the fill strategy
    cdef int fill_i
    if fill == "min":
        fill_i = 1
    elif fill == "avg":
        fill_i = 2
    elif fill == "oriented":
        fill_i = 3
    elif fill == "none":
        fill_i = 4
    else:
        raise ValueError("Unknown fill.")

    # Prepare flow input
    cdef np.ndarray[float, ndim=2, mode="c"] u = np.ascontiguousarray(flow[..., 1])
    cdef np.ndarray[float, ndim=2, mode="c"] v = np.ascontiguousarray(flow[..., 0])

    # Prepare image input
    cdef np.ndarray[float, ndim=2, mode="c"] i1r
    cdef np.ndarray[float, ndim=2, mode="c"] i1g
    cdef np.ndarray[float, ndim=2, mode="c"] i1b
    cdef np.ndarray[float, ndim=2, mode="c"] i2r
    cdef np.ndarray[float, ndim=2, mode="c"] i2g
    cdef np.ndarray[float, ndim=2, mode="c"] i2b

    if img1 is not None and img2 is not None:
        i1r = np.ascontiguousarray(img1[..., 0])
        i1g = np.ascontiguousarray(img1[..., 1])
        i1b = np.ascontiguousarray(img1[..., 2])
        i2r = np.ascontiguousarray(img2[..., 0])
        i2g = np.ascontiguousarray(img2[..., 1])
        i2b = np.ascontiguousarray(img2[..., 2])
    elif strategy in ["max_image", "avg_image"]:
        raise ValueError("Both images must be given for this strategy.")
    else:
        i1r = np.ascontiguousarray(np.zeros((1, 1), dtype=np.float32))
        i1g = np.ascontiguousarray(np.zeros((1, 1), dtype=np.float32))
        i1b = np.ascontiguousarray(np.zeros((1, 1), dtype=np.float32))
        i2r = np.ascontiguousarray(np.zeros((1, 1), dtype=np.float32))
        i2g = np.ascontiguousarray(np.zeros((1, 1), dtype=np.float32))
        i2b = np.ascontiguousarray(np.zeros((1, 1), dtype=np.float32))

    # Prepare output
    cdef np.ndarray[float, ndim=2, mode="c"] u_ = np.ascontiguousarray(
        np.zeros((h, w), dtype=np.float32)
    )
    cdef np.ndarray[float, ndim=2, mode="c"] v_ = np.ascontiguousarray(
        np.zeros((h, w), dtype=np.float32)
    )
    cdef np.ndarray[float, ndim=2, mode="c"] mask = np.ascontiguousarray(
        np.zeros((h, w), dtype=np.float32)
    )

    ret_val = backward_flow(
        &i1r[0, 0],
        &i1g[0, 0],
        &i1b[0, 0],
        &i2r[0, 0],
        &i2g[0, 0],
        &i2b[0, 0],
        &u[0, 0],
        &v[0, 0],
        &u_[0, 0],
        &v_[0, 0],
        &mask[0, 0],
        w,
        h,
        strategy_i,
        fill_i,
    )

    return np.stack([v_, u_], axis=-1), mask