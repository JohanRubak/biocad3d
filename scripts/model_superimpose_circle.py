from pathlib import Path

import numpy as np
import pyvista as pv
import vtk

from scipy.spatial import cKDTree


# ============================================================
# DATA LOCATION
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\au662213\OneDrive - Aarhus universitet"
    r"\Forskning - Tandlægeskolen\_IOOS_036_Johan"
    r"\BioCal3D\30-07-26 Pilot data"
)

FOLDERS = {
    "Before brushing": DATA_ROOT / "Before brushing",
    "After brushing": DATA_ROOT / "After brushing",
    "Extraoral scanner": DATA_ROOT / "Extraoral scanner",
    "White background": DATA_ROOT / "White background",
}


# ============================================================
# STAGE 1 — FULL MODEL ICP SETTINGS
# ============================================================

STAGE1_ITERATIONS = 300
STAGE1_LANDMARKS = 20000


# ============================================================
# CIRCLE LANDMARK SETTINGS
# ============================================================

N_CIRCLE_LANDMARKS = 8

CIRCLE_LANDMARK_NAMES = [
    "Top of circle",
    "Upper-right",
    "Right",
    "Lower-right",
    "Bottom",
    "Lower-left",
    "Left",
    "Upper-left",
]


# ============================================================
# FINAL CIRCLE-ROI ICP SETTINGS
# ============================================================

# IMPORTANT:
#
# The WHOLE circular top surface is used for final ICP.
#
# 1.00 means everything from the centre out to the
# landmark-defined circle radius.
FINAL_ICP_RADIUS_FRACTION = 1.00

# Allow geometry above/below fitted plane.
FINAL_ICP_HALF_THICKNESS = 2.0

# Only approximately top-facing triangles are used.
# This helps prevent the surrounding holder and vertical
# surfaces from entering the final ICP region.
FINAL_ICP_MIN_NORMAL_ALIGNMENT = 0.50

FINAL_ICP_ITERATIONS = 400
FINAL_ICP_LANDMARKS = 20000


# ============================================================
# FINAL ICP SAFETY SETTINGS
# ============================================================

# Because a circle is rotationally symmetric, prevent
# the local ICP from making a large additional rotation.
MAX_FINAL_ICP_ROTATION_DEG = 5.0

# Final ICP should not strongly worsen manually defined
# circle landmark correspondence.
MAX_LANDMARK_ERROR_INCREASE_RATIO = 1.15
MAX_LANDMARK_ERROR_ABS_INCREASE = 0.03


# ============================================================
# SURFACE-DIFFERENCE ROI SETTINGS
# ============================================================

# Surface difference is calculated over almost the entire
# circular sample.
#
# Only outermost 5% of radius is excluded.
ANALYSIS_RADIUS_FRACTION = 0.95

ANALYSIS_HALF_THICKNESS = 2.0

MIN_ANALYSIS_NORMAL_ALIGNMENT = 0.60


# ============================================================
# FIXED-LOCATION SURFACE COMPARISON SETTINGS
# ============================================================

# Number of neighbouring AFTER vertices used to interpolate
# AFTER surface height at each BEFORE (u,v) coordinate.
UV_INTERPOLATION_NEIGHBOURS = 6

# Automatic maximum interpolation radius:
#
# max UV distance =
# typical mesh spacing * this value.
UV_MAX_DISTANCE_FACTOR = 4.0


# ============================================================
# FIND FILES
# ============================================================

files = {}

for category, folder in FOLDERS.items():

    if folder.exists():

        files[category] = sorted(
            folder.glob("*.ply")
        )

    else:

        files[category] = []


# ============================================================
# LOAD MESH
# ============================================================

def load_mesh(path):

    mesh = pv.read(path)

    print("\n" + "=" * 80)
    print(f"Loaded: {path.name}")
    print("=" * 80)

    print(
        f"Points: {mesh.n_points:,}"
    )

    print(
        f"Cells:  {mesh.n_cells:,}"
    )

    mesh = mesh.clean()

    mesh = mesh.triangulate()

    mesh = mesh.compute_normals(
        point_normals=True,
        cell_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
    )

    return mesh


# ============================================================
# NORMALIZE VECTOR
# ============================================================

def normalize(vector):

    vector = np.asarray(
        vector,
        dtype=np.float64,
    )

    length = np.linalg.norm(
        vector
    )

    if length < 1e-12:

        raise RuntimeError(
            "Cannot normalize a near-zero vector."
        )

    return vector / length


# ============================================================
# PICK CIRCLE LANDMARKS
# ============================================================

def pick_circle_landmarks(
    mesh,
    title,
):

    picked_points = []

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(
        f"""
Select exactly {N_CIRCLE_LANDMARKS} landmarks on the
CIRCUMFERENCE of the circular specimen.

The full-model ICP has already approximately aligned
the two scans.

Select the SAME visual positions in this order:

1. Top
2. Upper-right
3. Right
4. Lower-right
5. Bottom
6. Lower-left
7. Left
8. Upper-left

Use the physical circular border as consistently as possible.

These landmarks are used to:

1. locally align the circular samples
2. fit the circular plane and radius
3. define the complete circle ROI for final ICP
4. define the subsequent surface-difference ROI

LEFT CLICK  = select
RIGHT CLICK = remove previous landmark

Close the window when all points are selected.
"""
    )

    plotter = pv.Plotter(
        window_size=(1400, 900)
    )

    plotter.add_mesh(
        mesh,
        rgb=True,
        smooth_shading=False,
        show_edges=False,
    )

    plotter.add_text(
        title
        + "\n\n"
        + "\n".join(
            f"{i + 1}: {name}"
            for i, name in enumerate(
                CIRCLE_LANDMARK_NAMES
            )
        )
        + "\n\nLEFT CLICK = select"
        + "\nRIGHT CLICK = remove last",
        position="upper_left",
        font_size=11,
        color="black",
    )

    plotter.add_axes()

    picker = vtk.vtkCellPicker()

    picker.SetTolerance(
        0.005
    )

    # --------------------------------------------------------
    # LEFT CLICK
    # --------------------------------------------------------

    def left_click_callback(
        obj,
        event,
    ):

        if (
            len(picked_points)
            >= N_CIRCLE_LANDMARKS
        ):

            obj.SetAbortFlag(1)

            return

        x, y = obj.GetEventPosition()

        picker.Pick(
            x,
            y,
            0,
            plotter.renderer,
        )

        if picker.GetCellId() < 0:

            print(
                "Nothing picked. "
                "Click directly on the mesh."
            )

            obj.SetAbortFlag(1)

            return

        pos = picker.GetPickPosition()

        point = np.array(
            [
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
            ],
            dtype=np.float64,
        )

        picked_points.append(
            point
        )

        idx = len(
            picked_points
        )

        print(
            f"\nLandmark "
            f"{idx}/{N_CIRCLE_LANDMARKS}"
        )

        print(
            f"  {CIRCLE_LANDMARK_NAMES[idx - 1]}"
        )

        print(
            f"  X = {point[0]:.5f}, "
            f"Y = {point[1]:.5f}, "
            f"Z = {point[2]:.5f}"
        )

        marker = pv.PolyData(
            point.reshape(
                1,
                3,
            )
        )

        plotter.add_mesh(
            marker,
            color="red",
            point_size=16,
            render_points_as_spheres=True,
            name=f"circle_landmark_{idx}",
        )

        plotter.add_point_labels(
            marker,
            [str(idx)],
            font_size=18,
            text_color="red",
            point_color="red",
            shape=None,
            always_visible=True,
            name=f"circle_label_{idx}",
        )

        plotter.render()

        obj.SetAbortFlag(1)

        if (
            len(picked_points)
            == N_CIRCLE_LANDMARKS
        ):

            print(
                "\nAll circle landmarks selected."
            )

            print(
                "Close the window to continue."
            )

    # --------------------------------------------------------
    # RIGHT CLICK
    # --------------------------------------------------------

    def right_click_callback(
        obj,
        event,
    ):

        if len(
            picked_points
        ) == 0:

            obj.SetAbortFlag(1)

            return

        idx = len(
            picked_points
        )

        removed = (
            picked_points.pop()
        )

        print(
            f"\nRemoved landmark {idx}: "
            f"{removed}"
        )

        try:

            plotter.remove_actor(
                f"circle_landmark_{idx}"
            )

        except Exception:

            pass

        try:

            plotter.remove_actor(
                f"circle_label_{idx}"
            )

        except Exception:

            pass

        plotter.render()

        obj.SetAbortFlag(1)

    plotter.iren.add_observer(
        "LeftButtonPressEvent",
        left_click_callback,
    )

    plotter.iren.add_observer(
        "RightButtonPressEvent",
        right_click_callback,
    )

    plotter.show()

    if (
        len(picked_points)
        != N_CIRCLE_LANDMARKS
    ):

        raise RuntimeError(
            f"Expected "
            f"{N_CIRCLE_LANDMARKS} landmarks "
            f"but got "
            f"{len(picked_points)}."
        )

    return np.asarray(
        picked_points,
        dtype=np.float64,
    )


# ============================================================
# FIT 3D CIRCLE
# ============================================================

def fit_circle_3d(points):

    """
    Fit a plane and circle to the manually selected
    circumference landmarks.
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    centroid = np.mean(
        points,
        axis=0,
    )

    centered = (
        points
        - centroid
    )

    # --------------------------------------------------------
    # BEST-FIT PLANE
    # --------------------------------------------------------

    _, _, Vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    normal = normalize(
        Vt[-1]
    )

    # --------------------------------------------------------
    # CREATE LOCAL PLANE AXES
    # --------------------------------------------------------

    x_axis = (
        points[0]
        - centroid
    )

    x_axis = (
        x_axis
        - np.dot(
            x_axis,
            normal,
        )
        * normal
    )

    x_axis = normalize(
        x_axis
    )

    y_axis = normalize(
        np.cross(
            normal,
            x_axis,
        )
    )

    # Maintain orientation according to landmark order.
    if (
        np.dot(
            points[1]
            - centroid,
            y_axis,
        )
        < 0
    ):

        normal = -normal

        y_axis = normalize(
            np.cross(
                normal,
                x_axis,
            )
        )

    # --------------------------------------------------------
    # PROJECT LANDMARKS INTO PLANE
    # --------------------------------------------------------

    x = (
        centered
        @ x_axis
    )

    y = (
        centered
        @ y_axis
    )

    # --------------------------------------------------------
    # LEAST-SQUARES CIRCLE FIT
    # --------------------------------------------------------

    A = np.column_stack(
        [
            2.0 * x,
            2.0 * y,
            np.ones(
                len(x)
            ),
        ]
    )

    b = (
        x**2
        + y**2
    )

    solution, _, _, _ = (
        np.linalg.lstsq(
            A,
            b,
            rcond=None,
        )
    )

    cx = solution[0]
    cy = solution[1]
    c = solution[2]

    radius_squared = (
        c
        + cx**2
        + cy**2
    )

    if radius_squared <= 0:

        raise RuntimeError(
            "Invalid fitted circle radius."
        )

    radius = np.sqrt(
        radius_squared
    )

    center = (
        centroid
        + cx * x_axis
        + cy * y_axis
    )

    # --------------------------------------------------------
    # REDEFINE AXES FROM FINAL CIRCLE CENTRE
    # --------------------------------------------------------

    x_axis = (
        points[0]
        - center
    )

    x_axis = (
        x_axis
        - np.dot(
            x_axis,
            normal,
        )
        * normal
    )

    x_axis = normalize(
        x_axis
    )

    y_axis = normalize(
        np.cross(
            normal,
            x_axis,
        )
    )

    if (
        np.dot(
            points[1]
            - center,
            y_axis,
        )
        < 0
    ):

        normal = -normal

        y_axis = normalize(
            np.cross(
                normal,
                x_axis,
            )
        )

    # --------------------------------------------------------
    # FIT QUALITY
    # --------------------------------------------------------

    vectors = (
        points
        - center
    )

    plane_coordinates = (
        vectors
        @ normal
    )

    plane_rms = np.sqrt(
        np.mean(
            plane_coordinates**2
        )
    )

    projected = (
        vectors
        - np.outer(
            plane_coordinates,
            normal,
        )
    )

    radial_distances = (
        np.linalg.norm(
            projected,
            axis=1,
        )
    )

    radial_rms = np.sqrt(
        np.mean(
            (
                radial_distances
                - radius
            )**2
        )
    )

    return {
        "center":
            center,

        "radius":
            radius,

        "normal":
            normal,

        "x_axis":
            x_axis,

        "y_axis":
            y_axis,

        "plane_rms":
            plane_rms,

        "radial_rms":
            radial_rms,
    }


# ============================================================
# PRINT CIRCLE FIT
# ============================================================

def print_circle_fit(
    circle,
    name,
):

    print("\n" + "=" * 80)
    print(f"{name} CIRCLE FIT")
    print("=" * 80)

    print(
        f"Centre:     "
        f"{circle['center']}"
    )

    print(
        f"Radius:     "
        f"{circle['radius']:.6f}"
    )

    print(
        f"Normal:     "
        f"{circle['normal']}"
    )

    print(
        f"Plane RMS:  "
        f"{circle['plane_rms']:.6f}"
    )

    print(
        f"Radial RMS: "
        f"{circle['radial_rms']:.6f}"
    )


# ============================================================
# CREATE CIRCLE OUTLINE
# ============================================================

def create_circle_outline(
    circle,
    radius_fraction=1.0,
):

    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        250,
        endpoint=False,
    )

    radius = (
        circle["radius"]
        * radius_fraction
    )

    points = (
        circle["center"][None, :]
        + radius
        * np.cos(angles)[:, None]
        * circle["x_axis"][None, :]
        + radius
        * np.sin(angles)[:, None]
        * circle["y_axis"][None, :]
    )

    points = np.vstack(
        [
            points,
            points[0],
        ]
    )

    return pv.lines_from_points(
        points
    )


# ============================================================
# VISUALIZE CIRCLE FIT
# ============================================================

def visualize_circle_fit(
    mesh,
    landmarks,
    circle,
    title,
):

    circle_line = (
        create_circle_outline(
            circle
        )
    )

    plotter = pv.Plotter(
        window_size=(1400, 900)
    )

    plotter.add_mesh(
        mesh,
        rgb=True,
        smooth_shading=False,
        show_edges=False,
    )

    plotter.add_mesh(
        circle_line,
        color="yellow",
        line_width=5,
    )

    plotter.add_points(
        landmarks,
        color="red",
        point_size=16,
        render_points_as_spheres=True,
    )

    plotter.add_points(
        circle["center"],
        color="black",
        point_size=18,
        render_points_as_spheres=True,
    )

    normal_end = (
        circle["center"]
        + circle["normal"]
        * circle["radius"]
        * 0.5
    )

    normal_line = (
        pv.lines_from_points(
            np.vstack(
                [
                    circle["center"],
                    normal_end,
                ]
            )
        )
    )

    plotter.add_mesh(
        normal_line,
        color="cyan",
        line_width=5,
    )

    plotter.add_text(
        f"{title}\n\n"
        f"Radius = {circle['radius']:.4f}\n"
        f"Plane RMS = {circle['plane_rms']:.4f}\n"
        f"Radial RMS = {circle['radial_rms']:.4f}\n\n"
        "YELLOW = fitted circle\n"
        "CYAN = plane normal",
        position="upper_left",
        font_size=14,
    )

    plotter.add_axes()

    plotter.show()


# ============================================================
# CALCULATE RIGID TRANSFORM
# ============================================================

def calculate_rigid_transform(
    source_points,
    target_points,
):

    source_points = np.asarray(
        source_points,
        dtype=np.float64,
    )

    target_points = np.asarray(
        target_points,
        dtype=np.float64,
    )

    source_centroid = np.mean(
        source_points,
        axis=0,
    )

    target_centroid = np.mean(
        target_points,
        axis=0,
    )

    source_centered = (
        source_points
        - source_centroid
    )

    target_centered = (
        target_points
        - target_centroid
    )

    H = (
        source_centered.T
        @ target_centered
    )

    U, _, Vt = (
        np.linalg.svd(H)
    )

    R = (
        Vt.T
        @ U.T
    )

    if np.linalg.det(R) < 0:

        Vt[-1, :] *= -1

        R = (
            Vt.T
            @ U.T
        )

    t = (
        target_centroid
        - R
        @ source_centroid
    )

    matrix = vtk.vtkMatrix4x4()

    matrix.Identity()

    for i in range(3):

        for j in range(3):

            matrix.SetElement(
                i,
                j,
                float(
                    R[i, j]
                ),
            )

        matrix.SetElement(
            i,
            3,
            float(
                t[i]
            ),
        )

    return matrix


# ============================================================
# TRANSFORM POINTS
# ============================================================

def transform_points(
    points,
    matrix,
):

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    output = np.zeros_like(
        points
    )

    for i, point in enumerate(
        points
    ):

        x = np.array(
            [
                point[0],
                point[1],
                point[2],
                1.0,
            ],
            dtype=np.float64,
        )

        y = np.zeros(
            4,
            dtype=np.float64,
        )

        for row in range(4):

            y[row] = sum(
                matrix.GetElement(
                    row,
                    col,
                )
                * x[col]
                for col in range(4)
            )

        output[i] = y[:3]

    return output


# ============================================================
# ICP
# ============================================================

def icp_register(
    source,
    target,
    max_iterations,
    max_landmarks,
    match_centroids=False,
):

    print(
        "\nRunning ICP..."
    )

    icp = (
        vtk.vtkIterativeClosestPointTransform()
    )

    icp.SetSource(
        source
    )

    icp.SetTarget(
        target
    )

    icp.GetLandmarkTransform().SetModeToRigidBody()

    icp.SetMaximumNumberOfIterations(
        max_iterations
    )

    icp.SetMaximumNumberOfLandmarks(
        min(
            max_landmarks,
            source.n_points,
        )
    )

    icp.SetMaximumMeanDistance(
        1e-6
    )

    icp.CheckMeanDistanceOn()

    if match_centroids:

        icp.StartByMatchingCentroidsOn()

    else:

        icp.StartByMatchingCentroidsOff()

    icp.Modified()

    icp.Update()

    transform = (
        icp.GetMatrix()
    )

    registered = (
        source.copy()
    )

    registered.transform(
        transform,
        inplace=True,
    )

    try:

        mean_distance = (
            icp.GetMeanDistance()
        )

    except Exception:

        mean_distance = None

    try:

        iterations = (
            icp.GetNumberOfIterations()
        )

    except Exception:

        iterations = None

    print(
        "ICP completed."
    )

    if mean_distance is not None:

        print(
            f"ICP mean distance: "
            f"{mean_distance:.8f}"
        )

    if iterations is not None:

        print(
            f"Iterations: "
            f"{iterations}"
        )

    return (
        registered,
        transform,
        mean_distance,
        iterations,
    )


# ============================================================
# TRANSFORM ROTATION ANGLE
# ============================================================

def transform_rotation_angle_deg(
    matrix,
):

    R = np.array(
        [
            [
                matrix.GetElement(
                    i,
                    j,
                )
                for j in range(3)
            ]
            for i in range(3)
        ],
        dtype=np.float64,
    )

    value = (
        (
            np.trace(R)
            - 1.0
        )
        / 2.0
    )

    value = np.clip(
        value,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(
                value
            )
        )
    )


# ============================================================
# VISUALIZE OVERLAY
# ============================================================

def visualize_overlay(
    before,
    after,
    title,
    before_landmarks=None,
    after_landmarks=None,
):

    plotter = pv.Plotter(
        window_size=(1500, 900)
    )

    plotter.add_mesh(
        before,
        color="blue",
        opacity=0.45,
        smooth_shading=False,
        show_edges=False,
    )

    plotter.add_mesh(
        after,
        color="red",
        opacity=0.45,
        smooth_shading=False,
        show_edges=False,
    )

    if before_landmarks is not None:

        plotter.add_points(
            before_landmarks,
            color="blue",
            point_size=16,
            render_points_as_spheres=True,
        )

    if after_landmarks is not None:

        plotter.add_points(
            after_landmarks,
            color="red",
            point_size=16,
            render_points_as_spheres=True,
        )

    plotter.add_text(
        f"{title}\n\n"
        "BEFORE = BLUE\n"
        "AFTER = RED\n\n"
        "Close window to continue",
        position="upper_left",
        font_size=14,
    )

    plotter.add_axes()

    plotter.show()


# ============================================================
# CREATE CIRCULAR CELL MASK
# ============================================================

def create_circle_cell_mask(
    mesh,
    circle,
    radial_min_fraction,
    radial_max_fraction,
    half_thickness,
    minimum_normal_alignment=None,
):

    faces = mesh.faces.reshape(
        -1,
        4,
    )

    triangles = (
        faces[:, 1:4]
    )

    points = (
        mesh.points
    )

    p0 = points[
        triangles[:, 0]
    ]

    p1 = points[
        triangles[:, 1]
    ]

    p2 = points[
        triangles[:, 2]
    ]

    centroids = (
        p0
        + p1
        + p2
    ) / 3.0

    vectors = (
        centroids
        - circle["center"]
    )

    plane_distance = (
        vectors
        @ circle["normal"]
    )

    projected = (
        vectors
        - np.outer(
            plane_distance,
            circle["normal"],
        )
    )

    radial_distance = (
        np.linalg.norm(
            projected,
            axis=1,
        )
    )

    minimum_radius = (
        circle["radius"]
        * radial_min_fraction
    )

    maximum_radius = (
        circle["radius"]
        * radial_max_fraction
    )

    mask = (
        (radial_distance >= minimum_radius)
        & (radial_distance <= maximum_radius)
        & (
            np.abs(
                plane_distance
            )
            <= half_thickness
        )
    )

    # --------------------------------------------------------
    # SURFACE ORIENTATION FILTER
    # --------------------------------------------------------

    if (
        minimum_normal_alignment
        is not None
    ):

        triangle_normals = np.cross(
            p1 - p0,
            p2 - p0,
        )

        lengths = (
            np.linalg.norm(
                triangle_normals,
                axis=1,
            )
        )

        valid = (
            lengths > 1e-12
        )

        unit_normals = (
            np.zeros_like(
                triangle_normals
            )
        )

        unit_normals[
            valid
        ] = (
            triangle_normals[
                valid
            ]
            / lengths[
                valid,
                None,
            ]
        )

        alignment = np.abs(
            unit_normals
            @ circle["normal"]
        )

        mask &= (
            alignment
            >= minimum_normal_alignment
        )

    return mask


# ============================================================
# EXTRACT CELL REGION
# ============================================================

def extract_cell_region(
    mesh,
    mask,
):

    indices = np.where(
        mask
    )[0]

    if len(indices) == 0:

        raise RuntimeError(
            "No triangles found in selected circular region."
        )

    region = (
        mesh.extract_cells(
            indices
        )
    )

    region = (
        region
        .extract_surface()
        .triangulate()
        .clean()
    )

    return region


# ============================================================
# EXTRACT COMPLETE CIRCLE ROI FOR FINAL ICP
# ============================================================

def extract_final_icp_region(
    mesh,
    circle,
):

    mask = (
        create_circle_cell_mask(
            mesh=mesh,
            circle=circle,

            radial_min_fraction=
                0.0,

            radial_max_fraction=
                FINAL_ICP_RADIUS_FRACTION,

            half_thickness=
                FINAL_ICP_HALF_THICKNESS,

            minimum_normal_alignment=
                FINAL_ICP_MIN_NORMAL_ALIGNMENT,
        )
    )

    return extract_cell_region(
        mesh,
        mask,
    )


# ============================================================
# EXTRACT SLIGHTLY SMALLER ANALYSIS ROI
# ============================================================

def extract_analysis_region(
    mesh,
    reference_circle,
):

    mask = (
        create_circle_cell_mask(
            mesh=mesh,
            circle=reference_circle,

            radial_min_fraction=
                0.0,

            radial_max_fraction=
                ANALYSIS_RADIUS_FRACTION,

            half_thickness=
                ANALYSIS_HALF_THICKNESS,

            minimum_normal_alignment=
                MIN_ANALYSIS_NORMAL_ALIGNMENT,
        )
    )

    return extract_cell_region(
        mesh,
        mask,
    )


# ============================================================
# VISUALIZE FINAL ICP REGIONS
# ============================================================

def visualize_final_icp_regions(
    before,
    after,
    before_roi,
    after_roi,
):

    plotter = pv.Plotter(
        shape=(1, 2),
        window_size=(1700, 750),
    )

    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    plotter.subplot(
        0,
        0,
    )

    plotter.add_mesh(
        before,
        color="lightgray",
        opacity=0.15,
    )

    plotter.add_mesh(
        before_roi,
        color="orange",
        opacity=1.0,
    )

    plotter.add_text(
        "BEFORE\n"
        "ORANGE = COMPLETE CIRCLE ROI\n"
        "USED FOR FINAL ICP",
        font_size=14,
    )

    plotter.add_axes()

    # --------------------------------------------------------
    # AFTER
    # --------------------------------------------------------

    plotter.subplot(
        0,
        1,
    )

    plotter.add_mesh(
        after,
        color="lightgray",
        opacity=0.15,
    )

    plotter.add_mesh(
        after_roi,
        color="orange",
        opacity=1.0,
    )

    plotter.add_text(
        "AFTER\n"
        "ORANGE = COMPLETE CIRCLE ROI\n"
        "USED FOR FINAL ICP",
        font_size=14,
    )

    plotter.add_axes()

    plotter.link_views()

    plotter.show()


# ============================================================
# VISUALIZE ANALYSIS REGIONS
# ============================================================

def visualize_analysis_regions(
    before,
    after,
    before_roi,
    after_roi,
):

    plotter = pv.Plotter(
        shape=(1, 2),
        window_size=(1700, 750),
    )

    # BEFORE
    plotter.subplot(
        0,
        0,
    )

    plotter.add_mesh(
        before,
        color="lightgray",
        opacity=0.15,
    )

    plotter.add_mesh(
        before_roi,
        color="orange",
        opacity=1.0,
    )

    plotter.add_text(
        "BEFORE\n"
        "ORANGE = SURFACE-DIFFERENCE ROI",
        font_size=14,
    )

    plotter.add_axes()

    # AFTER
    plotter.subplot(
        0,
        1,
    )

    plotter.add_mesh(
        after,
        color="lightgray",
        opacity=0.15,
    )

    plotter.add_mesh(
        after_roi,
        color="orange",
        opacity=1.0,
    )

    plotter.add_text(
        "AFTER\n"
        "ORANGE = SURFACE-DIFFERENCE ROI",
        font_size=14,
    )

    plotter.add_axes()

    plotter.link_views()

    plotter.show()


# ============================================================
# TRIANGLE AREAS
# ============================================================

def triangle_areas(
    mesh,
):

    faces = mesh.faces.reshape(
        -1,
        4,
    )

    triangles = (
        faces[:, 1:4]
    )

    points = (
        mesh.points
    )

    p0 = points[
        triangles[:, 0]
    ]

    p1 = points[
        triangles[:, 1]
    ]

    p2 = points[
        triangles[:, 2]
    ]

    cross = np.cross(
        p1 - p0,
        p2 - p0,
    )

    return (
        0.5
        * np.linalg.norm(
            cross,
            axis=1,
        )
    )


# ============================================================
# PROJECT SURFACE TO COMMON CIRCLE COORDINATES
# ============================================================

def project_to_circle_coordinates(
    points,
    circle,
    normal,
):

    """
    Convert 3D vertices into:

        u = coordinate across circle
        v = coordinate across circle
        h = height perpendicular to fitted circle plane

    BEFORE and AFTER are represented in exactly the same
    coordinate system.
    """

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    vectors = (
        points
        - circle["center"]
    )

    u = (
        vectors
        @ circle["x_axis"]
    )

    v = (
        vectors
        @ circle["y_axis"]
    )

    h = (
        vectors
        @ normal
    )

    return (
        u,
        v,
        h,
    )


# ============================================================
# ORIENT NORMAL TOWARD EXPOSED SURFACE
# ============================================================

def orient_normal_to_surface(
    mesh,
    normal,
):

    normal = normalize(
        normal
    )

    # Prefer VTK/PyVista cell normals if present.
    if (
        "Normals"
        in mesh.cell_data
    ):

        normals = np.asarray(
            mesh.cell_data[
                "Normals"
            ],
            dtype=np.float64,
        )

    else:

        faces = mesh.faces.reshape(
            -1,
            4,
        )

        triangles = (
            faces[:, 1:4]
        )

        points = (
            mesh.points
        )

        p0 = points[
            triangles[:, 0]
        ]

        p1 = points[
            triangles[:, 1]
        ]

        p2 = points[
            triangles[:, 2]
        ]

        normals = np.cross(
            p1 - p0,
            p2 - p0,
        )

        lengths = np.linalg.norm(
            normals,
            axis=1,
        )

        valid = (
            lengths > 1e-12
        )

        normals[
            valid
        ] /= lengths[
            valid,
            None,
        ]

        normals = (
            normals[
                valid
            ]
        )

    if len(normals) == 0:

        return normal

    alignment = (
        normals
        @ normal
    )

    # Flip the circle normal if it points predominantly
    # opposite to the top-surface normals.
    if np.median(
        alignment
    ) < 0:

        normal = -normal

    return normal


# ============================================================
# INTERPOLATE AFTER HEIGHT AT BEFORE U,V POSITIONS
# ============================================================

def interpolate_height_at_uv(
    query_uv,
    reference_uv,
    reference_height,
    k=UV_INTERPOLATION_NEIGHBOURS,
):

    """
    Instead of finding the closest 3D surface point,
    estimate AFTER height at exactly the same in-plane
    U,V position as each BEFORE vertex.
    """

    query_uv = np.asarray(
        query_uv,
        dtype=np.float64,
    )

    reference_uv = np.asarray(
        reference_uv,
        dtype=np.float64,
    )

    reference_height = np.asarray(
        reference_height,
        dtype=np.float64,
    )

    tree = cKDTree(
        reference_uv
    )

    # --------------------------------------------------------
    # ESTIMATE TYPICAL MESH SPACING
    # --------------------------------------------------------

    sample_count = min(
        len(reference_uv),
        10000,
    )

    if (
        sample_count
        < len(reference_uv)
    ):

        indices = np.linspace(
            0,
            len(reference_uv) - 1,
            sample_count,
        ).astype(int)

        sample_uv = (
            reference_uv[
                indices
            ]
        )

    else:

        sample_uv = (
            reference_uv
        )

    spacing_distances, _ = (
        tree.query(
            sample_uv,
            k=2,
        )
    )

    typical_spacing = (
        np.median(
            spacing_distances[
                :,
                1,
            ]
        )
    )

    max_distance = (
        typical_spacing
        * UV_MAX_DISTANCE_FACTOR
    )

    print(
        f"\nTypical projected mesh spacing: "
        f"{typical_spacing:.6f}"
    )

    print(
        f"Maximum allowed U,V interpolation distance: "
        f"{max_distance:.6f}"
    )

    # --------------------------------------------------------
    # FIND K NEAREST U,V NEIGHBOURS
    # --------------------------------------------------------

    k = min(
        k,
        len(reference_uv),
    )

    distances, indices = (
        tree.query(
            query_uv,
            k=k,
        )
    )

    if k == 1:

        distances = (
            distances[
                :,
                None,
            ]
        )

        indices = (
            indices[
                :,
                None,
            ]
        )

    # --------------------------------------------------------
    # INVERSE-DISTANCE HEIGHT INTERPOLATION
    # --------------------------------------------------------

    weights = (
        1.0
        / (
            distances
            + 1e-12
        )**2
    )

    neighbour_heights = (
        reference_height[
            indices
        ]
    )

    interpolated_height = (
        np.sum(
            weights
            * neighbour_heights,
            axis=1,
        )
        / np.sum(
            weights,
            axis=1,
        )
    )

    nearest_distance = (
        distances[
            :,
            0,
        ]
    )

    valid = (
        nearest_distance
        <= max_distance
    )

    return (
        interpolated_height,
        valid,
        nearest_distance,
        max_distance,
    )


# ============================================================
# POINT VALUES -> CELL VALUES WITH VALIDITY MASK
# ============================================================

def point_to_cell_values_valid(
    mesh,
    point_values,
    valid_points,
):

    faces = mesh.faces.reshape(
        -1,
        4,
    )

    triangles = (
        faces[:, 1:4]
    )

    # Require all three triangle vertices to have
    # valid correspondence.
    valid_cells = np.all(
        valid_points[
            triangles
        ],
        axis=1,
    )

    cell_values = np.full(
        len(triangles),
        np.nan,
        dtype=np.float64,
    )

    cell_values[
        valid_cells
    ] = np.mean(
        point_values[
            triangles[
                valid_cells
            ]
        ],
        axis=1,
    )

    return (
        cell_values,
        valid_cells,
    )


# ============================================================
# FIXED-LOCATION SURFACE DIFFERENCE
# ============================================================

def calculate_surface_difference(
    before_roi,
    after_roi,
    reference_circle,
    threshold,
):

    """
    Compare surface height at identical U,V locations.

    Positive:

        BEFORE is higher than AFTER

        -> candidate removed plaque/material

    Approximately zero:

        BEFORE and AFTER have similar surface height

        -> expected where plaque remains unchanged

    Negative:

        AFTER is higher than BEFORE

        -> NOT counted as plaque removal
        -> retained as diagnostic information
    """

    # ========================================================
    # COMMON PLANE NORMAL
    # ========================================================

    normal = (
        orient_normal_to_surface(
            before_roi,
            reference_circle[
                "normal"
            ],
        )
    )

    print(
        "\nAnalysis normal:"
    )

    print(
        normal
    )

    # ========================================================
    # BEFORE U,V,H
    # ========================================================

    (
        before_u,
        before_v,
        before_h,
    ) = project_to_circle_coordinates(
        before_roi.points,
        reference_circle,
        normal,
    )

    before_uv = np.column_stack(
        [
            before_u,
            before_v,
        ]
    )

    # ========================================================
    # AFTER U,V,H
    # ========================================================

    (
        after_u,
        after_v,
        after_h,
    ) = project_to_circle_coordinates(
        after_roi.points,
        reference_circle,
        normal,
    )

    after_uv = np.column_stack(
        [
            after_u,
            after_v,
        ]
    )

    # ========================================================
    # INTERPOLATE AFTER HEIGHT AT SAME U,V LOCATIONS
    # ========================================================

    (
        after_height_at_before,
        valid_points,
        uv_distance,
        max_uv_distance,
    ) = interpolate_height_at_uv(
        query_uv=
            before_uv,

        reference_uv=
            after_uv,

        reference_height=
            after_h,

        k=
            UV_INTERPOLATION_NEIGHBOURS,
    )

    # ========================================================
    # SIGNED HEIGHT CHANGE
    # ========================================================

    # Positive means BEFORE is higher than AFTER.
    signed_height_change = (
        before_h
        - after_height_at_before
    )

    # Positive component only:
    # candidate plaque/material removal.
    positive_removal = np.maximum(
        signed_height_change,
        0.0,
    )

    # Negative change retained for QC.
    negative_change = np.minimum(
        signed_height_change,
        0.0,
    )

    # ========================================================
    # CONVERT POINT VALUES TO TRIANGLE VALUES
    # ========================================================

    (
        signed_height_cell,
        valid_cells,
    ) = point_to_cell_values_valid(
        before_roi,
        signed_height_change,
        valid_points,
    )

    (
        removal_cell,
        _,
    ) = point_to_cell_values_valid(
        before_roi,
        positive_removal,
        valid_points,
    )

    (
        negative_cell,
        _,
    ) = point_to_cell_values_valid(
        before_roi,
        negative_change,
        valid_points,
    )

    (
        uv_distance_cell,
        _,
    ) = point_to_cell_values_valid(
        before_roi,
        uv_distance,
        valid_points,
    )

    # ========================================================
    # AREAS
    # ========================================================

    areas = triangle_areas(
        before_roi
    )

    valid_areas = (
        areas[
            valid_cells
        ]
    )

    valid_signed = (
        signed_height_cell[
            valid_cells
        ]
    )

    valid_removal = (
        removal_cell[
            valid_cells
        ]
    )

    valid_negative = (
        negative_cell[
            valid_cells
        ]
    )

    valid_uv = (
        uv_distance_cell[
            valid_cells
        ]
    )

    # ========================================================
    # THRESHOLDED REMOVAL
    # ========================================================

    changed_mask = (
        valid_removal
        > threshold
    )

    changed_area = np.sum(
        valid_areas[
            changed_mask
        ]
    )

    roi_area = np.sum(
        valid_areas
    )

    changed_fraction = (
        changed_area
        / roi_area
        if roi_area > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Threshold-corrected volume estimate
    # --------------------------------------------------------

    effective_height = np.maximum(
        valid_removal
        - threshold,
        0.0,
    )

    volume_difference = np.sum(
        effective_height
        * valid_areas
    )

    # --------------------------------------------------------
    # Negative area diagnostic
    # --------------------------------------------------------

    negative_mask = (
        valid_signed
        < -threshold
    )

    negative_area = np.sum(
        valid_areas[
            negative_mask
        ]
    )

    negative_fraction = (
        negative_area
        / roi_area
        if roi_area > 0
        else np.nan
    )

    return {
        "signed_height_cell":
            signed_height_cell,

        "removal_cell":
            removal_cell,

        "negative_cell":
            negative_cell,

        "uv_distance_cell":
            uv_distance_cell,

        "valid_cells":
            valid_cells,

        "roi_area":
            roi_area,

        "changed_area":
            changed_area,

        "changed_fraction":
            changed_fraction,

        "negative_area":
            negative_area,

        "negative_fraction":
            negative_fraction,

        "volume_difference":
            volume_difference,

        "mean_signed_height":
            np.mean(
                valid_signed
            ),

        "median_signed_height":
            np.median(
                valid_signed
            ),

        "mean_removal":
            np.mean(
                valid_removal
            ),

        "median_removal":
            np.median(
                valid_removal
            ),

        "removal_95":
            np.percentile(
                valid_removal,
                95,
            ),

        "removal_99":
            np.percentile(
                valid_removal,
                99,
            ),

        "maximum_removal":
            np.max(
                valid_removal
            ),

        "mean_negative":
            np.mean(
                valid_negative
            ),

        "mean_uv_distance":
            np.mean(
                valid_uv
            ),

        "median_uv_distance":
            np.median(
                valid_uv
            ),

        "valid_fraction":
            (
                np.sum(
                    valid_cells
                )
                / len(
                    valid_cells
                )
            ),

        "max_uv_distance":
            max_uv_distance,
    }


# ============================================================
# VISUALIZE SURFACE DIFFERENCE
# ============================================================

def visualize_surface_difference(
    before,
    after,
    before_roi,
    results,
    threshold,
):

    difference_mesh = (
        before_roi.copy()
    )

    removal = (
        results[
            "removal_cell"
        ].copy()
    )

    valid = (
        results[
            "valid_cells"
        ]
    )

    # Hide invalid interpolation regions.
    removal[
        ~valid
    ] = np.nan

    # Everything below threshold is displayed as zero.
    thresholded_removal = (
        removal.copy()
    )

    finite = np.isfinite(
        thresholded_removal
    )

    thresholded_removal[
        finite
        & (
            thresholded_removal
            <= threshold
        )
    ] = 0.0

    difference_mesh.cell_data[
        "Surface height removed"
    ] = thresholded_removal

    valid_values = (
        thresholded_removal[
            np.isfinite(
                thresholded_removal
            )
        ]
    )

    if len(
        valid_values
    ) > 0:

        vmax = np.percentile(
            valid_values,
            99,
        )

        vmax = max(
            vmax,
            threshold * 2.0,
        )

    else:

        vmax = (
            threshold
            * 2.0
        )

    plotter = pv.Plotter(
        shape=(1, 3),
        window_size=(1800, 700),
    )

    # ========================================================
    # BEFORE
    # ========================================================

    plotter.subplot(
        0,
        0,
    )

    plotter.add_mesh(
        before,
        rgb=True,
        smooth_shading=False,
        show_edges=False,
    )

    plotter.add_text(
        "BEFORE BRUSHING",
        font_size=14,
    )

    plotter.add_axes()

    # ========================================================
    # AFTER
    # ========================================================

    plotter.subplot(
        0,
        1,
    )

    plotter.add_mesh(
        after,
        rgb=True,
        smooth_shading=False,
        show_edges=False,
    )

    plotter.add_text(
        "AFTER BRUSHING\n"
        "FINAL REGISTERED",
        font_size=14,
    )

    plotter.add_axes()

    # ========================================================
    # SURFACE REMOVAL
    # ========================================================

    plotter.subplot(
        0,
        2,
    )

    plotter.add_mesh(
        before,
        color="lightgray",
        opacity=0.20,
        smooth_shading=False,
        show_edges=False,
    )

    plotter.add_mesh(
        difference_mesh,
        scalars=
            "Surface height removed",

        cmap="turbo",

        clim=[
            0.0,
            vmax,
        ],

        smooth_shading=False,
        show_edges=False,

        scalar_bar_args={
            "title":
                "Surface height removed"
        },
    )

    plotter.add_text(
        "FIXED-LOCATION SURFACE CHANGE\n"
        "CIRCULAR SAMPLE ROI",
        font_size=14,
    )

    plotter.add_axes()

    plotter.link_views()

    plotter.show()


# ============================================================
# COMPLETE BEFORE / AFTER ANALYSIS
# ============================================================

def analyze_before_after(
    index,
    threshold=0.01,
):

    before_files = (
        files[
            "Before brushing"
        ]
    )

    after_files = (
        files[
            "After brushing"
        ]
    )

    if (
        index
        >= len(before_files)
    ):

        raise RuntimeError(
            "Invalid BEFORE file."
        )

    if (
        index
        >= len(after_files)
    ):

        raise RuntimeError(
            "Invalid AFTER file."
        )

    before_path = (
        before_files[
            index
        ]
    )

    after_path = (
        after_files[
            index
        ]
    )

    print("\n" + "=" * 80)
    print("BEFORE / AFTER CIRCULAR SAMPLE ANALYSIS")
    print("=" * 80)

    print(
        f"Before: "
        f"{before_path.name}"
    )

    print(
        f"After:  "
        f"{after_path.name}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    before = load_mesh(
        before_path
    )

    after = load_mesh(
        after_path
    )

    # ========================================================
    # STAGE 1 — FULL MODEL ICP
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 1 — INITIAL FULL-MODEL ICP")
    print("=" * 80)

    print(
        """
The complete scan is used only for initial registration.

Purpose:

- approximate translation
- approximate tilt
- approximate rotation
- establish the otherwise ambiguous orientation of the circle
"""
    )

    (
        after_stage1,
        stage1_transform,
        stage1_mean,
        stage1_iterations,
    ) = icp_register(
        source=
            after,

        target=
            before,

        max_iterations=
            STAGE1_ITERATIONS,

        max_landmarks=
            STAGE1_LANDMARKS,

        match_centroids=
            True,
    )

    visualize_overlay(
        before,
        after_stage1,
        "STAGE 1 — FULL-MODEL ICP",
    )

    # ========================================================
    # STAGE 2 — BEFORE CIRCLE LANDMARKS
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 2 — DEFINE CIRCULAR SAMPLE")
    print("=" * 80)

    input(
        "\nPress ENTER to select "
        "BEFORE circle landmarks..."
    )

    before_landmarks = (
        pick_circle_landmarks(
            before,
            "BEFORE — CIRCLE LANDMARKS",
        )
    )

    before_circle = (
        fit_circle_3d(
            before_landmarks
        )
    )

    print_circle_fit(
        before_circle,
        "BEFORE",
    )

    visualize_circle_fit(
        before,
        before_landmarks,
        before_circle,
        "BEFORE CIRCLE FIT",
    )

    # ========================================================
    # STAGE 2 — AFTER CIRCLE LANDMARKS
    # ========================================================

    input(
        "\nPress ENTER to select "
        "AFTER circle landmarks..."
    )

    after_landmarks = (
        pick_circle_landmarks(
            after_stage1,
            "AFTER — CIRCLE LANDMARKS",
        )
    )

    after_circle_stage1 = (
        fit_circle_3d(
            after_landmarks
        )
    )

    print_circle_fit(
        after_circle_stage1,
        "AFTER",
    )

    visualize_circle_fit(
        after_stage1,
        after_landmarks,
        after_circle_stage1,
        "AFTER CIRCLE FIT",
    )

    # ========================================================
    # STAGE 3 — LANDMARK-BASED CIRCLE ALIGNMENT
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 3 — CIRCLE LANDMARK REGISTRATION")
    print("=" * 80)

    circle_transform = (
        calculate_rigid_transform(
            source_points=
                after_landmarks,

            target_points=
                before_landmarks,
        )
    )

    after_stage2 = (
        after_stage1.copy()
    )

    after_stage2.transform(
        circle_transform,
        inplace=True,
    )

    after_landmarks_stage2 = (
        transform_points(
            after_landmarks,
            circle_transform,
        )
    )

    circle_landmark_errors = (
        np.linalg.norm(
            before_landmarks
            - after_landmarks_stage2,
            axis=1,
        )
    )

    print(
        "\nCircle landmark errors:"
    )

    for i, error in enumerate(
        circle_landmark_errors
    ):

        print(
            f"{i + 1}. "
            f"{CIRCLE_LANDMARK_NAMES[i]}: "
            f"{error:.6f}"
        )

    pre_final_icp_error = (
        np.mean(
            circle_landmark_errors
        )
    )

    print(
        f"\nMean landmark error: "
        f"{pre_final_icp_error:.6f}"
    )

    print(
        f"Maximum landmark error: "
        f"{np.max(circle_landmark_errors):.6f}"
    )

    visualize_overlay(
        before,
        after_stage2,
        "STAGE 3 — CIRCLE LANDMARK ALIGNMENT",
        before_landmarks=
            before_landmarks,
        after_landmarks=
            after_landmarks_stage2,
    )

    # Refit AFTER circle after landmark transform.
    after_circle_stage2 = (
        fit_circle_3d(
            after_landmarks_stage2
        )
    )

    # ========================================================
    # STAGE 4 — COMPLETE CIRCLE ROI FOR FINAL ICP
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 4 — COMPLETE CIRCULAR ROI FOR FINAL ICP")
    print("=" * 80)

    print(
        f"""
The COMPLETE circular top surface is now used for final ICP.

ICP radius:
    0 <= r <= {FINAL_ICP_RADIUS_FRACTION:.2f} × R

This includes the plaque-bearing surface.

Only geometry inside the landmark-defined circular specimen
is used. The teeth and square holder are excluded.

Surface-difference analysis will subsequently use:

    r <= {ANALYSIS_RADIUS_FRACTION:.2f} × R
"""
    )

    before_final_icp_roi = (
        extract_final_icp_region(
            before,
            before_circle,
        )
    )

    after_final_icp_roi = (
        extract_final_icp_region(
            after_stage2,
            after_circle_stage2,
        )
    )

    print(
        f"BEFORE final ICP ROI points: "
        f"{before_final_icp_roi.n_points:,}"
    )

    print(
        f"BEFORE final ICP ROI cells: "
        f"{before_final_icp_roi.n_cells:,}"
    )

    print(
        f"AFTER final ICP ROI points: "
        f"{after_final_icp_roi.n_points:,}"
    )

    print(
        f"AFTER final ICP ROI cells: "
        f"{after_final_icp_roi.n_cells:,}"
    )

    visualize_final_icp_regions(
        before,
        after_stage2,
        before_final_icp_roi,
        after_final_icp_roi,
    )

    # ========================================================
    # STAGE 5 — FINAL ICP USING COMPLETE CIRCLE
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 5 — FINAL ICP USING COMPLETE CIRCLE ROI")
    print("=" * 80)

    (
        _,
        final_icp_transform,
        final_icp_mean,
        final_icp_iterations,
    ) = icp_register(
        source=
            after_final_icp_roi,

        target=
            before_final_icp_roi,

        max_iterations=
            FINAL_ICP_ITERATIONS,

        max_landmarks=
            FINAL_ICP_LANDMARKS,

        # IMPORTANT:
        #
        # The model has already been aligned.
        # Do not re-match centroids.
        match_centroids=
            False,
    )

    # ========================================================
    # FINAL ICP CANDIDATE
    # ========================================================

    after_candidate = (
        after_stage2.copy()
    )

    after_candidate.transform(
        final_icp_transform,
        inplace=True,
    )

    candidate_landmarks = (
        transform_points(
            after_landmarks_stage2,
            final_icp_transform,
        )
    )

    candidate_errors = (
        np.linalg.norm(
            before_landmarks
            - candidate_landmarks,
            axis=1,
        )
    )

    candidate_mean_error = (
        np.mean(
            candidate_errors
        )
    )

    final_icp_rotation = (
        transform_rotation_angle_deg(
            final_icp_transform
        )
    )

    # ========================================================
    # FINAL ICP SAFETY CHECK
    # ========================================================

    print("\n" + "=" * 80)
    print("FINAL CIRCLE ICP SAFETY CHECK")
    print("=" * 80)

    print(
        f"Landmark error before final ICP: "
        f"{pre_final_icp_error:.6f}"
    )

    print(
        f"Landmark error after final ICP:  "
        f"{candidate_mean_error:.6f}"
    )

    print(
        f"Additional final ICP rotation: "
        f"{final_icp_rotation:.3f} degrees"
    )

    allowed_error_by_ratio = (
        pre_final_icp_error
        * MAX_LANDMARK_ERROR_INCREASE_RATIO
    )

    allowed_error_absolute = (
        pre_final_icp_error
        + MAX_LANDMARK_ERROR_ABS_INCREASE
    )

    allowed_error = min(
        allowed_error_by_ratio,
        allowed_error_absolute,
    )

    accept_final_icp = (
        (
            candidate_mean_error
            <= allowed_error
        )
        and (
            final_icp_rotation
            <= MAX_FINAL_ICP_ROTATION_DEG
        )
    )

    if accept_final_icp:

        print(
            "\nFINAL CIRCLE ICP ACCEPTED."
        )

        after_final = (
            after_candidate
        )

        final_landmarks = (
            candidate_landmarks
        )

    else:

        print(
            "\nWARNING:"
        )

        print(
            "Final circle ICP was rejected."
        )

        if (
            final_icp_rotation
            > MAX_FINAL_ICP_ROTATION_DEG
        ):

            print(
                f"\nReason:"
                f"\nICP attempted an additional "
                f"{final_icp_rotation:.3f} degree rotation."
            )

            print(
                "This may reflect rotational symmetry "
                "of the circular sample."
            )

        if (
            candidate_mean_error
            > allowed_error
        ):

            print(
                "\nReason:"
                "\nICP worsened the manually corresponding "
                "circle landmarks too much."
            )

        print(
            "\nUsing circle-landmark registration instead."
        )

        after_final = (
            after_stage2
        )

        final_landmarks = (
            after_landmarks_stage2
        )

    # ========================================================
    # FINAL REGISTRATION QC
    # ========================================================

    final_landmark_errors = (
        np.linalg.norm(
            before_landmarks
            - final_landmarks,
            axis=1,
        )
    )

    print("\n" + "=" * 80)
    print("FINAL REGISTRATION QC")
    print("=" * 80)

    print(
        f"Final ICP accepted: "
        f"{accept_final_icp}"
    )

    print(
        f"Mean landmark error: "
        f"{np.mean(final_landmark_errors):.6f}"
    )

    print(
        f"Median landmark error: "
        f"{np.median(final_landmark_errors):.6f}"
    )

    print(
        f"Maximum landmark error: "
        f"{np.max(final_landmark_errors):.6f}"
    )

    visualize_overlay(
        before,
        after_final,
        "FINAL — COMPLETE CIRCLE ROI ICP",
        before_landmarks=
            before_landmarks,
        after_landmarks=
            final_landmarks,
    )

    # ========================================================
    # STAGE 6 — SURFACE-DIFFERENCE ROI
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 6 — SURFACE-DIFFERENCE ROI")
    print("=" * 80)

    print(
        f"""
Surface difference is calculated over almost the complete
circular plate.

Measurement radius:

    r <= {ANALYSIS_RADIUS_FRACTION:.2f} × R

Only the outermost
{(1.0 - ANALYSIS_RADIUS_FRACTION) * 100:.1f}% of the radius
is excluded to reduce edge artefacts.
"""
    )

    # IMPORTANT:
    #
    # Both surfaces are now cropped using the SAME BEFORE
    # circle after final registration.
    before_roi = (
        extract_analysis_region(
            before,
            before_circle,
        )
    )

    after_roi = (
        extract_analysis_region(
            after_final,
            before_circle,
        )
    )

    print(
        f"BEFORE analysis ROI points: "
        f"{before_roi.n_points:,}"
    )

    print(
        f"BEFORE analysis ROI cells: "
        f"{before_roi.n_cells:,}"
    )

    print(
        f"AFTER analysis ROI points: "
        f"{after_roi.n_points:,}"
    )

    print(
        f"AFTER analysis ROI cells: "
        f"{after_roi.n_cells:,}"
    )

    visualize_analysis_regions(
        before,
        after_final,
        before_roi,
        after_roi,
    )

    # ========================================================
    # STAGE 7 — FIXED U,V SURFACE DIFFERENCE
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 7 — FIXED-LOCATION SURFACE DIFFERENCE")
    print("=" * 80)

    print(
        """
BEFORE and AFTER are now compared at identical positions
across the circular plane.

For every BEFORE location (u,v):

    height change =
        BEFORE height(u,v) - AFTER height(u,v)

Positive:
    AFTER is lower -> possible removed plaque/material

Approximately zero:
    surfaces have same height -> expected for unchanged plaque

Negative:
    AFTER is higher -> diagnostic only, not counted as removal
"""
    )

    results = (
        calculate_surface_difference(
            before_roi=
                before_roi,

            after_roi=
                after_roi,

            reference_circle=
                before_circle,

            threshold=
                threshold,
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("FIXED-LOCATION CIRCULAR SURFACE CHANGE")
    print("=" * 80)

    print(
        f"\nValid ROI area: "
        f"{results['roi_area']:.6f}"
    )

    print(
        f"Valid correspondence: "
        f"{results['valid_fraction'] * 100:.2f}%"
    )

    print(
        "\n--- SIGNED HEIGHT CHANGE ---"
    )

    print(
        f"Mean signed height change: "
        f"{results['mean_signed_height']:.6f}"
    )

    print(
        f"Median signed height change: "
        f"{results['median_signed_height']:.6f}"
    )

    print(
        "\n--- POSITIVE SURFACE REMOVAL ---"
    )

    print(
        f"Mean removal: "
        f"{results['mean_removal']:.6f}"
    )

    print(
        f"Median removal: "
        f"{results['median_removal']:.6f}"
    )

    print(
        f"95th percentile: "
        f"{results['removal_95']:.6f}"
    )

    print(
        f"99th percentile: "
        f"{results['removal_99']:.6f}"
    )

    print(
        f"Maximum removal: "
        f"{results['maximum_removal']:.6f}"
    )

    print(
        "\n--- NEGATIVE CHANGE QC ---"
    )

    print(
        f"Mean negative change: "
        f"{results['mean_negative']:.6f}"
    )

    print(
        f"Area with change < -threshold: "
        f"{results['negative_area']:.6f}"
    )

    print(
        f"Negative-change fraction: "
        f"{results['negative_fraction'] * 100:.2f}%"
    )

    print(
        "\n--- U,V CORRESPONDENCE QC ---"
    )

    print(
        f"Mean UV interpolation distance: "
        f"{results['mean_uv_distance']:.6f}"
    )

    print(
        f"Median UV interpolation distance: "
        f"{results['median_uv_distance']:.6f}"
    )

    print(
        "\n--- THRESHOLDED PLAQUE / SURFACE REMOVAL ---"
    )

    print(
        f"Threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"Changed area: "
        f"{results['changed_area']:.6f}"
    )

    print(
        f"Changed fraction: "
        f"{results['changed_fraction'] * 100:.2f}%"
    )

    print(
        "\n--- VOLUME ---"
    )

    print(
        f"Threshold-corrected removed volume: "
        f"{results['volume_difference']:.6f}"
    )

    print(
        "\nIf PLY coordinates are in millimetres, "
        "this value is already in mm^3."
    )

    # ========================================================
    # FINAL VISUALIZATION
    # ========================================================

    visualize_surface_difference(
        before=
            before,

        after=
            after_final,

        before_roi=
            before_roi,

        results=
            results,

        threshold=
            threshold,
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "before":
            before,

        "after_final":
            after_final,

        "before_circle":
            before_circle,

        "before_landmarks":
            before_landmarks,

        "final_landmarks":
            final_landmarks,

        "before_final_icp_roi":
            before_final_icp_roi,

        "after_final_icp_roi":
            after_final_icp_roi,

        "before_analysis_roi":
            before_roi,

        "after_analysis_roi":
            after_roi,

        "stage1_transform":
            stage1_transform,

        "circle_transform":
            circle_transform,

        "final_icp_transform":
            final_icp_transform,

        "final_icp_accepted":
            accept_final_icp,

        "final_icp_rotation_deg":
            final_icp_rotation,

        "final_landmark_errors":
            final_landmark_errors,

        "stage1_mean_distance":
            stage1_mean,

        "stage1_iterations":
            stage1_iterations,

        "final_icp_mean_distance":
            final_icp_mean,

        "final_icp_iterations":
            final_icp_iterations,

        "surface_results":
            results,
    }


# ============================================================
# MAIN MENU
# ============================================================

while True:

    print("\n" + "=" * 80)
    print("BIOCAL3D — CIRCULAR SAMPLE REGISTRATION")
    print("=" * 80)

    print(
        "1 - Before / After circular surface difference"
    )

    print(
        "2 - Quit"
    )

    choice = input(
        "\nChoice: "
    ).strip()

    # ========================================================
    # ANALYSIS
    # ========================================================

    if choice == "1":

        print(
            "\nAvailable pairs:"
        )

        pairs = list(
            zip(
                files[
                    "Before brushing"
                ],
                files[
                    "After brushing"
                ],
            )
        )

        for i, (
            before_file,
            after_file,
        ) in enumerate(
            pairs
        ):

            print(
                f"[{i}] "
                f"{before_file.name} "
                f"<-> "
                f"{after_file.name}"
            )

        try:

            index = int(
                input(
                    "\nPair number: "
                )
            )

            threshold = float(
                input(
                    "Distance threshold "
                    "(same units as PLY): "
                )
            )

            result = (
                analyze_before_after(
                    index=index,
                    threshold=threshold,
                )
            )

        except ValueError:

            print(
                "\nInvalid input."
            )

        except Exception as e:

            print("\n" + "=" * 80)
            print("ANALYSIS ERROR")
            print("=" * 80)

            print(
                f"{type(e).__name__}: {e}"
            )

            print(
                "\nReturning to main menu..."
            )

    # ========================================================
    # QUIT
    # ========================================================

    elif choice == "2":

        print(
            "\nExiting BioCal3D."
        )

        break

    else:

        print(
            "\nPlease enter 1 or 2."
        )