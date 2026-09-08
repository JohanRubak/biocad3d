from pathlib import Path

import numpy as np
import pyvista as pv
import vtk


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
# INITIAL FULL-MODEL ICP
# ============================================================

STAGE1_ITERATIONS = 300
STAGE1_LANDMARKS = 20000


# ============================================================
# CIRCLE SETTINGS
# ============================================================

N_CIRCLE_LANDMARKS = 8

# Landmarks are NOT treated as corresponding physical points.
#
# They are only used to estimate:
#
# - circle centre
# - circle radius
# - circle plane
#
# Therefore simply distribute them around the circumference.
CIRCLE_LANDMARK_NAMES = [
    "Circumference point 1",
    "Circumference point 2",
    "Circumference point 3",
    "Circumference point 4",
    "Circumference point 5",
    "Circumference point 6",
    "Circumference point 7",
    "Circumference point 8",
]


# ============================================================
# ANALYSIS ROI
# ============================================================

# Almost the complete circular sample.
#
# Only outermost 5% is excluded to reduce boundary artefacts.
ANALYSIS_RADIUS_FRACTION = 0.95

# Distance above/below fitted circle plane that may be
# considered part of the sample.
ANALYSIS_HALF_THICKNESS = 2.0

# Only approximately top-facing triangles.
MIN_ANALYSIS_NORMAL_ALIGNMENT = 0.60


# ============================================================
# RAY-CASTING SETTINGS
# ============================================================

# Ray is cast:
#
# point + normal * RAY_HALF_LENGTH
#
# through
#
# point - normal * RAY_HALF_LENGTH
#
# This is in PLY coordinate units.
#
# If coordinates are mm, 2.0 = +/- 2 mm.
RAY_HALF_LENGTH = 2.0


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
# VECTOR NORMALIZATION
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
Select {N_CIRCLE_LANDMARKS} points distributed around the
CIRCUMFERENCE of the circular sample.

IMPORTANT:

These are NOT corresponding landmarks between BEFORE and AFTER.

Point 1 on BEFORE does NOT need to be the same physical point
as point 1 on AFTER.

The points are ONLY used to independently fit:

    - circle centre
    - circle radius
    - circle plane normal

Try to distribute the points evenly around the circumference.

LEFT CLICK  = select point
RIGHT CLICK = remove previous point

Close the window after all points are selected.
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
            f"\nPoint "
            f"{idx}/{N_CIRCLE_LANDMARKS}: "
            f"{point}"
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
            name=f"landmark_{idx}",
        )

        plotter.add_point_labels(
            marker,
            [str(idx)],
            font_size=18,
            text_color="red",
            point_color="red",
            shape=None,
            always_visible=True,
            name=f"label_{idx}",
        )

        plotter.render()

        obj.SetAbortFlag(1)

        if (
            len(picked_points)
            == N_CIRCLE_LANDMARKS
        ):

            print(
                "\nAll circumference points selected."
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
            f"\nRemoved point {idx}: "
            f"{removed}"
        )

        try:

            plotter.remove_actor(
                f"landmark_{idx}"
            )

        except Exception:

            pass

        try:

            plotter.remove_actor(
                f"label_{idx}"
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
            f"{N_CIRCLE_LANDMARKS} points "
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
    Fit a best-fit plane and circle to the manually selected
    circumference points.

    No point-to-point correspondence is assumed.
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
    # ARBITRARY IN-PLANE BASIS
    #
    # Only needed for the 2D circle fit.
    # It is NOT used to establish scan orientation.
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

    # --------------------------------------------------------
    # PROJECT INTO PLANE
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
    # 2D LEAST-SQUARES CIRCLE FIT
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
        f"Centre: "
        f"{circle['center']}"
    )

    print(
        f"Radius: "
        f"{circle['radius']:.6f}"
    )

    print(
        f"Normal: "
        f"{circle['normal']}"
    )

    print(
        f"Plane RMS: "
        f"{circle['plane_rms']:.6f}"
    )

    print(
        f"Radial RMS: "
        f"{circle['radial_rms']:.6f}"
    )


# ============================================================
# CREATE IN-PLANE AXES
# ============================================================

def create_plane_axes(
    normal,
):

    """
    Create reproducible U,V axes using only the final common
    plane normal.

    These are used for visualization / coordinates.

    They do NOT establish registration orientation.
    """

    normal = normalize(
        normal
    )

    candidate = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float64,
    )

    if abs(
        np.dot(
            candidate,
            normal,
        )
    ) > 0.9:

        candidate = np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float64,
        )

    u_axis = (
        candidate
        - np.dot(
            candidate,
            normal,
        )
        * normal
    )

    u_axis = normalize(
        u_axis
    )

    v_axis = normalize(
        np.cross(
            normal,
            u_axis,
        )
    )

    return (
        u_axis,
        v_axis,
    )


# ============================================================
# CREATE CIRCLE OUTLINE
# ============================================================

def create_circle_outline(
    circle,
    radius_fraction=1.0,
):

    (
        u_axis,
        v_axis,
    ) = create_plane_axes(
        circle["normal"]
    )

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
        * u_axis[None, :]
        + radius
        * np.sin(angles)[:, None]
        * v_axis[None, :]
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

    outline = create_circle_outline(
        circle
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
        outline,
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
        f"Radius = "
        f"{circle['radius']:.4f}\n"
        f"Plane RMS = "
        f"{circle['plane_rms']:.4f}\n"
        f"Radial RMS = "
        f"{circle['radial_rms']:.4f}\n\n"
        "YELLOW = fitted circumference\n"
        "CYAN = plane normal",
        position="upper_left",
        font_size=14,
    )

    plotter.add_axes()

    plotter.show()


# ============================================================
# ICP REGISTRATION
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
            f"ICP iterations: "
            f"{iterations}"
        )

    return (
        registered,
        transform,
        mean_distance,
        iterations,
    )


# ============================================================
# MINIMUM ROTATION BETWEEN TWO NORMALS
# ============================================================

def minimum_rotation_matrix(
    source_normal,
    target_normal,
):

    """
    Return the smallest possible 3D rotation that maps:

        source_normal -> target_normal

    No additional rotation around the target normal is added.

    This is important because the circular sample cannot
    determine its own in-plane rotation.
    """

    a = normalize(
        source_normal
    )

    b = normalize(
        target_normal
    )

    dot = np.clip(
        np.dot(
            a,
            b,
        ),
        -1.0,
        1.0,
    )

    # --------------------------------------------------------
    # ALREADY ALIGNED
    # --------------------------------------------------------

    if dot > 1.0 - 1e-10:

        return np.eye(
            3,
            dtype=np.float64,
        )

    # --------------------------------------------------------
    # OPPOSITE NORMALS
    # --------------------------------------------------------

    if dot < -1.0 + 1e-10:

        # Find any axis perpendicular to a.
        candidate = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        )

        if abs(
            np.dot(
                candidate,
                a,
            )
        ) > 0.9:

            candidate = np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float64,
            )

        axis = (
            candidate
            - np.dot(
                candidate,
                a,
            )
            * a
        )

        axis = normalize(
            axis
        )

        # 180 degree rotation:
        #
        # R = -I + 2 uu^T
        R = (
            -np.eye(3)
            + 2.0
            * np.outer(
                axis,
                axis,
            )
        )

        return R

    # --------------------------------------------------------
    # RODRIGUES ROTATION
    # --------------------------------------------------------

    cross = np.cross(
        a,
        b,
    )

    s = np.linalg.norm(
        cross
    )

    axis = (
        cross
        / s
    )

    K = np.array(
        [
            [
                0.0,
                -axis[2],
                axis[1],
            ],
            [
                axis[2],
                0.0,
                -axis[0],
            ],
            [
                -axis[1],
                axis[0],
                0.0,
            ],
        ],
        dtype=np.float64,
    )

    angle = np.arctan2(
        s,
        dot,
    )

    R = (
        np.eye(3)
        + np.sin(angle) * K
        + (
            1.0
            - np.cos(angle)
        )
        * (
            K @ K
        )
    )

    return R


# ============================================================
# BUILD VTK RIGID TRANSFORM
# ============================================================

def build_rigid_matrix(
    R,
    t,
):

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
# CONSTRAINED CIRCLE ALIGNMENT
# ============================================================

def calculate_circle_plane_transform(
    source_circle,
    target_circle,
):

    """
    Align AFTER circular sample to BEFORE circular sample by:

    1. matching plane normals using the MINIMUM rotation
    2. matching circle centres by translation

    No arbitrary in-plane rotation is introduced.

    Therefore the approximate in-plane orientation established
    by the initial full-model ICP is retained.
    """

    source_normal = (
        source_circle[
            "normal"
        ].copy()
    )

    target_normal = (
        target_circle[
            "normal"
        ].copy()
    )

    # --------------------------------------------------------
    # NORMAL DIRECTION IS AMBIGUOUS
    #
    # Choose the source sign that is closest to target.
    # --------------------------------------------------------

    if np.dot(
        source_normal,
        target_normal,
    ) < 0:

        source_normal = (
            -source_normal
        )

    R = minimum_rotation_matrix(
        source_normal,
        target_normal,
    )

    source_center_rotated = (
        R
        @ source_circle[
            "center"
        ]
    )

    t = (
        target_circle[
            "center"
        ]
        - source_center_rotated
    )

    matrix = build_rigid_matrix(
        R,
        t,
    )

    angle = np.degrees(
        np.arccos(
            np.clip(
                (
                    np.trace(R)
                    - 1.0
                )
                / 2.0,
                -1.0,
                1.0,
            )
        )
    )

    return (
        matrix,
        angle,
    )


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
# VISUALIZE OVERLAY
# ============================================================

def visualize_overlay(
    before,
    after,
    title,
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
    radius_fraction,
    half_thickness,
    minimum_normal_alignment,
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
        - circle[
            "center"
        ]
    )

    plane_distance = (
        vectors
        @ circle[
            "normal"
        ]
    )

    projected = (
        vectors
        - np.outer(
            plane_distance,
            circle[
                "normal"
            ],
        )
    )

    radial_distance = (
        np.linalg.norm(
            projected,
            axis=1,
        )
    )

    # --------------------------------------------------------
    # RADIAL + PLANE MASK
    # --------------------------------------------------------

    mask = (
        (
            radial_distance
            <= circle[
                "radius"
            ]
            * radius_fraction
        )
        & (
            np.abs(
                plane_distance
            )
            <= half_thickness
        )
    )

    # --------------------------------------------------------
    # SURFACE NORMAL FILTER
    # --------------------------------------------------------

    triangle_normals = np.cross(
        p1 - p0,
        p2 - p0,
    )

    lengths = np.linalg.norm(
        triangle_normals,
        axis=1,
    )

    valid_normals = (
        lengths > 1e-12
    )

    unit_normals = np.zeros_like(
        triangle_normals
    )

    unit_normals[
        valid_normals
    ] = (
        triangle_normals[
            valid_normals
        ]
        / lengths[
            valid_normals,
            None,
        ]
    )

    alignment = np.abs(
        unit_normals
        @ circle[
            "normal"
        ]
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
            "No triangles found inside circular ROI."
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
# EXTRACT ANALYSIS ROI
# ============================================================

def extract_analysis_region(
    mesh,
    reference_circle,
):

    mask = (
        create_circle_cell_mask(
            mesh=
                mesh,

            circle=
                reference_circle,

            radius_fraction=
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
# VISUALIZE ANALYSIS ROI
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
        "ORANGE = SURFACE-DIFFERENCE ROI",
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
        "ORANGE = SURFACE-DIFFERENCE ROI",
        font_size=14,
    )

    plotter.add_axes()

    plotter.link_views()

    plotter.show()


# ============================================================
# ORIENT COMMON NORMAL TOWARD EXPOSED SURFACE
# ============================================================

def orient_normal_to_surface(
    mesh,
    normal,
):

    normal = normalize(
        normal
    )

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

    triangle_normals = np.cross(
        p1 - p0,
        p2 - p0,
    )

    lengths = np.linalg.norm(
        triangle_normals,
        axis=1,
    )

    valid = (
        lengths > 1e-12
    )

    triangle_normals = (
        triangle_normals[
            valid
        ]
        / lengths[
            valid,
            None,
        ]
    )

    alignment = (
        triangle_normals
        @ normal
    )

    if np.median(
        alignment
    ) < 0:

        normal = -normal

    return normal


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
# RAY-CAST HEIGHT DIFFERENCE
# ============================================================

def raycast_surface_difference(
    before_roi,
    after_roi,
    plane_normal,
):

    """
    For every BEFORE vertex:

    1. Cast a line along the fitted circle normal.
    2. Intersect that line with the AFTER triangular mesh.
    3. Choose the closest AFTER intersection.
    4. Calculate signed height difference.

    This compares the surfaces at the SAME projected position.

    No nearest-vertex interpolation is used.
    """

    normal = orient_normal_to_surface(
        before_roi,
        plane_normal,
    )

    print(
        "\nAnalysis normal:"
    )

    print(
        normal
    )

    # --------------------------------------------------------
    # BUILD AFTER TRIANGLE INTERSECTION TREE
    # --------------------------------------------------------

    locator = vtk.vtkOBBTree()

    locator.SetDataSet(
        after_roi
    )

    locator.BuildLocator()

    before_points = (
        before_roi.points
    )

    signed_difference = np.full(
        len(before_points),
        np.nan,
        dtype=np.float64,
    )

    intersection_distance = np.full(
        len(before_points),
        np.nan,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # RAY-CAST EACH BEFORE POINT
    # --------------------------------------------------------

    for i, point in enumerate(
        before_points
    ):

        p_start = (
            point
            + normal
            * RAY_HALF_LENGTH
        )

        p_end = (
            point
            - normal
            * RAY_HALF_LENGTH
        )

        intersections = (
            vtk.vtkPoints()
        )

        cell_ids = (
            vtk.vtkIdList()
        )

        hit = locator.IntersectWithLine(
            p_start,
            p_end,
            intersections,
            cell_ids,
        )

        if hit == 0:

            continue

        n_intersections = (
            intersections
            .GetNumberOfPoints()
        )

        if n_intersections == 0:

            continue

        candidates = []

        for j in range(
            n_intersections
        ):

            q = np.array(
                intersections.GetPoint(j),
                dtype=np.float64,
            )

            # Signed height difference:
            #
            # BEFORE - AFTER along exposed normal.
            #
            # Positive:
            # BEFORE higher than AFTER
            # -> candidate removed material.
            difference = np.dot(
                point - q,
                normal,
            )

            absolute_axial_distance = abs(
                difference
            )

            candidates.append(
                (
                    absolute_axial_distance,
                    difference,
                    q,
                )
            )

        # Choose closest intersection to BEFORE point.
        candidates.sort(
            key=lambda item: item[0]
        )

        best = candidates[0]

        intersection_distance[i] = (
            best[0]
        )

        signed_difference[i] = (
            best[1]
        )

    valid_points = np.isfinite(
        signed_difference
    )

    print(
        f"\nValid ray intersections: "
        f"{np.sum(valid_points):,} / "
        f"{len(valid_points):,} "
        f"({np.mean(valid_points) * 100:.2f}%)"
    )

    return (
        signed_difference,
        intersection_distance,
        valid_points,
        normal,
    )


# ============================================================
# POINT VALUES -> TRIANGLE VALUES
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

    valid_cells = np.all(
        valid_points[
            triangles
        ],
        axis=1,
    )

    values = np.full(
        len(triangles),
        np.nan,
        dtype=np.float64,
    )

    values[
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
        values,
        valid_cells,
    )


# ============================================================
# CALCULATE RAY-CAST SURFACE DIFFERENCE
# ============================================================

def calculate_surface_difference(
    before_roi,
    after_roi,
    reference_circle,
):

    (
        signed_point,
        ray_distance_point,
        valid_points,
        normal,
    ) = raycast_surface_difference(
        before_roi=
            before_roi,

        after_roi=
            after_roi,

        plane_normal=
            reference_circle[
                "normal"
            ],
    )

    # --------------------------------------------------------
    # POINT -> CELL
    # --------------------------------------------------------

    (
        signed_cell,
        valid_cells,
    ) = point_to_cell_values_valid(
        before_roi,
        signed_point,
        valid_points,
    )

    (
        ray_distance_cell,
        _,
    ) = point_to_cell_values_valid(
        before_roi,
        ray_distance_point,
        valid_points,
    )

    # --------------------------------------------------------
    # DERIVED MAPS
    # --------------------------------------------------------

    absolute_cell = (
        np.abs(
            signed_cell
        )
    )

    positive_removal_cell = (
        np.maximum(
            signed_cell,
            0.0,
        )
    )

    negative_change_cell = (
        np.minimum(
            signed_cell,
            0.0,
        )
    )

    # --------------------------------------------------------
    # AREA
    # --------------------------------------------------------

    areas = triangle_areas(
        before_roi
    )

    valid_areas = (
        areas[
            valid_cells
        ]
    )

    valid_signed = (
        signed_cell[
            valid_cells
        ]
    )

    valid_absolute = (
        absolute_cell[
            valid_cells
        ]
    )

    valid_positive = (
        positive_removal_cell[
            valid_cells
        ]
    )

    valid_ray_distance = (
        ray_distance_cell[
            valid_cells
        ]
    )

    roi_area = np.sum(
        valid_areas
    )

    # --------------------------------------------------------
    # RAW VOLUME ESTIMATES
    #
    # No threshold is applied.
    # --------------------------------------------------------

    positive_volume = np.sum(
        valid_positive
        * valid_areas
    )

    signed_volume = np.sum(
        valid_signed
        * valid_areas
    )

    return {
        "signed_cell":
            signed_cell,

        "absolute_cell":
            absolute_cell,

        "positive_removal_cell":
            positive_removal_cell,

        "negative_change_cell":
            negative_change_cell,

        "ray_distance_cell":
            ray_distance_cell,

        "valid_cells":
            valid_cells,

        "normal":
            normal,

        "roi_area":
            roi_area,

        "valid_fraction":
            (
                np.sum(
                    valid_cells
                )
                / len(
                    valid_cells
                )
            ),

        "mean_signed":
            np.mean(
                valid_signed
            ),

        "median_signed":
            np.median(
                valid_signed
            ),

        "signed_5":
            np.percentile(
                valid_signed,
                5,
            ),

        "signed_95":
            np.percentile(
                valid_signed,
                95,
            ),

        "signed_99":
            np.percentile(
                valid_signed,
                99,
            ),

        "minimum_signed":
            np.min(
                valid_signed
            ),

        "maximum_signed":
            np.max(
                valid_signed
            ),

        "mean_absolute":
            np.mean(
                valid_absolute
            ),

        "median_absolute":
            np.median(
                valid_absolute
            ),

        "absolute_95":
            np.percentile(
                valid_absolute,
                95,
            ),

        "absolute_99":
            np.percentile(
                valid_absolute,
                99,
            ),

        "mean_positive":
            np.mean(
                valid_positive
            ),

        "median_positive":
            np.median(
                valid_positive
            ),

        "positive_95":
            np.percentile(
                valid_positive,
                95,
            ),

        "positive_99":
            np.percentile(
                valid_positive,
                99,
            ),

        "maximum_positive":
            np.max(
                valid_positive
            ),

        "positive_volume":
            positive_volume,

        "signed_volume":
            signed_volume,

        "mean_ray_distance":
            np.mean(
                valid_ray_distance
            ),

        "median_ray_distance":
            np.median(
                valid_ray_distance
            ),
    }


# ============================================================
# VISUALIZE SURFACE DIFFERENCE
# ============================================================

def visualize_surface_difference(
    before,
    after,
    before_roi,
    results,
):

    signed_mesh = (
        before_roi.copy()
    )

    positive_mesh = (
        before_roi.copy()
    )

    valid_cells = (
        results[
            "valid_cells"
        ]
    )

    signed_values = (
        results[
            "signed_cell"
        ].copy()
    )

    positive_values = (
        results[
            "positive_removal_cell"
        ].copy()
    )

    signed_values[
        ~valid_cells
    ] = np.nan

    positive_values[
        ~valid_cells
    ] = np.nan

    signed_mesh.cell_data[
        "Signed height change"
    ] = signed_values

    positive_mesh.cell_data[
        "Positive removal"
    ] = positive_values

    # --------------------------------------------------------
    # SIGNED MAP SCALE
    #
    # Symmetric around zero.
    # Use 99th percentile to prevent isolated outliers
    # dominating the visualization.
    # --------------------------------------------------------

    finite_signed = signed_values[
        np.isfinite(
            signed_values
        )
    ]

    if len(
        finite_signed
    ) > 0:

        signed_limit = np.percentile(
            np.abs(
                finite_signed
            ),
            99,
        )

        if signed_limit < 1e-9:

            signed_limit = 1e-6

    else:

        signed_limit = 0.01

    # --------------------------------------------------------
    # POSITIVE MAP SCALE
    # --------------------------------------------------------

    finite_positive = positive_values[
        np.isfinite(
            positive_values
        )
    ]

    if len(
        finite_positive
    ) > 0:

        positive_limit = np.percentile(
            finite_positive,
            99,
        )

        if positive_limit < 1e-9:

            positive_limit = 1e-6

    else:

        positive_limit = 0.01

    # ========================================================
    # VISUALIZATION
    # ========================================================

    plotter = pv.Plotter(
        shape=(1, 4),
        window_size=(2100, 650),
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
        rgb=True,
        smooth_shading=False,
        show_edges=False,
    )

    plotter.add_text(
        "BEFORE BRUSHING",
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
        rgb=True,
        smooth_shading=False,
        show_edges=False,
    )

    plotter.add_text(
        "AFTER BRUSHING\n"
        "CIRCLE-PLANE ALIGNED",
        font_size=14,
    )

    plotter.add_axes()

    # --------------------------------------------------------
    # RAW SIGNED DIFFERENCE
    # --------------------------------------------------------

    plotter.subplot(
        0,
        2,
    )

    plotter.add_mesh(
        before,
        color="lightgray",
        opacity=0.15,
        smooth_shading=False,
    )

    plotter.add_mesh(
        signed_mesh,
        scalars=
            "Signed height change",

        cmap=
            "coolwarm",

        clim=[
            -signed_limit,
            signed_limit,
        ],

        smooth_shading=False,
        show_edges=False,

        scalar_bar_args={
            "title":
                "BEFORE - AFTER"
        },
    )

    plotter.add_text(
        "RAW SIGNED HEIGHT CHANGE\n"
        "RED = BEFORE HIGHER\n"
        "BLUE = AFTER HIGHER",
        font_size=13,
    )

    plotter.add_axes()

    # --------------------------------------------------------
    # POSITIVE REMOVAL
    # --------------------------------------------------------

    plotter.subplot(
        0,
        3,
    )

    plotter.add_mesh(
        before,
        color="lightgray",
        opacity=0.15,
        smooth_shading=False,
    )

    plotter.add_mesh(
        positive_mesh,
        scalars=
            "Positive removal",

        cmap=
            "turbo",

        clim=[
            0.0,
            positive_limit,
        ],

        smooth_shading=False,
        show_edges=False,

        scalar_bar_args={
            "title":
                "Positive height removal"
        },
    )

    plotter.add_text(
        "POSITIVE SURFACE REMOVAL\n"
        "NO THRESHOLD APPLIED",
        font_size=13,
    )

    plotter.add_axes()

    plotter.link_views()

    plotter.show()


# ============================================================
# COMPLETE BEFORE / AFTER ANALYSIS
# ============================================================

def analyze_before_after(
    index,
):

    before_files = files[
        "Before brushing"
    ]

    after_files = files[
        "After brushing"
    ]

    if index >= len(
        before_files
    ):

        raise RuntimeError(
            "Invalid BEFORE file."
        )

    if index >= len(
        after_files
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
    # STAGE 1
    # FULL-MODEL ICP
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 1 — INITIAL FULL-MODEL ICP")
    print("=" * 80)

    print(
        """
The complete BEFORE and AFTER scans are used here.

Purpose:

- establish overall orientation
- establish approximate in-plane rotation
- approximate translation
- approximate tilt

This is the ONLY ICP step.

The plaque-bearing circular surface will NOT subsequently
be optimized with ICP.
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
    # STAGE 2
    # DEFINE BEFORE CIRCLE
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 2 — DEFINE CIRCULAR SAMPLE")
    print("=" * 80)

    input(
        "\nPress ENTER to select "
        "BEFORE circumference points..."
    )

    before_landmarks = (
        pick_circle_landmarks(
            before,
            "BEFORE — DEFINE CIRCLE",
        )
    )

    before_circle = fit_circle_3d(
        before_landmarks
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
    # DEFINE AFTER CIRCLE
    # ========================================================

    input(
        "\nPress ENTER to select "
        "AFTER circumference points..."
    )

    after_landmarks = (
        pick_circle_landmarks(
            after_stage1,
            "AFTER — DEFINE CIRCLE",
        )
    )

    after_circle = fit_circle_3d(
        after_landmarks
    )

    # --------------------------------------------------------
    # Match normal direction to BEFORE for reporting.
    # --------------------------------------------------------

    if np.dot(
        after_circle[
            "normal"
        ],
        before_circle[
            "normal"
        ],
    ) < 0:

        after_circle[
            "normal"
        ] = (
            -after_circle[
                "normal"
            ]
        )

    print_circle_fit(
        after_circle,
        "AFTER",
    )

    visualize_circle_fit(
        after_stage1,
        after_landmarks,
        after_circle,
        "AFTER CIRCLE FIT",
    )

    # ========================================================
    # CIRCLE FIT QC BEFORE ALIGNMENT
    # ========================================================

    centre_error_before = np.linalg.norm(
        before_circle[
            "center"
        ]
        - after_circle[
            "center"
        ]
    )

    normal_dot = np.clip(
        np.dot(
            before_circle[
                "normal"
            ],
            after_circle[
                "normal"
            ],
        ),
        -1.0,
        1.0,
    )

    normal_angle_before = np.degrees(
        np.arccos(
            normal_dot
        )
    )

    radius_difference = abs(
        before_circle[
            "radius"
        ]
        - after_circle[
            "radius"
        ]
    )

    print("\n" + "=" * 80)
    print("CIRCLE FIT QC BEFORE LOCAL CORRECTION")
    print("=" * 80)

    print(
        f"Centre difference: "
        f"{centre_error_before:.6f}"
    )

    print(
        f"Plane-normal angle: "
        f"{normal_angle_before:.6f} degrees"
    )

    print(
        f"Radius difference: "
        f"{radius_difference:.6f}"
    )

    # ========================================================
    # STAGE 3
    # CONSTRAINED CIRCLE / PLANE ALIGNMENT
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 3 — CONSTRAINED CIRCLE-PLANE ALIGNMENT")
    print("=" * 80)

    print(
        """
The two circular samples are now aligned WITHOUT performing ICP.

The correction:

1. matches the fitted plane normals
2. matches the fitted circle centres
3. uses the MINIMUM possible rotation

No arbitrary rotation around the circular plane normal is added.

Therefore the in-plane orientation established by the initial
full-model ICP is retained.
"""
    )

    (
        circle_transform,
        circle_rotation_angle,
    ) = calculate_circle_plane_transform(
        source_circle=
            after_circle,

        target_circle=
            before_circle,
    )

    print(
        f"\nAdditional plane-alignment rotation: "
        f"{circle_rotation_angle:.6f} degrees"
    )

    after_final = (
        after_stage1.copy()
    )

    after_final.transform(
        circle_transform,
        inplace=True,
    )

    transformed_after_landmarks = (
        transform_points(
            after_landmarks,
            circle_transform,
        )
    )

    after_circle_final = fit_circle_3d(
        transformed_after_landmarks
    )

    # Match normal sign again.
    if np.dot(
        after_circle_final[
            "normal"
        ],
        before_circle[
            "normal"
        ],
    ) < 0:

        after_circle_final[
            "normal"
        ] = (
            -after_circle_final[
                "normal"
            ]
        )

    # ========================================================
    # FINAL CIRCLE QC
    # ========================================================

    centre_error_final = np.linalg.norm(
        before_circle[
            "center"
        ]
        - after_circle_final[
            "center"
        ]
    )

    final_dot = np.clip(
        np.dot(
            before_circle[
                "normal"
            ],
            after_circle_final[
                "normal"
            ],
        ),
        -1.0,
        1.0,
    )

    normal_angle_final = np.degrees(
        np.arccos(
            final_dot
        )
    )

    print("\n" + "=" * 80)
    print("FINAL CIRCLE / PLANE ALIGNMENT QC")
    print("=" * 80)

    print(
        f"Circle centre error: "
        f"{centre_error_final:.8f}"
    )

    print(
        f"Plane-normal angle error: "
        f"{normal_angle_final:.8f} degrees"
    )

    print(
        f"BEFORE radius: "
        f"{before_circle['radius']:.6f}"
    )

    print(
        f"AFTER radius: "
        f"{after_circle_final['radius']:.6f}"
    )

    print(
        f"Radius difference: "
        f"{abs(before_circle['radius'] - after_circle_final['radius']):.6f}"
    )

    visualize_overlay(
        before,
        after_final,
        "FINAL — CONSTRAINED CIRCLE/PLANE ALIGNMENT",
    )

    # ========================================================
    # STAGE 4
    # ANALYSIS ROI
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 4 — SURFACE-DIFFERENCE ROI")
    print("=" * 80)

    print(
        f"""
The SAME circular aperture is now applied to both registered scans.

Analysis radius:

    r <= {ANALYSIS_RADIUS_FRACTION:.2f} × R

Only the outermost
{(1.0 - ANALYSIS_RADIUS_FRACTION) * 100:.1f}% of the radius
is excluded.

No surface ICP is performed inside this region.
"""
    )

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
        f"BEFORE ROI points: "
        f"{before_roi.n_points:,}"
    )

    print(
        f"BEFORE ROI cells: "
        f"{before_roi.n_cells:,}"
    )

    print(
        f"AFTER ROI points: "
        f"{after_roi.n_points:,}"
    )

    print(
        f"AFTER ROI cells: "
        f"{after_roi.n_cells:,}"
    )

    visualize_analysis_regions(
        before,
        after_final,
        before_roi,
        after_roi,
    )

    # ========================================================
    # STAGE 5
    # RAY-CAST SURFACE DIFFERENCE
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 5 — RAY-CAST SURFACE DIFFERENCE")
    print("=" * 80)

    print(
        """
For every BEFORE surface point:

1. A line is cast through the AFTER mesh along the circular
   plane normal.

2. The closest intersection with the AFTER triangular surface
   is identified.

3. The signed difference is calculated along the normal.

Positive:
    BEFORE is higher than AFTER
    -> candidate material / plaque removal

Zero:
    same geometric height

Negative:
    AFTER is higher than BEFORE

No threshold is applied at this stage.
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
        )
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("RAY-CAST CIRCULAR SURFACE CHANGE")
    print("=" * 80)

    print(
        f"\nValid ROI area: "
        f"{results['roi_area']:.6f}"
    )

    print(
        f"Valid ray-cast cells: "
        f"{results['valid_fraction'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # SIGNED
    # --------------------------------------------------------

    print(
        "\n--- RAW SIGNED HEIGHT CHANGE ---"
    )

    print(
        f"Mean: "
        f"{results['mean_signed']:.6f}"
    )

    print(
        f"Median: "
        f"{results['median_signed']:.6f}"
    )

    print(
        f"5th percentile: "
        f"{results['signed_5']:.6f}"
    )

    print(
        f"95th percentile: "
        f"{results['signed_95']:.6f}"
    )

    print(
        f"99th percentile: "
        f"{results['signed_99']:.6f}"
    )

    print(
        f"Minimum: "
        f"{results['minimum_signed']:.6f}"
    )

    print(
        f"Maximum: "
        f"{results['maximum_signed']:.6f}"
    )

    # --------------------------------------------------------
    # ABSOLUTE
    # --------------------------------------------------------

    print(
        "\n--- ABSOLUTE HEIGHT DIFFERENCE ---"
    )

    print(
        f"Mean: "
        f"{results['mean_absolute']:.6f}"
    )

    print(
        f"Median: "
        f"{results['median_absolute']:.6f}"
    )

    print(
        f"95th percentile: "
        f"{results['absolute_95']:.6f}"
    )

    print(
        f"99th percentile: "
        f"{results['absolute_99']:.6f}"
    )

    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    print(
        "\n--- POSITIVE SURFACE REMOVAL ---"
    )

    print(
        f"Mean: "
        f"{results['mean_positive']:.6f}"
    )

    print(
        f"Median: "
        f"{results['median_positive']:.6f}"
    )

    print(
        f"95th percentile: "
        f"{results['positive_95']:.6f}"
    )

    print(
        f"99th percentile: "
        f"{results['positive_99']:.6f}"
    )

    print(
        f"Maximum: "
        f"{results['maximum_positive']:.6f}"
    )

    # --------------------------------------------------------
    # RAY QC
    # --------------------------------------------------------

    print(
        "\n--- RAY-CAST QC ---"
    )

    print(
        f"Mean BEFORE-to-AFTER ray distance: "
        f"{results['mean_ray_distance']:.6f}"
    )

    print(
        f"Median BEFORE-to-AFTER ray distance: "
        f"{results['median_ray_distance']:.6f}"
    )

    # --------------------------------------------------------
    # RAW VOLUME
    # --------------------------------------------------------

    print(
        "\n--- RAW VOLUME ESTIMATES ---"
    )

    print(
        f"Positive removed volume: "
        f"{results['positive_volume']:.6f}"
    )

    print(
        f"Signed volume difference: "
        f"{results['signed_volume']:.6f}"
    )

    print(
        "\nNo noise threshold has been applied."
    )

    print(
        "If PLY coordinates are in millimetres, "
        "volume is already in mm^3."
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

        "after_circle_before_correction":
            after_circle,

        "after_circle_final":
            after_circle_final,

        "before_landmarks":
            before_landmarks,

        "after_landmarks":
            after_landmarks,

        "transformed_after_landmarks":
            transformed_after_landmarks,

        "before_roi":
            before_roi,

        "after_roi":
            after_roi,

        "stage1_transform":
            stage1_transform,

        "circle_transform":
            circle_transform,

        "stage1_mean_distance":
            stage1_mean,

        "stage1_iterations":
            stage1_iterations,

        "circle_rotation_angle":
            circle_rotation_angle,

        "centre_error_before":
            centre_error_before,

        "centre_error_final":
            centre_error_final,

        "normal_angle_before":
            normal_angle_before,

        "normal_angle_final":
            normal_angle_final,

        "surface_results":
            results,
    }


# ============================================================
# MAIN MENU
# ============================================================

while True:

    print("\n" + "=" * 80)
    print("BIOCAL3D — CIRCULAR SURFACE CHANGE")
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

            result = (
                analyze_before_after(
                    index=index,
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