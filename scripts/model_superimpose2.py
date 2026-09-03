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
# LANDMARK SETTINGS
# ============================================================

N_LANDMARKS = 4

LANDMARK_NAMES = [
    "TOP-LEFT corner of square mount",
    "TOP-RIGHT corner of square mount",
    "BOTTOM-RIGHT corner of square mount",
    "BOTTOM-LEFT corner of square mount",
]


# ============================================================
# ICP SETTINGS
# ============================================================

# Initial full-model ICP
STAGE1_ITERATIONS = 300
STAGE1_LANDMARKS = 20000

# Final ICP using square mount only
STAGE2_ITERATIONS = 500
STAGE2_LANDMARKS = 20000

# Extra exclusion around circular sample
ICP_EXCLUSION_BUFFER = 2.0

# Margin around four mount landmarks
MOUNT_MARGIN = 1.5


# ============================================================
# FIND FILES
# ============================================================

files = {}

for category, folder in FOLDERS.items():

    if folder.exists():
        files[category] = sorted(folder.glob("*.ply"))
    else:
        files[category] = []


# ============================================================
# LOAD MESH
# ============================================================

def load_mesh(path):

    mesh = pv.read(path)

    print(f"\nLoaded: {path.name}")
    print(f"Points: {mesh.n_points:,}")
    print(f"Cells: {mesh.n_cells:,}")

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
# PICK LANDMARKS
# ============================================================

def pick_landmarks(mesh, title):

    picked_points = []

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(
        f"""
Select exactly {N_LANDMARKS} landmarks in this order:

1. {LANDMARK_NAMES[0]}
2. {LANDMARK_NAMES[1]}
3. {LANDMARK_NAMES[2]}
4. {LANDMARK_NAMES[3]}

LEFT CLICK  = select point
RIGHT CLICK = remove last point

Select the SAME physical corners on BEFORE and AFTER.

Close the window after all points have been selected.
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
            for i, name in enumerate(LANDMARK_NAMES)
        )
        + "\n\nLEFT CLICK = select\n"
        "RIGHT CLICK = remove last",
        position="upper_left",
        font_size=12,
        color="black",
    )

    plotter.add_axes()

    picker = vtk.vtkCellPicker()
    picker.SetTolerance(0.005)

    def left_click_callback(obj, event):

        if len(picked_points) >= N_LANDMARKS:
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
                "Nothing picked. Click directly on the mesh."
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

        picked_points.append(point)

        idx = len(picked_points)

        print(
            f"\nLandmark {idx}/{N_LANDMARKS}"
        )

        print(
            f"  {LANDMARK_NAMES[idx - 1]}"
        )

        print(
            f"  X = {point[0]:.5f}"
            f"  Y = {point[1]:.5f}"
            f"  Z = {point[2]:.5f}"
        )

        marker = pv.PolyData(
            point.reshape(1, 3)
        )

        plotter.add_mesh(
            marker,
            color="red",
            point_size=15,
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

        if len(picked_points) == N_LANDMARKS:

            print(
                "\nAll four square-mount landmarks selected."
            )

            print(
                "Close the window to continue."
            )

    def right_click_callback(obj, event):

        if len(picked_points) == 0:

            obj.SetAbortFlag(1)
            return

        idx = len(picked_points)

        removed = picked_points.pop()

        print(
            f"\nRemoved landmark {idx}: "
            f"{LANDMARK_NAMES[idx - 1]}"
        )

        print(
            f"Coordinates were: {removed}"
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

    if len(picked_points) != N_LANDMARKS:

        raise RuntimeError(
            f"Expected {N_LANDMARKS} landmarks "
            f"but got {len(picked_points)}."
        )

    return np.asarray(
        picked_points,
        dtype=np.float64,
    )


# ============================================================
# CALCULATE RIGID TRANSFORM
# ============================================================

def calculate_rigid_transform(
    source_points,
    target_points,
):

    source_centroid = np.mean(
        source_points,
        axis=0,
    )

    target_centroid = np.mean(
        target_points,
        axis=0,
    )

    source_centered = (
        source_points - source_centroid
    )

    target_centered = (
        target_points - target_centroid
    )

    H = (
        source_centered.T
        @ target_centered
    )

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:

        Vt[-1, :] *= -1

        R = Vt.T @ U.T

    t = (
        target_centroid
        - R @ source_centroid
    )

    matrix = vtk.vtkMatrix4x4()

    for i in range(3):

        for j in range(3):

            matrix.SetElement(
                i,
                j,
                float(R[i, j]),
            )

        matrix.SetElement(
            i,
            3,
            float(t[i]),
        )

    matrix.SetElement(3, 0, 0.0)
    matrix.SetElement(3, 1, 0.0)
    matrix.SetElement(3, 2, 0.0)
    matrix.SetElement(3, 3, 1.0)

    return matrix


# ============================================================
# TRANSFORM POINTS
# ============================================================

def transform_points(
    points,
    matrix,
):

    output = np.zeros_like(points)

    for i, p in enumerate(points):

        x = np.array(
            [
                p[0],
                p[1],
                p[2],
                1.0,
            ]
        )

        y = np.zeros(4)

        for row in range(4):

            y[row] = sum(
                matrix.GetElement(row, col)
                * x[col]
                for col in range(4)
            )

        output[i] = y[:3]

    return output


# ============================================================
# VISUALIZE LANDMARK ALIGNMENT
# ============================================================

def visualize_landmark_alignment(
    before,
    after_registered,
    before_landmarks,
    after_landmarks,
):

    print("\n" + "=" * 80)
    print("LANDMARK ALIGNMENT QC")
    print("=" * 80)

    plotter = pv.Plotter(
        window_size=(1400, 900)
    )

    plotter.add_mesh(
        before,
        color="blue",
        opacity=0.45,
        smooth_shading=False,
        show_edges=False,
    )

    plotter.add_mesh(
        after_registered,
        color="red",
        opacity=0.45,
        smooth_shading=False,
    )

    before_poly = pv.PolyData(
        before_landmarks
    )

    plotter.add_mesh(
        before_poly,
        color="blue",
        point_size=18,
        render_points_as_spheres=True,
    )

    plotter.add_point_labels(
        before_poly,
        [
            f"B{i + 1}"
            for i in range(
                len(before_landmarks)
            )
        ],
        font_size=16,
        text_color="blue",
        shape=None,
        always_visible=True,
    )

    after_poly = pv.PolyData(
        after_landmarks
    )

    plotter.add_mesh(
        after_poly,
        color="red",
        point_size=18,
        render_points_as_spheres=True,
    )

    plotter.add_point_labels(
        after_poly,
        [
            f"A{i + 1}"
            for i in range(
                len(after_landmarks)
            )
        ],
        font_size=16,
        text_color="red",
        shape=None,
        always_visible=True,
    )

    plotter.add_text(
        "LANDMARK REGISTRATION QC\n\n"
        "BEFORE = BLUE\n"
        "AFTER = RED\n\n"
        "Close window to continue",
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
    max_iterations=200,
    max_landmarks=10000,
    start_by_matching_centroids=False,
):

    print("\nRunning ICP registration...")

    icp = vtk.vtkIterativeClosestPointTransform()

    icp.SetSource(source)
    icp.SetTarget(target)

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

    if start_by_matching_centroids:

        icp.StartByMatchingCentroidsOn()

    else:

        icp.StartByMatchingCentroidsOff()

    icp.Modified()
    icp.Update()

    transform = icp.GetMatrix()

    registered = source.copy()

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

    print("\nICP completed.")

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
# VISUALIZE ICP OVERLAY
# ============================================================

def visualize_icp_overlay(
    before,
    after_registered,
    title="ICP OVERLAY",
):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    plotter = pv.Plotter(
        window_size=(1400, 900)
    )

    plotter.add_mesh(
        before,
        color="blue",
        opacity=0.50,
        smooth_shading=False,
        show_edges=False,
    )

    plotter.add_mesh(
        after_registered,
        color="red",
        opacity=0.50,
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
# SELECT CIRCULAR ROI
# ============================================================

def select_circular_roi(mesh):

    picked_points = []

    print("\n" + "=" * 80)
    print("SELECT CIRCULAR MEASUREMENT ROI")
    print("=" * 80)

    print(
        """
Select exactly 3 points on the BEFORE scan:

1. Centre of circular sample
2. Point on circular edge
3. Another point on circular edge

This circular ROI is:

- EXCLUDED from final ICP
- USED for surface-difference analysis
- USED for volume analysis
"""
    )

    def callback(point, picker):

        if point is not None:

            if len(picked_points) >= 3:
                return

            picked_points.append(
                np.array(point)
            )

            print(
                f"Point {len(picked_points)}: "
                f"{np.array(point)}"
            )

    plotter = pv.Plotter()

    plotter.add_mesh(
        mesh,
        rgb=True,
        smooth_shading=False,
        show_edges=False,
    )

    plotter.add_text(
        "Select 3 points:\n"
        "1 = centre\n"
        "2 = edge\n"
        "3 = edge",
        font_size=14,
    )

    plotter.add_axes()

    plotter.enable_surface_point_picking(
        callback=callback,
        show_point=True,
        show_message=True,
        left_clicking=True,
        use_picker=True,
    )

    plotter.show()

    if len(picked_points) != 3:

        raise RuntimeError(
            f"Expected 3 points, "
            f"but received "
            f"{len(picked_points)}."
        )

    center = picked_points[0]

    edge1 = picked_points[1]

    edge2 = picked_points[2]

    radius1 = np.linalg.norm(
        edge1 - center
    )

    radius2 = np.linalg.norm(
        edge2 - center
    )

    radius = (
        radius1 + radius2
    ) / 2.0

    v1 = edge1 - center
    v2 = edge2 - center

    normal = np.cross(
        v1,
        v2,
    )

    normal_length = np.linalg.norm(
        normal
    )

    if normal_length < 1e-8:

        raise RuntimeError(
            "The three selected points are "
            "too close to being collinear."
        )

    normal /= normal_length

    print("\nROI defined:")

    print(
        f"Centre: {center}"
    )

    print(
        f"Radius: {radius:.6f}"
    )

    print(
        f"Normal: {normal}"
    )

    return {
        "center": center,
        "radius": radius,
        "normal": normal,
    }


# ============================================================
# CREATE ROI MASK
# ============================================================

def create_roi_mask(mesh, roi):

    center = roi["center"]
    radius = roi["radius"]
    normal = roi["normal"]

    faces = mesh.faces.reshape(
        -1,
        4,
    )

    triangles = faces[:, 1:4]

    points = mesh.points

    centroids = (
        points[triangles[:, 0]]
        + points[triangles[:, 1]]
        + points[triangles[:, 2]]
    ) / 3.0

    vectors = (
        centroids - center
    )

    perpendicular = (
        np.sum(
            vectors * normal,
            axis=1,
        )[:, None]
        * normal
    )

    projected = (
        vectors - perpendicular
    )

    radial_distance = np.linalg.norm(
        projected,
        axis=1,
    )

    mask = (
        radial_distance <= radius
    )

    return mask


# ============================================================
# EXTRACT SQUARE MOUNT REGION
# ============================================================

def extract_mount_region(
    mesh,
    mount_landmarks,
    roi,
    margin=MOUNT_MARGIN,
):

    """
    Extract only the square-mount region for final ICP.

    The region is defined using the four manually selected
    mount corners.

    The circular sample is explicitly excluded.
    """

    points = mesh.points

    landmarks = np.asarray(
        mount_landmarks
    )

    # --------------------------------------------------------
    # Bounding box around square mount
    # --------------------------------------------------------

    xmin = (
        landmarks[:, 0].min()
        - margin
    )

    xmax = (
        landmarks[:, 0].max()
        + margin
    )

    ymin = (
        landmarks[:, 1].min()
        - margin
    )

    ymax = (
        landmarks[:, 1].max()
        + margin
    )

    zmin = (
        landmarks[:, 2].min()
        - margin
    )

    zmax = (
        landmarks[:, 2].max()
        + margin
    )

    mask = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
        & (points[:, 2] >= zmin)
        & (points[:, 2] <= zmax)
    )

    # --------------------------------------------------------
    # Exclude circular ROI
    # --------------------------------------------------------

    center = roi["center"]
    radius = roi["radius"]
    normal = roi["normal"]

    vectors = (
        points - center
    )

    perpendicular = (
        np.sum(
            vectors * normal,
            axis=1,
        )[:, None]
        * normal
    )

    projected = (
        vectors - perpendicular
    )

    radial_distance = np.linalg.norm(
        projected,
        axis=1,
    )

    mask &= (
        radial_distance
        > radius + ICP_EXCLUSION_BUFFER
    )

    indices = np.where(mask)[0]

    if len(indices) == 0:

        raise RuntimeError(
            "No points found in square-mount ICP region."
        )

    mount = mesh.extract_points(
        indices,
        adjacent_cells=True,
    )

    mount = mount.clean()
    mount = mount.triangulate()

    return mount


# ============================================================
# VISUALIZE MOUNT REGION
# ============================================================

def visualize_mount_region(
    mesh,
    mount,
    landmarks,
):

    print("\n" + "=" * 80)
    print("FINAL ICP REGION")
    print("=" * 80)

    print(
        """
ORANGE = geometry used for final ICP

The circular sample should NOT be orange.

The teeth should NOT be orange.
"""
    )

    plotter = pv.Plotter(
        window_size=(1400, 900)
    )

    plotter.add_mesh(
        mesh,
        color="lightgray",
        opacity=0.20,
        smooth_shading=False,
    )

    plotter.add_mesh(
        mount,
        color="orange",
        opacity=1.0,
        smooth_shading=False,
    )

    plotter.add_points(
        landmarks,
        color="red",
        point_size=18,
        render_points_as_spheres=True,
    )

    plotter.add_text(
        "FINAL ICP REGION\n"
        "ORANGE = SQUARE MOUNT ONLY",
        position="upper_left",
        font_size=14,
    )

    plotter.add_axes()

    plotter.show()


# ============================================================
# CALCULATE SURFACE DISTANCES
# ============================================================

def calculate_surface_distances(
    source,
    target,
):

    locator = vtk.vtkStaticCellLocator()

    locator.SetDataSet(
        target
    )

    locator.BuildLocator()

    points = source.points

    distances = np.zeros(
        len(points)
    )

    closest_points = np.zeros_like(
        points
    )

    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)

    for i, point in enumerate(points):

        closest = [
            0.0,
            0.0,
            0.0,
        ]

        locator.FindClosestPoint(
            point,
            closest,
            cell_id,
            sub_id,
            dist2,
        )

        closest_points[i] = closest

        distances[i] = np.sqrt(
            float(dist2)
        )

    return (
        distances,
        closest_points,
    )


# ============================================================
# TRIANGLE AREAS
# ============================================================

def triangle_areas(mesh):

    points = mesh.points

    faces = mesh.faces.reshape(
        -1,
        4,
    )

    triangles = faces[:, 1:4]

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

    areas = (
        0.5
        * np.linalg.norm(
            cross,
            axis=1,
        )
    )

    return areas


# ============================================================
# POINT DISTANCE -> CELL DISTANCE
# ============================================================

def point_to_cell_distances(
    mesh,
    point_distances,
):

    faces = mesh.faces.reshape(
        -1,
        4,
    )

    triangles = faces[:, 1:4]

    cell_distances = np.mean(
        point_distances[
            triangles
        ],
        axis=1,
    )

    return cell_distances


# ============================================================
# CALCULATE ROI SURFACE DIFFERENCE + VOLUME
# ============================================================

def calculate_volume_difference_roi(
    before,
    point_distances,
    roi_mask,
    threshold=0.01,
):

    """
    Calculate surface difference and estimated volume
    difference ONLY inside the circular ROI.

    Registration is already complete.

    point_distances:
        Distance from every BEFORE point to the
        registered AFTER surface.

    roi_mask:
        Triangle mask for circular ROI.
    """

    # --------------------------------------------------------
    # Point distances -> triangle distances
    # --------------------------------------------------------

    cell_distances = (
        point_to_cell_distances(
            before,
            point_distances,
        )
    )

    # --------------------------------------------------------
    # Triangle areas
    # --------------------------------------------------------

    areas = triangle_areas(
        before
    )

    # --------------------------------------------------------
    # Select ROI only
    # --------------------------------------------------------

    roi_cell_distances = (
        cell_distances[
            roi_mask
        ]
    )

    roi_areas = (
        areas[
            roi_mask
        ]
    )

    if len(roi_cell_distances) == 0:

        raise RuntimeError(
            "No surface triangles found inside "
            "the circular ROI."
        )

    # --------------------------------------------------------
    # Surface statistics
    # --------------------------------------------------------

    mean_distance = np.mean(
        roi_cell_distances
    )

    median_distance = np.median(
        roi_cell_distances
    )

    percentile_95 = np.percentile(
        roi_cell_distances,
        95,
    )

    percentile_99 = np.percentile(
        roi_cell_distances,
        99,
    )

    maximum_distance = np.max(
        roi_cell_distances
    )

    # --------------------------------------------------------
    # Changed surface area
    # --------------------------------------------------------

    changed_mask = (
        roi_cell_distances
        > threshold
    )

    changed_area = np.sum(
        roi_areas[
            changed_mask
        ]
    )

    total_roi_area = np.sum(
        roi_areas
    )

    # --------------------------------------------------------
    # Volume difference
    # --------------------------------------------------------

    effective_distance = np.maximum(
        roi_cell_distances - threshold,
        0,
    )

    volume_difference = np.sum(
        effective_distance
        * roi_areas
    )

    return {
        "cell_distances":
            cell_distances,

        "roi_distances":
            roi_cell_distances,

        "roi_areas":
            roi_areas,

        "roi_area":
            total_roi_area,

        "changed_area":
            changed_area,

        "mean_distance":
            mean_distance,

        "median_distance":
            median_distance,

        "percentile_95":
            percentile_95,

        "percentile_99":
            percentile_99,

        "maximum_distance":
            maximum_distance,

        "volume_difference":
            volume_difference,
    }


# ============================================================
# REGISTRATION ERROR
# OUTSIDE ROI — DIAGNOSTIC ONLY
# ============================================================

def calculate_registration_error(
    before,
    after_registered,
    roi,
):

    """
    Registration error outside the circular ROI.

    This is diagnostic only.

    It does NOT determine the final registration.
    """

    # --------------------------------------------------------
    # Build point mask outside circular ROI
    # --------------------------------------------------------

    center = roi["center"]
    radius = roi["radius"]
    normal = roi["normal"]

    vectors = (
        before.points - center
    )

    perpendicular = (
        np.sum(
            vectors * normal,
            axis=1,
        )[:, None]
        * normal
    )

    projected = (
        vectors - perpendicular
    )

    radial_distance = np.linalg.norm(
        projected,
        axis=1,
    )

    stable_mask = (
        radial_distance
        > radius
    )

    stable_indices = np.where(
        stable_mask
    )[0]

    if len(stable_indices) == 0:

        raise RuntimeError(
            "No points outside ROI "
            "for registration diagnostic."
        )

    before_stable = (
        before.extract_points(
            stable_indices,
            adjacent_cells=True,
        )
    )

    after_stable = (
        after_registered.extract_points(
            stable_indices,
            adjacent_cells=True,
        )
    )

    distances, _ = (
        calculate_surface_distances(
            before_stable,
            after_stable,
        )
    )

    print("\n" + "=" * 80)
    print("REGISTRATION ERROR — OUTSIDE ROI")
    print("=" * 80)

    print(
        f"Mean:   "
        f"{np.mean(distances):.6f}"
    )

    print(
        f"Median: "
        f"{np.median(distances):.6f}"
    )

    print(
        f"95th:   "
        f"{np.percentile(distances, 95):.6f}"
    )

    print(
        f"99th:   "
        f"{np.percentile(distances, 99):.6f}"
    )

    print(
        f"Maximum:"
        f" {np.max(distances):.6f}"
    )

    return distances


# ============================================================
# VISUALIZE ROI
# ============================================================

def visualize_roi(
    mesh,
    roi,
    roi_mask,
):

    display_mesh = mesh.copy()

    display_mesh.cell_data[
        "ROI"
    ] = (
        roi_mask.astype(float)
    )

    plotter = pv.Plotter()

    plotter.add_mesh(
        display_mesh,
        scalars="ROI",
        cmap="coolwarm",
        clim=[0, 1],
        show_scalar_bar=False,
        smooth_shading=False,
    )

    plotter.add_points(
        roi["center"],
        color="black",
        point_size=15,
        render_points_as_spheres=True,
    )

    plotter.add_text(
        f"CIRCULAR ROI\n"
        f"Radius = {roi['radius']:.4f}",
        font_size=14,
    )

    plotter.add_axes()

    plotter.show()


# ============================================================
# VISUALIZE SURFACE DIFFERENCE
# ============================================================

def visualize_difference_roi(
    before,
    after,
    cell_distances,
    roi_mask,
    threshold,
):

    display_mesh = before.copy()

    display_distances = (
        cell_distances.astype(float)
    )

    # IMPORTANT:
    # Everything outside the circular ROI
    # is hidden from the difference map.
    display_distances[
        ~roi_mask
    ] = np.nan

    display_mesh.cell_data[
        "Surface difference"
    ] = display_distances

    valid_distances = (
        display_distances[
            np.isfinite(
                display_distances
            )
        ]
    )

    if len(valid_distances) > 0:

        vmax = np.percentile(
            valid_distances,
            99,
        )

        vmax = max(
            vmax,
            threshold * 1.1,
        )

    else:

        vmax = threshold * 2

    plotter = pv.Plotter(
        shape=(1, 3),
        window_size=(1800, 700),
    )

    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    plotter.subplot(0, 0)

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

    plotter.subplot(0, 1)

    plotter.add_mesh(
        after,
        rgb=True,
        smooth_shading=False,
        show_edges=False,
    )

    plotter.add_text(
        "AFTER BRUSHING\n"
        "REGISTERED",
        font_size=14,
    )

    plotter.add_axes()

    # --------------------------------------------------------
    # DIFFERENCE
    # --------------------------------------------------------

    plotter.subplot(0, 2)

    plotter.add_mesh(
        display_mesh,
        scalars="Surface difference",
        cmap="turbo",
        clim=[
            threshold,
            vmax,
        ],
        smooth_shading=False,
        show_edges=False,
        scalar_bar_args={
            "title":
                "Surface difference"
        },
    )

    plotter.add_text(
        "SURFACE DIFFERENCE\n"
        "CIRCULAR ROI ONLY",
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

    before_files = files[
        "Before brushing"
    ]

    after_files = files[
        "After brushing"
    ]

    if index >= len(before_files):

        print(
            "Invalid before file."
        )

        return

    if index >= len(after_files):

        print(
            "Invalid after file."
        )

        return

    before_path = (
        before_files[index]
    )

    after_path = (
        after_files[index]
    )

    print("\n" + "=" * 80)
    print("BEFORE / AFTER ANALYSIS")
    print("=" * 80)

    print(
        f"Before: "
        f"{before_path.name}"
    )

    print(
        f"After: "
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
    # STAGE 1 ICP
    # FULL MODEL
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 1 — FULL-MESH ICP")
    print("=" * 80)

    print(
        """
The full model is used ONLY for approximate
initial alignment.
"""
    )

    (
        after_stage1,
        transform1,
        mean1,
        iterations1,
    ) = icp_register(
        source=after,
        target=before,
        max_iterations=STAGE1_ITERATIONS,
        max_landmarks=STAGE1_LANDMARKS,
        start_by_matching_centroids=False,
    )

    visualize_icp_overlay(
        before,
        after_stage1,
        title="STAGE 1 — INITIAL FULL-MESH ICP",
    )

    # ========================================================
    # FOUR SQUARE-MOUNT LANDMARKS
    # ========================================================

    print("\n" + "=" * 80)
    print("SQUARE-MOUNT LANDMARK REGISTRATION")
    print("=" * 80)

    input(
        "\nPress ENTER to select "
        "BEFORE landmarks..."
    )

    before_landmarks = (
        pick_landmarks(
            before,
            "BEFORE — Select 4 square-mount corners",
        )
    )

    input(
        "\nPress ENTER to select "
        "AFTER landmarks..."
    )

    after_landmarks = (
        pick_landmarks(
            after_stage1,
            "AFTER — Select the SAME 4 square-mount corners",
        )
    )

    # ========================================================
    # LANDMARK RIGID TRANSFORM
    # ========================================================

    print("\n" + "=" * 80)
    print("CALCULATING LANDMARK TRANSFORM")
    print("=" * 80)

    landmark_transform = (
        calculate_rigid_transform(
            source_points=after_landmarks,
            target_points=before_landmarks,
        )
    )

    after_landmark_registered = (
        after_stage1.copy()
    )

    after_landmark_registered.transform(
        landmark_transform,
        inplace=True,
    )

    after_landmarks_registered = (
        transform_points(
            after_landmarks,
            landmark_transform,
        )
    )

    # ========================================================
    # LANDMARK ERROR
    # ========================================================

    landmark_errors = np.linalg.norm(
        before_landmarks
        - after_landmarks_registered,
        axis=1,
    )

    print("\nSquare-mount landmark errors:")

    for i, error in enumerate(
        landmark_errors
    ):

        print(
            f"{i + 1}. "
            f"{LANDMARK_NAMES[i]}: "
            f"{error:.6f}"
        )

    print(
        f"\nMean landmark error: "
        f"{np.mean(landmark_errors):.6f}"
    )

    print(
        f"Maximum landmark error: "
        f"{np.max(landmark_errors):.6f}"
    )

    visualize_landmark_alignment(
        before=before,
        after_registered=after_landmark_registered,
        before_landmarks=before_landmarks,
        after_landmarks=after_landmarks_registered,
    )

    # ========================================================
    # SELECT CIRCULAR ROI
    # ========================================================

    roi = select_circular_roi(
        before
    )

    roi_mask_before = (
        create_roi_mask(
            before,
            roi,
        )
    )

    visualize_roi(
        before,
        roi,
        roi_mask_before,
    )

    # ========================================================
    # EXTRACT SQUARE MOUNT
    # ========================================================

    print("\n" + "=" * 80)
    print("EXTRACTING SQUARE-MOUNT ICP REGION")
    print("=" * 80)

    before_mount = extract_mount_region(
        mesh=before,
        mount_landmarks=before_landmarks,
        roi=roi,
    )

    after_mount = extract_mount_region(
        mesh=after_landmark_registered,
        mount_landmarks=before_landmarks,
        roi=roi,
    )

    print(
        f"BEFORE mount points: "
        f"{before_mount.n_points:,}"
    )

    print(
        f"AFTER mount points: "
        f"{after_mount.n_points:,}"
    )

    visualize_mount_region(
        mesh=before,
        mount=before_mount,
        landmarks=before_landmarks,
    )

    # ========================================================
    # STAGE 2 ICP
    # SQUARE MOUNT ONLY
    # ========================================================

    print("\n" + "=" * 80)
    print("STAGE 2 — FINAL ICP USING SQUARE MOUNT ONLY")
    print("=" * 80)

    (
        after_mount_registered,
        transform2,
        mean2,
        iterations2,
    ) = icp_register(
        source=after_mount,
        target=before_mount,
        max_iterations=STAGE2_ITERATIONS,
        max_landmarks=STAGE2_LANDMARKS,
        start_by_matching_centroids=False,
    )

    # ========================================================
    # APPLY FINAL ICP TRANSFORM TO ENTIRE AFTER
    # ========================================================

    after_final = (
        after_landmark_registered.copy()
    )

    after_final.transform(
        transform2,
        inplace=True,
    )

    # ========================================================
    # FINAL REGISTRATION VISUALIZATION
    # ========================================================

    visualize_icp_overlay(
        before,
        after_final,
        title="FINAL — SQUARE-MOUNT ICP ONLY",
    )

    # ========================================================
    # FINAL LANDMARK QC
    # ========================================================

    final_landmarks = (
        transform_points(
            after_landmarks_registered,
            transform2,
        )
    )

    final_landmark_errors = np.linalg.norm(
        before_landmarks
        - final_landmarks,
        axis=1,
    )

    print("\n" + "=" * 80)
    print("FINAL SQUARE-MOUNT LANDMARK QC")
    print("=" * 80)

    for i, error in enumerate(
        final_landmark_errors
    ):

        print(
            f"{i + 1}. "
            f"{LANDMARK_NAMES[i]}: "
            f"{error:.6f}"
        )

    print(
        f"\nMean final landmark error: "
        f"{np.mean(final_landmark_errors):.6f}"
    )

    print(
        f"Maximum final landmark error: "
        f"{np.max(final_landmark_errors):.6f}"
    )

    print("\nRegistration comparison:")

    print(
        f"Landmark-only mean error: "
        f"{np.mean(landmark_errors):.6f}"
    )

    print(
        f"Mount-ICP mean error:      "
        f"{np.mean(final_landmark_errors):.6f}"
    )

    if (
        np.mean(final_landmark_errors)
        > np.mean(landmark_errors)
    ):

        print(
            "\nWARNING:"
            "\nMount-only ICP increased the "
            "square-mount landmark error."
        )

    else:

        print(
            "\nMount-only ICP improved or "
            "maintained the square-mount alignment."
        )

    # ========================================================
    # REGISTRATION ERROR
    # OUTSIDE ROI
    # ========================================================

    registration_distances = (
        calculate_registration_error(
            before,
            after_final,
            roi,
        )
    )

    # ========================================================
    # SURFACE DISTANCES
    #
    # IMPORTANT:
    #
    # Distances are calculated between the COMPLETE
    # registered BEFORE and AFTER models.
    #
    # The ROI is applied AFTER this step.
    #
    # Therefore the circular sample is still available
    # for quantitative surface analysis.
    # ========================================================

    print("\n" + "=" * 80)
    print("CALCULATING SURFACE DIFFERENCE")
    print("=" * 80)

    distances, closest_points = (
        calculate_surface_distances(
            before,
            after_final,
        )
    )

    # ========================================================
    # ROI SURFACE DIFFERENCE + VOLUME
    # ========================================================

    roi_results = (
        calculate_volume_difference_roi(
            before=before,
            point_distances=distances,
            roi_mask=roi_mask_before,
            threshold=threshold,
        )
    )

    # ========================================================
    # ROI RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("CIRCULAR ROI SURFACE DIFFERENCE")
    print("=" * 80)

    print(
        f"\nROI area: "
        f"{roi_results['roi_area']:.6f}"
    )

    print(
        f"Mean surface difference: "
        f"{roi_results['mean_distance']:.6f}"
    )

    print(
        f"Median surface difference: "
        f"{roi_results['median_distance']:.6f}"
    )

    print(
        f"95th percentile: "
        f"{roi_results['percentile_95']:.6f}"
    )

    print(
        f"99th percentile: "
        f"{roi_results['percentile_99']:.6f}"
    )

    print(
        f"Maximum surface difference: "
        f"{roi_results['maximum_distance']:.6f}"
    )

    print(
        f"\nSurface area above threshold "
        f"({threshold}): "
        f"{roi_results['changed_area']:.6f}"
    )

    print(
        f"\nEstimated volume difference: "
        f"{roi_results['volume_difference']:.6f}"
    )

    print(
        f"Estimated volume difference "
        f"in mm³: "
        f"{roi_results['volume_difference'] * 1000:.3f}"
    )

    # ========================================================
    # FINAL SURFACE DIFFERENCE VISUALIZATION
    # ========================================================

    visualize_difference_roi(
        before=before,
        after=after_final,
        cell_distances=roi_results[
            "cell_distances"
        ],
        roi_mask=roi_mask_before,
        threshold=threshold,
    )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "before":
            before,

        "after":
            after_final,

        "distances":
            distances,

        "closest_points":
            closest_points,

        "cell_distances":
            roi_results[
                "cell_distances"
            ],

        "roi_distances":
            roi_results[
                "roi_distances"
            ],

        "roi":
            roi,

        "roi_mask":
            roi_mask_before,

        "roi_area":
            roi_results[
                "roi_area"
            ],

        "changed_area":
            roi_results[
                "changed_area"
            ],

        "mean_roi_distance":
            roi_results[
                "mean_distance"
            ],

        "median_roi_distance":
            roi_results[
                "median_distance"
            ],

        "roi_95th_percentile":
            roi_results[
                "percentile_95"
            ],

        "roi_99th_percentile":
            roi_results[
                "percentile_99"
            ],

        "maximum_roi_distance":
            roi_results[
                "maximum_distance"
            ],

        "volume_difference":
            roi_results[
                "volume_difference"
            ],

        "registration_distances":
            registration_distances,

        "before_landmarks":
            before_landmarks,

        "after_landmarks":
            after_landmarks,

        "after_landmarks_registered":
            after_landmarks_registered,

        "final_landmarks":
            final_landmarks,

        "landmark_errors":
            landmark_errors,

        "final_landmark_errors":
            final_landmark_errors,

        "landmark_transform":
            landmark_transform,

        "stage1_transform":
            transform1,

        "stage2_transform":
            transform2,

        "stage1_mean_distance":
            mean1,

        "stage2_mean_distance":
            mean2,

        "stage1_iterations":
            iterations1,

        "stage2_iterations":
            iterations2,
    }


# ============================================================
# MAIN MENU
# ============================================================

while True:

    print("\n" + "=" * 80)
    print("BIOCAL3D ANALYSIS")
    print("=" * 80)

    print(
        "1 - Before / After surface difference"
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

        print("\nAvailable pairs:")

        pairs = list(
            zip(
                files["Before brushing"],
                files["After brushing"],
            )
        )

        for i, (before_file, after_file) in enumerate(
            pairs
        ):

            print(
                f"[{i}] "
                f"{before_file.name} <-> "
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

            result = analyze_before_after(
                index=index,
                threshold=threshold,
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