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
    "Extraoral scanner": DATA_ROOT / "Scans from extraoral scanner",
    "White background": DATA_ROOT / "White background -available only for the post brushing sample",
}


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
# LOAD AND CLEAN MESH
# ============================================================

def load_mesh(path):

    mesh = pv.read(path)

    print(f"\nLoaded: {path.name}")
    print(f"Points: {mesh.n_points:,}")
    print(f"Cells:  {mesh.n_cells:,}")

    # Clean duplicate points
    mesh = mesh.clean()

    # Triangulate
    mesh = mesh.triangulate()

    # Compute normals
    mesh = mesh.compute_normals(
        point_normals=True,
        cell_normals=True,
        consistent_normals=True,
        auto_orient_normals=True
    )

    return mesh


# ============================================================
# ICP REGISTRATION
# ============================================================

def icp_register(source, target, max_iterations=100):

    """
    Register source mesh onto target mesh using ICP.

    source = mesh that will be transformed
    target = reference mesh
    """

    print("\nRunning ICP registration...")

    source_vtk = source
    target_vtk = target

    icp = vtk.vtkIterativeClosestPointTransform()

    icp.SetSource(source_vtk)
    icp.SetTarget(target_vtk)

    icp.GetLandmarkTransform().SetModeToRigidBody()

    icp.SetMaximumNumberOfIterations(max_iterations)

    # Maximum distance for correspondence
    icp.SetMaximumMeanDistance(1e-4)

    icp.StartByMatchingCentroidsOn()

    icp.Modified()
    icp.Update()

    transform = icp.GetMatrix()

    # Apply transformation
    registered = source.copy()
    registered.transform(transform, inplace=True)

    print("ICP completed.")

    return registered, transform

# ============================================================
# VISUALIZE ICP OVERLAY
# ============================================================

def visualize_icp_overlay(before, after_registered):
    """
    Visually verify the full-mesh ICP registration.

    BEFORE = blue transparent mesh
    AFTER  = red transparent mesh

    The full models are shown here because ICP itself is based
    on the full model. This step is purely for visual QC.
    """

    print("\n" + "=" * 80)
    print("ICP VERIFICATION")
    print("=" * 80)

    print("""
The full BEFORE and AFTER models are shown after ICP.

Inspect the registration by:
  - rotating the model
  - zooming in/out
  - looking for areas where the surfaces do not overlap

Close the window when you are satisfied.

IMPORTANT:
This visualization is ONLY a registration check.
The quantitative analysis will still use the circular ROI.
""")

    plotter = pv.Plotter()

    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    plotter.add_mesh(
        before,
        color="blue",
        opacity=0.50,
        smooth_shading=False,
        show_edges=False,
        name="before"
    )

    # --------------------------------------------------------
    # AFTER
    # --------------------------------------------------------

    plotter.add_mesh(
        after_registered,
        color="red",
        opacity=0.50,
        smooth_shading=False,
        show_edges=False,
        name="after"
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    plotter.add_text(
        "ICP OVERLAY\n"
        "BEFORE = BLUE\n"
        "AFTER = RED\n\n"
        "Close window to continue",
        position="upper_left",
        font_size=14
    )

    plotter.add_axes()

    plotter.show()


# ============================================================
# CALCULATE POINT-TO-SURFACE DISTANCE
# ============================================================

def calculate_surface_distances(source, target):

    """
    Calculate closest-point distance from every source point
    to the target surface.
    """

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(target)
    locator.BuildLocator()

    points = source.points

    distances = np.zeros(len(points))

    closest_points = np.zeros_like(points)

    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)

    for i, point in enumerate(points):

        closest = [0.0, 0.0, 0.0]

        locator.FindClosestPoint(
            point,
            closest,
            cell_id,
            sub_id,
            dist2
        )

        closest_points[i] = closest

        distances[i] = np.sqrt(float(dist2))

    return distances, closest_points


# ============================================================
# CALCULATE TRIANGLE AREAS
# ============================================================

def triangle_areas(mesh):

    """
    Calculate area of every triangle.
    """

    points = mesh.points
    faces = mesh.faces.reshape(-1, 4)

    triangles = faces[:, 1:4]

    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]

    cross = np.cross(
        p1 - p0,
        p2 - p0
    )

    areas = 0.5 * np.linalg.norm(cross, axis=1)

    return areas


# ============================================================
# POINT DISTANCE -> CELL DISTANCE
# ============================================================

def point_to_cell_distances(mesh, point_distances):

    """
    Convert point distances to triangle distances
    using the mean distance of the three vertices.
    """

    faces = mesh.faces.reshape(-1, 4)

    triangles = faces[:, 1:4]

    cell_distances = np.mean(
        point_distances[triangles],
        axis=1
    )

    return cell_distances


# ============================================================
# ESTIMATE VOLUME DIFFERENCE
# ============================================================

def calculate_volume_difference(
    before,
    distances,
    threshold=0.05
):

    """
    Estimate removed volume using:

        volume = distance * surface area

    Distances below threshold are ignored as noise.
    """

    # Convert point distances to triangle distances
    cell_distances = point_to_cell_distances(
        before,
        distances
    )

    # Calculate triangle areas
    areas = triangle_areas(before)

    # Ignore small differences
    valid = cell_distances > threshold

    effective_distance = np.zeros_like(cell_distances)

    effective_distance[valid] = (
        cell_distances[valid] - threshold
    )

    # Local volume contribution
    volumes = (
        effective_distance *
        areas
    )

    total_volume = np.sum(volumes)

    return total_volume, cell_distances, volumes


# ============================================================
# CREATE DIFFERENCE MESH
# ============================================================

def create_difference_mesh(
    mesh,
    cell_distances
):

    difference_mesh = mesh.copy()

    difference_mesh.cell_data["Distance"] = (
        cell_distances
    )

    return difference_mesh


# ============================================================
# VISUALIZE DIFFERENCE
# ============================================================

def visualize_difference(
    before,
    after,
    difference_mesh,
    threshold
):

    plotter = pv.Plotter(
        shape=(1, 3)
    )

    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    plotter.subplot(0, 0)

    plotter.add_mesh(
        before,
        rgb=True,
        smooth_shading=False
    )

    plotter.add_text(
        "Before brushing",
        font_size=14
    )

    plotter.add_axes()

    # --------------------------------------------------------
    # AFTER
    # --------------------------------------------------------

    plotter.subplot(0, 1)

    plotter.add_mesh(
        after,
        rgb=True,
        smooth_shading=False
    )

    plotter.add_text(
        "After brushing",
        font_size=14
    )

    plotter.add_axes()

    # --------------------------------------------------------
    # DIFFERENCE
    # --------------------------------------------------------

    plotter.subplot(0, 2)

    plotter.add_mesh(
        difference_mesh,
        scalars="Distance",
        cmap="turbo",
        clim=[
            threshold,
            np.percentile(
                difference_mesh.cell_data["Distance"],
                99
            )
        ],
        scalar_bar_args={
            "title": "Surface difference"
        }
    )

    plotter.add_text(
        f"Surface difference\n"
        f"Threshold = {threshold:.3f}",
        font_size=14
    )

    plotter.add_axes()

    plotter.link_views()

    plotter.show()

# ============================================================
# CIRCULAR ROI
# ============================================================

def select_circular_roi(mesh):
    """
    Interactively define the circular sample ROI.

    Pick 3 points on the BEFORE mesh:

        Point 1 = centre of circular sample
        Point 2 = point on the circular edge
        Point 3 = another point on the circular edge

    The three points define:
        - circle centre
        - circle radius
        - plane normal

    The ROI is then used for all quantitative analysis.
    """

    picked_points = []

    print("\n" + "=" * 80)
    print("SELECT CIRCULAR ROI")
    print("=" * 80)

    print("""
You will select 3 points on the BEFORE scan:

  1. Centre of the circular sample
  2. Point on the edge of the circular sample
  3. Another point on the edge of the circular sample

Click the points in the 3D window.
Close the window when finished.
""")

    def callback(point, picker):
        if point is not None:
            picked_points.append(np.array(point))

            print(
                f"Point {len(picked_points)}: "
                f"{np.array(point)}"
            )

    plotter = pv.Plotter()

    plotter.add_mesh(
        mesh,
        rgb=True,
        smooth_shading=False,
        show_edges=False
    )

    plotter.add_text(
        "Click 3 points:\n"
        "1 = centre\n"
        "2 = edge\n"
        "3 = edge",
        font_size=14
    )

    plotter.add_axes()

    plotter.enable_surface_point_picking(
        callback=callback,
        show_point=True,
        show_message=True,
        left_clicking=True,
        use_picker=True
    )

    plotter.show()

    if len(picked_points) != 3:

        raise RuntimeError(
            f"Expected 3 points, "
            f"but received {len(picked_points)}."
        )

    # --------------------------------------------------------
    # POINTS
    # --------------------------------------------------------

    center = picked_points[0]
    edge1 = picked_points[1]
    edge2 = picked_points[2]

    # --------------------------------------------------------
    # RADIUS
    # --------------------------------------------------------

    radius1 = np.linalg.norm(edge1 - center)
    radius2 = np.linalg.norm(edge2 - center)

    radius = (radius1 + radius2) / 2.0

    # --------------------------------------------------------
    # PLANE NORMAL
    # --------------------------------------------------------

    v1 = edge1 - center
    v2 = edge2 - center

    normal = np.cross(v1, v2)

    normal_length = np.linalg.norm(normal)

    if normal_length < 1e-8:

        raise RuntimeError(
            "The three selected points are too "
            "close to being collinear."
        )

    normal /= normal_length

    print("\nROI defined:")
    print(f"Centre:  {center}")
    print(f"Radius:  {radius:.4f}")
    print(f"Normal:  {normal}")

    return {
        "center": center,
        "radius": radius,
        "normal": normal
    }


# ============================================================
# ROI MASK
# ============================================================

def create_roi_mask(mesh, roi):
    """
    Determine which cells of a mesh are inside the circular ROI.

    The calculation is performed using the centre of each
    triangle projected onto the ROI plane.
    """

    center = roi["center"]
    radius = roi["radius"]
    normal = roi["normal"]

    # Triangle connectivity
    faces = mesh.faces.reshape(-1, 4)
    triangles = faces[:, 1:4]

    # Triangle centres
    points = mesh.points

    centroids = (
        points[triangles[:, 0]]
        + points[triangles[:, 1]]
        + points[triangles[:, 2]]
    ) / 3.0

    # Vector from ROI centre to triangle centre
    vectors = centroids - center

    # Remove component perpendicular to ROI plane
    perpendicular = (
        np.sum(vectors * normal, axis=1)[:, None]
        * normal
    )

    projected = vectors - perpendicular

    # Radial distance from ROI centre
    radial_distance = np.linalg.norm(
        projected,
        axis=1
    )

    # Inside circle
    mask = radial_distance <= radius

    return mask


# ============================================================
# ROI VISUALIZATION
# ============================================================

def visualize_roi(mesh, roi, roi_mask):

    """
    Show the mesh with the selected ROI highlighted.
    """

    faces = mesh.faces.reshape(-1, 4)

    # Create a copy for visualization
    display_mesh = mesh.copy()

    # Assign ROI value to cells
    display_mesh.cell_data["ROI"] = (
        roi_mask.astype(float)
    )

    plotter = pv.Plotter()

    # Full mesh
    plotter.add_mesh(
        display_mesh,
        scalars="ROI",
        cmap="coolwarm",
        clim=[0, 1],
        show_scalar_bar=False,
        smooth_shading=False
    )

    # Add ROI centre
    plotter.add_points(
        roi["center"],
        color="black",
        point_size=15,
        render_points_as_spheres=True
    )

    plotter.add_text(
        f"ROI radius = {roi['radius']:.2f}",
        font_size=14
    )

    plotter.add_axes()

    plotter.show()


# ============================================================
# ROI-RESTRICTED VOLUME
# ============================================================

def calculate_volume_difference_roi(
    before,
    point_distances,
    roi_mask,
    threshold=0.05
):
    """
    Calculate volume difference ONLY inside the circular ROI.
    """

    # Convert point distances to cell distances
    cell_distances = point_to_cell_distances(
        before,
        point_distances
    )

    # Triangle areas
    areas = triangle_areas(before)

    # --------------------------------------------------------
    # Apply noise threshold
    # --------------------------------------------------------

    effective_distance = np.maximum(
        cell_distances - threshold,
        0
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # ONLY ROI CONTRIBUTES
    # --------------------------------------------------------

    effective_distance[
        ~roi_mask
    ] = 0.0

    # Local volume
    volumes = (
        effective_distance *
        areas
    )

    total_volume = np.sum(volumes)

    # ROI area
    roi_area = np.sum(
        areas[roi_mask]
    )

    # Area showing a difference
    changed = (
        roi_mask &
        (cell_distances > threshold)
    )

    changed_area = np.sum(
        areas[changed]
    )

    return (
        total_volume,
        roi_area,
        changed_area,
        cell_distances,
        volumes
    )


# ============================================================
# ROI DIFFERENCE VISUALIZATION
# ============================================================

def visualize_difference_roi(
    before,
    after,
    cell_distances,
    roi_mask,
    threshold
):

    display_mesh = before.copy()

    # --------------------------------------------------------
    # Mask everything outside ROI
    # --------------------------------------------------------

    display_distances = cell_distances.copy()

    display_distances[~roi_mask] = np.nan

    display_mesh.cell_data["Distance"] = (
        display_distances
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plotter = pv.Plotter(
        shape=(1, 3)
    )

    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    plotter.subplot(0, 0)

    plotter.add_mesh(
        before,
        rgb=True,
        smooth_shading=False
    )

    plotter.add_text(
        "Before brushing",
        font_size=14
    )

    plotter.add_axes()

    # --------------------------------------------------------
    # AFTER
    # --------------------------------------------------------

    plotter.subplot(0, 1)

    plotter.add_mesh(
        after,
        rgb=True,
        smooth_shading=False
    )

    plotter.add_text(
        "After brushing\nRegistered",
        font_size=14
    )

    plotter.add_axes()

    # --------------------------------------------------------
    # DIFFERENCE
    # --------------------------------------------------------

    plotter.subplot(0, 2)

    valid_distances = display_distances[
        ~np.isnan(display_distances)
    ]

    if len(valid_distances) > 0:

        vmax = np.percentile(
            valid_distances,
            99
        )

    else:

        vmax = threshold * 2

    plotter.add_mesh(
        display_mesh,
        scalars="Distance",
        cmap="turbo",
        clim=[
            threshold,
            vmax
        ],
        smooth_shading=False,
        scalar_bar_args={
            "title": "Surface difference"
        }
    )

    plotter.add_text(
        "Difference\n"
        "ROI ONLY",
        font_size=14
    )

    plotter.add_axes()

    plotter.link_views()

    plotter.show()


# ============================================================
# BEFORE / AFTER ANALYSIS WITH ROI
# ============================================================

def analyze_before_after(
    index,
    threshold=0.05
):

    before_files = files["Before brushing"]
    after_files = files["After brushing"]

    if index >= len(before_files):
        print("Invalid before file.")
        return

    if index >= len(after_files):
        print("Invalid after file.")
        return

    before_path = before_files[index]
    after_path = after_files[index]

    print("\n" + "=" * 80)
    print("BEFORE / AFTER ANALYSIS")
    print("=" * 80)

    print(f"Before: {before_path.name}")
    print(f"After:  {after_path.name}")

    # ========================================================
    # LOAD
    # ========================================================

    before = load_mesh(before_path)
    after = load_mesh(after_path)

    # ========================================================
    # ICP
    #
    # IMPORTANT:
    # FULL MESHES ARE USED HERE
    # ========================================================

    print("\nInitial alignment...")

    before_center = np.array(before.center)
    after_center = np.array(after.center)

    after.translate(
        before_center - after_center
    )

    print("\nRunning FULL-MESH ICP...")

    after_registered, transform = icp_register(
        after,
        before
    )

    # ========================================================
    # VERIFY ICP REGISTRATION
    # ========================================================

    visualize_icp_overlay(
        before=before,
        after_registered=after_registered
    )

    # ========================================================
    # SELECT ROI
    #
    # IMPORTANT:
    # ROI IS DEFINED AFTER REGISTRATION
    # ========================================================

    roi = select_circular_roi(before)

    # ========================================================
    # CREATE ROI MASK
    # ========================================================

    roi_mask = create_roi_mask(
        before,
        roi
    )

    print("\nROI statistics:")

    print(
        f"ROI triangles: "
        f"{np.sum(roi_mask):,}"
    )

    print(
        f"Total triangles: "
        f"{len(roi_mask):,}"
    )

    print(
        f"ROI percentage: "
        f"{100 * np.mean(roi_mask):.2f}%"
    )

    print(
        f"ROI radius: "
        f"{roi['radius']:.4f}"
    )

    # ========================================================
    # VISUALIZE ROI
    # ========================================================

    visualize_roi(
        before,
        roi,
        roi_mask
    )

    # ========================================================
    # SURFACE DISTANCE
    #
    # Distances are calculated from BEFORE to AFTER
    # ========================================================

    print("\nCalculating surface distances...")

    distances, closest_points = (
        calculate_surface_distances(
            before,
            after_registered
        )
    )

    # ========================================================
    # GENERAL DISTANCE STATISTICS
    # ========================================================

    print("\nFull-mesh distance statistics:")

    print(
        f"Mean:   {np.mean(distances):.4f}"
    )

    print(
        f"Median: {np.median(distances):.4f}"
    )

    print(
        f"95th:   {np.percentile(distances, 95):.4f}"
    )

    print(
        f"Max:    {np.max(distances):.4f}"
    )

    # ========================================================
    # ROI VOLUME
    # ========================================================

    (
        volume,
        roi_area,
        changed_area,
        cell_distances,
        volumes
    ) = calculate_volume_difference_roi(
        before=before,
        point_distances=distances,
        roi_mask=roi_mask,
        threshold=threshold
    )

    # ========================================================
    # ROI DISTANCE STATISTICS
    # ========================================================

    roi_point_mask = np.zeros(
        before.n_points,
        dtype=bool
    )

    # Points belonging to ROI triangles
    faces = before.faces.reshape(-1, 4)
    triangles = faces[:, 1:4]

    roi_points = np.unique(
        triangles[roi_mask].ravel()
    )

    roi_point_mask[roi_points] = True

    roi_distances = distances[
        roi_point_mask
    ]

    print("\n" + "=" * 80)
    print("ROI RESULTS")
    print("=" * 80)

    print(
        f"ROI area: "
        f"{roi_area:.4f} square units"
    )

    print(
        f"Area above threshold: "
        f"{changed_area:.4f} square units"
    )

    print(
        f"Mean ROI distance: "
        f"{np.mean(roi_distances):.4f}"
    )

    print(
        f"Median ROI distance: "
        f"{np.median(roi_distances):.4f}"
    )

    print(
        f"95th percentile ROI distance: "
        f"{np.percentile(roi_distances, 95):.4f}"
    )

    print(
        f"\nEstimated ROI volume difference: "
        f"{volume:.6f} cubic units"
    )

    print(
        f"Estimated volume difference: "
        f"{volume * 1000:.3f} mm³"
    )

    # ========================================================
    # VISUALIZE DIFFERENCE
    # ========================================================

    visualize_difference_roi(
        before=before,
        after=after_registered,
        cell_distances=cell_distances,
        roi_mask=roi_mask,
        threshold=threshold
    )

    return {
        "before": before,
        "after": after_registered,
        "distances": distances,
        "cell_distances": cell_distances,
        "roi": roi,
        "roi_mask": roi_mask,
        "roi_area": roi_area,
        "changed_area": changed_area,
        "volume": volume,
        "difference_mesh": before.copy(),
        "transform": transform
    }

# ============================================================
# SIMPLE MENU
# ============================================================

while True:

    print("\n" + "=" * 80)
    print("BIOCAL3D ANALYSIS")
    print("=" * 80)

    print("1 - Before / After surface difference")
    print("2 - Quit")

    choice = input("\nChoice: ").strip()

    if choice == "1":

        print("\nAvailable pairs:")

        for i, (before, after) in enumerate(
            zip(
                files["Before brushing"],
                files["After brushing"]
            )
        ):
            print(
                f"[{i}] "
                f"{before.name}  <->  {after.name}"
            )

        try:

            index = int(
                input("\nPair number: ")
            )

            threshold = float(
                input(
                    "Distance threshold "
                    "(same units as PLY): "
                )
            )

            result = analyze_before_after(
                index=index,
                threshold=threshold
            )

        except ValueError:

            print("\nInvalid input.")

        except Exception as e:

            print("\n" + "=" * 80)
            print("ANALYSIS ERROR")
            print("=" * 80)
            print(f"{type(e).__name__}: {e}")

            print(
                "\nReturning to main menu..."
            )

    elif choice == "2":

        print("\nExiting BioCal3D.")
        break

    else:

        print("\nPlease enter 1 or 2.")