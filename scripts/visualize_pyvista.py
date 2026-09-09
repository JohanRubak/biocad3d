from pathlib import Path

import numpy as np
import pyvista as pv


# ============================================================
# FILE
# ============================================================

PLY_FILE = Path(
    r"C:\Users\au662213\repos\biocad3d\data\02-09-2026-Data collected\iTERO\G2\BCG2T0-1\315628974_shell_occlusion_l.ply"
)


# ============================================================
# LOAD PLY
# ============================================================

if not PLY_FILE.exists():
    raise FileNotFoundError(f"Could not find file:\n{PLY_FILE}")

mesh = pv.read(PLY_FILE)

print("\n" + "=" * 60)
print("PLY INFORMATION")
print("=" * 60)

print(f"File:   {PLY_FILE.name}")
print(f"Points: {mesh.n_points:,}")
print(f"Cells:  {mesh.n_cells:,}")
print(f"Bounds: {mesh.bounds}")
print(f"Center: {mesh.center}")

print("\nPoint data:")
for name in mesh.point_data.keys():
    array = mesh.point_data[name]
    print(
        f"  {name}: "
        f"shape={array.shape}, "
        f"dtype={array.dtype}"
    )

print("\nCell data:")
for name in mesh.cell_data.keys():
    array = mesh.cell_data[name]
    print(
        f"  {name}: "
        f"shape={array.shape}, "
        f"dtype={array.dtype}"
    )


# ============================================================
# FIND RGB COLORS
# ============================================================

rgb_name = None

# Common ways RGB can be stored in PLY files
for candidate in [
    "RGB",
    "rgb",
    "RGBA",
    "rgba",
    "Colors",
    "colors",
    "Color",
    "color",
]:
    if candidate in mesh.point_data:
        rgb_name = candidate
        break


# Sometimes red, green, blue are stored separately
if rgb_name is None:
    point_keys = list(mesh.point_data.keys())

    if all(x in point_keys for x in ["red", "green", "blue"]):
        rgb = np.column_stack(
            [
                mesh.point_data["red"],
                mesh.point_data["green"],
                mesh.point_data["blue"],
            ]
        )

        mesh.point_data["RGB_combined"] = rgb.astype(np.uint8)
        rgb_name = "RGB_combined"


# ============================================================
# VISUALIZE
# ============================================================

plotter = pv.Plotter()

if rgb_name is not None:

    print(f"\nUsing vertex colors: {rgb_name}")

    plotter.add_mesh(
        mesh,
        scalars=rgb_name,
        rgb=True,
        smooth_shading=True,
    )

else:

    print("\nNo RGB vertex colors detected.")
    print("Displaying mesh using a uniform color.")

    plotter.add_mesh(
        mesh,
        color="lightgray",
        smooth_shading=True,
    )


# Coordinate axes
plotter.add_axes()

# White background
plotter.set_background("white")

# Fit camera to mesh
plotter.reset_camera()

plotter.show()