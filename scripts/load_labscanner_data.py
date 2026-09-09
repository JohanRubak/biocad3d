from pathlib import Path

import pyvista as pv


# ============================================================
# FILES
# ============================================================

DATA_DIR = Path(
    r"O:\HE_BioCal-3D-2026\02-09-2026-Data collected"
    r"\LABscanner\G1-OK"
)

PLY_FILE = DATA_DIR / "BCG1T0-1.ply"
TEXTURE_FILE = DATA_DIR / "BCG1T0-1.jpg"


# ============================================================
# CHECK FILES
# ============================================================

if not PLY_FILE.exists():
    raise FileNotFoundError(f"PLY file not found:\n{PLY_FILE}")

if not TEXTURE_FILE.exists():
    raise FileNotFoundError(f"Texture file not found:\n{TEXTURE_FILE}")


# ============================================================
# LOAD PLY
# ============================================================

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


# ============================================================
# TEXTURE COORDINATES
# ============================================================

if "TCoords" not in mesh.point_data:
    raise RuntimeError(
        "The PLY file does not contain TCoords.\n"
        "The JPG therefore cannot be mapped directly onto the mesh."
    )

print("\nTexture coordinates found.")

tcoords = mesh.point_data["TCoords"]

print(f"TCoords min: {tcoords.min(axis=0)}")
print(f"TCoords max: {tcoords.max(axis=0)}")


# ============================================================
# IMPORTANT:
# Make TCoords the active VTK texture coordinates
# ============================================================

vtk_tcoords = mesh.GetPointData().GetArray("TCoords")

if vtk_tcoords is None:
    raise RuntimeError("VTK could not access the TCoords array.")

mesh.GetPointData().SetTCoords(vtk_tcoords)


# ============================================================
# LOAD JPG TEXTURE
# ============================================================

texture = pv.read_texture(TEXTURE_FILE)

print("\nTexture loaded:")
print(TEXTURE_FILE.name)


# ============================================================
# VISUALIZE
# ============================================================

plotter = pv.Plotter(
    window_size=(1400, 900)
)

plotter.set_background("white")

plotter.add_mesh(
    mesh,
    texture=texture,
    smooth_shading=True,
)

plotter.add_axes()

plotter.reset_camera()

plotter.show()