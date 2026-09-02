import open3d as o3d
import numpy as np

# --------------------------------------------------
# 1. Load PLY as a triangle mesh
# --------------------------------------------------

ply_path = r"C:\Users\johan\OneDrive - Aarhus universitet\Forskning - Tandlægeskolen\_IOOS_036_Johan\BioCal3D\30-07-26 Pilot data\Before brushing\1-2 LowerJawScan.ply"


mesh = o3d.io.read_triangle_mesh(ply_path)

# --------------------------------------------------
# 2. Inspect the mesh
# --------------------------------------------------

print(mesh)

print(f"Number of vertices: {len(mesh.vertices)}")
print(f"Number of triangles: {len(mesh.triangles)}")
print(f"Has vertex colours: {mesh.has_vertex_colors()}")
print(f"Has vertex normals: {mesh.has_vertex_normals()}")

# --------------------------------------------------
# 3. Extract data
# --------------------------------------------------

vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
colors = np.asarray(mesh.vertex_colors)

print("\nData shapes:")
print(f"Vertices:  {vertices.shape}")
print(f"Triangles: {triangles.shape}")
print(f"Colours:   {colors.shape}")

# --------------------------------------------------
# 4. Show coordinate range
# --------------------------------------------------

print("\nCoordinate ranges:")

for i, axis in enumerate(["X", "Y", "Z"]):
    print(
        f"{axis}: "
        f"{vertices[:, i].min():.4f} → "
        f"{vertices[:, i].max():.4f}"
    )

# --------------------------------------------------
# 5. Show RGB range
# --------------------------------------------------

print("\nRGB range:")
print(f"R: {colors[:, 0].min():.3f} → {colors[:, 0].max():.3f}")
print(f"G: {colors[:, 1].min():.3f} → {colors[:, 1].max():.3f}")
print(f"B: {colors[:, 2].min():.3f} → {colors[:, 2].max():.3f}")

# --------------------------------------------------
# 6. Visualize original mesh
# --------------------------------------------------

o3d.visualization.draw_geometries(
    [mesh],
    window_name="Original PLY Mesh",
    mesh_show_back_face=True
)