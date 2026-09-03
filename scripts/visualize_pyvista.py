from pathlib import Path
import numpy as np
import pyvista as pv


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
# PRINT DATA
# ============================================================

print("=" * 80)
print("BIOCAL3D PILOT DATA")
print("=" * 80)

for category, category_files in files.items():

    print(f"\n{category}")
    print("-" * 80)

    for i, path in enumerate(category_files):
        print(f"[{i}] {path.name}")

    print(f"Total: {len(category_files)}")


# ============================================================
# LOAD
# ============================================================

def load_mesh(path):

    mesh = pv.read(path)

    print(f"\nLoaded: {path.name}")
    print(f"Points: {mesh.n_points:,}")
    print(f"Cells:  {mesh.n_cells:,}")

    return mesh


# ============================================================
# SINGLE MESH
# ============================================================

def visualize_single(category, index):

    path = files[category][index]

    mesh = load_mesh(path)

    plotter = pv.Plotter()

    plotter.add_mesh(
        mesh,
        rgb=True,
        smooth_shading=False,
        show_edges=False
    )

    plotter.add_text(
        f"{category}\n{path.name}",
        font_size=12
    )

    plotter.add_axes()

    plotter.show()


# ============================================================
# TWO MESHES SIDE-BY-SIDE
# ============================================================

def visualize_two(category1, index1, category2, index2):

    path1 = files[category1][index1]
    path2 = files[category2][index2]

    mesh1 = load_mesh(path1)
    mesh2 = load_mesh(path2)

    # Center both meshes
    mesh1 = mesh1.copy()
    mesh2 = mesh2.copy()

    mesh1.translate(-np.array(mesh1.center))
    mesh2.translate(-np.array(mesh2.center))

    # Put second mesh to the right
    width = mesh1.bounds[1] - mesh1.bounds[0]

    mesh2.translate((width * 1.5, 0, 0))

    plotter = pv.Plotter(
        shape=(1, 2)
    )

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    plotter.subplot(0, 0)

    plotter.add_mesh(
        mesh1,
        rgb=True,
        smooth_shading=False,
        show_edges=False
    )

    plotter.add_text(
        f"{category1}\n{path1.name}",
        font_size=12
    )

    plotter.add_axes()

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    plotter.subplot(0, 1)

    # We don't want the translation used above to affect
    # the camera separation, so display original mesh2
    mesh2_display = load_mesh(path2)

    mesh2_display = mesh2_display.copy()
    mesh2_display.translate(-np.array(mesh2_display.center))

    plotter.add_mesh(
        mesh2_display,
        rgb=True,
        smooth_shading=False,
        show_edges=False
    )

    plotter.add_text(
        f"{category2}\n{path2.name}",
        font_size=12
    )

    plotter.add_axes()

    plotter.link_views()

    plotter.show()


# ============================================================
# BEFORE / AFTER
# ============================================================

def visualize_before_after(index):

    before = files["Before brushing"]
    after = files["After brushing"]

    if index >= len(before) or index >= len(after):
        print("Invalid pair.")
        return

    visualize_two(
        "Before brushing",
        index,
        "After brushing",
        index
    )


# ============================================================
# MENU
# ============================================================

categories = list(FOLDERS.keys())

while True:

    print("\n" + "=" * 80)
    print("PYVISTA MENU")
    print("=" * 80)

    print("1 - Before vs After")
    print("2 - Compare any two scans")
    print("3 - View single scan")
    print("q - Quit")

    choice = input("\nChoice: ").strip()

    if choice == "1":

        for i, path in enumerate(files["Before brushing"]):
            print(f"[{i}] {path.name}")

        try:
            index = int(input("\nPair number: "))
            visualize_before_after(index)
        except ValueError:
            print("Invalid input.")

    elif choice == "2":

        print("\nCategories:")

        for i, category in enumerate(categories):
            print(f"[{i}] {category}")

        try:
            c1 = categories[int(input("\nFirst category: "))]
            i1 = int(input("First file: "))

            c2 = categories[int(input("\nSecond category: "))]
            i2 = int(input("Second file: "))

            visualize_two(c1, i1, c2, i2)

        except (ValueError, IndexError):
            print("Invalid selection.")

    elif choice == "3":

        print("\nCategories:")

        for i, category in enumerate(categories):
            print(f"[{i}] {category}")

        try:
            c = categories[int(input("\nCategory: "))]
            i = int(input("File: "))

            visualize_single(c, i)

        except (ValueError, IndexError):
            print("Invalid selection.")

    elif choice.lower() == "q":
        break