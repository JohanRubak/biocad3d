from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image


# ============================================================
# CHANGE THESE TWO PATHS
# ============================================================


BAD_G1_PLY = Path(
    r"C:\Users\au662213\repos\biocad3d\data\02-09-2026-Data collected\iTERO\G1-OK\BCG1T0-1\315552846_shell_occlusion_l.ply"
)

BAD_G1_JPG = Path(
    r"C:\Users\au662213\repos\biocad3d\data\02-09-2026-Data collected\iTERO\G1-OK\BCG1T0-1\315552846_shell_occlusion_l_texture.jpg"
)


GOOD_G2_PLY = Path(
    r"C:\Users\au662213\repos\biocad3d\data\02-09-2026-Data collected\iTERO\G2\BCG2T0-1\315628974_shell_occlusion_l.ply"
)

GOOD_G2_JPG = Path(
    r"C:\Users\au662213\repos\biocad3d\data\02-09-2026-Data collected\iTERO\G2\BCG2T0-1\315628974_shell_occlusion_l_texture.jpg"
)


# ============================================================
# INSPECTION
# ============================================================

def inspect_scan(label, ply_path, jpg_path):

    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)

    print(f"\nPLY:\n{ply_path}")
    print(f"\nJPG:\n{jpg_path}")

    mesh = pv.read(ply_path)

    print("\nGEOMETRY")
    print("-" * 80)

    print(f"Points: {mesh.n_points:,}")
    print(f"Cells:  {mesh.n_cells:,}")

    print("\nBounds:")
    print(mesh.bounds)


    # ========================================================
    # POINT DATA
    # ========================================================

    print("\nPOINT DATA")
    print("-" * 80)

    if len(mesh.point_data.keys()) == 0:

        print("None")

    else:

        for name in mesh.point_data.keys():

            arr = np.asarray(
                mesh.point_data[name]
            )

            print(
                f"{name}: "
                f"shape={arr.shape}, "
                f"dtype={arr.dtype}"
            )

            if (
                arr.ndim == 2
                and
                arr.shape[1] == 2
            ):

                print(
                    f"    min = "
                    f"{np.nanmin(arr, axis=0)}"
                )

                print(
                    f"    max = "
                    f"{np.nanmax(arr, axis=0)}"
                )

                print(
                    f"    mean = "
                    f"{np.nanmean(arr, axis=0)}"
                )

                print(
                    f"    first 10:"
                )

                print(
                    arr[:10]
                )


    # ========================================================
    # CELL DATA
    # ========================================================

    print("\nCELL DATA")
    print("-" * 80)

    if len(mesh.cell_data.keys()) == 0:

        print("None")

    else:

        for name in mesh.cell_data.keys():

            arr = np.asarray(
                mesh.cell_data[name]
            )

            print(
                f"{name}: "
                f"shape={arr.shape}, "
                f"dtype={arr.dtype}"
            )


    # ========================================================
    # FIELD DATA
    # ========================================================

    print("\nFIELD DATA")
    print("-" * 80)

    if len(mesh.field_data.keys()) == 0:

        print("None")

    else:

        for name in mesh.field_data.keys():

            arr = np.asarray(
                mesh.field_data[name]
            )

            print(
                f"{name}: "
                f"shape={arr.shape}, "
                f"dtype={arr.dtype}"
            )


    # ========================================================
    # ACTIVE TCOORDS
    # ========================================================

    print("\nVTK ACTIVE TCOORDS")
    print("-" * 80)

    vtk_tcoords = (
        mesh.GetPointData()
        .GetTCoords()
    )

    if vtk_tcoords is None:

        print(
            "No active texture coordinates."
        )

    else:

        tcoords = np.array(
            [
                vtk_tcoords.GetTuple2(i)
                for i in range(
                    min(
                        vtk_tcoords.GetNumberOfTuples(),
                        10,
                    )
                )
            ]
        )

        print(
            f"Name: "
            f"{vtk_tcoords.GetName()}"
        )

        print(
            f"Number of tuples: "
            f"{vtk_tcoords.GetNumberOfTuples()}"
        )

        print(
            "First 10:"
        )

        print(
            tcoords
        )


    # ========================================================
    # JPG
    # ========================================================

    print("\nTEXTURE IMAGE")
    print("-" * 80)

    if not jpg_path.exists():

        print(
            "JPG NOT FOUND"
        )

    else:

        with Image.open(
            jpg_path
        ) as image:

            print(
                f"Image size: "
                f"{image.size}"
            )

            print(
                f"Mode: "
                f"{image.mode}"
            )


    # ========================================================
    # RAW PLY HEADER
    # ========================================================

    print("\nPLY HEADER")
    print("-" * 80)

    try:

        with open(
            ply_path,
            "rb",
        ) as file:

            header_bytes = b""

            while True:

                line = file.readline()

                if not line:
                    break

                header_bytes += line

                if line.strip() == b"end_header":
                    break


        header = header_bytes.decode(
            "ascii",
            errors="replace",
        )

        print(
            header
        )

    except Exception as exc:

        print(
            f"Could not read PLY header: "
            f"{exc}"
        )


# ============================================================
# RUN
# ============================================================

inspect_scan(
    "BAD iTERO G1",
    BAD_G1_PLY,
    BAD_G1_JPG,
)

inspect_scan(
    "GOOD iTERO G2",
    GOOD_G2_PLY,
    GOOD_G2_JPG,
)