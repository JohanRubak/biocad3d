from pathlib import Path
from collections import defaultdict
import csv
import math
import re

import numpy as np
import pyvista as pv

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# DATA ROOT
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\au662213\repos\biocad3d"
    r"\data\02-09-2026-Data collected"
)


# ============================================================
# OUTPUT
# ============================================================
#
# New folder name so the previous QC results are not mixed
# with this revised version.
# ============================================================

OUTPUT_ROOT = (
    DATA_ROOT
    / "_loading_QC_v2"
)

PREVIEW_ROOT = (
    OUTPUT_ROOT
    / "previews"
)

OVERVIEW_ROOT = (
    OUTPUT_ROOT
    / "overviews"
)

MANIFEST_FILE = (
    OUTPUT_ROOT
    / "loading_manifest.csv"
)


for folder in [
    OUTPUT_ROOT,
    PREVIEW_ROOT,
    OVERVIEW_ROOT,
]:

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# OPTIONAL FILTERS
# ============================================================
#
# None = everything
#
# Examples:
#
# SCANNERS_TO_PROCESS = {"iTERO"}
# GROUPS_TO_PROCESS = {1}
# MAX_FILES = 16
#
# ============================================================

SCANNERS_TO_PROCESS = None

GROUPS_TO_PROCESS = None

MAX_FILES = None


# ============================================================
# PREVIEW SETTINGS
# ============================================================
#
# Each individual preview contains:
#
#     Isometric | XY
#     XZ        | YZ
#
# ============================================================

PREVIEW_WIDTH = 1000
PREVIEW_HEIGHT = 1000


# ============================================================
# CONTACT SHEETS
# ============================================================

CONTACT_SHEET_COLUMNS = 3
CONTACT_SHEET_ROWS = 3

CONTACT_SHEET_TILE_WIDTH = 1020
CONTACT_SHEET_TILE_HEIGHT = 1070


# ============================================================
# SAMPLE METADATA REGEX
# ============================================================

SAMPLE_PATTERN = re.compile(
    r"BCG(?P<group>\d+)"
    r"T(?P<timepoint>\d+)"
    r"-(?P<sample>\d+)",
    re.IGNORECASE,
)


GROUP_PATTERN = re.compile(
    r"^G(?P<group>\d+)",
    re.IGNORECASE,
)


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_name(text):

    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(text),
    )


# ============================================================
# EMBEDDED RGB
# ============================================================

def prepare_embedded_rgb(mesh):
    """
    Detect RGB/RGBA stored directly in the PLY.

    This is the preferred color source.

    Known examples:

        TRIOS3:
            RGB, shape=(N, 3), uint8

        TRIOS5:
            RGB, shape=(N, 3), uint8

        iTERO:
            vertex red/green/blue in raw PLY,
            loaded by PyVista as RGB.
    """

    preferred_names = [
        "RGB",
        "rgb",
        "RGBA",
        "rgba",
        "Colors",
        "colors",
        "Color",
        "color",
    ]


    # ========================================================
    # EXISTING Nx3 / Nx4 ARRAY
    # ========================================================

    for name in preferred_names:

        if name not in mesh.point_data:
            continue


        array = np.asarray(
            mesh.point_data[name]
        )


        if not (
            array.ndim == 2
            and
            array.shape[0] == mesh.n_points
            and
            array.shape[1] in (3, 4)
        ):

            continue


        # ----------------------------------------------------
        # Convert floating RGB to uint8 if needed
        # ----------------------------------------------------

        if np.issubdtype(
            array.dtype,
            np.floating,
        ):

            converted = array.copy()


            if np.nanmax(
                converted
            ) <= 1.0:

                converted *= 255.0


            converted = np.clip(
                converted,
                0,
                255,
            ).astype(
                np.uint8
            )


            mesh.point_data[
                "RGB"
            ] = converted


            return (
                mesh,
                "RGB",
            )


        return (
            mesh,
            name,
        )


    # ========================================================
    # SEPARATE R / G / B ARRAYS
    # ========================================================

    triplets = [
        (
            "red",
            "green",
            "blue",
        ),
        (
            "Red",
            "Green",
            "Blue",
        ),
        (
            "RED",
            "GREEN",
            "BLUE",
        ),
        (
            "diffuse_red",
            "diffuse_green",
            "diffuse_blue",
        ),
    ]


    point_keys = set(
        mesh.point_data.keys()
    )


    for (
        r_name,
        g_name,
        b_name,
    ) in triplets:

        if not all(
            name in point_keys
            for name in (
                r_name,
                g_name,
                b_name,
            )
        ):

            continue


        rgb = np.column_stack(
            [
                mesh.point_data[
                    r_name
                ],
                mesh.point_data[
                    g_name
                ],
                mesh.point_data[
                    b_name
                ],
            ]
        )


        if np.issubdtype(
            rgb.dtype,
            np.floating,
        ):

            if np.nanmax(
                rgb
            ) <= 1.0:

                rgb *= 255.0


        rgb = np.clip(
            rgb,
            0,
            255,
        ).astype(
            np.uint8
        )


        mesh.point_data[
            "RGB"
        ] = rgb


        return (
            mesh,
            "RGB",
        )


    return (
        mesh,
        None,
    )


# ============================================================
# UV / TEXTURE COORDINATES
# ============================================================

def activate_tcoords(mesh):
    """
    Activate UV coordinates for meshes using external JPGs.

    IMPORTANT:
    We only use these if embedded RGB is NOT available.
    """

    possible_names = [
        "TCoords",
        "tcoords",
        "Texture Coordinates",
        "TextureCoordinates",
        "texture_coordinates",
        "UV",
        "UVs",
        "uv",
        "uvs",
        "TexCoord",
        "TexCoords",
        "texcoord",
        "texcoords",
    ]


    # ========================================================
    # KNOWN NAMES ONLY
    # ========================================================
    #
    # Deliberately no generic "any Nx2 array" fallback here.
    #
    # That is safer for this dataset.
    # ========================================================

    for name in possible_names:

        if name not in mesh.point_data:
            continue


        array = np.asarray(
            mesh.point_data[name]
        )


        if not (
            array.ndim == 2
            and
            array.shape[0] == mesh.n_points
            and
            array.shape[1] == 2
        ):

            continue


        vtk_array = (
            mesh.GetPointData()
            .GetArray(name)
        )


        if vtk_array is not None:

            mesh.GetPointData().SetTCoords(
                vtk_array
            )


            return (
                mesh,
                name,
            )


    return (
        mesh,
        None,
    )


def has_active_tcoords(mesh):

    return (
        mesh.GetPointData()
        .GetTCoords()
        is not None
    )


# ============================================================
# LOAD MESH
# ============================================================

def load_mesh(path):
    """
    Load the mesh while retaining both:

        - embedded RGB
        - TCoords

    No transformation or registration is performed.
    """

    mesh = pv.read(
        path
    )


    # --------------------------------------------------------
    # Convert to surface / triangles
    # --------------------------------------------------------

    mesh = (
        mesh.extract_surface()
    )


    mesh = (
        mesh.triangulate()
    )


    mesh = (
        mesh.clean()
    )


    # --------------------------------------------------------
    # Embedded color
    # --------------------------------------------------------

    mesh, rgb_name = (
        prepare_embedded_rgb(
            mesh
        )
    )


    # --------------------------------------------------------
    # External texture coordinates
    # --------------------------------------------------------

    mesh, uv_name = (
        activate_tcoords(
            mesh
        )
    )


    return (
        mesh,
        rgb_name,
        uv_name,
    )


# ============================================================
# FIND EXTERNAL TEXTURE
# ============================================================

def find_texture(
    ply_path,
    sample_id=None,
):
    """
    Search for an external image associated with the PLY.

    Supports examples such as:

    LABscanner
    ----------
    BCG1T0-1.ply
    BCG1T0-1.jpg


    iTERO
    -----
    315628974_shell_occlusion_l.ply
    315628974_shell_occlusion_l_texture.jpg
    """

    folder = (
        ply_path.parent
    )


    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }


    images = sorted(
        [
            path

            for path in folder.iterdir()

            if (
                path.is_file()
                and
                path.suffix.lower()
                in image_extensions
            )
        ]
    )


    if len(
        images
    ) == 0:

        return None


    ply_stem = (
        ply_path.stem.lower()
    )


    # ========================================================
    # 1. EXACT STEM
    # ========================================================

    for image in images:

        if (
            image.stem.lower()
            ==
            ply_stem
        ):

            return image


    # ========================================================
    # 2. STEM + TEXTURE
    # ========================================================

    expected_names = {
        ply_stem
        + "_texture",

        ply_stem
        + "-texture",

        ply_stem
        + "texture",
    }


    for image in images:

        if (
            image.stem.lower()
            in expected_names
        ):

            return image


    # ========================================================
    # 3. SAMPLE ID
    # ========================================================

    if sample_id is not None:

        sample_lower = (
            sample_id.lower()
        )


        sample_matches = [
            image

            for image in images

            if sample_lower
            in image.stem.lower()
        ]


        if len(
            sample_matches
        ) == 1:

            return (
                sample_matches[0]
            )


        texture_matches = [
            image

            for image in sample_matches

            if "texture"
            in image.stem.lower()
        ]


        if len(
            texture_matches
        ) == 1:

            return (
                texture_matches[0]
            )


    # ========================================================
    # 4. PARTIAL PLY STEM
    # ========================================================

    partial_matches = [
        image

        for image in images

        if (
            ply_stem
            in image.stem.lower()
            or
            image.stem.lower()
            in ply_stem
        )
    ]


    if len(
        partial_matches
    ) == 1:

        return (
            partial_matches[0]
        )


    # ========================================================
    # 5. DEDICATED SAMPLE FOLDER
    # ========================================================

    if len(
        images
    ) == 1:

        return (
            images[0]
        )


    return None


# ============================================================
# LOAD EXTERNAL TEXTURE
# ============================================================

def load_texture(
    texture_path,
):

    if texture_path is None:

        return None


    try:

        return pv.read_texture(
            texture_path
        )


    except Exception as exc:

        print(
            f"WARNING: could not load texture:\n"
            f"{texture_path}\n"
            f"{exc}"
        )


        return None


# ============================================================
# COLOR MODE
# ============================================================

def determine_color_mode(
    mesh,
    rgb_name,
    texture,
):
    """
    IMPORTANT PRIORITY:

        1. embedded RGB
        2. external texture
        3. no color
    """

    if (
        rgb_name is not None
        and
        rgb_name
        in mesh.point_data
    ):

        return (
            "embedded_rgb"
        )


    if (
        texture is not None
        and
        has_active_tcoords(
            mesh
        )
    ):

        return (
            "external_texture"
        )


    return (
        "NO_COLOR"
    )


# ============================================================
# ADD COLORED MESH TO PLOTTER
# ============================================================

def add_scan_mesh(
    plotter,
    mesh,
    rgb_name,
    texture,
):
    """
    Visualization priority:

        1. embedded RGB
        2. external texture
        3. neutral gray
    """


    # ========================================================
    # EMBEDDED RGB
    # ========================================================

    if (
        rgb_name is not None
        and
        rgb_name
        in mesh.point_data
    ):

        plotter.add_mesh(
            mesh,

            scalars=rgb_name,

            rgb=True,

            preference="point",

            lighting=False,
        )


        return


    # ========================================================
    # EXTERNAL JPG
    # ========================================================

    if (
        texture is not None
        and
        has_active_tcoords(
            mesh
        )
    ):

        plotter.add_mesh(
            mesh,

            texture=texture,

            lighting=False,
        )


        return


    # ========================================================
    # FALLBACK
    # ========================================================

    plotter.add_mesh(
        mesh,

        color="lightgray",

        smooth_shading=True,
    )


# ============================================================
# EXTRACT SAMPLE ID
# ============================================================

def extract_sample_match(
    ply_path,
):

    # --------------------------------------------------------
    # Filename
    # --------------------------------------------------------

    match = SAMPLE_PATTERN.search(
        ply_path.stem
    )


    if match is not None:

        return match


    # --------------------------------------------------------
    # Parent folder
    # --------------------------------------------------------

    for parent in ply_path.parents:

        if parent == DATA_ROOT:
            break


        match = SAMPLE_PATTERN.search(
            parent.name
        )


        if match is not None:

            return match


    return None


# ============================================================
# EXTRACT SCANNER NAME
# ============================================================

def extract_scanner(
    ply_path,
):

    relative = (
        ply_path.relative_to(
            DATA_ROOT
        )
    )


    if len(
        relative.parts
    ) < 2:

        return (
            "UNKNOWN"
        )


    return (
        relative.parts[0]
    )


# ============================================================
# EXTRACT GROUP FOLDER
# ============================================================

def extract_group_folder(
    ply_path,
):

    relative = (
        ply_path.relative_to(
            DATA_ROOT
        )
    )


    for part in relative.parts:

        match = (
            GROUP_PATTERN.match(
                part
            )
        )


        if match is not None:

            return (
                int(
                    match.group(
                        "group"
                    )
                ),

                part,
            )


    return (
        None,
        None,
    )


# ============================================================
# DISCOVER DATASET
# ============================================================

def discover_dataset():

    excluded_top_level = {
        "_loading_QC",
        "_loading_QC_v2",
        "_processed_ROI",
    }


    ply_files = []


    for path in DATA_ROOT.rglob("*"):

        if not path.is_file():
            continue


        if (
            path.suffix.lower()
            != ".ply"
        ):

            continue


        relative = (
            path.relative_to(
                DATA_ROOT
            )
        )


        if (
            len(relative.parts) > 0
            and
            relative.parts[0]
            in excluded_top_level
        ):

            continue


        ply_files.append(
            path
        )


    records = []


    for ply_path in ply_files:

        scanner = (
            extract_scanner(
                ply_path
            )
        )


        (
            folder_group,
            group_folder,
        ) = (
            extract_group_folder(
                ply_path
            )
        )


        match = (
            extract_sample_match(
                ply_path
            )
        )


        # ====================================================
        # PARSE SAMPLE METADATA
        # ====================================================

        if match is not None:

            group = int(
                match.group(
                    "group"
                )
            )


            timepoint_number = int(
                match.group(
                    "timepoint"
                )
            )


            sample_number = int(
                match.group(
                    "sample"
                )
            )


            timepoint = (
                f"T{timepoint_number}"
            )


            sample_id = (
                f"BCG{group}"
                f"T{timepoint_number}"
                f"-{sample_number}"
            )


        else:

            group = (
                folder_group
            )


            timepoint_number = (
                None
            )


            sample_number = (
                None
            )


            timepoint = (
                None
            )


            sample_id = (
                ply_path.stem
            )


        # ====================================================
        # EXTERNAL TEXTURE
        # ====================================================

        texture_path = (
            find_texture(
                ply_path,
                sample_id=sample_id,
            )
        )


        records.append(
            {
                "scanner": scanner,

                "group": group,

                "group_folder": (
                    group_folder
                ),

                "timepoint": (
                    timepoint
                ),

                "timepoint_number": (
                    timepoint_number
                ),

                "sample_number": (
                    sample_number
                ),

                "sample_id": (
                    sample_id
                ),

                "ply_path": (
                    ply_path
                ),

                "texture_path": (
                    texture_path
                ),
            }
        )


    # ========================================================
    # SORT
    # ========================================================

    records.sort(
        key=lambda record: (

            record[
                "scanner"
            ].lower(),

            (
                record[
                    "group"
                ]
                if
                record[
                    "group"
                ]
                is not None
                else 999
            ),

            (
                record[
                    "timepoint_number"
                ]
                if
                record[
                    "timepoint_number"
                ]
                is not None
                else 999
            ),

            (
                record[
                    "sample_number"
                ]
                if
                record[
                    "sample_number"
                ]
                is not None
                else 999
            ),
        )
    )


    return (
        records
    )


# ============================================================
# SAVE FOUR-VIEW PREVIEW
# ============================================================

def save_loading_preview(
    mesh,
    rgb_name,
    texture,
    output_file,
):
    """
    Create:

        Isometric | XY
        XZ        | YZ
    """

    plotter = pv.Plotter(
        shape=(
            2,
            2,
        ),

        off_screen=True,

        window_size=(
            PREVIEW_WIDTH,
            PREVIEW_HEIGHT,
        ),
    )


    # ========================================================
    # ISOMETRIC
    # ========================================================

    plotter.subplot(
        0,
        0,
    )


    plotter.set_background(
        "white"
    )


    add_scan_mesh(
        plotter,
        mesh,
        rgb_name,
        texture,
    )


    plotter.add_text(
        "Isometric",
        font_size=8,
    )


    plotter.view_isometric()

    plotter.reset_camera()


    # ========================================================
    # XY
    # ========================================================

    plotter.subplot(
        0,
        1,
    )


    plotter.set_background(
        "white"
    )


    add_scan_mesh(
        plotter,
        mesh,
        rgb_name,
        texture,
    )


    plotter.add_text(
        "XY",
        font_size=8,
    )


    plotter.view_xy()

    plotter.camera.parallel_projection = True

    plotter.reset_camera()


    # ========================================================
    # XZ
    # ========================================================

    plotter.subplot(
        1,
        0,
    )


    plotter.set_background(
        "white"
    )


    add_scan_mesh(
        plotter,
        mesh,
        rgb_name,
        texture,
    )


    plotter.add_text(
        "XZ",
        font_size=8,
    )


    plotter.view_xz()

    plotter.camera.parallel_projection = True

    plotter.reset_camera()


    # ========================================================
    # YZ
    # ========================================================

    plotter.subplot(
        1,
        1,
    )


    plotter.set_background(
        "white"
    )


    add_scan_mesh(
        plotter,
        mesh,
        rgb_name,
        texture,
    )


    plotter.add_text(
        "YZ",
        font_size=8,
    )


    plotter.view_yz()

    plotter.camera.parallel_projection = True

    plotter.reset_camera()


    # ========================================================
    # SAVE
    # ========================================================

    plotter.screenshot(
        str(
            output_file
        )
    )


    plotter.close()


# ============================================================
# CONTACT SHEETS
# ============================================================

def make_contact_sheets(
    records,
    output_directory,
    prefix,
):

    if len(
        records
    ) == 0:

        return


    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    per_page = (
        CONTACT_SHEET_COLUMNS
        *
        CONTACT_SHEET_ROWS
    )


    number_of_pages = (
        math.ceil(
            len(records)
            /
            per_page
        )
    )


    font = (
        ImageFont.load_default()
    )


    for page in range(
        number_of_pages
    ):

        start = (
            page
            *
            per_page
        )


        end = min(
            start
            +
            per_page,

            len(records),
        )


        page_records = (
            records[
                start:end
            ]
        )


        sheet_width = (
            CONTACT_SHEET_COLUMNS
            *
            CONTACT_SHEET_TILE_WIDTH
        )


        sheet_height = (
            CONTACT_SHEET_ROWS
            *
            CONTACT_SHEET_TILE_HEIGHT
        )


        sheet = Image.new(
            "RGB",

            (
                sheet_width,
                sheet_height,
            ),

            "white",
        )


        draw = (
            ImageDraw.Draw(
                sheet
            )
        )


        # ====================================================
        # EACH TILE
        # ====================================================

        for i, record in enumerate(
            page_records
        ):

            row = (
                i
                //
                CONTACT_SHEET_COLUMNS
            )


            col = (
                i
                %
                CONTACT_SHEET_COLUMNS
            )


            x0 = (
                col
                *
                CONTACT_SHEET_TILE_WIDTH
            )


            y0 = (
                row
                *
                CONTACT_SHEET_TILE_HEIGHT
            )


            preview_path = (
                record.get(
                    "preview"
                )
            )


            # ------------------------------------------------
            # Preview image
            # ------------------------------------------------

            if (
                preview_path
                and
                Path(
                    preview_path
                ).exists()
            ):

                image = (
                    Image.open(
                        preview_path
                    )
                    .convert(
                        "RGB"
                    )
                )


                image.thumbnail(
                    (
                        CONTACT_SHEET_TILE_WIDTH
                        - 20,

                        CONTACT_SHEET_TILE_HEIGHT
                        - 78,
                    )
                )


                image_x = (
                    x0
                    +
                    (
                        CONTACT_SHEET_TILE_WIDTH
                        -
                        image.width
                    )
                    //
                    2
                )


                image_y = (
                    y0
                    + 5
                )


                sheet.paste(
                    image,

                    (
                        image_x,
                        image_y,
                    ),
                )


            # ------------------------------------------------
            # Information
            # ------------------------------------------------

            label = (
                f"{record['scanner']} | "
                f"{record['sample_id']}\n"
                f"{record['status']} | "
                f"{record.get('color_mode', '')}"
            )


            draw.text(
                (
                    x0 + 8,

                    y0
                    +
                    CONTACT_SHEET_TILE_HEIGHT
                    - 65,
                ),

                label,

                fill="black",

                font=font,
            )


            # ------------------------------------------------
            # Border
            # ------------------------------------------------

            draw.rectangle(
                [
                    x0,
                    y0,

                    x0
                    +
                    CONTACT_SHEET_TILE_WIDTH
                    - 1,

                    y0
                    +
                    CONTACT_SHEET_TILE_HEIGHT
                    - 1,
                ],

                outline="gray",

                width=1,
            )


        # ====================================================
        # SAVE PAGE
        # ====================================================

        output_file = (
            output_directory
            /
            f"{prefix}_{page + 1:03d}.jpg"
        )


        sheet.save(
            output_file,
            quality=92,
        )


        print(
            f"Created overview: "
            f"{output_file}"
        )


# ============================================================
# DISCOVER DATASET
# ============================================================

dataset = (
    discover_dataset()
)


print(
    "\n" + "=" * 80
)

print(
    "DATASET DISCOVERY"
)

print(
    "=" * 80
)


print(
    f"\nFound "
    f"{len(dataset)} PLY files."
)


# ============================================================
# DISCOVERY SUMMARY
# ============================================================

scanner_counts = (
    defaultdict(
        int
    )
)


group_counts = (
    defaultdict(
        int
    )
)


for record in dataset:

    scanner_counts[
        record[
            "scanner"
        ]
    ] += 1


    group_counts[
        (
            record[
                "scanner"
            ],

            record[
                "group"
            ],
        )
    ] += 1


print(
    "\nScanner counts:"
)


for scanner in sorted(
    scanner_counts
):

    print(
        f"  {scanner:<20} "
        f"{scanner_counts[scanner]:>4}"
    )


print(
    "\nScanner/group counts:"
)


for (
    scanner,
    group,
), count in sorted(
    group_counts.items(),

    key=lambda item: (
        item[0][0].lower(),

        (
            item[0][1]
            if item[0][1]
            is not None
            else 999
        ),
    ),
):

    print(
        f"  {scanner:<20} "
        f"G{group}: "
        f"{count}"
    )


# ============================================================
# OPTIONAL FILTERING
# ============================================================

processing_dataset = []


for record in dataset:

    scanner = (
        record[
            "scanner"
        ]
    )


    group = (
        record[
            "group"
        ]
    )


    if (
        SCANNERS_TO_PROCESS
        is not None
        and
        scanner
        not in
        SCANNERS_TO_PROCESS
    ):

        continue


    if (
        GROUPS_TO_PROCESS
        is not None
        and
        group
        not in
        GROUPS_TO_PROCESS
    ):

        continue


    processing_dataset.append(
        record
    )


if MAX_FILES is not None:

    processing_dataset = (
        processing_dataset[
            :MAX_FILES
        ]
    )


print(
    f"\nGenerating QC for "
    f"{len(processing_dataset)} scans."
)


# ============================================================
# PROCESS ALL MODELS
# ============================================================

manifest_rows = []


for index, record in enumerate(
    processing_dataset,
    start=1,
):

    scanner = (
        record[
            "scanner"
        ]
    )


    group = (
        record[
            "group"
        ]
    )


    sample_id = (
        record[
            "sample_id"
        ]
    )


    ply_path = (
        record[
            "ply_path"
        ]
    )


    texture_path = (
        record[
            "texture_path"
        ]
    )


    print(
        "\n" + "=" * 80
    )

    print(
        f"[{index}/"
        f"{len(processing_dataset)}] "
        f"{scanner} | "
        f"{sample_id}"
    )

    print(
        "=" * 80
    )


    try:

        # ====================================================
        # LOAD MODEL
        # ====================================================

        (
            mesh,
            rgb_name,
            uv_name,
        ) = load_mesh(
            ply_path
        )


        # ====================================================
        # LOAD EXTERNAL TEXTURE
        # ====================================================

        texture = (
            load_texture(
                texture_path
            )
        )


        # ====================================================
        # COLOR SOURCE
        # ====================================================

        color_mode = (
            determine_color_mode(
                mesh,
                rgb_name,
                texture,
            )
        )


        # ====================================================
        # STATUS
        # ====================================================

        status = (
            "OK"
        )


        if (
            color_mode
            ==
            "NO_COLOR"
        ):

            status = (
                "CHECK"
            )


        # ====================================================
        # PREVIEW DIRECTORY
        # ====================================================

        group_name = (
            f"G{group}"
            if group
            is not None
            else "G_UNKNOWN"
        )


        preview_directory = (
            PREVIEW_ROOT
            / scanner
            / group_name
        )


        preview_directory.mkdir(
            parents=True,
            exist_ok=True,
        )


        preview_file = (
            preview_directory
            /
            (
                safe_name(
                    sample_id
                )
                +
                "_loading_QC.png"
            )
        )


        # ====================================================
        # RENDER
        # ====================================================

        save_loading_preview(
            mesh,
            rgb_name,
            texture,
            preview_file,
        )


        # ====================================================
        # RGB INFORMATION
        # ====================================================

        rgb_shape = ""
        rgb_dtype = ""


        if (
            rgb_name is not None
            and
            rgb_name
            in mesh.point_data
        ):

            rgb_array = np.asarray(
                mesh.point_data[
                    rgb_name
                ]
            )


            rgb_shape = str(
                rgb_array.shape
            )


            rgb_dtype = str(
                rgb_array.dtype
            )


        # ====================================================
        # CONSOLE
        # ====================================================

        print(
            f"Points:       "
            f"{mesh.n_points:,}"
        )


        print(
            f"Cells:        "
            f"{mesh.n_cells:,}"
        )


        print(
            f"Color mode:   "
            f"{color_mode}"
        )


        print(
            f"Embedded RGB: "
            f"{rgb_name}"
        )


        print(
            f"UV array:     "
            f"{uv_name}"
        )


        print(
            f"JPG:          "
            f"{texture_path}"
        )


        if (
            rgb_name is not None
        ):

            print(
                f"RGB shape:    "
                f"{rgb_shape}"
            )


            print(
                f"RGB dtype:    "
                f"{rgb_dtype}"
            )


        # ====================================================
        # MANIFEST
        # ====================================================

        manifest_rows.append(
            {
                "scanner": (
                    scanner
                ),

                "group": (
                    group
                ),

                "group_folder": (
                    record[
                        "group_folder"
                    ]
                ),

                "timepoint": (
                    record[
                        "timepoint"
                    ]
                ),

                "sample_number": (
                    record[
                        "sample_number"
                    ]
                ),

                "sample_id": (
                    sample_id
                ),

                "status": (
                    status
                ),

                "color_mode": (
                    color_mode
                ),

                "points": (
                    mesh.n_points
                ),

                "cells": (
                    mesh.n_cells
                ),

                "rgb_array": (
                    rgb_name
                    if rgb_name
                    is not None
                    else ""
                ),

                "rgb_shape": (
                    rgb_shape
                ),

                "rgb_dtype": (
                    rgb_dtype
                ),

                "uv_array": (
                    uv_name
                    if uv_name
                    is not None
                    else ""
                ),

                "external_texture_found": (
                    texture_path
                    is not None
                ),

                "external_texture_loaded": (
                    texture
                    is not None
                ),

                "source_ply": str(
                    ply_path
                ),

                "source_texture": (
                    str(
                        texture_path
                    )
                    if texture_path
                    is not None
                    else ""
                ),

                "preview": str(
                    preview_file
                ),
            }
        )


    # ========================================================
    # FAILED
    # ========================================================

    except Exception as exc:

        print(
            f"FAILED: "
            f"{exc}"
        )


        manifest_rows.append(
            {
                "scanner": (
                    scanner
                ),

                "group": (
                    group
                ),

                "group_folder": (
                    record[
                        "group_folder"
                    ]
                ),

                "timepoint": (
                    record[
                        "timepoint"
                    ]
                ),

                "sample_number": (
                    record[
                        "sample_number"
                    ]
                ),

                "sample_id": (
                    sample_id
                ),

                "status": (
                    "FAILED"
                ),

                "color_mode": "",

                "source_ply": str(
                    ply_path
                ),

                "source_texture": (
                    str(
                        texture_path
                    )
                    if texture_path
                    is not None
                    else ""
                ),

                "preview": "",

                "error": str(
                    exc
                ),
            }
        )


# ============================================================
# SAVE MANIFEST CSV
# ============================================================

if len(
    manifest_rows
) > 0:

    fields = sorted(
        {
            key

            for row in manifest_rows

            for key in row.keys()
        }
    )


    with open(
        MANIFEST_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )


        writer.writeheader()

        writer.writerows(
            manifest_rows
        )


# ============================================================
# OVERVIEW: ALL
# ============================================================

make_contact_sheets(
    manifest_rows,

    OVERVIEW_ROOT
    / "ALL",

    "all_loading_QC",
)


# ============================================================
# OVERVIEW: PER SCANNER
# ============================================================

by_scanner = (
    defaultdict(
        list
    )
)


for row in manifest_rows:

    by_scanner[
        row[
            "scanner"
        ]
    ].append(
        row
    )


for scanner, rows in (
    by_scanner.items()
):

    make_contact_sheets(
        rows,

        OVERVIEW_ROOT
        / scanner
        / "ALL",

        (
            safe_name(
                scanner
            )
            +
            "_loading_QC"
        ),
    )


# ============================================================
# OVERVIEW: SCANNER + GROUP
# ============================================================

by_scanner_group = (
    defaultdict(
        list
    )
)


for row in manifest_rows:

    group = (
        row.get(
            "group"
        )
    )


    group_name = (
        f"G{group}"
        if group
        is not None
        else "G_UNKNOWN"
    )


    by_scanner_group[
        (
            row[
                "scanner"
            ],

            group_name,
        )
    ].append(
        row
    )


for (
    scanner,
    group_name,
), rows in (
    by_scanner_group.items()
):

    make_contact_sheets(
        rows,

        OVERVIEW_ROOT
        / scanner
        / group_name,

        (
            safe_name(
                scanner
            )
            +
            "_"
            +
            group_name
            +
            "_loading_QC"
        ),
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

n_ok = sum(
    row.get(
        "status"
    )
    ==
    "OK"

    for row in manifest_rows
)


n_check = sum(
    row.get(
        "status"
    )
    ==
    "CHECK"

    for row in manifest_rows
)


n_failed = sum(
    row.get(
        "status"
    )
    ==
    "FAILED"

    for row in manifest_rows
)


color_counts = (
    defaultdict(
        int
    )
)


for row in manifest_rows:

    color_counts[
        row.get(
            "color_mode",
            ""
        )
    ] += 1


print(
    "\n" + "=" * 80
)

print(
    "LOADING QC FINISHED"
)

print(
    "=" * 80
)


print(
    f"\nTotal:   "
    f"{len(manifest_rows)}"
)


print(
    f"OK:      "
    f"{n_ok}"
)


print(
    f"CHECK:   "
    f"{n_check}"
)


print(
    f"FAILED:  "
    f"{n_failed}"
)


print(
    "\nColor modes:"
)


for (
    mode,
    count,
) in sorted(
    color_counts.items()
):

    print(
        f"  {mode:<20} "
        f"{count}"
    )


print(
    f"\nManifest:\n"
    f"{MANIFEST_FILE}"
)


print(
    f"\nIndividual previews:\n"
    f"{PREVIEW_ROOT}"
)


print(
    f"\nOverview files:\n"
    f"{OVERVIEW_ROOT}"
)