from pathlib import Path
from collections import defaultdict
from itertools import combinations
import re
import warnings

import numpy as np
import pandas as pd
import pyvista as pv

from PIL import Image

import matplotlib.pyplot as plt

from skimage.color import rgb2lab
from skimage.color import deltaE_ciede2000

from vtk.util.numpy_support import vtk_to_numpy


# ============================================================
# DATA
# ============================================================

DATA_ROOT = Path(
    r"C:\Users\au662213\repos\biocad3d\data"
    r"\02-09-2026-Data collected"
)


PROCESSED_ROOT = (
    DATA_ROOT
    / "_processed_ROI"
)


OUTPUT_ROOT = (
    DATA_ROOT
    / "_raw_color_analysis"
)


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# OUTPUT FILES
# ============================================================

ROI_SUMMARY_FILE = (
    OUTPUT_ROOT
    / "roi_color_summary.csv"
)


SCANNER_SUMMARY_FILE = (
    OUTPUT_ROOT
    / "scanner_color_summary.csv"
)


PAIRED_FILE = (
    OUTPUT_ROOT
    / "paired_scanner_differences.csv"
)


# ============================================================
# SCANNER ORDER
# ============================================================

PREFERRED_SCANNER_ORDER = [
    "LABscanner",
    "iTERO",
    "TRIOS3",
    "TRIOS5",
]


# ============================================================
# EXTERNAL TEXTURE SETTINGS
# ============================================================
#
# VTK texture coordinates conventionally use:
#
#     v = 0 at bottom
#
# whereas PIL / NumPy images use:
#
#     row = 0 at top
#
# Therefore V is normally flipped when directly reading
# pixels from the JPG.
#
# If you later find a LABscanner texture is vertically
# inverted during direct texture sampling, change to False.
# ============================================================

FLIP_TEXTURE_V = True


# ============================================================
# SAMPLE REGEX
# ============================================================

SAMPLE_PATTERN = re.compile(
    r"BCG(?P<group>\d+)"
    r"T(?P<timepoint>\d+)"
    r"-(?P<sample>\d+)",
    re.IGNORECASE,
)


# ============================================================
# HELPERS
# ============================================================

def npz_scalar_to_python(value):
    """
    Convert scalar values loaded from NPZ into normal Python
    strings/numbers.
    """

    array = np.asarray(value)


    if array.ndim == 0:

        return array.item()


    if array.size == 1:

        return array.reshape(
            -1
        )[0].item()


    return value


# ============================================================
# LOAD GEOMETRY METADATA
# ============================================================

def load_geometry_metadata(
    geometry_file,
):

    data = np.load(
        geometry_file,
        allow_pickle=True,
    )


    metadata = {}


    for key in data.files:

        value = data[
            key
        ]


        if (
            np.asarray(
                value
            ).size
            == 1
        ):

            try:

                value = (
                    npz_scalar_to_python(
                        value
                    )
                )

            except Exception:
                pass


        metadata[
            key
        ] = value


    return metadata


# ============================================================
# FIND RGB ARRAY
# ============================================================

def find_rgb_array(
    mesh,
):

    possible_names = [
        "RGB",
        "rgb",
        "RGBA",
        "rgba",
        "Colors",
        "colors",
        "Color",
        "color",
    ]


    for name in possible_names:

        if name not in mesh.point_data:
            continue


        rgb = np.asarray(
            mesh.point_data[
                name
            ]
        )


        if (
            rgb.ndim == 2
            and
            rgb.shape[0]
            ==
            mesh.n_points
            and
            rgb.shape[1]
            in (
                3,
                4,
            )
        ):

            rgb = (
                rgb[
                    :,
                    :3
                ]
            )


            if np.issubdtype(
                rgb.dtype,
                np.floating,
            ):

                rgb = (
                    rgb.copy()
                )


                if np.nanmax(
                    rgb
                ) <= 1.0:

                    rgb *= 255.0


            rgb = np.clip(
                rgb,
                0,
                255,
            )


            return (
                rgb.astype(
                    np.uint8
                ),
                name,
            )


    return (
        None,
        None,
    )


# ============================================================
# FIND UV/TCOORDS
# ============================================================

def find_tcoords(
    mesh,
):

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


    for name in possible_names:

        if name not in mesh.point_data:
            continue


        uv = np.asarray(
            mesh.point_data[
                name
            ]
        )


        if (
            uv.ndim == 2
            and
            uv.shape
            ==
            (
                mesh.n_points,
                2,
            )
        ):

            return (
                uv.astype(
                    np.float64
                ),
                name,
            )


    # --------------------------------------------------------
    # Check active VTK TCoords
    # --------------------------------------------------------

    vtk_tcoords = (
        mesh.GetPointData()
        .GetTCoords()
    )


    if vtk_tcoords is not None:

        uv = vtk_to_numpy(
            vtk_tcoords
        )


        if (
            uv.ndim == 2
            and
            uv.shape[1] == 2
        ):

            return (
                uv.astype(
                    np.float64
                ),
                (
                    vtk_tcoords.GetName()
                    or
                    "VTK_TCoords"
                ),
            )


    return (
        None,
        None,
    )


# ============================================================
# BILINEAR IMAGE SAMPLING
# ============================================================

def bilinear_sample_rgb(
    image,
    uv,
    flip_v=True,
):
    """
    Sample RGB image at UV positions.

    Parameters
    ----------
    image
        H x W x 3 uint8 image

    uv
        N x 2 array between approximately 0 and 1

    Returns
    -------
    N x 3 uint8
    """

    height, width, _ = (
        image.shape
    )


    u = np.clip(
        uv[:, 0],
        0.0,
        1.0,
    )


    v = np.clip(
        uv[:, 1],
        0.0,
        1.0,
    )


    if flip_v:

        v = (
            1.0
            - v
        )


    x = (
        u
        *
        (
            width
            - 1
        )
    )


    y = (
        v
        *
        (
            height
            - 1
        )
    )


    x0 = np.floor(
        x
    ).astype(
        int
    )

    y0 = np.floor(
        y
    ).astype(
        int
    )


    x1 = np.minimum(
        x0 + 1,
        width - 1,
    )

    y1 = np.minimum(
        y0 + 1,
        height - 1,
    )


    wx = (
        x - x0
    )[:, None]


    wy = (
        y - y0
    )[:, None]


    image_float = (
        image.astype(
            np.float64
        )
    )


    c00 = image_float[
        y0,
        x0,
    ]

    c10 = image_float[
        y0,
        x1,
    ]

    c01 = image_float[
        y1,
        x0,
    ]

    c11 = image_float[
        y1,
        x1,
    ]


    upper = (
        c00
        *
        (
            1.0
            - wx
        )
        +
        c10
        *
        wx
    )


    lower = (
        c01
        *
        (
            1.0
            - wx
        )
        +
        c11
        *
        wx
    )


    sampled = (
        upper
        *
        (
            1.0
            - wy
        )
        +
        lower
        *
        wy
    )


    return np.clip(
        sampled,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# GET RAW ROI COLORS
# ============================================================

def get_roi_rgb(
    mesh,
    metadata,
):
    """
    Color source priority:

    1. embedded vertex RGB
    2. original external texture + VTP UV coordinates
    """


    # ========================================================
    # EMBEDDED RGB
    # ========================================================

    rgb, rgb_name = (
        find_rgb_array(
            mesh
        )
    )


    if rgb is not None:

        return (
            rgb,
            "embedded_rgb",
            rgb_name,
        )


    # ========================================================
    # EXTERNAL TEXTURE
    # ========================================================

    uv, uv_name = (
        find_tcoords(
            mesh
        )
    )


    if uv is None:

        raise RuntimeError(
            "ROI has neither embedded RGB nor valid UV coordinates."
        )


    source_texture = str(
        metadata.get(
            "source_texture",
            "",
        )
    ).strip()


    if not source_texture:

        raise RuntimeError(
            "UV coordinates exist but geometry.npz "
            "contains no source_texture."
        )


    texture_path = Path(
        source_texture
    )


    if not texture_path.exists():

        raise FileNotFoundError(
            f"Source texture does not exist:\n"
            f"{texture_path}"
        )


    image = np.asarray(
        Image.open(
            texture_path
        ).convert(
            "RGB"
        )
    )


    rgb = bilinear_sample_rgb(
        image,
        uv,
        flip_v=(
            FLIP_TEXTURE_V
        ),
    )


    return (
        rgb,
        "external_texture",
        uv_name,
    )


# ============================================================
# SAMPLE IDENTIFIERS
# ============================================================

def parse_sample_id(
    sample_id,
):

    match = SAMPLE_PATTERN.search(
        str(
            sample_id
        )
    )


    if match is None:

        return {
            "group": None,
            "timepoint_number": None,
            "sample_number": None,
            "physical_sample_id": None,
        }


    group = int(
        match.group(
            "group"
        )
    )


    timepoint = int(
        match.group(
            "timepoint"
        )
    )


    sample_number = int(
        match.group(
            "sample"
        )
    )


    return {
        "group": group,

        "timepoint_number": (
            timepoint
        ),

        "sample_number": (
            sample_number
        ),

        # Same physical disk independent of cleaning stage.
        "physical_sample_id": (
            f"BCG{group}-"
            f"{sample_number}"
        ),
    }


# ============================================================
# COLOR STATISTICS FOR ONE ROI
# ============================================================

def calculate_color_statistics(
    rgb_uint8,
):
    """
    Calculate statistics within ONE scan.

    This avoids allowing dense scanners to dominate the
    analysis simply because they contain more mesh vertices.
    """

    rgb = (
        rgb_uint8.astype(
            np.float64
        )
        /
        255.0
    )


    # --------------------------------------------------------
    # CIELAB
    # --------------------------------------------------------

    lab = rgb2lab(
        rgb.reshape(
            -1,
            1,
            3,
        )
    ).reshape(
        -1,
        3,
    )


    L = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]


    chroma = np.sqrt(
        a**2
        +
        b**2
    )


    hue = np.degrees(
        np.arctan2(
            b,
            a,
        )
    )


    hue = (
        hue
        + 360.0
    ) % 360.0


    R = (
        rgb_uint8[
            :,
            0
        ].astype(
            float
        )
    )

    G = (
        rgb_uint8[
            :,
            1
        ].astype(
            float
        )
    )

    B = (
        rgb_uint8[
            :,
            2
        ].astype(
            float
        )
    )


    values = {
        "n_color_points": (
            len(rgb_uint8)
        ),
    }


    # ========================================================
    # RGB
    # ========================================================

    for name, channel in [
        ("R", R),
        ("G", G),
        ("B", B),
    ]:

        values[
            f"mean_{name}"
        ] = float(
            np.mean(
                channel
            )
        )


        values[
            f"median_{name}"
        ] = float(
            np.median(
                channel
            )
        )


        values[
            f"std_{name}"
        ] = float(
            np.std(
                channel
            )
        )


    # ========================================================
    # LAB
    # ========================================================

    for name, channel in [
        ("L", L),
        ("a", a),
        ("b", b),
    ]:

        values[
            f"mean_{name}"
        ] = float(
            np.mean(
                channel
            )
        )


        values[
            f"median_{name}"
        ] = float(
            np.median(
                channel
            )
        )


        values[
            f"std_{name}"
        ] = float(
            np.std(
                channel
            )
        )


        values[
            f"p10_{name}"
        ] = float(
            np.percentile(
                channel,
                10,
            )
        )


        values[
            f"p90_{name}"
        ] = float(
            np.percentile(
                channel,
                90,
            )
        )


    # ========================================================
    # OTHER COLOR PROPERTIES
    # ========================================================

    values[
        "mean_chroma"
    ] = float(
        np.mean(
            chroma
        )
    )


    values[
        "median_chroma"
    ] = float(
        np.median(
            chroma
        )
    )


    # Hue mean is avoided because hue is circular.
    values[
        "median_hue_deg"
    ] = float(
        np.median(
            hue
        )
    )


    # Useful crude indicator of clipping.
    values[
        "fraction_near_black"
    ] = float(
        np.mean(
            np.max(
                rgb_uint8,
                axis=1,
            )
            <= 5
        )
    )


    values[
        "fraction_near_white"
    ] = float(
        np.mean(
            np.min(
                rgb_uint8,
                axis=1,
            )
            >= 250
        )
    )


    return values


# ============================================================
# DISCOVER PROCESSED ROIs
# ============================================================

def discover_processed_rois():

    records = []


    for roi_file in (
        PROCESSED_ROOT.rglob(
            "roi.vtp"
        )
    ):

        sample_folder = (
            roi_file.parent
        )


        geometry_file = (
            sample_folder
            / "geometry.npz"
        )


        if not geometry_file.exists():

            print(
                f"Skipping, geometry.npz missing:\n"
                f"{roi_file}"
            )

            continue


        records.append(
            (
                roi_file,
                geometry_file,
            )
        )


    return sorted(
        records,
        key=lambda x:
        str(
            x[0]
        ).lower(),
    )


# ============================================================
# PROCESS ALL ROIs
# ============================================================

roi_files = (
    discover_processed_rois()
)


print(
    "\n" + "=" * 80
)

print(
    "RAW COLOR CHARACTERIZATION"
)

print(
    "=" * 80
)


print(
    f"\nFound "
    f"{len(roi_files)} processed ROIs."
)


rows = []


for index, (
    roi_file,
    geometry_file,
) in enumerate(
    roi_files,
    start=1,
):

    print(
        f"[{index:>3}/"
        f"{len(roi_files)}] "
        f"{roi_file.parent.name}"
    )


    try:

        metadata = (
            load_geometry_metadata(
                geometry_file
            )
        )


        mesh = pv.read(
            roi_file
        )


        sample_id = str(
            metadata.get(
                "sample_id",
                roi_file.parent.name,
            )
        )


        scanner = str(
            metadata.get(
                "scanner",
                roi_file.parents[
                    2
                ].name,
            )
        )


        parsed = (
            parse_sample_id(
                sample_id
            )
        )


        (
            rgb,
            color_source,
            color_array,
        ) = get_roi_rgb(
            mesh,
            metadata,
        )


        stats = (
            calculate_color_statistics(
                rgb
            )
        )


        row = {
            "scanner": scanner,

            "sample_id": sample_id,

            "physical_sample_id": (
                parsed[
                    "physical_sample_id"
                ]
            ),

            "group": (
                parsed[
                    "group"
                ]
            ),

            "timepoint_number": (
                parsed[
                    "timepoint_number"
                ]
            ),

            "sample_number": (
                parsed[
                    "sample_number"
                ]
            ),

            "color_source": (
                color_source
            ),

            "color_array_or_uv": (
                color_array
            ),

            "roi_file": str(
                roi_file
            ),

            "geometry_file": str(
                geometry_file
            ),
        }


        row.update(
            stats
        )


        rows.append(
            row
        )


    except Exception as exc:

        print(
            f"    FAILED: {exc}"
        )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    rows
)


if df.empty:

    raise RuntimeError(
        "No ROI color data could be extracted."
    )


df.to_csv(
    ROI_SUMMARY_FILE,
    index=False,
)


print(
    f"\nSaved ROI-level summary:\n"
    f"{ROI_SUMMARY_FILE}"
)


# ============================================================
# ORDER SCANNERS
# ============================================================

available_scanners = list(
    df[
        "scanner"
    ].unique()
)


scanner_order = [
    scanner

    for scanner
    in PREFERRED_SCANNER_ORDER

    if scanner
    in available_scanners
]


scanner_order += sorted(
    [
        scanner

        for scanner
        in available_scanners

        if scanner
        not in scanner_order
    ]
)


# ============================================================
# SCANNER-LEVEL SUMMARY
# ============================================================

summary_variables = [
    "median_R",
    "median_G",
    "median_B",
    "median_L",
    "median_a",
    "median_b",
    "median_chroma",
]


scanner_summary_rows = []


for scanner in scanner_order:

    subset = df[
        df[
            "scanner"
        ]
        ==
        scanner
    ]


    row = {
        "scanner": scanner,
        "n_scans": len(
            subset
        ),
    }


    for variable in (
        summary_variables
    ):

        values = (
            subset[
                variable
            ]
            .dropna()
            .values
        )


        row[
            f"{variable}_mean"
        ] = float(
            np.mean(
                values
            )
        )


        row[
            f"{variable}_sd"
        ] = float(
            np.std(
                values,
                ddof=1,
            )
        )


        row[
            f"{variable}_median"
        ] = float(
            np.median(
                values
            )
        )


        row[
            f"{variable}_iqr"
        ] = float(
            np.percentile(
                values,
                75,
            )
            -
            np.percentile(
                values,
                25,
            )
        )


    scanner_summary_rows.append(
        row
    )


scanner_summary = pd.DataFrame(
    scanner_summary_rows
)


scanner_summary.to_csv(
    SCANNER_SUMMARY_FILE,
    index=False,
)


print(
    f"\nSaved scanner summary:\n"
    f"{SCANNER_SUMMARY_FILE}"
)


# ============================================================
# PAIRED SAME-SAMPLE SCANNER COMPARISONS
# ============================================================
#
# IMPORTANT:
#
# Compare exactly the same BCGxTy-z sample between scanners.
#
# This is much more informative than comparing unrelated
# samples.
# ============================================================

paired_rows = []


for sample_id, sample_df in (
    df.groupby(
        "sample_id"
    )
):

    available = {
        row[
            "scanner"
        ]:
        row

        for _, row
        in sample_df.iterrows()
    }


    for (
        scanner_a,
        scanner_b,
    ) in combinations(
        sorted(
            available.keys()
        ),
        2,
    ):

        A = available[
            scanner_a
        ]

        B = available[
            scanner_b
        ]


        # ----------------------------------------------------
        # Use ROI median Lab as robust whole-sample color.
        # ----------------------------------------------------

        lab_A = np.array(
            [
                A[
                    "median_L"
                ],
                A[
                    "median_a"
                ],
                A[
                    "median_b"
                ],
            ],
            dtype=float,
        )


        lab_B = np.array(
            [
                B[
                    "median_L"
                ],
                B[
                    "median_a"
                ],
                B[
                    "median_b"
                ],
            ],
            dtype=float,
        )


        delta_e = float(
            deltaE_ciede2000(
                lab_A.reshape(
                    1,
                    1,
                    3,
                ),
                lab_B.reshape(
                    1,
                    1,
                    3,
                ),
            )[
                0,
                0
            ]
        )


        paired_rows.append(
            {
                "sample_id": sample_id,

                "physical_sample_id": (
                    A[
                        "physical_sample_id"
                    ]
                ),

                "group": (
                    A[
                        "group"
                    ]
                ),

                "timepoint_number": (
                    A[
                        "timepoint_number"
                    ]
                ),

                "sample_number": (
                    A[
                        "sample_number"
                    ]
                ),

                "scanner_A": scanner_a,

                "scanner_B": scanner_b,

                "scanner_pair": (
                    f"{scanner_a} vs "
                    f"{scanner_b}"
                ),

                "delta_L": (
                    B[
                        "median_L"
                    ]
                    -
                    A[
                        "median_L"
                    ]
                ),

                "delta_a": (
                    B[
                        "median_a"
                    ]
                    -
                    A[
                        "median_a"
                    ]
                ),

                "delta_b": (
                    B[
                        "median_b"
                    ]
                    -
                    A[
                        "median_b"
                    ]
                ),

                "delta_chroma": (
                    B[
                        "median_chroma"
                    ]
                    -
                    A[
                        "median_chroma"
                    ]
                ),

                "deltaE00_median_color": (
                    delta_e
                ),
            }
        )


paired_df = pd.DataFrame(
    paired_rows
)


paired_df.to_csv(
    PAIRED_FILE,
    index=False,
)


print(
    f"\nSaved paired scanner comparisons:\n"
    f"{PAIRED_FILE}"
)


# ============================================================
# PLOT FUNCTION
# ============================================================

def save_scanner_boxplot(
    variable,
    ylabel,
    filename,
):

    data = [
        df.loc[
            df[
                "scanner"
            ]
            ==
            scanner,
            variable,
        ].dropna().values

        for scanner
        in scanner_order
    ]


    fig, ax = plt.subplots(
        figsize=(
            8,
            5,
        )
    )


    ax.boxplot(
        data,
        labels=(
            scanner_order
        ),
        showfliers=True,
    )


    ax.set_xlabel(
        "Scanner"
    )


    ax.set_ylabel(
        ylabel
    )


    ax.set_title(
        f"Raw ROI {ylabel} by scanner"
    )


    ax.grid(
        axis="y",
        alpha=0.25,
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_ROOT
        / filename,

        dpi=200,
    )


    plt.close(
        fig
    )


# ============================================================
# RGB / LAB BOX PLOTS
# ============================================================

save_scanner_boxplot(
    "median_L",
    "Median L*",
    "scanner_Lstar_boxplot.png",
)


save_scanner_boxplot(
    "median_a",
    "Median a*",
    "scanner_astar_boxplot.png",
)


save_scanner_boxplot(
    "median_b",
    "Median b*",
    "scanner_bstar_boxplot.png",
)


save_scanner_boxplot(
    "median_chroma",
    "Median C*ab",
    "scanner_chroma_boxplot.png",
)


# ============================================================
# PAIRED DELTA-E BOXPLOT
# ============================================================

if not paired_df.empty:

    pair_order = sorted(
        paired_df[
            "scanner_pair"
        ].unique()
    )


    data = [
        paired_df.loc[
            paired_df[
                "scanner_pair"
            ]
            ==
            pair,
            "deltaE00_median_color",
        ].dropna().values

        for pair
        in pair_order
    ]


    fig, ax = plt.subplots(
        figsize=(
            11,
            5,
        )
    )


    ax.boxplot(
        data,
        labels=(
            pair_order
        ),
        showfliers=True,
    )


    ax.set_ylabel(
        "CIEDE2000 ΔE00"
    )


    ax.set_xlabel(
        "Scanner pair"
    )


    ax.set_title(
        "Raw color difference for matched physical scans"
    )


    ax.tick_params(
        axis="x",
        rotation=35,
    )


    ax.grid(
        axis="y",
        alpha=0.25,
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_ROOT
        / "paired_scanner_deltaE00_boxplot.png",

        dpi=200,
    )


    plt.close(
        fig
    )


# ============================================================
# PAIRWISE MEDIAN DELTA-E MATRIX
# ============================================================

if not paired_df.empty:

    matrix = np.full(
        (
            len(
                scanner_order
            ),
            len(
                scanner_order
            ),
        ),
        np.nan,
    )


    np.fill_diagonal(
        matrix,
        0.0,
    )


    for i, scanner_a in enumerate(
        scanner_order
    ):

        for j, scanner_b in enumerate(
            scanner_order
        ):

            if j <= i:
                continue


            subset = paired_df[
                (
                    (
                        paired_df[
                            "scanner_A"
                        ]
                        ==
                        scanner_a
                    )
                    &
                    (
                        paired_df[
                            "scanner_B"
                        ]
                        ==
                        scanner_b
                    )
                )
                |
                (
                    (
                        paired_df[
                            "scanner_A"
                        ]
                        ==
                        scanner_b
                    )
                    &
                    (
                        paired_df[
                            "scanner_B"
                        ]
                        ==
                        scanner_a
                    )
                )
            ]


            if subset.empty:
                continue


            value = float(
                subset[
                    "deltaE00_median_color"
                ].median()
            )


            matrix[
                i,
                j
            ] = value


            matrix[
                j,
                i
            ] = value


    fig, ax = plt.subplots(
        figsize=(
            7,
            6,
        )
    )


    image = ax.imshow(
        matrix,
    )


    ax.set_xticks(
        np.arange(
            len(
                scanner_order
            )
        )
    )


    ax.set_yticks(
        np.arange(
            len(
                scanner_order
            )
        )
    )


    ax.set_xticklabels(
        scanner_order,
        rotation=35,
        ha="right",
    )


    ax.set_yticklabels(
        scanner_order
    )


    ax.set_title(
        "Median paired raw ΔE00"
    )


    for i in range(
        len(
            scanner_order
        )
    ):

        for j in range(
            len(
                scanner_order
            )
        ):

            if np.isfinite(
                matrix[
                    i,
                    j
                ]
            ):

                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                )


    fig.colorbar(
        image,
        ax=ax,
        label="Median ΔE00",
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_ROOT
        / "paired_scanner_deltaE00_matrix.png",

        dpi=200,
    )


    plt.close(
        fig
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n" + "=" * 80
)

print(
    "FINISHED"
)

print(
    "=" * 80
)


print(
    f"\nAnalysed ROIs: "
    f"{len(df)}"
)


print(
    "\nColor source:"
)


for (
    source,
    count,
) in (
    df[
        "color_source"
    ].value_counts()
    .items()
):

    print(
        f"  {source:<20} "
        f"{count}"
    )


print(
    "\nROIs per scanner:"
)


for scanner in scanner_order:

    count = int(
        np.sum(
            df[
                "scanner"
            ]
            ==
            scanner
        )
    )


    print(
        f"  {scanner:<15} "
        f"{count}"
    )


if not paired_df.empty:

    print(
        "\nMedian paired ΔE00:"
    )


    for pair, subset in (
        paired_df.groupby(
            "scanner_pair"
        )
    ):

        values = (
            subset[
                "deltaE00_median_color"
            ]
        )


        print(
            f"  {pair:<30} "
            f"{values.median():6.2f} "
            f"(IQR "
            f"{values.quantile(0.25):.2f}"
            f"–"
            f"{values.quantile(0.75):.2f})"
        )


print(
    f"\nAll outputs saved to:\n"
    f"{OUTPUT_ROOT}"
)