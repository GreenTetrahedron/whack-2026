"""WorldPop raster → population per H3 cell (CLAUDE.md §6, 02)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from khatt.geo import cells_for_points


def raster_to_cells(path, bbox, resolutions=(8, 9), chunk_rows: int = 512):
    """Sum populated pixel centres into H3 cells.

    Returns ({res: DataFrame(h3, pop)}, raster_total_in_bbox). Pixel centres are assigned to the
    cell containing them, so totals are preserved exactly (up to float error).
    """
    import rasterio
    from rasterio.windows import from_bounds

    lng_min, lat_min, lng_max, lat_max = bbox
    sums: dict[int, dict[str, float]] = {r: {} for r in resolutions}
    total = 0.0
    with rasterio.open(path) as src:
        window = from_bounds(lng_min, lat_min, lng_max, lat_max, src.transform).round_offsets()
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        nodata = src.nodata
        for row0 in range(int(window.row_off), int(window.row_off + window.height), chunk_rows):
            h = min(chunk_rows, int(window.row_off + window.height) - row0)
            win = rasterio.windows.Window(window.col_off, row0, window.width, h)
            data = src.read(1, window=win, masked=False).astype(np.float64)
            mask = np.isfinite(data) & (data > 0)
            if nodata is not None:
                mask &= data != nodata
            rows, cols = np.nonzero(mask)
            if len(rows) == 0:
                continue
            xs, ys = rasterio.transform.xy(src.transform, rows + row0, cols + int(window.col_off))
            lng, lat = np.asarray(xs), np.asarray(ys)
            inside = (lng >= lng_min) & (lng <= lng_max) & (lat >= lat_min) & (lat <= lat_max)
            vals = data[rows, cols][inside]
            lat, lng = lat[inside], lng[inside]
            total += float(vals.sum())
            for res in resolutions:
                cells = cells_for_points(lat, lng, res)
                agg = pd.Series(vals).groupby(cells).sum()
                d = sums[res]
                for cell, v in agg.items():
                    d[cell] = d.get(cell, 0.0) + float(v)
    out = {res: pd.DataFrame({"h3": list(d.keys()), "pop": list(d.values())})
           for res, d in sums.items()}
    return out, total
