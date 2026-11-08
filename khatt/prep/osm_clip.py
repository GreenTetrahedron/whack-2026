"""Clip an OSM extract to the study bbox for r5py (pyosmium; no osmium CLI needed).

R5 refuses street layers wider than 975,000 km² (all of Egypt is ~1.8 M km²). Keeps every
`highway=*` way with at least one node inside the bbox, plus the nodes those ways reference
(so ways crossing the edge stay whole). Boundaries, rivers etc. are dropped.
"""

from __future__ import annotations

from pathlib import Path


def clip_highways(src: Path, dst: Path, bbox: tuple[float, float, float, float]) -> int:
    """bbox = (lng_min, lat_min, lng_max, lat_max). Returns the number of ways written."""
    import osmium

    lng_min, lat_min, lng_max, lat_max = bbox
    tmp = Path(dst).with_suffix(".part.pbf")
    processor = (osmium.FileProcessor(str(src), osmium.osm.NODE | osmium.osm.WAY)
                 .with_locations()
                 .with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
                 .with_filter(osmium.filter.KeyFilter("highway")))
    n = 0
    with osmium.BackReferenceWriter(str(tmp), ref_src=str(src), overwrite=True,
                                    remove_tags=True) as writer:
        for way in processor:
            for node in way.nodes:
                loc = node.location
                if loc.valid() and lng_min <= loc.lon <= lng_max and lat_min <= loc.lat <= lat_max:
                    writer.add(way)
                    n += 1
                    break
    tmp.rename(dst)
    return n
