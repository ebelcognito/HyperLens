"""Result-pool IO: Parquet-first, with a JSON-compatible shim.

A *pool* is a list of design dicts (each e.g. {theta, fitness, rms_um, metrics,
...}) plus an optional top-level ``meta`` dict (name, n_evaluations, config, ...).

`write_pool` stores the designs as a columnar Parquet table:
  * scalar fields  -> typed columns (fast to filter/query with duckdb/polars),
  * ``theta``/list-of-number fields -> Arrow list<double> columns,
  * any remaining nested/dict fields -> JSON-encoded string columns,
and the top-level meta in the Parquet schema metadata. This is ~5-20x smaller
than JSON and partially readable (no full-file load to query).

`read_pool` auto-detects Parquet vs JSON (by extension, then by magic bytes), so
existing ``*.json`` pools keep working unchanged. If pyarrow is unavailable, or
the path ends in ``.json``, everything degrades to plain JSON.

Round-trips losslessly (modulo None<->null and int/float): the design dicts you
write come back equal.

    from io_results import write_pool, read_pool
    write_pool("pool.parquet", designs, meta={"name": "cmaes", "n_evaluations": 100000})
    designs = read_pool("pool.parquet")            # list[dict]
    designs, meta = read_pool("pool.parquet", with_meta=True)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_PARQUET = True
except Exception:  # pragma: no cover - pyarrow optional
    _HAVE_PARQUET = False

PathLike = Union[str, Path]


def _is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def _is_num_list(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and all(
        isinstance(x, (bool, int, float)) for x in v
    )


def _use_parquet(path: PathLike) -> bool:
    suf = Path(path).suffix.lower()
    if suf == ".json":
        return False
    if suf in (".parquet", ".pq"):
        return True
    return _HAVE_PARQUET  # default to parquet when available


# --------------------------------------------------------------------------- #
# Write
# --------------------------------------------------------------------------- #
def write_pool(
    path: PathLike,
    designs: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a pool to ``path``. Returns the path actually written."""
    meta = dict(meta or {})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not _use_parquet(path) or not _HAVE_PARQUET:
        out = {**meta, "designs": designs}
        jpath = path if path.suffix.lower() == ".json" else path.with_suffix(".json")
        jpath.write_text(json.dumps(out, indent=2))
        return str(jpath)

    # Union of keys, order-preserving (first-seen).
    cols: List[str] = []
    seen = set()
    for d in designs:
        for k in d:
            if k not in seen:
                seen.add(k)
                cols.append(k)

    arrays: List[Any] = []
    json_cols: List[str] = []
    for col in cols:
        vals = [d.get(col, None) for d in designs]
        nonnull = [v for v in vals if v is not None]
        if all(_is_scalar(v) for v in nonnull):
            arrays.append(pa.array(vals))
        elif all(_is_num_list(v) for v in nonnull):
            arrays.append(
                pa.array(
                    [None if v is None else [float(x) for x in v] for v in vals],
                    type=pa.list_(pa.float64()),
                )
            )
        else:  # nested / dict / heterogeneous -> JSON string column
            json_cols.append(col)
            arrays.append(
                pa.array([None if v is None else json.dumps(v) for v in vals],
                         type=pa.string())
            )

    table = pa.table(dict(zip(cols, arrays))) if cols else pa.table({})
    md = {b"pool_meta": json.dumps(meta).encode(),
          b"json_cols": json.dumps(json_cols).encode(),
          b"n_designs": str(len(designs)).encode()}
    table = table.replace_schema_metadata(md)
    ppath = path if path.suffix.lower() in (".parquet", ".pq") else path.with_suffix(".parquet")
    pq.write_table(table, ppath, compression="zstd")
    return str(ppath)


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #
def _looks_parquet(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"PAR1"
    except Exception:
        return False


def read_pool(
    path: PathLike, with_meta: bool = False
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """Read a pool. Returns designs (list[dict]); (designs, meta) if with_meta."""
    path = Path(path)
    suf = path.suffix.lower()
    is_parquet = suf in (".parquet", ".pq") or (suf not in (".json",) and _looks_parquet(path))

    if is_parquet and _HAVE_PARQUET:
        table = pq.read_table(path)
        md = table.schema.metadata or {}
        meta = json.loads(md[b"pool_meta"]) if b"pool_meta" in md else {}
        json_cols = set(json.loads(md[b"json_cols"])) if b"json_cols" in md else set()
        pydict = table.to_pydict()
        n = table.num_rows
        designs: List[Dict[str, Any]] = []
        for i in range(n):
            row: Dict[str, Any] = {}
            for col, vals in pydict.items():
                v = vals[i]
                if v is not None and col in json_cols:
                    v = json.loads(v)
                row[col] = v
            designs.append(row)
        return (designs, meta) if with_meta else designs

    # JSON path
    blob = json.loads(path.read_text())
    if isinstance(blob, list):
        return (blob, {}) if with_meta else blob
    designs = blob.get("designs", [])
    meta = {k: v for k, v in blob.items() if k != "designs"}
    return (designs, meta) if with_meta else designs
