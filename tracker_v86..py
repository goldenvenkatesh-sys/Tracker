# =============================================================================
# MASRAINMAN — INDIA TROPICAL TRACKER
# V89 — ECMWF IFS HRES + AIFS Single + IFS ENS + NOAA AI-GFS + NOAA OISST SST
# Auto-refresh / latest-run detection every 1 hour
# THEME: "Turbo-Muted" Oceanic SST Palette
# FEATURES: Capture-Phase Zoom Engine + Widescreen Aspect Ratio (Edge-to-Edge)
# =============================================================================

import json
import os
import webbrowser
import time
import io
import base64
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import xarray as xr
import geopandas as gpd
from PIL import Image
import plotly.graph_objects as go
from shapely.geometry import Point
from shapely.prepared import prep

from ecmwf.opendata import Client

# -----------------------------------------------------------------------------
# Survey of India / GOI map boundary source
# -----------------------------------------------------------------------------
ALL_INDIA_DIR = Path(os.environ.get("MASRAINMAN_ALL_INDIA", r"D:\\ALL INDIA"))
STATE_BOUNDARY_SHP = ALL_INDIA_DIR / "STATE_BOUNDARY.shp"

from eccodes import (
    codes_bufr_new_from_file,
    codes_grib_new_from_file,
    codes_get,
    codes_get_array,
    codes_set,
    codes_release,
    CODES_MISSING_DOUBLE,
    CODES_MISSING_LONG,
    CodesInternalError,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

APP_NAME = "MASRAINMAN INDIA TROPICAL TRACKER"

AUTO_REFRESH_HOURS = 1
AUTO_REFRESH_SECONDS = AUTO_REFRESH_HOURS * 60 * 60

ROOT_DIR = Path(os.environ.get("MASRAINMAN_TRACK_ROOT", r"D:\Track\MR_Tropical_Tracker"))
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Broad domain for tracking storms
TRACK_WEST, TRACK_EAST = 35.0, 120.0
TRACK_SOUTH, TRACK_NORTH = -10.0, 40.0

# Strict layout domain for Map and SST (Widescreen ratio to fill the page)
MAP_WEST, MAP_EAST = 40.0, 115.0
MAP_SOUTH, MAP_NORTH = -5.0, 38.0

MAX_HOURS = 360
DISPLAY_STEP = 6
IFS_RESOLUTION = "0p25"
CYCLES = [0, 6, 12, 18]
IFS_STEPS = {0: 360, 6: 144, 12: 360, 18: 144}
AIFS_STEPS = {0: 360, 6: 360, 12: 360, 18: 360}
AIGFS_MAX_HOURS = 384
AIGFS_STEP = 6
AIGFS_BASE_URL = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/aigfs/prod"

HTML_OUT = OUTPUT_DIR / "MASRAINMAN_IFS_HRES_AIFS_ENS_AIGFS_Tropical_Tracker.html"
AIGFS_DIAG_DIR = DATA_DIR / "aigfs_diagnostics"
AIGFS_DIAG_DIR.mkdir(parents=True, exist_ok=True)
SST_CACHE_DIR = DATA_DIR / "sst"
SST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUT = OUTPUT_DIR / "MASRAINMAN_IFS_HRES_AIFS_ENS_AIGFS_Tropical_Tracks.json"

# =============================================================================
# HELPERS
# =============================================================================

def utc_now():
    return datetime.now(timezone.utc)

def print_banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)

def cycle_label(hh):
    return f"{hh:02d} Z"

def normalize_lon(lon):
    lon = float(lon)
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon

def in_domain(df):
    if df.empty:
        return df
    return df[
        (df["lat"] >= TRACK_SOUTH) & (df["lat"] <= TRACK_NORTH) &
        (df["lon"] >= TRACK_WEST) & (df["lon"] <= TRACK_EAST)
    ].copy()

def first_valid(a):
    for x in np.asarray(a).ravel():
        try:
            xf = float(x)
            if np.isfinite(xf) and xf not in (
                float(CODES_MISSING_LONG),
                float(CODES_MISSING_DOUBLE),
            ):
                return xf
        except Exception:
            pass
    return np.nan

def arr(bufr, key):
    try:
        v = codes_get_array(bufr, key)
    except Exception:
        try:
            v = [codes_get(bufr, key)]
        except Exception:
            return np.array([], dtype=float)
    a = np.asarray(v)
    if a.ndim == 0:
        a = a.reshape(1)
    return a

def align(a, n, fill=np.nan):
    a = np.asarray(a).ravel()
    if n <= 0:
        return np.array([], dtype=float)
    if a.size == n:
        return a
    if a.size == 0:
        return np.full(n, fill)
    if a.size == 1:
        return np.repeat(a, n)
    return np.resize(a, n)

# =============================================================================
# ECMWF CYCLE AVAILABILITY
# =============================================================================

def cycle_candidates(now=None, search_cycles=12):
    now = now or utc_now()
    anchor_hour = (now.hour // 6) * 6
    anchor = datetime(now.year, now.month, now.day, anchor_hour, tzinfo=timezone.utc)
    return [anchor - pd.Timedelta(hours=6*i) for i in range(search_cycles)]

def run_key(run_dt):
    return run_dt.strftime("%Y%m%d%H")

def cycle_datetime_from_key(key):
    key = str(key)
    return datetime.strptime(key, "%Y%m%d%H").replace(tzinfo=timezone.utc)

def run_label(run_dt):
    return f"{run_dt:%d.%m.%Y} {run_dt:%H}Z"

def check_and_download_cycle(run_dt):
    hh = run_dt.hour
    date_obj = run_dt.date()
    step = IFS_STEPS[hh]
    target = DATA_DIR / f"ifs_tc_{date_obj:%Y%m%d}_{hh:02d}z.bufr"

    print(f"Checking IFS TC track product: {run_label(run_dt)} | step={step}h")
    if target.exists() and target.stat().st_size > 0:
        return target

    try:
        client = Client(source="ecmwf", model="ifs", resol=IFS_RESOLUTION, preserve_request_order=True)
        client.retrieve(date=date_obj.strftime("%Y-%m-%d"), time=hh, stream="oper", type="tf", step=step, target=str(target))
    except Exception as e:
        if target.exists():
            try: target.unlink()
            except Exception: pass
        return None

    if not target.exists() or target.stat().st_size < 100:
        try: target.unlink()
        except Exception: pass
        return None
    return target

def check_and_download_ens_cycle(run_dt):
    hh = run_dt.hour
    date_obj = run_dt.date()
    step = IFS_STEPS[hh]
    target = DATA_DIR / f"ifs_ens_tc_{date_obj:%Y%m%d}_{hh:02d}z.bufr"

    if target.exists() and target.stat().st_size > 100:
        return target

    try:
        client = Client(source="ecmwf", model="ifs", resol=IFS_RESOLUTION, preserve_request_order=True)
        client.retrieve(date=date_obj.strftime("%Y-%m-%d"), time=hh, stream="enfo", type="tf", step=step, target=str(target))
    except Exception as e:
        if target.exists():
            try: target.unlink()
            except Exception: pass
        return None

    if not target.exists() or target.stat().st_size <= 100:
        return None
    return target

def check_and_download_aifs_cycle(run_dt):
    hh = run_dt.hour
    date_obj = run_dt.date()
    step = AIFS_STEPS[hh]
    target = DATA_DIR / f"aifs_single_tc_{date_obj:%Y%m%d}_{hh:02d}z.bufr"

    if target.exists() and target.stat().st_size > 100:
        return target

    try:
        client = Client(source="ecmwf", model="aifs-single", resol=IFS_RESOLUTION, preserve_request_order=True)
        client.retrieve(date=date_obj.strftime("%Y-%m-%d"), time=hh, stream="oper", type="tf", step=step, target=str(target))
    except Exception as e:
        if target.exists():
            try: target.unlink()
            except Exception: pass
        return None

    if not target.exists() or target.stat().st_size <= 100:
        try:
            if target.exists(): target.unlink()
        except Exception: pass
        return None
    return target

def find_latest_four_runs():
    now = utc_now()
    print_banner(f"IFS LATEST 4 RUNS — {now:%d %b %Y %H:%M UTC}")
    found = []
    for run_dt in cycle_candidates(now, search_cycles=32):
        if len(found) >= 4:
            break
        path = check_and_download_cycle(run_dt)
        if path is not None:
            found.append((run_dt, path))
            print(f"  FOUND #{len(found)}: {run_label(run_dt)}")
    return found

# =============================================================================
# ECMWF TC BUFR DECODER — HRES + ENS
# =============================================================================

def decode_ecmwf_tc_bufr(path, run_dt):
    rows = []
    message_count = 0

    def add_rows_direct(bufr, storm_id, storm_name, member_values):
        tp = arr(bufr, "timePeriod")
        lat = arr(bufr, "latitude")
        lon = arr(bufr, "longitude")
        p = arr(bufr, "pressureReducedToMeanSeaLevel")
        wind = arr(bufr, "windSpeedAt10M")

        n = max(len(tp), len(lat), len(lon))
        if n == 0:
            return 0
        tp, lat, lon, p, wind = align(tp, n), align(lat, n), align(lon, n), align(p, n), align(wind, n)
        members = np.asarray(member_values).ravel()
        if members.size == 0:
            members = np.array([0])

        if members.size > 1 and n % members.size == 0:
            nper = n // members.size
            arrays = [tp.reshape(members.size, nper), lat.reshape(members.size, nper), lon.reshape(members.size, nper), p.reshape(members.size, nper), wind.reshape(members.size, nper)]
            for k, mem in enumerate(members):
                for j in range(nper):
                    rows.append({"lat": arrays[1][k, j], "lon": arrays[2][k, j], "tau": arrays[0][k, j], "mslp": arrays[3][k, j], "wind_kt": arrays[4][k, j], "storm_id": storm_id, "storm_name": storm_name, "member": mem})
            return n

        for j in range(n):
            rows.append({"lat": lat[j], "lon": lon[j], "tau": tp[j], "mslp": p[j], "wind_kt": wind[j], "storm_id": storm_id, "storm_name": storm_name, "member": members[0]})
        return n

    with open(path, "rb") as f:
        while True:
            bufr = codes_bufr_new_from_file(f)
            if bufr is None:
                break
            message_count += 1
            try:
                codes_set(bufr, "unpack", 1)
                try: storm_id = codes_get(bufr, "stormIdentifier")
                except Exception: storm_id = "UNKNOWN"
                try: storm_name = codes_get(bufr, "longStormName")
                except Exception: storm_name = ""

                member_values = arr(bufr, "ensembleMemberNumber")
                if member_values.size == 0: member_values = np.array([0])
                n_members = len(member_values)

                n_periods = 0
                while True:
                    try:
                        codes_get_array(bufr, f"#{n_periods + 1}#timePeriod")
                        n_periods += 1
                    except (CodesInternalError, Exception):
                        break

                if n_periods > 0:
                    lat0 = align(arr(bufr, "#2#latitude"), n_members)
                    lon0 = align(arr(bufr, "#2#longitude"), n_members)
                    p0 = align(arr(bufr, "#1#pressureReducedToMeanSeaLevel"), n_members)
                    wlat0 = align(arr(bufr, "#3#latitude"), n_members)
                    wlon0 = align(arr(bufr, "#3#longitude"), n_members)
                    wind0 = align(arr(bufr, "#1#windSpeedAt10M"), n_members)
                    periods = [0]
                    period_rows = [(lat0, lon0, p0, wlat0, wlon0, wind0)]

                    for i in range(1, n_periods):
                        tp = arr(bufr, f"#{i}#timePeriod")
                        tau = first_valid(tp)
                        if not np.isfinite(tau): continue

                        rank_centre = i * 2 + 2
                        rank_wind = i * 2 + 3

                        period_rows.append((
                            align(arr(bufr, f"#{rank_centre}#latitude"), n_members),
                            align(arr(bufr, f"#{rank_centre}#longitude"), n_members),
                            align(arr(bufr, f"#{i + 1}#pressureReducedToMeanSeaLevel"), n_members),
                            align(arr(bufr, f"#{rank_wind}#latitude"), n_members),
                            align(arr(bufr, f"#{rank_wind}#longitude"), n_members),
                            align(arr(bufr, f"#{i + 1}#windSpeedAt10M"), n_members),
                        ))
                        periods.append(tau)

                    for k in range(n_members):
                        for tau, values in zip(periods, period_rows):
                            lat, lon, pressure, lat_w, lon_w, wind = values
                            rows.append({"lat": lat[k], "lon": lon[k], "tau": tau, "mslp": pressure[k], "wind_kt": wind[k], "wind_lat": lat_w[k], "wind_lon": lon_w[k], "storm_id": storm_id, "storm_name": storm_name, "member": member_values[k]})
                else:
                    add_rows_direct(bufr, storm_id, storm_name, member_values)
            except Exception as e:
                pass
            finally:
                codes_release(bufr)

    if not rows: return pd.DataFrame()
    out = pd.DataFrame(rows)
    for c in ["lat", "lon", "tau", "mslp", "wind_kt", "wind_lat", "wind_lon"]:
        if c in out: out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.replace([float(CODES_MISSING_DOUBLE), 1e20, -1e20, np.inf, -np.inf], np.nan)
    if out["mslp"].dropna().size:
        med_p = out["mslp"].dropna().median()
        if pd.notna(med_p) and med_p > 2000: out["mslp"] /= 100.0
    if out["wind_kt"].dropna().size:
        med_w = out["wind_kt"].dropna().median()
        if pd.notna(med_w) and med_w < 100: out["wind_kt"] *= 1.943844
    if out["lat"].dropna().size and out["lat"].abs().max() > 90: out["lat"] /= 10.0
    if out["lon"].dropna().size and out["lon"].abs().max() > 360: out["lon"] /= 10.0

    out["lon"] = out["lon"].map(normalize_lon)
    out = out.dropna(subset=["lat", "lon", "tau"])
    out = out[(out["tau"] >= 0) & (out["tau"] <= MAX_HOURS)].copy()
    out = in_domain(out)
    out = out.drop_duplicates(subset=["storm_id", "member", "tau", "lat", "lon"])
    return out.reset_index(drop=True)

# =============================================================================
# NOAA AI-GFS — MSLP-DERIVED LOW-CENTER TRACK
# =============================================================================

def aigfs_file_url(run_dt, forecast_hour):
    hh = run_dt.hour
    date_str = run_dt.strftime("%Y%m%d")
    cc = f"{hh:02d}"
    ff = f"{int(forecast_hour):03d}"
    return f"{AIGFS_BASE_URL}/aigfs.{date_str}/{cc}/model/atmos/grib2/aigfs.t{cc}z.sfc.f{ff}.grib2"

def aigfs_candidate_runs(now=None, search_cycles=16):
    now = now or utc_now()
    anchor_hour = (now.hour // 6) * 6
    anchor = datetime(now.year, now.month, now.day, anchor_hour, tzinfo=timezone.utc)
    return [anchor - pd.Timedelta(hours=6*i) for i in range(search_cycles)]

def aigfs_url_exists(url):
    try:
        r = requests.head(url, timeout=20, allow_redirects=True)
        if r.status_code == 200: return True
    except Exception: pass
    return False

# AIGFS operational runs can appear before all forecast frames are published.
# Require the final F384 frame before treating a cycle as complete.
AIGFS_COMPLETION_HOUR = 384

def find_latest_aigfs_runs(max_runs=4):
    found = []
    for run_dt in aigfs_candidate_runs(search_cycles=32):
        if aigfs_url_exists(aigfs_file_url(run_dt, AIGFS_COMPLETION_HOUR)):
            found.append(run_dt)
            if len(found) >= max_runs:
                break
    return found

def download_aigfs_sfc(run_dt, forecast_hour):
    ff = int(forecast_hour)
    target = DATA_DIR / f"aigfs_sfc_{run_dt:%Y%m%d}_{run_dt:%H}z_f{ff:03d}.grib2"
    if target.exists() and target.stat().st_size > 1000: return target
    url = aigfs_file_url(run_dt, ff)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk: f.write(chunk)
        if tmp.stat().st_size <= 1000: raise RuntimeError("Downloaded file too small")
        tmp.replace(target)
        return target
    except Exception as e:
        try: tmp.unlink(missing_ok=True)
        except Exception: pass
        return None

def decode_aigfs_sfc_prmsl(path):
    fh = open(path, "rb")
    try:
        prmsl = u10 = v10 = prmsl_lats = prmsl_lons = None
        while True:
            h = codes_grib_new_from_file(fh)
            if h is None: break
            try:
                short_name = str(codes_get(h, "shortName"))
                if short_name == "prmsl":
                    prmsl = np.asarray(codes_get_array(h, "values"), dtype=float).ravel()
                    prmsl_lats = np.asarray(codes_get_array(h, "latitudes"), dtype=float).ravel()
                    prmsl_lons = np.asarray(codes_get_array(h, "longitudes"), dtype=float).ravel()
                elif short_name == "10u":
                    u10 = np.asarray(codes_get_array(h, "values"), dtype=float).ravel()
                elif short_name == "10v":
                    v10 = np.asarray(codes_get_array(h, "values"), dtype=float).ravel()
            finally:
                codes_release(h)
    finally:
        fh.close()

    if prmsl is None or prmsl_lats is None or prmsl_lons is None:
        raise RuntimeError("PRMSL/grid not found")

    n = min(prmsl.size, prmsl_lats.size, prmsl_lons.size)
    prmsl, prmsl_lats, prmsl_lons = prmsl[:n], prmsl_lats[:n], prmsl_lons[:n]
    if u10 is not None: u10 = u10[:n]
    if v10 is not None: v10 = v10[:n]

    return prmsl_lats, prmsl_lons, prmsl, u10, v10

_AIGFS_LAND_PREP = None
def _aigfs_india_land_prepared():
    global _AIGFS_LAND_PREP
    if _AIGFS_LAND_PREP is not None: return _AIGFS_LAND_PREP
    try:
        if not STATE_BOUNDARY_SHP.exists(): return None
        gdf = gpd.read_file(STATE_BOUNDARY_SHP)
        if not gdf.empty:
            geom = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
            _AIGFS_LAND_PREP = prep(geom)
        return _AIGFS_LAND_PREP
    except Exception: return None

def choose_aigfs_low_center(lats, lons, prmsl, u10, v10, prev=None):
    """Robust AI-GFS MSLP low-center detector.

    The previous version could reject valid frames because it first clipped the
    native grid and then assumed the clipped 1-D arrays could always be safely
    reshaped.  This version reconstructs the regular grid explicitly and uses
    progressively relaxed candidate selection so one weak/flat low does not
    terminate the track.
    """
    lats = np.asarray(lats, dtype=float).ravel()
    lons = np.asarray(lons, dtype=float).ravel()
    p = np.asarray(prmsl, dtype=float).ravel()
    n = min(lats.size, lons.size, p.size)
    if n == 0:
        return None
    lats, lons, p = lats[:n], lons[:n], p[:n]

    good = (
        np.isfinite(lats) & np.isfinite(lons) & np.isfinite(p) &
        (lons >= TRACK_WEST) & (lons <= TRACK_EAST) &
        (lats >= TRACK_SOUTH) & (lats <= TRACK_NORTH)
    )
    if good.sum() < 100:
        return None

    idx = np.where(good)[0]
    la = lats[idx]
    lo = lons[idx]
    pp = p[idx]

    # Explicitly rebuild the regular geographic grid from the coordinate values.
    # This avoids relying on GRIB point ordering after domain clipping.
    lat_vals = np.unique(np.round(la, 6))
    lon_vals = np.unique(np.round(lo, 6))
    if lat_vals.size < 10 or lon_vals.size < 10:
        return None

    lat_vals.sort()
    lon_vals.sort()
    grid = np.full((lat_vals.size, lon_vals.size), np.nan, dtype=float)
    lat_ix = np.searchsorted(lat_vals, np.round(la, 6))
    lon_ix = np.searchsorted(lon_vals, np.round(lo, 6))
    grid[lat_ix, lon_ix] = pp

    # Local pressure minima.  50 Pa = 0.5 hPa is deliberately modest:
    # AI-GFS can contain broad/weak lows where a 2 hPa depth test is too strict.
    try:
        from scipy.ndimage import minimum_filter, maximum_filter
        local_min = minimum_filter(np.nan_to_num(grid, nan=np.nanmax(grid)), size=9, mode="nearest")
        broad_max = maximum_filter(np.nan_to_num(grid, nan=np.nanmax(grid)), size=25, mode="nearest")
    except Exception:
        return None

    depth = broad_max - grid
    LAT, LON = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    valid = np.isfinite(grid)

    # Exclude the outermost grid ring where filter edge effects can create false minima.
    valid[:2, :] = False
    valid[-2:, :] = False
    valid[:, :2] = False
    valid[:, -2:] = False

    candidate = valid & (grid <= local_min + 1e-6) & (depth >= 50.0)
    cand = np.where(candidate)

    # If no 0.5-hPa local minimum exists, fall back to the lowest pressure in
    # the valid domain. This keeps the frame rather than dropping it.
    if cand[0].size == 0:
        flat = np.where(valid.ravel())[0]
        if flat.size == 0:
            return None
        jflat = flat[np.nanargmin(grid.ravel()[flat])]
        ci, cj = np.unravel_index(jflat, grid.shape)
        cand_i = np.array([ci])
        cand_j = np.array([cj])
    else:
        cand_i, cand_j = cand

    cand_lat = LAT[cand_i, cand_j].astype(float)
    cand_lon = LON[cand_i, cand_j].astype(float)
    cand_p = grid[cand_i, cand_j].astype(float)
    cand_depth = depth[cand_i, cand_j].astype(float)

    # Candidate ocean mask.  Apply it preferentially for initial selection;
    # after a track exists, continuity is more important than a perfect mask.
    land = _aigfs_india_land_prepared()
    ocean = np.ones(cand_lat.size, dtype=bool)
    if land is not None:
        try:
            ocean = np.array([not land.covers(Point(float(lo0), float(la0)))
                              for la0, lo0 in zip(cand_lat, cand_lon)], dtype=bool)
        except Exception:
            pass

    if prev is None:
        # Prefer the tropical/subtropical oceanic belt around the Bay of Bengal,
        # Arabian Sea and adjoining Indian Ocean, but retain a broader fallback.
        initial = (
            ocean & (cand_lat >= 3.0) & (cand_lat <= 28.0) &
            (cand_lon >= 40.0) & (cand_lon <= 105.0) &
            (cand_p <= 102000.0)
        )
        if not np.any(initial):
            initial = (
                ocean & (cand_lat >= TRACK_SOUTH) & (cand_lat <= TRACK_NORTH) &
                (cand_lon >= TRACK_WEST) & (cand_lon <= TRACK_EAST) &
                (cand_p <= 102500.0)
            )
        if not np.any(initial):
            initial = np.ones(cand_lat.size, dtype=bool)

        ii = np.where(initial)[0]
        # Prefer deeper lows, with a small preference for the Indian Ocean/tropics.
        tropical_bonus = np.where(
            (cand_lat[ii] >= 5.0) & (cand_lat[ii] <= 22.0) &
            (cand_lon[ii] >= 55.0) & (cand_lon[ii] <= 105.0), 0.0, 500.0
        )
        score = cand_p[ii] + tropical_bonus - 25.0 * np.clip(cand_depth[ii], 0.0, 1000.0) / 1000.0
        j = ii[np.argmin(score)]
    else:
        plat, plon = prev
        distance = np.sqrt(
            (cand_lat - plat) ** 2 +
            ((cand_lon - plon) * np.cos(np.deg2rad(plat))) ** 2
        )

        # Normal continuity radius first, then a wider recovery radius.
        nearby = distance <= 7.0
        if not np.any(nearby):
            nearby = distance <= 12.0
        if not np.any(nearby):
            # Do not lose the track because a weak low temporarily disappears.
            nearby = np.ones(cand_lat.size, dtype=bool)

        ii = np.where(nearby)[0]
        score = (distance[ii] / 7.0) - 0.20 * np.clip(cand_depth[ii] / 500.0, 0.0, 1.0)
        # Mild pressure preference prevents jumping to a shallow minimum at the edge.
        score += 0.002 * np.maximum(cand_p[ii] - np.nanmin(cand_p[ii]), 0.0)
        j = ii[np.argmin(score)]

    lat = float(cand_lat[j])
    lon = float(cand_lon[j])
    mslp = float(cand_p[j] / 100.0)

    # Wind is sampled from the original GRIB point nearest the selected low.
    wind_kt = np.nan
    if u10 is not None and v10 is not None:
        uu = np.asarray(u10, dtype=float).ravel()[:n]
        vv = np.asarray(v10, dtype=float).ravel()[:n]
        if uu.size and vv.size:
            d2 = (lats - lat) ** 2 + ((lons - lon) * np.cos(np.deg2rad(lat))) ** 2
            d2[~np.isfinite(uu) | ~np.isfinite(vv)] = np.inf
            k = int(np.argmin(d2))
            if np.isfinite(d2[k]):
                uv = np.hypot(uu[k], vv[k])
                if np.isfinite(uv):
                    wind_kt = float(uv * 1.94384449)

    return lat, lon, mslp, wind_kt

def load_aigfs_track(run_dt):
    rows, prev = [], None
    for fhour in range(0, AIGFS_MAX_HOURS + 1, AIGFS_STEP):
        path = download_aigfs_sfc(run_dt, fhour)
        if path is None:
            print(f"  AI-GFS {run_label(run_dt)}: F{fhour:03d} unavailable — skipping frame.")
            continue
        try:
            lats, lons, p, u10, v10 = decode_aigfs_sfc_prmsl(path)
            center = choose_aigfs_low_center(lats, lons, p, u10, v10, prev)
            if center is None:
                print(f"  AI-GFS {run_label(run_dt)}: no valid low center at F{fhour:03d} — skipping frame.")
                continue
            lat, lon, mslp, wind_kt = center
            rows.append({"storm_id": "AIGFS_LOW", "storm_name": "AI-GFS MSLP-DERIVED LOW", "member": "AI-GFS", "tau": fhour, "lat": lat, "lon": lon, "mslp": mslp, "wind_kt": wind_kt})
            prev = (lat, lon)
        except Exception as exc:
            print(f"  AI-GFS {run_label(run_dt)}: decode failed at F{fhour:03d} — {exc}")
            continue
    return pd.DataFrame(rows)

# =============================================================================
# DOWNLOAD IFS TC BUFR
# =============================================================================

def load_cycles():
    runs=find_latest_four_runs()
    hres_data, aifs_data, ens_data, hres_availability, aifs_availability, ens_availability, run_meta = {}, {}, {}, {}, {}, {}, {}

    for run_dt, hres_path in runs:
        key = run_key(run_dt)
        hres_df = decode_ecmwf_tc_bufr(hres_path, run_dt)
        if not hres_df.empty:
            hres_data[key], hres_availability[key] = hres_df, True
        run_meta[key] = {"date": run_dt.strftime("%Y-%m-%d"), "hour": run_dt.hour, "label": run_label(run_dt)}

        aifs_path = check_and_download_aifs_cycle(run_dt)
        if aifs_path:
            aifs_df = decode_ecmwf_tc_bufr(aifs_path, run_dt)
            if not aifs_df.empty: aifs_data[key], aifs_availability[key] = aifs_df, True

        ens_path = check_and_download_ens_cycle(run_dt)
        if ens_path:
            ens_df = decode_ecmwf_tc_bufr(ens_path, run_dt)
            if not ens_df.empty: ens_data[key], ens_availability[key] = ens_df, True

    if not hres_data: raise RuntimeError("No usable IFS HRES data found.")
    return hres_data, aifs_data, ens_data, hres_availability, aifs_availability, ens_availability, run_meta

def load_aigfs(run_meta):
    aigfs_data, aigfs_availability = {}, {}
    aigfs_runs = find_latest_aigfs_runs(max_runs=4)
    for aigfs_run in aigfs_runs:
        key = run_key(aigfs_run)
        df = load_aigfs_track(aigfs_run)
        if not df.empty:
            aigfs_data[key], aigfs_availability[key] = df, True
            run_meta[key] = {"date": aigfs_run.strftime("%Y-%m-%d"), "hour": aigfs_run.hour, "label": run_label(aigfs_run)}
    return aigfs_data, aigfs_availability, aigfs_runs

# =============================================================================
# NOAA OISST v2.1 — DAILY OCEAN SST BACKGROUND
# =============================================================================

OISST_DAILY_BASE = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr"
SST_DISPLAY_RES = 0.125 
SST_GAUSSIAN_SIGMA = 2.8 

def _oisst_daily_urls(date_obj):
    yyyymm, yyyymmdd = date_obj.strftime("%Y%m"), date_obj.strftime("%Y%m%d")
    base = f"{OISST_DAILY_BASE}/{yyyymm}/"
    return [("NRT", f"{base}oisst-avhrr-v02r01.{yyyymmdd}_preliminary.nc"), ("FINAL", f"{base}oisst-avhrr-v02r01.{yyyymmdd}.nc")]

def _sst_palette_rgb(values):
    """Muted Turbo-style palette: easy on the eyes, professional oceanic tones."""
    v = np.asarray(values, dtype=float)
    stops = np.array([
        [0.00,  48,  18,  59],  # 20 C: Deep dark indigo
        [0.12,  70, 100, 220],  # 21.5 C: Muted Blue
        [0.25,  40, 180, 200],  # 23 C: Soft Teal
        [0.40,  90, 210, 140],  # 25 C: Sea Green
        [0.55, 180, 220,  90],  # 27 C: Soft Green-Yellow
        [0.70, 240, 190,  60],  # 29 C: Muted Sand/Gold
        [0.85, 220,  90,  50],  # 31 C: Brick Red
        [0.95, 150,  30,  30],  # 32 C: Dark Crimson
        [1.00,  90,  10,  15],  # 33 C: Deep Burgundy
    ], dtype=float)
    
    x = np.clip((v - 20.0) / 13.0, 0.0, 1.0)
    
    rgb = np.stack([
        np.interp(x, stops[:, 0], stops[:, 1]),
        np.interp(x, stops[:, 0], stops[:, 2]),
        np.interp(x, stops[:, 0], stops[:, 3]),
    ], axis=-1)
    
    rgb_out = np.rint(rgb).astype(np.uint8)
    
    nan_mask = np.isnan(v)
    rgb_out[nan_mask] = [0, 0, 0]
    
    return rgb_out

def _mercator_y(lat):
    lat = np.clip(np.asarray(lat, dtype=float), -85.0, 85.0)
    return np.log(np.tan(np.pi / 4.0 + np.deg2rad(lat) / 2.0))

def _read_oisst_daily_netcdf(nc_path):
    try: ds = xr.open_dataset(nc_path)
    except Exception as exc: raise RuntimeError("Could not open NOAA OISST NetCDF.") from exc

    try:
        lat_name, lon_name = ("lat" if "lat" in ds.coords else "latitude"), ("lon" if "lon" in ds.coords else "longitude")
        da = ds["sst"].squeeze(drop=True).transpose(lat_name, lon_name)

        lat_values, lon_values = np.asarray(da[lat_name].values, dtype=float), np.asarray(da[lon_name].values, dtype=float)
        lat_mask = (lat_values >= MAP_SOUTH) & (lat_values <= MAP_NORTH)
        lon_mask = (lon_values >= MAP_WEST) & (lon_values <= MAP_EAST)
        da = da.isel({lat_name: np.where(lat_mask)[0], lon_name: np.where(lon_mask)[0]}).sortby(lat_name).sortby(lon_name).astype("float32")
        da = da.where(np.isfinite(da)).where((da > -3.0) & (da < 45.0))

        lat_native, lon_native = np.asarray(da[lat_name].values, dtype=float), np.asarray(da[lon_name].values, dtype=float)
        native = np.asarray(da.values, dtype=np.float32)

        fine_lat = np.arange(max(MAP_SOUTH, lat_native.min()), min(MAP_NORTH, lat_native.max()) + 0.0001, SST_DISPLAY_RES)
        fine_lon = np.arange(max(MAP_WEST, lon_native.min()), min(MAP_EAST, lon_native.max()) + 0.0001, SST_DISPLAY_RES)

        da_interp = xr.DataArray(native, coords={lat_name: lat_native, lon_name: lon_native}, dims=(lat_name, lon_name))
        grid = np.asarray(da_interp.interp({lat_name: fine_lat, lon_name: fine_lon}, method="linear").values, dtype=np.float32)

        valid_da = xr.DataArray(np.isfinite(native).astype(np.float32), coords={lat_name: lat_native, lon_name: lon_native}, dims=(lat_name, lon_name))
        valid_fine = np.asarray(valid_da.interp({lat_name: fine_lat, lon_name: fine_lon}, method="nearest").values, dtype=np.float32) > 0.5

        from scipy.ndimage import gaussian_filter, binary_dilation
        valid_fine_expanded = binary_dilation(valid_fine, iterations=15)

        valid = valid_fine & np.isfinite(grid)
        raw = np.where(valid, grid, 0.0).astype(np.float32)
        weights = valid.astype(np.float32)
        smooth_raw = gaussian_filter(raw, sigma=SST_GAUSSIAN_SIGMA, mode="nearest")
        smooth_w = gaussian_filter(weights, sigma=SST_GAUSSIAN_SIGMA, mode="nearest")
        smooth = np.divide(smooth_raw, smooth_w, out=np.full_like(smooth_raw, np.nan, dtype=np.float32), where=smooth_w > 0.001)
        smooth[~valid_fine_expanded] = np.nan

        hover_lat, hover_lon, hover_sst = [], [], []
        for i in range(0, len(fine_lat), 3):
            for j in range(0, len(fine_lon), 3):
                if np.isfinite(smooth[i, j]) and valid_fine_expanded[i, j]:
                    hover_lat.append(round(float(fine_lat[i]), 3))
                    hover_lon.append(round(float(fine_lon[j]), 3))
                    hover_sst.append(round(float(smooth[i, j]), 1))

        y0, y1 = float(_mercator_y(MAP_SOUTH)), float(_mercator_y(MAP_NORTH))
        merc_height = y1 - y0
        merc_width = np.deg2rad(MAP_EAST - MAP_WEST)
        RASTER_W = 1400
        RASTER_H = int(RASTER_W * (merc_height / merc_width)) 

        x_lon = np.linspace(MAP_WEST, MAP_EAST, RASTER_W)
        y_merc = np.linspace(y1, y0, RASTER_H)
        lat_for_rows = np.rad2deg(2.0 * np.arctan(np.exp(y_merc)) - np.pi / 2.0)

        lon_stage = np.full((smooth.shape[0], RASTER_W), np.nan, dtype=np.float32)
        mask_stage = np.zeros((valid_fine.shape[0], RASTER_W), dtype=np.float32)
        for i in range(smooth.shape[0]):
            row = smooth[i]
            good = np.isfinite(row)
            if good.sum() >= 2: lon_stage[i] = np.interp(x_lon, fine_lon[good], row[good], left=np.nan, right=np.nan)
            mask_stage[i] = np.interp(x_lon, fine_lon, valid_fine_expanded[i].astype(np.float32), left=0.0, right=0.0)

        raster = np.full((RASTER_H, RASTER_W), np.nan, dtype=np.float32)
        raster_mask = np.zeros((RASTER_H, RASTER_W), dtype=bool)
        for j in range(RASTER_W):
            col = lon_stage[:, j]
            good = np.isfinite(col)
            if good.sum() >= 2: raster[:, j] = np.interp(lat_for_rows, fine_lat[good], col[good], left=np.nan, right=np.nan)
            raster_mask[:, j] = np.interp(lat_for_rows, fine_lat, mask_stage[:, j], left=0.0, right=0.0) > 0.5

        valid_out = np.isfinite(raster) & raster_mask
        rgba = np.zeros((RASTER_H, RASTER_W, 4), dtype=np.uint8)
        clipped = np.clip(np.nan_to_num(raster, nan=20.0), 20.0, 33.0)
        rgb = _sst_palette_rgb(clipped)
        rgba[..., :3][valid_out] = rgb[valid_out]
        rgba[..., 3][valid_out] = 225  

        png_buffer = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(png_buffer, format="PNG", optimize=True)
        
        return {
            "date": None,
            "source": None,
            "image": "data:image/png;base64," + base64.b64encode(png_buffer.getvalue()).decode("ascii"),
            "min": float(np.nanmin(raster[valid_out])),
            "max": float(np.nanmax(raster[valid_out])),
            "resolution": 0.25,
            "cells": int(valid_out.sum()),
            "hover_lat": hover_lat,
            "hover_lon": hover_lon,
            "hover_sst": hover_sst,
        }
    finally: ds.close()


def load_latest_oisst_sst():
    today = utc_now().date()
    for back in range(14 + 1):
        date_obj = today - pd.Timedelta(days=back).to_pytimedelta()
        for source_name, url in _oisst_daily_urls(date_obj):
            cache = SST_CACHE_DIR / Path(url).name
            try:
                if not cache.exists() or cache.stat().st_size < 500_000:
                    r = requests.get(url, timeout=60, headers={"User-Agent": "MASRAINMAN-India-Tropical-Tracker/86"})
                    if r.status_code == 200 and len(r.content) > 500_000: cache.write_bytes(r.content)
                if cache.exists():
                    sst_raster = _read_oisst_daily_netcdf(cache)
                    sst_raster["date"] = date_obj.strftime("%Y-%m-%d")
                    sst_raster["source"] = f"NOAA/NCEI OISST v2.1 {source_name}"
                    return sst_raster
            except Exception: pass
    return {"date": None, "source": None, "image": None, "hover_lat": [], "hover_lon": [], "hover_sst": []}

def load_state_boundary_segments():
    try:
        gdf = gpd.read_file(STATE_BOUNDARY_SHP)
        if gdf.crs.to_epsg() != 4326: gdf = gdf.to_crs(4326)
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf["geometry"] = gdf.geometry.simplify(0.003, preserve_topology=True)

        segments = []
        for geom in gdf.geometry:
            if not geom or geom.is_empty: continue
            lines = list(geom.boundary.geoms) if hasattr(geom.boundary, "geoms") else [geom.boundary]
            for line in lines:
                if line and not line.is_empty and hasattr(line, "coords"):
                    coords = list(line.coords)
                    segments.append({"lon": [round(float(x), 5) for x, y in coords], "lat": [round(float(y), 5) for x, y in coords]})
        return segments
    except Exception: return []

# =============================================================================
# HTML APPLICATION
# =============================================================================

def create_html(hres_data, aifs_data, ens_data, aigfs_data, hres_availability, aifs_availability, ens_availability, aigfs_availability, initial_cycle, run_meta, sst_data):
    def serialize_dataset(dataset):
        data, max_hours = {}, {}
        for key,df in dataset.items():
            records=[]
            if not df.empty:
                for _,r in df.iterrows():
                    records.append({
                        "tau":int(round(float(r["tau"]))), "lat":float(r["lat"]), "lon":float(r["lon"]),
                        "mslp":float(r["mslp"]) if pd.notna(r.get("mslp",np.nan)) else None,
                        "wind_kt":float(r["wind_kt"]) if pd.notna(r.get("wind_kt",np.nan)) else None,
                        "member":str(r["member"]), "storm_id":str(r["storm_id"]), "storm_name":str(r.get("storm_name","")),
                    })
                max_hours[key]=int(np.nanmax(df["tau"]))
            else: max_hours[key]=0
            data[str(key)]=records
        return data, max_hours

    data,hres_max=serialize_dataset(hres_data)
    aifs_data_obj,aifs_max=serialize_dataset(aifs_data)
    ens_data_obj,ens_max=serialize_dataset(ens_data)
    aigfs_data_obj,aigfs_max=serialize_dataset(aigfs_data)
    
    data_json = json.dumps(data)
    aifs_data_json = json.dumps(aifs_data_obj)
    ens_data_json = json.dumps(ens_data_obj)
    aigfs_data_json = json.dumps(aigfs_data_obj)
    state_boundary_json = json.dumps(load_state_boundary_segments(), separators=(",", ":"))
    sst_json = json.dumps(sst_data, separators=(",", ":"))
    avail_json = json.dumps({str(k): bool(v) for k, v in hres_availability.items()})
    aifs_avail_json = json.dumps({str(k): bool(v) for k, v in aifs_availability.items()})
    ens_avail_json = json.dumps({str(k): bool(v) for k, v in ens_availability.items()})
    aigfs_avail_json = json.dumps({str(k): bool(v) for k, v in aigfs_availability.items()})
    max_json = json.dumps({str(k): int(v) for k, v in hres_max.items()})
    aifs_max_json = json.dumps({str(k): int(v) for k, v in aifs_max.items()})
    ens_max_json = json.dumps({str(k): int(v) for k, v in ens_max.items()})
    aigfs_max_json = json.dumps({str(k): int(v) for k, v in aigfs_max.items()})
    runs_json = json.dumps(run_meta, separators=(",", ":"))
    cycle_keys_json = json.dumps(list(run_meta.keys()))
    
    html = r'''<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="7200">
<title>MASRAINMAN AI + IFS Tropical Tracker</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
<style>
* { box-sizing: border-box; }
body { margin:0; font-family:Arial,sans-serif; background:#ffffff; overflow:hidden; }
#app { display:flex; width:100vw; height:100vh; }
#sidebar { width:290px; min-width:290px; background:#111; color:#fff; padding:14px 16px; overflow-y:auto; }
.brand { font-size:21px; font-weight:900; letter-spacing:1px; line-height:1.05; }
.subtitle { color:#d0d0d0; font-size:10px; font-weight:700; line-height:1.35; margin-top:5px; }
.section { color:#ffd400; font-weight:900; font-size:12px; margin:14px 0 7px; letter-spacing:.5px; }
.grid { display:grid; grid-template-columns:repeat(2,1fr); gap:6px; }
button { border:0; border-radius:5px; padding:7px 5px; background:#292929; color:#fff; cursor:pointer; font-weight:800; font-size:11px; }
button:hover:not(:disabled) { background:#444; }
button.active { background:#ffd400; color:#111; }
button:disabled { background:#191919; color:#555; cursor:not-allowed; }
#exportPngBtn { width:100%; background:#ffd400; color:#111; font-size:11px; min-height:34px; }
#exportPngBtn:hover:not(:disabled) { background:#ffe34d; }
#exportPngBtn.busy { background:#777; color:#fff; cursor:wait; }
#hourButtons { max-height:390px; overflow-y:auto; padding-right:2px; }
.status { margin-top:12px; padding:9px 10px; background:#1d1d1d; border-radius:6px; font-size:10px; font-weight:700; line-height:1.55; }
.play { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
.footer { margin-top:10px; color:#a0a0a0; font-size:8.5px; font-weight:600; line-height:1.5; }
#mapWrap {
  position:relative;
  flex:1;
  min-width:0;
  height:100vh;
  background:#ffffff;
  overflow:hidden;
  user-select: none;
}
#zoomUI {
  position: absolute;
  top: 55px; 
  left: 15px;
  z-index: 200;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.zoom-btn {
  width: 32px; height: 32px;
  background: rgba(255,255,255,0.95);
  border: 1px solid #777;
  border-radius: 6px;
  color: #111;
  font-size: 18px; font-weight: bold;
  cursor: pointer;
  box-shadow: 0 2px 5px rgba(0,0,0,0.2);
  display: flex; align-items: center; justify-content: center;
}
.zoom-btn:hover { background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
#zoomTarget {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  transform-origin: 0 0; 
  transition: transform 0.1s ease-out;
}
#map {
  position:absolute;
  inset:0;
  width:100%;
  height:100%;
  z-index:2;
}
#sstOverlay {
  position:absolute;
  inset:0;
  width:100%;
  height:100%;
  display:none;
  pointer-events:none;
  z-index:1;
  object-fit:contain;
  opacity:0.82;
}
#trackerHeader {
  position:absolute;
  top:0; left:0; right:0; height:43px; box-sizing:border-box;
  padding:3px 10px 2px; text-align:center; background:rgba(255,255,255,0.98);
  z-index:15; pointer-events:none; overflow:hidden;
}
.main-title { margin:0; font-size:18px; line-height:21px; font-weight:900; color:#111; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.sub-title { margin:0; font-size:10px; line-height:14px; font-weight:900; color:#111; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
#mapLegend {
  position:absolute; left:2.5%; right:2.5%; bottom:48px; min-height:74px;
  display:flex; align-items:center; box-sizing:border-box; padding:9px 14px;
  background:rgba(255,255,255,0.97); border:1px solid #999; border-radius:7px;
  box-shadow:0 2px 8px rgba(0,0,0,0.18); z-index:20; color:#172033;
}
#operationalLegend, #ensembleLegend { width:100%; display:grid; align-items:center; column-gap:18px; }
#operationalLegend { grid-template-columns:repeat(4,minmax(0,1fr)); grid-template-rows:repeat(2,minmax(25px,auto)); row-gap:5px; }
#ensembleLegend { grid-template-columns:repeat(5,minmax(0,1fr)); grid-template-rows:1fr; }
.legend-item { display:flex; align-items:center; justify-content:flex-start; gap:7px; min-width:0; overflow:hidden; line-height:1.05; }
.legend-line { width:24px; height:5px; border-radius:4px; position:relative; flex:0 0 24px; border:1px solid #333; box-sizing:border-box; }
.legend-line::after { content:''; position:absolute; width:11px; height:11px; border-radius:50%; left:8px; top:-4px; background:inherit; border:1px solid #111; box-sizing:border-box; }
.legend-label { display:flex; flex-direction:column; align-items:flex-start; justify-content:center; min-width:0; overflow:hidden; color:#111; }
.legend-label b { font-size:9px; font-weight:900; white-space:nowrap; }
.legend-label span { font-size:7px; font-weight:700; margin-top:2px; white-space:nowrap; color:#333; }
#sstColorbar { position:absolute; top:51px; right:12px; width:54px; padding:7px 7px 8px; background:rgba(255,255,255,0.92); border:1px solid #777; border-radius:5px; box-shadow:0 1px 5px rgba(0,0,0,0.16); z-index:12; display:none; pointer-events:none; }
#sstColorbar.on { display:block; }
.sst-cb-title { text-align:center; font-size:11px; font-weight:900; color:#111; margin-bottom:5px; }
.sst-cb-body { display:flex; align-items:stretch; justify-content:center; height:242px; }
.sst-cb-gradient { 
  width:16px; height:100%; border:1px solid #555; 
  background:linear-gradient(to bottom, 
    rgb(90,10,15) 0%, 
    rgb(150,30,30) 5%, 
    rgb(220,90,50) 15%, 
    rgb(240,190,60) 30%, 
    rgb(180,220,90) 45%, 
    rgb(90,210,140) 60%, 
    rgb(40,180,200) 75%, 
    rgb(70,100,220) 88%, 
    rgb(48,18,59) 100%);
}
.sst-cb-labels { height:100%; display:flex; flex-direction:column; justify-content:space-between; margin-left:5px; font-size:8px; color:#111; line-height:1; }
#mapFooter { position:absolute; left:0; right:0; bottom:0; height:42px; display:flex; align-items:center; justify-content:space-between; padding:0 10px 0 16px; background:rgba(255,255,255,0.98); border-top:1px solid #d2b36c; font-size:9px; font-weight:600; line-height:1.2; color:#3e3a30; z-index:10; }
#mapFooter .left { text-align:left; max-width:65%; }
#dataAttribution { font-weight:800; color:#3e3a30; }
#mapFooter .right { text-align:right; font-weight:900; color:#172033; font-size:10px; }
#mapFooter .disclaimer { font-weight:600; color:#d12b20; }
</style>
</head>
<body>
<div id="app">
<aside id="sidebar">
  <div class="brand">MASRAINMAN</div>
  <div class="subtitle">India Tropical Tracker<br>ECMWF IFS + AIFS + AI-GFS</div>
  <div class="section">MODEL SET</div>
  <div class="grid">
    <button id="setDeterministic" class="active" onclick="selectSet('deterministic')">DETERMINISTIC</button>
    <button id="setIFS" onclick="selectSet('ifs')">ECMWF IFS</button>
  </div>
  <div class="section">MODEL PRODUCT</div>
  <div id="productButtons"></div>
  <div class="section">MODEL CYCLE</div>
  <div id="cycleButtons" class="grid"></div>
  <div class="section">FORECAST HOUR</div>
  <div id="hourButtons" class="grid"></div>
  <div class="section">OCEAN SST</div>
  <button id="sstToggleBtn" onclick="toggleSST()" style="width:100%;margin-bottom:4px;">🌊 SST: OFF</button>
  <div class="section">PLAYBACK</div>
  <div class="play"><button onclick="play()">▶ PLAY</button><button onclick="pause()">⏸ PAUSE</button></div>
  <div class="section">EXPORT</div>
  <button id="exportPngBtn" onclick="downloadFullPNG()">⬇ DOWNLOAD FULL PNG</button>
  <div id="pointInfo" style="margin-top:8px;padding:8px;background:#fff;border:1px solid #d2b36c;border-radius:6px;color:#172033;font-size:10px;line-height:1.35;display:none;">
    <b>SELECTED TRACK POINT</b><br><span id="pointInfoText">Click a forecast point on the map.</span>
  </div>
  <div class="status">
    <div><b>Product:</b> <span id="productStatus">IFS HRES / Control</span></div>
    <div><b>Cycle:</b> <span id="cycleStatus"></span></div>
    <div><b>Valid date &amp; time:</b><br><span id="validStatus"></span></div>
    <div><b>Intensity:</b> <span id="intensityStatus">—</span></div>
    <div><b>Track points:</b> <span id="pointsStatus"></span></div>
    <div><b>Max available:</b> <span id="maxStatus"></span></div>
    <div><b>SST:</b> <span id="sstStatus">—</span></div>
  </div>
</aside>
<div id="mapWrap">
  <div id="trackerHeader">
    <div class="main-title" id="trackerTitle">MASRAINMAN – ECMWF IFS HRES – INDIA TROPICAL TRACKER</div>
    <div class="sub-title" id="trackerSubTitle">ECMWF IFS HRES | Valid date/time</div>
  </div>
  <div id="zoomUI">
     <button class="zoom-btn" onclick="window.doZoom(1.2)" title="Zoom In">+</button>
     <button class="zoom-btn" onclick="window.doZoom(0.8)" title="Zoom Out">−</button>
     <button class="zoom-btn" onclick="window.resetZoom()" title="Reset Map">↺</button>
  </div>
  <div id="zoomTarget">
    <img id="sstOverlay" alt="" aria-hidden="true">
    <div id="map"></div>
  </div>
  <div id="mapLegend">
    <div id="operationalLegend">
      <div class="legend-item"><span class="legend-line" style="background:#111111"></span><span class="legend-label"><b>Low Pressure Area (LPA)</b><span>≤17 kt / ≤31 km/h</span></span></div>
      <div class="legend-item"><span class="legend-line" style="background:#ffcc00"></span><span class="legend-label"><b>Depression (D)</b><span>18–27 kt / 33–50 km/h</span></span></div>
      <div class="legend-item"><span class="legend-line" style="background:#ff8c00"></span><span class="legend-label"><b>Deep Depression (DD)</b><span>28–33 kt / 50–61 km/h</span></span></div>
      <div class="legend-item"><span class="legend-line" style="background:#ff6b6b"></span><span class="legend-label"><b>Cyclonic Storm (CS)</b><span>34–47 kt / 62–88 km/h</span></span></div>
      <div class="legend-item"><span class="legend-line" style="background:#d32f2f"></span><span class="legend-label"><b>Severe Cyclonic Storm (SCS)</b><span>48–63 kt / 89–117 km/h</span></span></div>
      <div class="legend-item"><span class="legend-line" style="background:#9b1c1c"></span><span class="legend-label"><b>Very Severe Cyclonic Storm (VSCS)</b><span>64–89 kt / 118–167 km/h</span></span></div>
      <div class="legend-item"><span class="legend-line" style="background:#650000"></span><span class="legend-label"><b>Extremely Severe Cyclonic Storm (ESCS)</b><span>90–119 kt / 168–221 km/h</span></span></div>
      <div class="legend-item"><span class="legend-line" style="background:#3b0010"></span><span class="legend-label"><b>Super Cyclonic Storm (SuCS)</b><span>≥120 kt / ≥222 km/h</span></span></div>
    </div>
    <div id="ensembleLegend" style="display:none;">
      <div class="legend-item"><span class="legend-line" style="background:#444"></span><span class="legend-label"><b>&gt;1000 hPa</b><span>MSLP</span></span></div>
      <div class="legend-item"><span class="legend-line" style="background:#1683ff"></span><span class="legend-label"><b>990–1000 hPa</b><span>MSLP</span></span></div>
      <div class="legend-item"><span class="legend-line" style="background:#f39c12"></span><span class="legend-label"><b>970–990 hPa</b><span>MSLP</span></span></div>
      <div class="legend-item"><span class="legend-line" style="background:#e91e63"></span><span class="legend-label"><b>950–970 hPa</b><span>MSLP</span></span></div>
      <div class="legend-item"><span class="legend-line" style="background:#9b0000"></span><span class="legend-label"><b>&lt;950 hPa</b><span>MSLP</span></span></div>
    </div>
  </div>
  <div id="sstColorbar" aria-label="Sea surface temperature scale">
    <div class="sst-cb-title">SST (°C)</div>
    <div class="sst-cb-body">
      <div class="sst-cb-gradient"></div>
      <div class="sst-cb-labels"><span>33</span><span>32</span><span>31</span><span>30</span><span>29</span><span>28</span><span>27</span><span>26</span><span>25</span><span>24</span><span>23</span><span>22</span><span>21</span><span>20</span></div>
    </div>
  </div>
  <div id="mapFooter">
    <div class="left">
      <b id="dataAttribution">Data attribution:</b><br>
      <span class="disclaimer">Not an official forecast or warning. For official information, follow IMD and relevant government agencies.</span><br>
      <span id="autoRefreshStatus" style="font-size:10px;color:#777;font-weight:700;">AUTO REFRESH: EVERY 1 HOUR</span>
    </div>
  </div>
</div>
</div>
<script>
const DATA = __DATA__;
const AIFS_DATA = __AIFS_DATA__;
const ENS_DATA = __ENS_DATA__;
const AIGFS_DATA = __AIGFS_DATA__;
const STATE_BOUNDARY = __STATE_BOUNDARY__;
const SST_DATA = __SST_DATA__;
let showSST = false;
const sstStatusEl = document.getElementById('sstStatus');
if(sstStatusEl) sstStatusEl.textContent = (SST_DATA && SST_DATA.date) ? (SST_DATA.date + ' ' + (SST_DATA.source || '')) : 'Unavailable';
const AVAIL = __AVAIL__;
const AIFS_AVAIL = __AIFS_AVAIL__;
const ENS_AVAIL = __ENS_AVAIL__;
const AIGFS_AVAIL = __AIGFS_AVAIL__;
const RUNS = __RUNS__;
const CYCLES = __CYCLE_KEYS__;
let cycle = "__INITIAL__";
let hour = 0;
let timer = null;
let selectedSet='deterministic';
let selectedProduct='hres';

// --- ROBUST CAPTURE-PHASE ZOOM ENGINE ---
let currentScale = 1;
let transX = 0, transY = 0;
let isDraggingMap = false;
let startCX, startCY, startTX, startTY;
const zTarget = document.getElementById('zoomTarget');
const mWrap = document.getElementById('mapWrap');

function applyZoomTransform() {
    zTarget.style.transform = `translate(${transX}px, ${transY}px) scale(${currentScale}) translateZ(0)`;
}

window.doZoom = function(factor, centerX, centerY) {
    const newScale = currentScale * factor;
    if (newScale < 1) { window.resetZoom(); return; }
    if (newScale > 10) return; 
    
    const rect = mWrap.getBoundingClientRect();
    const cx = centerX !== undefined ? centerX : rect.width / 2;
    const cy = centerY !== undefined ? centerY : rect.height / 2;
    
    transX = cx - (cx - transX) * factor;
    transY = cy - (cy - transY) * factor;
    currentScale = newScale;
    applyZoomTransform();
};

window.resetZoom = function() {
    currentScale = 1; transX = 0; transY = 0;
    applyZoomTransform();
};

// Using TRUE for useCapture ensures we intercept events BEFORE Plotly can cancel them.
mWrap.addEventListener('mousedown', (e) => {
    // Let clicks on the sidebars and UI pass through normally
    if(e.target.closest('#zoomUI') || e.target.closest('#sidebar') || e.target.closest('#mapLegend') || e.target.closest('#sstColorbar')) return;
    
    if(currentScale > 1) {
        isDraggingMap = true;
        startCX = e.clientX; startCY = e.clientY;
        startTX = transX; startTY = transY;
        mWrap.style.cursor = 'grabbing';
        e.stopPropagation(); // Stop Plotly from interfering with the drag
    }
}, true);

window.addEventListener('mouseup', (e) => { 
    if(isDraggingMap) {
        isDraggingMap = false; 
        mWrap.style.cursor = 'default'; 
    }
}, true);

window.addEventListener('mousemove', (e) => {
    if(isDraggingMap && currentScale > 1) {
        transX = startTX + (e.clientX - startCX);
        transY = startTY + (e.clientY - startCY);
        applyZoomTransform();
        e.stopPropagation();
        e.preventDefault(); // Stop accidental text selection while dragging
    }
}, true);

mWrap.addEventListener('wheel', (e) => {
    if(e.target.closest('#sidebar') || e.target.closest('#zoomUI') || e.target.closest('#mapLegend') || e.target.closest('#sstColorbar')) return;
    e.preventDefault();
    e.stopPropagation(); // Stop Plotly from catching the scroll wheel
    const rect = mWrap.getBoundingClientRect();
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    window.doZoom(factor, e.clientX - rect.left, e.clientY - rect.top);
}, {passive: false, capture: true});
// ----------------------------------------

function availableProductsForSet(){
  return selectedSet==='ifs' ? [{id:'hres', label:'IFS HRES / CONTROL'}, {id:'ensemble', label:'IFS ENS — MULTI-MEMBER'}]
    : [{id:'hres', label:'IFS HRES / CONTROL'}, {id:'aifs', label:'AIFS SINGLE'}, {id:'aigfs', label:'AI-GFS'}];
}

function selectSet(set){
  selectedSet=set;
  const products=availableProductsForSet();
  if(!products.some(p=>p.id===selectedProduct)) selectedProduct=products[0].id;
  document.getElementById('setDeterministic').classList.toggle('active',selectedSet==='deterministic');
  document.getElementById('setIFS').classList.toggle('active',selectedSet==='ifs');
  pause(); buildProductButtons(); ensureValidCycle(); buildCycles(); buildHours(); updateMap();
}

function buildProductButtons(){
  const box=document.getElementById('productButtons'); box.innerHTML='';
  availableProductsForSet().forEach(p=>{
    const b=document.createElement('button');
    b.style.width='100%'; b.style.marginBottom='6px'; b.textContent=p.label;
    b.className='product-option'+(selectedProduct===p.id?' active':'');
    b.onclick=()=>selectProduct(p.id); box.appendChild(b);
  });
}

function selectProduct(product){ selectedProduct=product; pause(); ensureValidCycle(); buildProductButtons(); buildCycles(); buildHours(); updateMap(); }

function ensureValidCycle(){
  const avail = selectedProduct==='aifs'?AIFS_AVAIL:(selectedProduct==='aigfs'?AIGFS_AVAIL:(selectedProduct==='ensemble'?ENS_AVAIL:AVAIL));
  const dataObj = selectedProduct==='aifs'?AIFS_DATA:(selectedProduct==='aigfs'?AIGFS_DATA:(selectedProduct==='ensemble'?ENS_DATA:DATA));
  if(!avail[String(cycle)] || !(dataObj[String(cycle)]||[]).length){
    const ec=CYCLES.filter(c=>avail[String(c)] && (dataObj[String(c)]||[]).length);
    if(ec.length) cycle=ec[0];
  }
}

function rows(){
  if(selectedProduct==='aifs') return AIFS_DATA[String(cycle)]||[];
  if(selectedProduct==='aigfs') return AIGFS_DATA[String(cycle)]||[];
  if(selectedProduct==='ensemble') return ENS_DATA[String(cycle)]||[];
  return DATA[String(cycle)]||[];
}

function buildCycles(){
  const box=document.getElementById('cycleButtons'); box.innerHTML='';
  const avail = selectedProduct==='aifs'?AIFS_AVAIL:(selectedProduct==='aigfs'?AIGFS_AVAIL:(selectedProduct==='ensemble'?ENS_AVAIL:AVAIL));
  const dataObj = selectedProduct==='aifs'?AIFS_DATA:(selectedProduct==='aigfs'?AIGFS_DATA:(selectedProduct==='ensemble'?ENS_DATA:DATA));
  const orderedCycles=[...CYCLES].sort((a,b)=>Date.parse((RUNS[String(b)]?.date||'1970')+'T'+String(RUNS[String(b)]?.hour||0).padStart(2,'0')+':00:00Z') - Date.parse((RUNS[String(a)]?.date||'1970')+'T'+String(RUNS[String(a)]?.hour||0).padStart(2,'0')+':00:00Z'));

  orderedCycles.forEach(c=>{
    const b=document.createElement('button');
    const hh=String(RUNS[String(c)]?.hour??'').padStart(2,'0');
    b.textContent=hh+' Z'; b.title=(RUNS[String(c)]?.date||'')+' '+hh+'Z';
    const ok = !!avail[String(c)] && (dataObj[String(c)]||[]).length>0;
    b.disabled = !ok;
    if(String(c)===String(cycle)) b.classList.add('active');
    if(ok) b.onclick=()=>{ cycle=c; pause(); hour=hours()[0]||0; buildCycles(); buildHours(); updateMap(); };
    box.appendChild(b);
  });
}

function hours() {
  const vals = [...new Set(rows().map(r => Number(r.tau)).filter(v => Number.isFinite(v)).map(Math.round))].sort((a,b) => a-b);
  const sixHourly = vals.filter(v => v % 6 === 0);
  return sixHourly.length ? sixHourly : vals;
}

function buildHours() {
  const box=document.getElementById('hourButtons'); box.innerHTML='';
  const hs=hours();
  if(!hs.length){ box.innerHTML='<span style="color:#777">No track data</span>'; return; }
  if(!hs.includes(hour)) hour=hs[0];
  hs.forEach(h=>{
    const b=document.createElement('button');
    b.textContent=String(h).padStart(3,'0');
    if(h===hour) b.classList.add('active');
    b.onclick=()=>{ hour=h; buildHours(); updateMap(); };
    box.appendChild(b);
  });
}

function modelAndValidTimes() {
  const meta=RUNS[String(cycle)] || {};
  const base=new Date((meta.date || '1970-01-01')+'T'+String(meta.hour ?? 0).padStart(2,'0')+':00:00Z');
  const rs = rows();
  const taus = rs.map(r => Number(r.tau)).filter(v => Number.isFinite(v));
  const minTau = taus.length ? Math.min(...taus) : 0;
  const maxTau = taus.length ? Math.max(...taus) : Number(hour || 0);

  const validFrom = new Date(base.getTime() + minTau*3600000);
  const validTo   = new Date(base.getTime() + maxTau*3600000);
  const opts={day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',hour12:false};
  return {
      modelUTC: base.toLocaleString('en-GB',{...opts,timeZone:'UTC'}), 
      modelIST: base.toLocaleString('en-GB',{...opts,timeZone:'Asia/Kolkata'}),
      fromUTC: validFrom.toLocaleString('en-GB',{...opts,timeZone:'UTC'}),
      fromIST: validFrom.toLocaleString('en-GB',{...opts,timeZone:'Asia/Kolkata'}),
      toUTC: validTo.toLocaleString('en-GB',{...opts,timeZone:'UTC'}),
      toIST: validTo.toLocaleString('en-GB',{...opts,timeZone:'Asia/Kolkata'})
  };
}

function classifyWind(v) {
  if (!Number.isFinite(v)) return {name:'UNKNOWN', color:'#222222'};
  if (v <= 17)  return {name:'LOW PRESSURE AREA', color:'#111111'};
  if (v < 28)   return {name:'DEPRESSION', color:'#ffcc00'};
  if (v < 34)   return {name:'DEEP DEPRESSION', color:'#ff8c00'};
  if (v < 48)   return {name:'CYCLONIC STORM', color:'#ff6b6b'};
  if (v < 64)   return {name:'SEVERE CYCLONIC STORM', color:'#d32f2f'};
  if (v < 90)   return {name:'VERY SEVERE CYCLONIC STORM', color:'#9b1c1c'};
  if (v < 120)  return {name:'EXTREMELY SEVERE CYCLONIC STORM', color:'#650000'};
  return {name:'SUPER CYCLONIC STORM', color:'#3b0010'};
}

function ensPressureClass(p){
  if(!Number.isFinite(p)) return {name:'UNKNOWN',color:'#666'};
  if(p>1000) return {name:'>1000 hPa',color:'#444'};
  if(p>=990) return {name:'990–1000 hPa',color:'#1683ff'};
  if(p>=970) return {name:'970–990 hPa',color:'#f39c12'};
  if(p>=950) return {name:'950–970 hPa',color:'#e91e63'};
  return {name:'<950 hPa',color:'#9b0000'};
}

function toggleSST(){
  showSST=!showSST;
  const b=document.getElementById('sstToggleBtn');
  if(b) { b.textContent = showSST ? '🌊 SST: ON' : '🌊 SST: OFF'; b.classList.toggle('active', showSST); }
  document.getElementById('sstColorbar')?.classList.toggle('on',showSST);
  
  const img = document.getElementById('sstOverlay');
  if(img){
      if(showSST && SST_DATA && SST_DATA.image) { img.src = SST_DATA.image; img.style.display = 'block'; }
      else { img.style.display = 'none'; img.removeAttribute('src'); }
  }
  updateMap();
}

function updateMap() {
  const rs = rows();
  const t = modelAndValidTimes();
  const ens = selectedProduct==='ensemble', aifs = selectedProduct==='aifs', aigfs = selectedProduct==='aigfs';
  
  document.getElementById('trackerTitle').textContent = ens?'MASRAINMAN – ECMWF IFS ENS':'MASRAINMAN – '+(aigfs?'NOAA AI-GFS':(aifs?'ECMWF AIFS SINGLE':'ECMWF IFS HRES'))+' – INDIA TROPICAL TRACKER';
  document.getElementById('operationalLegend').style.display=ens?'none':'grid';
  document.getElementById('ensembleLegend').style.display=ens?'grid':'none';
  document.getElementById('productStatus').textContent = ens ? 'IFS ENS Multi-Member Ensemble' : (aigfs ? 'AI-GFS MSLP-Derived Low-Center Track' : (aifs ? 'AIFS Single Deterministic' : 'IFS HRES / Control Deterministic'));

  const attributionEl = document.getElementById('dataAttribution');
  if (attributionEl) {
    if (ens) {
      attributionEl.innerHTML = 'Data attribution: ECMWF IFS ENS © ECMWF, CC BY 4.0; NOAA/NCEI OISST v2.1.';
    } else if (aifs) {
      attributionEl.innerHTML = 'Data attribution: ECMWF AIFS © ECMWF, CC BY 4.0; NOAA/NCEI OISST v2.1.';
    } else if (aigfs) {
      attributionEl.innerHTML = 'Data attribution: NOAA/NCEP AI-GFS; NOAA/NCEI OISST v2.1.';
    } else {
      attributionEl.innerHTML = 'Data attribution: ECMWF IFS HRES © ECMWF, CC BY 4.0; NOAA/NCEI OISST v2.1.';
    }
  }

  const meta = RUNS[String(cycle)] || {};
  const modelRunHour = String(meta.hour ?? 0).padStart(2,'0');
  const modelRunDateObj = new Date((meta.date || '1970-01-01') + 'T' + modelRunHour + ':00:00Z');
  
  const formatPointTimes = (tau) => {
    const valid = new Date(modelRunDateObj.getTime() + Number(tau) * 3600000);
    const opts = {day:'2-digit',month:'2-digit',year:'numeric',hour:'2-digit',minute:'2-digit',hour12:false};
    return {
      ist: valid.toLocaleString('en-GB',{...opts,timeZone:'Asia/Kolkata'}),
      utc: valid.toLocaleString('en-GB',{...opts,timeZone:'UTC'})
    };
  };

  const selectedTau=Number(hour);
  const selectedValid=new Date(modelRunDateObj.getTime()+selectedTau*3600000);
  const selectedIST=selectedValid.toLocaleString('en-GB',{day:'2-digit',month:'2-digit',year:'numeric',hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'Asia/Kolkata'});
  const selectedUTC=selectedValid.toLocaleString('en-GB',{day:'2-digit',month:'2-digit',year:'numeric',hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'UTC'});

  document.getElementById('trackerSubTitle').textContent = `MODEL RUN: ${modelRunHour}Z | VALID: ${selectedIST} IST / ${selectedUTC} UTC`;
  document.getElementById('cycleStatus').textContent = modelRunHour + ' Z';
  document.getElementById('validStatus').innerHTML = `FROM ${t.fromIST} IST<br><span style="color:#888">${t.fromUTC} UTC</span><br><b>TO ${t.toIST} IST</b><br><span style="color:#888">${t.toUTC} UTC</span>`;
  document.getElementById('pointsStatus').textContent = rs.length;
  
  const actualTaus = rs.map(r => Number(r.tau)).filter(v => Number.isFinite(v));
  document.getElementById('maxStatus').textContent = actualTaus.length ? 'F' + String(Math.floor(Math.max(...actualTaus) / 6) * 6).padStart(3,'0') : '—';

  const traces = [];
  if (showSST && SST_DATA && SST_DATA.hover_lat && SST_DATA.hover_lat.length) {
    traces.push({ type: 'scattergeo', lon: SST_DATA.hover_lon, lat: SST_DATA.hover_lat, mode: 'markers', marker: {size: 24, color: 'rgba(0,0,0,0)'}, showlegend: false, customdata: SST_DATA.hover_sst, hovertemplate: '<b>SST: %{customdata:.1f} °C</b><br>Lat: %{lat:.2f}°<br>Lon: %{lon:.2f}°<extra></extra>' });
  }

  STATE_BOUNDARY.forEach((seg) => {
    traces.push({ type: 'scattergeo', lon: seg.lon, lat: seg.lat, mode: 'lines', line: {color: '#c9953d', width: 1.15}, hoverinfo: 'skip', showlegend: false });
  });

  const groups={};
  rs.forEach(r=>{ const key=String(r.storm_id)+'|'+r.member; if(!groups[key]) groups[key]=[]; groups[key].push(r); });

  Object.keys(groups).forEach(key=>{
    const fullTr=groups[key].sort((a,b)=>Number(a.tau)-Number(b.tau));
    const tr=fullTr.filter(x=>Number(x.tau)<=selectedTau);
    if(!tr.length) return;
    let name=String(tr[0].storm_name||'').trim(); if(!name || name==='nan') name='System '+tr[0].storm_id;

    if(ens){
      for(let i=0;i<tr.length-1;i++){
        traces.push({ type:'scattergeo',lon:[tr[i].lon,tr[i+1].lon],lat:[tr[i].lat,tr[i+1].lat],mode:'lines',line:{width:0.9,color:ensPressureClass(tr[i].mslp).color},showlegend:false,hoverinfo:'skip' });
      }
      traces.push({ type:'scattergeo',lon:tr.map(x=>x.lon),lat:tr.map(x=>x.lat),mode:'markers', marker:{size:5.5,symbol:'circle-open',color:tr.map(x=>ensPressureClass(x.mslp).color),line:{width:0.9,color:'#333'}}, showlegend:false, customdata:tr.map(x=>{const t=formatPointTimes(x.tau); return [x.tau,x.member,x.mslp,x.wind_kt,ensPressureClass(x.mslp).name,t.ist,t.utc]}), hovertemplate: '<b>'+name+'</b><br>Forecast: F%{customdata[0]}h<br>Valid: %{customdata[5]} IST<br>UTC: %{customdata[6]}<br>MSLP: %{customdata[2]:.0f} hPa<br>Lat: %{lat:.2f}°<br>Lon: %{lon:.2f}°<extra></extra>' });
    }else{
      for(let i=0;i<tr.length-1;i++){
        traces.push({ type:'scattergeo',lon:[tr[i].lon,tr[i+1].lon],lat:[tr[i].lat,tr[i+1].lat],mode:'lines',line:{width:3,color:classifyWind(tr[i].wind_kt).color},showlegend:false,hoverinfo:'skip' });
      }
      traces.push({ type:'scattergeo',lon:tr.map(x=>x.lon),lat:tr.map(x=>x.lat),mode:'markers', marker:{size:8,symbol:'circle',color:tr.map(x=>classifyWind(x.wind_kt).color),line:{width:1.1,color:'#111'}}, showlegend:false, customdata:tr.map(x=>{const t=formatPointTimes(x.tau); return [x.tau,x.wind_kt,x.mslp,classifyWind(x.wind_kt).name,t.ist,t.utc]}), hovertemplate: '<b>'+name+'</b><br>Forecast: F%{customdata[0]}h<br>Valid: %{customdata[4]} IST<br>UTC: %{customdata[5]}<br>Intensity: %{customdata[3]}<br>Wind: %{customdata[1]:.1f} kt<br>MSLP: %{customdata[2]:.0f} hPa<br>Lat: %{lat:.2f}°<br>Lon: %{lon:.2f}°<extra></extra>' });
    }
  });

  const current=rs.filter(x=>Math.round(Number(x.tau))===selectedTau);
  if(current.length){
    if(ens){
      traces.push({ type:'scattergeo', lon:current.map(x=>x.lon),lat:current.map(x=>x.lat),mode:'markers', marker:{size:8,color:current.map(x=>ensPressureClass(x.mslp).color),line:{width:1.2,color:'#111'}}, showlegend:false, customdata:current.map(x=>{const t=formatPointTimes(x.tau); return [x.tau,x.member,x.mslp,x.wind_kt,ensPressureClass(x.mslp).name,t.ist,t.utc]}), hovertemplate: '<b>IFS ENS Member</b><br>Forecast: F%{customdata[0]}h<br>Valid: %{customdata[5]} IST<br>UTC: %{customdata[6]}<br>MSLP: %{customdata[2]:.0f} hPa<extra></extra>' });
      document.getElementById('intensityStatus').textContent=current.length+' ensemble members'; document.getElementById('intensityStatus').style.color='#1683ff';
    }else{
      const selectedColors=current.map(x=>classifyWind(x.wind_kt).color), selectedNames=current.map(x=>classifyWind(x.wind_kt).name);
      traces.push({ type:'scattergeo', lon:current.map(x=>x.lon),lat:current.map(x=>x.lat),mode:'markers+text', text:current.map(x=>'F'+String(x.tau).padStart(3,'0')), textposition:'top center',textfont:{size:11,color:'#111'}, marker:{size:16,symbol:'circle',color:selectedColors,line:{width:2,color:'#111'}}, showlegend:false, customdata:current.map((x,i)=>{const t=formatPointTimes(x.tau); return [x.tau,x.wind_kt,x.mslp,selectedNames[i],t.ist,t.utc]}), hovertemplate: '<b>Selected Position</b><br>Forecast: F%{customdata[0]}h<br>Valid: %{customdata[4]} IST<br>UTC: %{customdata[5]}<br>Intensity: %{customdata[3]}<br>Wind: %{customdata[1]:.1f} kt<extra></extra>' });
      document.getElementById('intensityStatus').textContent=classifyWind(current[0].wind_kt).name; document.getElementById('intensityStatus').style.color=classifyWind(current[0].wind_kt).color;
    }
  }

  if (actualTaus.length) {
    const first = rs.filter(x => Math.round(Number(x.tau)) === Math.round(Math.min(...actualTaus)));
    if (first.length) traces.push({ type: 'scattergeo', lon: first.map(x => x.lon), lat: first.map(x => x.lat), mode: 'markers', marker: { size: 9, symbol: 'circle-open', color: '#111111', line: {width: 1.5, color: '#111111'} }, showlegend: false, customdata: first.map(x => x.tau), hovertemplate: '<b>Track Start</b><br>Forecast: F%{customdata}h<extra></extra>' });
  }

  Plotly.react('map', traces, {
    dragmode: false,
    margin: {l: 0, r: 0, t: 0, b: 0},
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    geo: {
      projection: {type: 'mercator'},
      lonaxis: { range: [40.0, 115.0], showgrid: true, dtick: 10, gridcolor: '#d8ad62', gridwidth: 1, griddash: 'dash', tickfont: {color: '#9b7a3f', size: 10} },
      lataxis: { range: [-5.0, 38.0], showgrid: true, dtick: 5, gridcolor: '#d8ad62', gridwidth: 1, griddash: 'dash', tickfont: {color: '#9b7a3f', size: 10} },
      showcountries: false, showcoastlines: false,
      showland: true, landcolor: '#e9dcae',
      showocean: false, oceancolor: '#e8e5f7',
      showlakes: true, lakecolor: '#e8e5f7',
      bgcolor: 'rgba(0,0,0,0)'
    },
    showlegend: false
  }, { responsive: true, displaylogo: false, scrollZoom: false, displayModeBar: false }).then(() => {
    const mapEl = document.getElementById('map');
    if(!mapEl) return;
    mapEl.removeAllListeners('plotly_click');
    mapEl.on('plotly_click', (ev) => {
      if(!ev || !ev.points || !ev.points.length) return;
      const pt = ev.points[0];
      const cd = pt.customdata;
      let tau = NaN, wind = NaN, mslp = NaN, intensity = '';
      if(Array.isArray(cd)) {
        tau = Number(cd[0]);
        if(selectedProduct === 'ensemble') {
          mslp = Number(cd[2]);
          wind = Number(cd[3]);
          intensity = ensPressureClass(mslp).name;
        } else {
          wind = Number(cd[1]);
          mslp = Number(cd[2]);
          intensity = String(cd[3] || classifyWind(wind).name);
        }
      } else {
        tau = Number(cd);
      }
      if(!Number.isFinite(tau)) return;

      const meta = RUNS[String(cycle)] || {};
      const runHour = String(meta.hour ?? 0).padStart(2,'0');
      const base = new Date((meta.date || '1970-01-01') + 'T' + runHour + ':00:00Z');
      const valid = new Date(base.getTime() + tau * 3600000);
      const fmt = (tz) => valid.toLocaleString('en-GB',{day:'2-digit',month:'2-digit',year:'numeric',hour:'2-digit',minute:'2-digit',hour12:false,timeZone:tz});
      const ist = fmt('Asia/Kolkata');
      const utc = fmt('UTC');
      const info = document.getElementById('pointInfo');
      const text = document.getElementById('pointInfoText');
      if(info && text) {
        const parts = [
          '<b>Forecast:</b> F' + String(Math.round(tau)).padStart(3,'0') + 'h',
          '<b>Valid:</b> ' + ist + ' IST',
          '<b>UTC:</b> ' + utc,
          Number.isFinite(mslp) ? '<b>MSLP:</b> ' + mslp.toFixed(0) + ' hPa' : '',
          Number.isFinite(wind) ? '<b>Wind:</b> ' + wind.toFixed(1) + ' kt' : '',
          intensity ? '<b>Intensity:</b> ' + intensity : '',
          Number.isFinite(pt.lat) ? '<b>Position:</b> ' + Number(pt.lat).toFixed(2) + '°N, ' + Number(pt.lon).toFixed(2) + '°E' : ''
        ].filter(Boolean);
        text.innerHTML = parts.join('<br>');
        info.style.display = 'block';
      }
    });
  });
}

async function downloadFullPNG(){
  const btn = document.getElementById('exportPngBtn');
  const target = document.getElementById('mapWrap');
  if(!target || typeof html2canvas === 'undefined') return;
  const oldText = btn.textContent; btn.disabled = true; btn.classList.add('busy'); btn.textContent = '⏳ CREATING PNG...';
  try{
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const canvas = await html2canvas(target, { backgroundColor: '#ffffff', scale: 2, useCORS: true, allowTaint: false, logging: false, imageTimeout: 15000, removeContainer: true });
    const runHour = String(RUNS[String(cycle)]?.hour ?? cycle).padStart(2,'0');
    canvas.toBlob(blob => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href = url; a.download = `MASRAINMAN_${selectedProduct}_${runHour}Z_FULL.png`;
      document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    }, 'image/png');
  }catch(err){ alert('Full PNG export failed.'); }finally{ btn.disabled = false; btn.classList.remove('busy'); btn.textContent = oldText; }
}

function updateAutoRefreshStatus(){
  const el = document.getElementById('autoRefreshStatus');
  if(!el) return;
  const next = new Date(Date.now() + 60*60*1000);
  const ist = next.toLocaleString('en-GB',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false,timeZone:'Asia/Kolkata'});
  el.textContent = 'AUTO REFRESH: EVERY 1 HOUR | NEXT: ' + ist + ' IST';
}
updateAutoRefreshStatus();
setInterval(updateAutoRefreshStatus, 60000);

function play(){ if(timer!==null) return; const hs=hours(); if(!hs.length) return; timer=setInterval(()=>{ const i=hs.indexOf(hour); if(i<0 || i>=hs.length-1){ pause(); return; } hour=hs[i+1]; buildHours(); updateMap(); },700); }
function pause(){ if(timer!==null){clearInterval(timer);timer=null;} }
buildProductButtons(); buildCycles(); buildHours(); updateMap();
</script>
</body>
</html>
'''
    
    html = (
        html.replace("__DATA__", data_json)
            .replace("__AIFS_DATA__", aifs_data_json)
            .replace("__ENS_DATA__", ens_data_json)
            .replace("__AIGFS_DATA__", aigfs_data_json)
            .replace("__STATE_BOUNDARY__", state_boundary_json)
            .replace("__SST_DATA__", sst_json)
            .replace("__AVAIL__", avail_json)
            .replace("__AIFS_AVAIL__", aifs_avail_json)
            .replace("__ENS_AVAIL__", ens_avail_json)
            .replace("__AIGFS_AVAIL__", aigfs_avail_json)
            .replace("__MAXH__", max_json)
            .replace("__AIFS_MAXH__", aifs_max_json)
            .replace("__ENS_MAXH__", ens_max_json)
            .replace("__AIGFS_MAXH__", aigfs_max_json)
            .replace("__INITIAL__", str(initial_cycle))
            .replace("__RUNS__", runs_json)
            .replace("__CYCLE_KEYS__", cycle_keys_json)
    )

    HTML_OUT.write_text(html, encoding="utf-8")
    return HTML_OUT

# =============================================================================
# MAIN
# =============================================================================

def build_tracker_once():
    """Check latest model availability, download new data, and rebuild HTML."""
    check_started = utc_now()
    print_banner(f"RUN CHECK — {check_started:%d.%m.%Y %H:%M UTC}")
    print("Checking latest IFS / AIFS / ENS / AI-GFS run availability...")

    hres_data, aifs_data, ens_data, hres_availability, aifs_availability, ens_availability, run_meta = load_cycles()
    aigfs_data, aigfs_availability, aigfs_runs = load_aigfs(run_meta)
    sst_data = load_latest_oisst_sst()

    usable = sorted(hres_data.keys(), key=lambda x: cycle_datetime_from_key(x), reverse=True)
    initial_cycle = usable[0] if usable else ""

    html = create_html(
        hres_data, aifs_data, ens_data, aigfs_data,
        hres_availability, aifs_availability, ens_availability, aigfs_availability,
        initial_cycle, run_meta, sst_data
    )

    latest_ifs = usable[0] if usable else "—"
    latest_aigfs = run_key(aigfs_runs[0]) if aigfs_runs else "—"
    print(f"Latest usable IFS run : {latest_ifs}")
    print(f"Latest complete AI-GFS: {latest_aigfs} (F384 required)")
    print(f"Tracker HTML updated   : {html}")
    return html


def main():
    print_banner(APP_NAME)
    print("Models: ECMWF IFS HRES + ECMWF AIFS Single + ECMWF IFS ENS + NOAA AI-GFS")
    print(f"Tracking Domain: {TRACK_WEST}E–{TRACK_EAST}E / {TRACK_SOUTH}N–{TRACK_NORTH}N")
    print(f"Map Focus Domain: {MAP_WEST}E–{MAP_EAST}E / {MAP_SOUTH}N–{MAP_NORTH}N")
    print("SST: NOAA OISST v2.1 — HTML overlay using strictly projected dimensions to guarantee alignment.")
    print(f"AUTO REFRESH: every {AUTO_REFRESH_HOURS} hour(s)")
    print("AI-GFS rule: only a completed cycle with F384 is accepted as a full run.")

    browser_opened = False
    while True:
        try:
            html = build_tracker_once()
            if not browser_opened:
                try:
                    webbrowser.open(html.resolve().as_uri())
                    browser_opened = True
                except Exception:
                    print(f"Open this file manually: {html}")

            next_check = utc_now() + pd.Timedelta(seconds=AUTO_REFRESH_SECONDS)
            print_banner(f"MASRAINMAN V88 READY — NEXT RUN CHECK {next_check:%d.%m.%Y %H:%M UTC}")
            print(f"The page will auto-refresh every {AUTO_REFRESH_HOURS} hour(s).")
            print("If a new completed model cycle is available at the next check, downloading and plotting starts automatically.")
            time.sleep(AUTO_REFRESH_SECONDS)

        except KeyboardInterrupt:
            print("\nAuto-refresh stopped by user.")
            break
        except Exception as exc:
            print(f"\nAUTO-REFRESH ERROR: {exc}")
            print(f"Retrying in {AUTO_REFRESH_HOURS} hour(s)...")
            time.sleep(AUTO_REFRESH_SECONDS)

if __name__ == "__main__":
    main()
