"""
blink_detector.py — Real-time blink detection using the BLINK unsupervised algorithm.

Paper: Agarwal M. & Bhatt P., "Blink: A Fully Automated Unsupervised Algorithm for
       Eye-Blink Detection in EEG Signals", ISCAS 2019.
Code:  github.com/meagmohit/BLINK.Codes  (original by Mohit Agarwal, Georgia Tech)

How the algorithm works
-----------------------
1. Low-pass filter at 10 Hz (4th-order Butterworth) to isolate slow blink envelope.
2. Candidate extraction — sequential peak-detector (peakdet) finds all troughs, then
   find_expoints locates stable baseline recovery on both sides of each trough.
3. Pairwise cross-correlation matrix — each candidate waveform is compared against
   every other (shape + amplitude) via Pearson r after resampling to equal length.
4. Hierarchical clustering — separates the high-amplitude repetitive blink cluster
   from low-amplitude noise; learns delta_new (the user's characteristic blink depth).
5. Second pass — re-run peakdet with delta_new/3, build a new correlation matrix, and
   select final blinks via the same clustering step.

Demo workflow (recommended)
---------------------------
  Before the demo:
    python tools/calibrate_blink.py --headset cyton --label user2
    python tools/calibrate_blink.py --headset gtech --label user1

  At demo time:
    det = BlinkDetector(fs=250, frontal_ch=0,
                        profile="eeg/models/blink_user2.npz")
    det.feed(chunk)      # feed every EEG chunk as it arrives
    if det.check():      # returns True on blink rising-edge, clears flag
        ...

  Fallback — if no profile is supplied, BlinkDetector accumulates CALIB_SECS
  of live data and calibrates automatically in a background thread.
"""

from __future__ import annotations

import collections
import os
import threading
import time
from typing import Optional

import numpy as np
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import linkage, fcluster


# ── Paper constants (universal — do not tune per user) ───────────────────────
CORR_THRESH_1   = 0.2    # pass-1: fingerprint membership threshold
CORR_THRESH_2   = 0.7    # pass-2: blink confirmation threshold
DELTA_INIT_UV   = 100.0  # µV — initial peak-detection sensitivity
BLINK_LEN_MAX_S = 2.0    # seconds — candidates longer than this are discarded
BLINK_LEN_MIN_S = 0.1    # seconds — candidates shorter than this are discarded

# ── Real-time adaptation parameters ──────────────────────────────────────────
CALIB_SECS    = 60.0   # seconds of live data before auto-calibration triggers
DETECT_WIN_S  = 4.0    # sliding detection window (BrainFlow recommends ≥ 4 s)
SCAN_EVERY_S  = 0.25   # how often the scanner runs after calibration
REFRACT_S     = 0.4    # minimum gap between consecutive reported blinks


# ══════════════════════════════════════════════════════════════════════════════
# Core algorithm helpers (private)
# ══════════════════════════════════════════════════════════════════════════════

def _lowpass(sig: np.ndarray, fc: float = 10.0, fs: float = 250.0,
             order: int = 4) -> np.ndarray:
    """Causal 4th-order Butterworth low-pass filter (matches the paper)."""
    B, A = butter(order, fc / (fs / 2.0), btype="low")
    return lfilter(B, A, sig.astype(np.float64))


def _running_std(sig: np.ndarray, fs: float,
                 window_s: float = 0.5) -> np.ndarray:
    """
    Per-sample running standard deviation over a window_s-second window.
    Values are floored at 1.0 to prevent zero-division in stable_threshold.
    """
    n = len(sig)
    w = int(window_s * fs)
    out = np.ones(n)
    for i in range(n - w):
        v = float(np.std(sig[i:i + w]))
        out[i] = max(v, 1.0)
    if n > w:
        out[n - w:] = out[n - w - 1]
    return out


class _PeakDetState:
    """
    Incremental sequential peak/trough detector.

    Directly ported from peakdet() in BLINK.Codes.  Call feed(t, v) for every
    sample; mintab accumulates [time, value] of detected troughs.
    """

    def __init__(self, delta: float) -> None:
        self.delta = max(abs(delta), 1e-6)
        self.mintab: list[list[float]] = []
        self._mn = float("inf")
        self._mx = -float("inf")
        self._mnpos: Optional[float] = None
        self._mxpos: Optional[float] = None
        self._lookformax = True

    def feed(self, t: float, v: float) -> bool:
        """Returns True when a new trough is committed to mintab."""
        if v > self._mx:
            self._mx, self._mxpos = v, t
        if v < self._mn:
            self._mn, self._mnpos = v, t

        if self._lookformax:
            if v < self._mx - self.delta:
                self._mn, self._mnpos = v, t
                self._lookformax = False
        else:
            if v > self._mn + self.delta:
                self.mintab.append([self._mnpos, self._mn])
                self._mx, self._mxpos = v, t
                self._lookformax = True
                return True
        return False


def _find_expoints(
    min_pts: np.ndarray,
    sig: np.ndarray,
    running_std: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each candidate trough in min_pts, walk outward in both directions to
    find the stable baseline recovery points on the left and right.

    Directly ported from find_expoints() in BLINK.Codes.

    Parameters
    ----------
    min_pts     : (N, 2)  [[trough_time_s, trough_amp], ...]
    sig         : (M,)    1D low-pass filtered EEG (single frontal channel)
    running_std : (M,)    running std of sig
    fs          : float   sample rate

    Returns
    -------
    p_t : (K, 3)  [[left_t, trough_t, right_t], ...]  K ≤ N
    p_v : (K, 3)  [[left_v, trough_v, right_v], ...]
    """
    N          = len(sig)
    win_size   = 25
    win_offset = 10
    iters      = int(1.5 * fs) // win_offset
    std_win    = int(5 * fs)

    p_t, p_v = [], []

    for trough_t, trough_v in min_pts:
        xR = xL = int(fs * trough_t)
        s0 = max(0, xR - std_win)
        s1 = min(N, xR + std_win)
        stable_thr = 2.0 * max(float(np.min(running_std[s0:s1])), 1.0)

        max_v  = trough_v
        found1 = found2 = 0
        state1 = state2 = 0

        for _ in range(iters):
            xR = min(xR, N - win_size - 1)
            xL = max(xL, 0)

            sr = sig[xR : xR + win_size]
            sl = sig[xL : xL + win_size]
            if len(sr) < win_size or len(sl) < win_size:
                break

            # ── right side ──────────────────────────────────────────────
            std_r = float(np.std(sr))
            if std_r > 2.5 * stable_thr and state1 == 0:
                state1 = 1
            if std_r < stable_thr and state1 == 1 and sig[xR] > trough_v:
                found1 = 1
                max_v  = max(float(sig[xR]), max_v)
            if found1 == 1 and sig[xR] < (max_v + 2.0 * trough_v) / 3.0:
                found1 = 0

            # ── left side ───────────────────────────────────────────────
            std_l = float(np.std(sl))
            right_of_sl = float(sig[min(N - 1, xL + win_size)])
            if std_l > 2.5 * stable_thr and state2 == 0:
                state2 = 1
            if std_l < stable_thr and state2 == 1 and right_of_sl > trough_v:
                found2 = 1
                max_v  = max(right_of_sl, max_v)
            if found2 == 1 and right_of_sl < (max_v + 2.0 * trough_v) / 3.0:
                found2 = 0

            if found1 == 0:
                xR += win_offset
            if found2 == 0:
                xL -= win_offset

            if found1 == 1 and found2 == 1:
                break

        if found1 == 1 and found2 == 1:
            # Determine left boundary (the sample just right of the stable window)
            left_i  = xL + win_size
            left_t  = left_i / fs
            if left_t > trough_t:        # edge case: window overlaps trough
                left_i = xL
                left_t = xL / fs
            right_t = xR / fs
            left_v  = float(sig[min(N - 1, left_i)])
            right_v = float(sig[min(N - 1, xR)])

            dur = right_t - left_t
            if BLINK_LEN_MIN_S <= dur <= BLINK_LEN_MAX_S:
                p_t.append([left_t, trough_t, right_t])
                p_v.append([left_v, trough_v, right_v])

    if p_t:
        return np.array(p_t), np.array(p_v)
    return np.zeros((0, 3)), np.zeros((0, 3))


def _compute_correlation(
    p_t: np.ndarray,
    sig: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the NxN pairwise cross-correlation matrix and power-ratio matrix for
    all candidate blink waveforms.  Each waveform is split at the trough into
    left and right halves; the left half of candidate i is resampled to the
    length of candidate j's left half before computing Pearson r.

    Directly ported from compute_correlation() in BLINK.Codes.
    """
    n     = len(p_t)
    corr  = np.ones((n, n))
    power = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            li = sig[int(fs * p_t[i, 0]) : int(fs * p_t[i, 1])]
            ri = sig[int(fs * p_t[i, 1]) : int(fs * p_t[i, 2])]
            lj = sig[int(fs * p_t[j, 0]) : int(fs * p_t[j, 1])]
            rj = sig[int(fs * p_t[j, 1]) : int(fs * p_t[j, 2])]

            if any(len(x) < 2 for x in (li, ri, lj, rj)):
                continue

            # Resample i halves to j lengths
            li_r = interp1d(np.arange(len(li)), li)(
                np.linspace(0, len(li) - 1, len(lj)))
            ri_r = interp1d(np.arange(len(ri)), ri)(
                np.linspace(0, len(ri) - 1, len(rj)))

            sigA = np.concatenate([li_r, ri_r])
            sigB = np.concatenate([lj,   rj  ])

            c = np.corrcoef(sigA, sigB)[0, 1]
            if np.isfinite(c):
                corr[i, j] = corr[j, i] = c

            si, sj = float(np.std(sigA)), float(np.std(sigB))
            if si > 0 and sj > 0:
                pr = (si / sj) if si > sj else (sj / si)
                if np.isfinite(pr):
                    power[i, j] = power[j, i] = pr

    return corr, power


def _cluster_select(
    candidate_idx: list[int],
    corr: np.ndarray,
    power: np.ndarray,
    blink_var: list[float],
    *,
    var_ratio_thr: Optional[float] = None,
) -> list[int]:
    """
    Hierarchical clustering on the correlation/power submatrix, then select
    the group with higher mean amplitude (variance).

    var_ratio_thr=None (pass-1): always pick the higher-variance group.
    var_ratio_thr=10   (pass-2): only split if ratio > 10; otherwise keep all.
    """
    if len(candidate_idx) < 2:
        return candidate_idx

    sub_c = corr[np.ix_(candidate_idx, candidate_idx)]
    sub_p = power[np.ix_(candidate_idx, candidate_idx)]
    combined = sub_c / np.maximum(sub_p, 1e-6)

    try:
        Z      = linkage(combined, "complete", metric="correlation")
        groups = fcluster(Z, 2, criterion="maxclust")
    except Exception:
        return candidate_idx

    g1v = [blink_var[k] for k, g in enumerate(groups) if g == 1]
    g2v = [blink_var[k] for k, g in enumerate(groups) if g == 2]
    m1  = float(np.mean(g1v)) if g1v else 0.0
    m2  = float(np.mean(g2v)) if g2v else 0.0

    if var_ratio_thr is not None:
        # Pass-2: only split if one group is clearly dominant
        if m1 > 0 and m2 > 0:
            if m1 / m2 > var_ratio_thr:
                return [candidate_idx[k] for k, g in enumerate(groups) if g == 1]
            if m2 / m1 > var_ratio_thr:
                return [candidate_idx[k] for k, g in enumerate(groups) if g == 2]
        return candidate_idx  # no dominant group → keep all
    else:
        # Pass-1: always select the higher-variance group (true blinks)
        sel = 1 if m1 >= m2 else 2
        return [candidate_idx[k] for k, g in enumerate(groups) if g == sel]


# ══════════════════════════════════════════════════════════════════════════════
# Public batch function — full two-pass BLINK algorithm
# ══════════════════════════════════════════════════════════════════════════════

def run_blink_algorithm(sig: np.ndarray, fs: float) -> dict:
    """
    Run the full two-pass BLINK algorithm on a 1D low-pass-filtered EEG signal
    from a single frontal channel.

    Parameters
    ----------
    sig : 1D float array — already low-pass filtered at 10 Hz
    fs  : float          — sample rate (e.g. 250 Hz)

    Returns
    -------
    dict with:
        delta_new     : float            — learned blink depth threshold (µV)
        template_wavs : list[np.ndarray] — extracted blink waveforms
        final_blinks  : (K, 3) array    — [[left_t, trough_t, right_t], ...]
    """
    rstd  = _running_std(sig, fs)
    times = np.arange(len(sig)) / fs

    empty = {"delta_new": DELTA_INIT_UV, "template_wavs": [], "final_blinks": np.zeros((0, 3))}

    # ── Pass 1: learn blink fingerprint and delta_new ──────────────────────
    det1 = _PeakDetState(DELTA_INIT_UV)
    for t, v in zip(times, sig):
        det1.feed(t, v)

    if not det1.mintab:
        return empty

    p_t1, p_v1 = _find_expoints(np.array(det1.mintab), sig, rstd, fs)
    if len(p_t1) == 0:
        return empty

    corr1, pow1 = _compute_correlation(p_t1, sig, fs)

    fp_idx     = int(np.argmax(corr1.sum(axis=1)))
    blink_idx1 = list(np.where(corr1[fp_idx, :] > CORR_THRESH_1)[0])

    if len(blink_idx1) < 2:
        delta_new = DELTA_INIT_UV
    else:
        bv1       = [float(np.var(sig[int(fs * p_t1[i, 0]) : int(fs * p_t1[i, 2])])) for i in blink_idx1]
        tmpl1     = _cluster_select(blink_idx1, corr1, pow1, bv1, var_ratio_thr=None)
        delta_new = float(np.mean([
            min(p_v1[i, 0], p_v1[i, 2]) - p_v1[i, 1]
            for i in tmpl1 if i < len(p_v1)
        ])) if tmpl1 else DELTA_INIT_UV

    if not (np.isfinite(delta_new) and delta_new > 5.0):
        delta_new = DELTA_INIT_UV

    # ── Pass 2: refine with learned delta ─────────────────────────────────
    det2 = _PeakDetState(delta_new / 3.0)
    for t, v in zip(times, sig):
        det2.feed(t, v)

    if not det2.mintab:
        return {"delta_new": delta_new, "template_wavs": [], "final_blinks": np.zeros((0, 3))}

    p_t2, p_v2 = _find_expoints(np.array(det2.mintab), sig, rstd, fs)
    if len(p_t2) == 0:
        return {"delta_new": delta_new, "template_wavs": [], "final_blinks": np.zeros((0, 3))}

    corr2, pow2 = _compute_correlation(p_t2, sig, fs)

    s_fc = corr2.sum(axis=1)
    top3 = list(np.argsort(s_fc)[-3:])
    blink_set: set[int] = set()
    for ki in top3:
        blink_set |= set(np.where(corr2[ki, :] > CORR_THRESH_2)[0].tolist())
    blink_idx2 = list(blink_set)

    if len(blink_idx2) >= 2:
        bv2        = [float(np.var(sig[int(fs * p_t2[i, 0]) : int(fs * p_t2[i, 2])])) for i in blink_idx2]
        blink_idx2 = _cluster_select(blink_idx2, corr2, pow2, bv2, var_ratio_thr=10.0)

    final_blinks = p_t2[blink_idx2] if blink_idx2 else np.zeros((0, 3))

    templates: list[np.ndarray] = []
    for i in blink_idx2:
        if i < len(p_t2):
            l = int(fs * p_t2[i, 0])
            r = int(fs * p_t2[i, 2])
            w = sig[l:r]
            if len(w) >= 5:
                templates.append(w.copy())

    return {
        "delta_new":     delta_new,
        "template_wavs": templates,
        "final_blinks":  final_blinks,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Real-time detector class
# ══════════════════════════════════════════════════════════════════════════════

class BlinkDetector:
    """
    Real-time blink detector backed by the BLINK unsupervised algorithm.

    Typical usage with pre-computed profile (no warm-up at demo time):

        det = BlinkDetector(fs=250, frontal_ch=0,
                            profile="eeg/models/blink_user2.npz")

        # in your data-feed loop:
        det.feed(eeg_chunk)   # (n_samples, n_channels) or (n_samples,)

        # in your decode loop (call at ≥ 10 Hz):
        if det.check():
            trigger_capture()

    Without a profile, the detector will auto-calibrate after CALIB_SECS (60 s)
    of live data and print a message when ready.

    Parameters
    ----------
    fs          : int   — sample rate in Hz (default 250)
    frontal_ch  : int   — column index of the frontal EEG channel in your chunk
    profile     : str   — path to a .npz profile saved by save_profile() /
                          tools/calibrate_blink.py.  None → auto-calibrate live.
    """

    def __init__(
        self,
        fs: int = 250,
        frontal_ch: int = 0,
        profile: Optional[str] = None,
    ) -> None:
        self._fs  = int(fs)
        self._ch  = frontal_ch
        self._lock = threading.Lock()

        # Raw single-channel ring buffer
        self._buf: collections.deque = collections.deque()

        self._calibrated  = False
        self._calibrating = False

        # Learned profile
        self._delta_new: float          = DELTA_INIT_UV
        self._templates: list[np.ndarray] = []

        # Detection state
        self._blink_flag: bool  = False
        self._last_blink_t: float = 0.0   # monotonic time of last reported blink
        self._next_scan_t:  float = 0.0

        if profile:
            if os.path.exists(profile):
                self._load_profile(profile)
            else:
                print(
                    f"  [!] BlinkDetector: profile not found at '{profile}' — "
                    f"live calibration ({CALIB_SECS:.0f} s). Run tools/calibrate_blink.py first.",
                    flush=True,
                )
        else:
            print(
                f"  [!] BlinkDetector: no profile — live calibration ({CALIB_SECS:.0f} s). "
                "Run tools/calibrate_blink.py before the demo for instant start.",
                flush=True,
            )

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        """True once the profile is loaded or live calibration is complete."""
        return self._calibrated

    def feed(self, chunk: np.ndarray) -> None:
        """
        Feed a new EEG chunk into the detector's internal buffer.

        chunk : (n_samples, n_channels) or (n_samples,) — raw µV values.
        Call this from every chunk as data arrives (e.g. from _poll_loop).
        """
        samples = (chunk[:, self._ch] if chunk.ndim == 2 else chunk).astype(np.float64)
        with self._lock:
            self._buf.extend(samples)
            if not self._calibrated and not self._calibrating:
                if len(self._buf) >= int(CALIB_SECS * self._fs):
                    self._start_calibration()

    def check(self) -> bool:
        """
        Return True if a blink was detected since the last call (rising-edge).
        Clears the internal flag.  Safe to call from any thread at any rate.
        """
        if self._calibrated:
            now = time.monotonic()
            if now >= self._next_scan_t:
                self._next_scan_t = now + SCAN_EVERY_S
                if self._scan():
                    with self._lock:
                        self._blink_flag = True

        with self._lock:
            v = self._blink_flag
            self._blink_flag = False
        return v

    # ── Profile persistence ───────────────────────────────────────────────────

    def save_profile(self, path: str) -> None:
        """Save the learned profile to a .npz file for future sessions."""
        if not self._calibrated:
            raise RuntimeError("Detector not calibrated yet — nothing to save.")
        tmpl_arr = np.empty(len(self._templates), dtype=object)
        for i, t in enumerate(self._templates):
            tmpl_arr[i] = t
        np.savez(path, delta_new=np.array([self._delta_new]), templates=tmpl_arr)
        print(f"  [✓] Blink profile saved → {path}", flush=True)

    def _load_profile(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self._delta_new  = float(data["delta_new"][0])
        self._templates  = list(data["templates"])
        self._calibrated = True
        print(
            f"  [✓] BlinkDetector: loaded '{os.path.basename(path)}'  "
            f"delta={self._delta_new:.1f} µV  templates={len(self._templates)}",
            flush=True,
        )

    # ── Live calibration ──────────────────────────────────────────────────────

    def _start_calibration(self) -> None:
        self._calibrating = True
        cal = np.array(list(self._buf))
        print("  [ ] BlinkDetector: calibrating …", flush=True)
        threading.Thread(
            target=self._do_calibrate, args=(cal,),
            daemon=True, name="blink-calib"
        ).start()

    def _do_calibrate(self, raw: np.ndarray) -> None:
        try:
            sig    = _lowpass(raw, fs=float(self._fs))
            result = run_blink_algorithm(sig, float(self._fs))
            with self._lock:
                self._delta_new   = result["delta_new"]
                self._templates   = result["template_wavs"][:12]
                self._calibrated  = True
                self._calibrating = False
            print(
                f"  [✓] BlinkDetector calibrated: "
                f"delta={result['delta_new']:.1f} µV  "
                f"templates={len(result['template_wavs'])}  "
                f"blinks_in_calib={len(result['final_blinks'])}",
                flush=True,
            )
        except Exception as exc:
            with self._lock:
                self._calibrated  = True
                self._calibrating = False
            print(f"  [!] BlinkDetector calibration failed ({exc}) — using default delta", flush=True)

    # ── Real-time scanner ─────────────────────────────────────────────────────

    def _scan(self) -> bool:
        """
        Run peakdet on the last DETECT_WIN_S of data, check each new trough
        in the latter half against stored templates.
        Returns True if a fresh blink is found.
        """
        detect_n = int(DETECT_WIN_S * self._fs)
        with self._lock:
            if len(self._buf) < detect_n:
                return False
            raw       = np.array(list(self._buf)[-detect_n:])
            delta     = self._delta_new
            templates = list(self._templates)

        sig  = _lowpass(raw, fs=float(self._fs))
        det  = _PeakDetState(delta / 3.0)
        half_t = DETECT_WIN_S / 2.0
        now    = time.monotonic()

        for t, v in zip(np.arange(len(sig)) / float(self._fs), sig):
            det.feed(t, v)

        for trough_t, trough_v in det.mintab:
            if trough_t < half_t:
                continue                         # ignore old data in window

            abs_t = now - (DETECT_WIN_S - trough_t)
            if abs_t - self._last_blink_t < REFRACT_S:
                continue                         # refractory period

            if self._is_blink(sig, trough_t, trough_v, templates, delta):
                self._last_blink_t = abs_t
                return True

        return False

    def _is_blink(
        self,
        sig: np.ndarray,
        trough_t: float,
        trough_v: float,
        templates: list[np.ndarray],
        delta: float,
    ) -> bool:
        """
        Verify a candidate trough against stored templates via cross-correlation.
        Falls back to amplitude threshold if no templates are available yet.
        """
        ti   = int(trough_t * self._fs)
        hw   = int(REFRACT_S * self._fs)          # half-window around trough
        wave = sig[max(0, ti - hw) : min(len(sig), ti + hw)]

        if not templates:
            # Amplitude fallback (used during live-calib warm-up period)
            return float(np.ptp(wave)) > delta * 0.6

        for tmpl in templates:
            if len(tmpl) < 2 or len(wave) < 2:
                continue
            if len(tmpl) != len(wave):
                tmpl_r = interp1d(np.arange(len(tmpl)), tmpl)(
                    np.linspace(0, len(tmpl) - 1, len(wave))
                )
            else:
                tmpl_r = tmpl
            c = np.corrcoef(wave, tmpl_r)[0, 1]
            if np.isfinite(c) and c >= CORR_THRESH_2:
                return True
        return False
