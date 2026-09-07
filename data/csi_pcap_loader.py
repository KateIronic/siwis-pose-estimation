"""
CSI .pcap loader for this project's actual hardware: a Raspberry Pi 4B
(bcm43455c0) running nexmon_csi against a TP-Link Archer AX10 (2x2,
80MHz, channel 36).

IMPORTANT -- read before using:
SiWiS's DNN (Fig. 13, §3) is trained on the feature tensor
    S_k = [t_k, Y_1(k), ..., Y_M(k)]              (Eqn. 11)
Y_m(k) is the output of SiWiS's CUSTOM SELF-MIXING RF HARDWARE
(§2.3-2.4): the signal reflected off the scene is mixed with a local
copy of the host WiFi device's own TX signal, giving a signal whose
*phase* is a linear function of reflector distance (Lemma 2.1). This
loader does not, and cannot, reproduce that -- there is no self-mixing
circuit here, just standard WiFi CSI (H_k, §2.1) extracted from a COTS
Raspberry Pi. Per the paper's own §2.1-2.2, that CSI's phase is
corrupted by carrier phase offset (CPO) between two independent,
unsynchronized oscillators, and is expected to be close to random
packet-to-packet -- this is exactly what was measured on this hardware
(see point 6 below).

This module builds S_k anyway, as the best available substitute, using
a preprocessing recipe ported from a working, empirically-validated
pipeline for this *exact* hardware (Raspberry Pi 4B / bcm43455c0 / this
router), documented in the sibling Collision-detection project
(PROJECT_NOTES.md, and first implemented in
custom_pipeline/dataset/pcap_to_doppler.py). Every step below is here
because leaving it out was measured to make the signal substantially
worse on this rig, not because it's the SiWiS paper's approach:

  1. Parse with `csiread.Nexmon` (a maintained, tested nexmon CSI
     parser) instead of hand-rolled struct parsing.
  2. Filter to one source MAC address (`src_mac`). Without this, CSI
     from every device on the channel gets mixed together as if it
     were one continuous link.
  3. Drop dead (all-zero) packets.
  4. Split interleaved transmit-chain frames apart via KMeans
     clustering on the per-subcarrier amplitude profile. This router
     alternates between two transmit chains per packet, each
     presenting a *different* channel to the receiver; averaging them
     together measured as "a large artificial noise source" (43%
     higher per-subcarrier temporal std) and halved the achievable
     motion-detection contrast. The chain identity isn't labeled
     anywhere in the CSI metadata, so it has to be recovered this way.
  5. Delete guard/pilot/DC subcarriers before any averaging -- at
     80MHz they hold ~94% of total power and are constant over time,
     so leaving them in swamps the real, informative variation with a
     fake DC-dominated signal.
  6. Use amplitude, not the complex value. Measured consecutive-sample
     coherence on a static-room capture from this rig: 0.826
     (amplitude) vs. 0.258 (complex/phase). Every packet carries an
     independent, uncorrectable carrier phase offset, and there's only
     one physical receive antenna here -- no second stream to
     reference it against and cancel that offset.
  7. Resample each recovered stream onto a shared uniform time grid.
     Real packet arrival is bursty (jitter measured at 2.85ms median,
     35ms gaps on this hardware); the model's Conv1d front end and the
     windowing in data/dataset.py both implicitly assume even spacing.
     Streams are resampled onto one *shared* grid (not independently,
     unlike the reference pipeline this is ported from) because SiWiS's
     S_k design needs all M streams sampled in sync at each t_k.
"""

from __future__ import annotations

from typing import Optional, Tuple

import csiread
import numpy as np
from sklearn.cluster import KMeans

import config

# Guard / pilot / DC subcarriers at 80MHz (256 subcarriers) -- carry no
# usable channel info and are constant over time. Same indices SHARP
# deletes in its own preprocessing; confirmed load-bearing on this
# hardware (Collision-detection/PROJECT_NOTES.md §4.6). These indices
# are specific to 80MHz/256 subcarriers -- re-derive them before using
# a 20 or 40MHz capture.
GUARD_SUBCARRIER_IDXS_80MHZ = np.asarray(
    [0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255], dtype=int
)


def _parse_mac(mac_str: str) -> np.ndarray:
    return np.array([int(b, 16) for b in mac_str.split(":")], dtype=int)


def load_csi_pcap(
    pcap_path: str,
    chip: str = "43455c0",
    bw: int = 80,
    variant: str = "legacy",
    src_mac: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a nexmon_csi pcap into a raw (complex) CSI matrix and matching
    packet timestamps. Ported from the validated recipe in
    Collision-detection/custom_pipeline/dataset/pcap_to_doppler.py
    (load_pcap()).

    variant selects the nexmon_csi "pull" mode ('legacy' is the default
    used by this project's collect_csi.sh; see csiread's docs for
    'pull46'/'pull256' if your firmware differs).

    Returns (csi, ts): csi is complex, shape (K_packet, num_subcarriers);
    ts is float seconds, shape (K_packet,). Dead (all-zero) packets are
    already dropped, and, if src_mac is given, packets from any other
    transmitter are already filtered out.
    """
    reader_cls = {
        "legacy": csiread.Nexmon,
        "pull46": csiread.NexmonPull46,
        "pull256": csiread.NexmonPull256,
    }[variant]
    d = reader_cls(pcap_path, chip=chip, bw=bw)
    d.read()

    csi = np.asarray(d.csi)
    if d.nano:
        ts = d.sec.astype(np.float64) + d.usec.astype(np.float64) * 1e-9
    else:
        ts = d.sec.astype(np.float64) + d.usec.astype(np.float64) * 1e-6

    if src_mac is not None:
        src = np.asarray(d.src_addr).reshape(len(csi), -1)
        want = _parse_mac(src_mac)
        mask = np.all(src[:, : len(want)] == want, axis=1)
        kept = int(mask.sum())
        print(f"MAC filter {src_mac}: keeping {kept} of {len(csi)} packets "
              f"({100.0 * kept / max(len(csi), 1):.1f}%)")
        if kept == 0:
            raise ValueError(f"No packets from {src_mac}. Check the MAC, or omit --src_mac.")
        csi, ts = csi[mask], ts[mask]

    nonzero = np.sum(np.abs(csi), axis=1) != 0
    if (~nonzero).sum():
        print(f"dropping {(~nonzero).sum()} zero/corrupt packets")
        csi, ts = csi[nonzero], ts[nonzero]

    return csi, ts


def split_tx_chains(csi: np.ndarray, n_chains: int) -> np.ndarray:
    """
    Recover which of the AP's transmit chains sent each frame, by
    clustering the normalized per-subcarrier amplitude profile -- the
    chain identity isn't labeled anywhere in the CSI metadata. See the
    module docstring, point 4; this was the single most impactful fix
    measured on this hardware. n_chains <= 1 is a no-op (single stream).

    Returns an (K_packet,) int array of chain labels in [0, n_chains).
    """
    if n_chains <= 1:
        return np.zeros(csi.shape[0], dtype=int)
    prof = np.abs(csi)
    prof = prof / np.maximum(prof.mean(axis=1, keepdims=True), 1e-12)
    return KMeans(n_clusters=n_chains, n_init=5, random_state=0).fit_predict(prof)


def _clean_amplitude(csi: np.ndarray, bw: int) -> np.ndarray:
    """
    Per-packet cleanup before collapsing to a scalar feature (module
    docstring, points 5-6): drop guard/pilot/DC subcarriers, normalize
    each packet by its own mean amplitude (removes AGC / TX-power-
    adaptation steps, which would otherwise look like broadband motion),
    then keep amplitude only.

    csi: complex, shape (K, F). Returns real, shape (K, F - num_guard).

    Note: after this normalization, each packet's OWN mean across
    subcarriers is exactly 1 by construction -- collapsing further with
    another .mean(axis=1) is a no-op that always returns 1.0 (a bug
    caught by actually running this against real data: it produced
    zero-variance streams). The informative quantity left after
    normalizing away overall packet power is the *shape* of the
    frequency response -- how much it deviates from flat across
    subcarriers -- so build_feature_sequence collapses with .std(axis=1),
    not .mean(axis=1).
    """
    if bw != 80:
        raise NotImplementedError(
            "GUARD_SUBCARRIER_IDXS_80MHZ is only derived for 80MHz/256 subcarriers -- "
            "re-derive the guard/pilot/DC indices for this bandwidth before using it."
        )
    csi = np.fft.fftshift(csi, axes=1)
    csi = np.delete(csi, GUARD_SUBCARRIER_IDXS_80MHZ, axis=1)
    amp = np.abs(csi)
    mean_amp = amp.mean(axis=1, keepdims=True)
    return amp / np.where(mean_amp == 0, 1, mean_amp)


def build_feature_sequence(
    pcap_path: str,
    chip: str = "43455c0",
    bw: int = 80,
    variant: str = "legacy",
    src_mac: Optional[str] = None,
    n_streams: int = config.NUM_SENSING_ANTENNAS,
    resample_rate_hz: float = config.RESAMPLE_RATE_HZ,
) -> np.ndarray:
    """
    Assemble S_k = [t_k, Y_1(k), ..., Y_M(k)] (Eqn. 11), M = n_streams,
    for an entire capture -- the real-valued (amplitude-only) analog
    described in the module docstring. Each Y_m(k) is the per-packet
    standard deviation, across the cleaned subcarrier set, of the
    packet's own-mean-normalized amplitude for recovered transmit-chain
    m (see _clean_amplitude's docstring for why std, not mean),
    resampled onto a shared uniform time grid across all M streams so
    one synchronized S_k row is meaningful (module docstring, point 7).

    Returns a real float64 array of shape (K_resampled, 1 + n_streams):
    column 0 is timestamp t_k, columns 1..M are Y_1(k)..Y_M(k).
    """
    csi, ts = load_csi_pcap(pcap_path, chip=chip, bw=bw, variant=variant, src_mac=src_mac)
    chain_labels = split_tx_chains(csi, n_streams)

    per_stream_ts = []
    per_stream_y = []
    for chain in range(n_streams):
        sel = chain_labels == chain
        csi_c, ts_c = csi[sel], ts[sel]
        if len(ts_c) < 2:
            raise ValueError(
                f"Chain {chain} has only {len(ts_c)} packet(s) after splitting -- "
                "capture too short, wrong src_mac, or n_streams doesn't match this capture."
            )
        order = np.argsort(ts_c)
        csi_c, ts_c = csi_c[order], ts_c[order]
        amp_c = _clean_amplitude(csi_c, bw)
        per_stream_ts.append(ts_c)
        # std, not mean: amp_c's own per-packet mean is already 1.0 by
        # construction (see _clean_amplitude docstring) -- std across
        # subcarriers is what's actually left carrying information.
        per_stream_y.append(amp_c.std(axis=1))

    t_start = max(t[0] for t in per_stream_ts)
    t_end = min(t[-1] for t in per_stream_ts)
    if t_end <= t_start:
        raise ValueError("Recovered streams do not overlap in time -- check the capture / n_streams.")
    t_grid = np.arange(t_start, t_end, 1.0 / resample_rate_hz)

    y_resampled = np.stack(
        [np.interp(t_grid, per_stream_ts[m], per_stream_y[m]) for m in range(n_streams)],
        axis=1,
    )  # (K_resampled, n_streams)

    return np.concatenate([t_grid[:, None], y_resampled], axis=1)
