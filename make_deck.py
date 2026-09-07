"""
Generates the project summary deck (SiWiS_Reproduction_Summary.pptx).

Regenerate any time with:
    venv/Scripts/python.exe make_deck.py
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image
import os

# Palette (validated reference instance from the dataviz skill)
INK = RGBColor(0x0B, 0x0B, 0x0B)
SECOND = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
SURFACE = RGBColor(0xFC, 0xFC, 0xFB)
S1 = RGBColor(0x2A, 0x78, 0xD6)   # blue
S2 = RGBColor(0xEB, 0x68, 0x34)   # orange
GOOD = RGBColor(0x0C, 0xA3, 0x0C)
CRIT = RGBColor(0xD0, 0x3B, 0x3B)
RULE = RGBColor(0xE1, 0xE0, 0xD9)

W, H = Inches(13.333), Inches(7.5)
prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]


def slide():
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = SURFACE
    return s


def tb(s, x, y, w, h, align=PP_ALIGN.LEFT):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].alignment = align
    return tf


def para(tf, text, size, color, bold=False, space_before=0, space_after=6, first=False, bullet_color=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = "Segoe UI"
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    return p


def header(s, title, sub=None):
    tf = tb(s, 0.62, 0.42, 12.1, 1.0)
    para(tf, title, 30, INK, bold=True, first=True, space_after=2)
    if sub:
        para(tf, sub, 13.5, MUTED, space_after=0)
    line = s.shapes.add_shape(1, Inches(0.62), Inches(1.52), Inches(12.1), Pt(1.2))
    line.fill.solid(); line.fill.fore_color.rgb = RULE
    line.line.fill.background(); line.shadow.inherit = False
    return s


def bullets(s, items, x=0.62, y=1.95, w=12.1, size=15, gap=13):
    """items: list of (text, color, bold) or plain strings."""
    tf = tb(s, x, y, w, 5.0)
    for i, it in enumerate(items):
        text, color, bold = (it if isinstance(it, tuple) else (it, SECOND, False))
        para(tf, text, size, color, bold=bold, first=(i == 0), space_after=gap)
    return tf


def picture(s, path, x, y, max_w, max_h):
    iw, ih = Image.open(path).size
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    return s.shapes.add_picture(path, Inches(x + (max_w - w) / 2), Inches(y + (max_h - h) / 2),
                                 Inches(w), Inches(h))


def table(s, rows, x, y, w, col_w, size=12.5, header_bold=True):
    n_r, n_c = len(rows), len(rows[0])
    shape = s.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w), Inches(0.4 * n_r))
    t = shape.table
    for i, cw in enumerate(col_w):
        t.columns[i].width = Inches(cw)
    for r, row in enumerate(rows):
        t.rows[r].height = Inches(0.36)
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = SURFACE if r else RGBColor(0xF0, 0xEF, 0xEC)
            cell.margin_left, cell.margin_right = Inches(0.09), Inches(0.09)
            cell.margin_top, cell.margin_bottom = Inches(0.03), Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            # style EVERY paragraph: a "\n" in the cell text creates additional
            # paragraphs, which would otherwise keep the 18pt default
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(size)
                p.font.name = "Segoe UI"
                p.font.bold = bool(r == 0 and header_bold)
                p.font.color.rgb = INK if r == 0 else SECOND
                p.space_after = Pt(0)
    return t


# ─────────────────────────── 1. Title ───────────────────────────
s = slide()
tf = tb(s, 0.9, 2.35, 11.5, 3.0)
para(tf, "Seeing People Through WiFi", 46, INK, bold=True, first=True, space_after=6)
para(tf, "Reproducing SiWiS (MobiCom '24) on commodity Raspberry Pi hardware", 20, SECOND, space_after=26)
para(tf, "Human body-mask segmentation and 2D pose estimation from WiFi CSI alone", 15, MUTED, space_after=4)
para(tf, "Full pipeline: capture → preprocessing → cross-modal labels → training → evaluation", 15, MUTED)
bar = s.shapes.add_shape(1, Inches(0.9), Inches(2.15), Inches(1.6), Pt(4))
bar.fill.solid(); bar.fill.fore_color.rgb = S1
bar.line.fill.background(); bar.shadow.inherit = False

# ─────────────────────────── 2. The goal ───────────────────────────
s = slide()
header(s, "What we set out to do", "Reproduce a MobiCom '24 paper's results using hardware we actually have")
bullets(s, [
    ("The paper: SiWiS — fine-grained human detection using a single WiFi device", INK, True),
    "A dual-branch DNN predicts a human body mask and 17 COCO keypoints from WiFi signals alone. "
    "Video is used only during training, as a supervision signal — never at inference.",
    ("Our starting point", INK, True),
    "An existing Python reproduction of the paper's architecture: signal encoder (conv + self-attention), "
    "two cross-attention decoders, and the paper's loss functions (Eqns. 12–14).",
    ("Our hardware", INK, True),
    "Raspberry Pi 4B (bcm43455c0) running nexmon_csi against a TP-Link Archer AX10 router, "
    "plus a webcam for label generation. No GPU.",
    ("The task", INK, True),
    "Verify the code is correct, adapt it to real hardware, collect data, train it, and measure whether "
    "it actually learns anything.",
], size=14.5, gap=11)

# ─────────────────────────── 3. Hardware gap ───────────────────────────
s = slide()
header(s, "The fundamental hardware gap", "Named up front, because it shapes every result that follows")
rows = [
    ["", "SiWiS paper", "This reproduction"],
    ["Sensing front end", "Custom self-mixing RF circuit", "None — stock WiFi NIC"],
    ["Antennas", "4 patch antennas + LO reference", "1 built-in Pi antenna"],
    ["Signal used", "Self-mixed Y(k), Eqn. 5–11", "Standard CSI H(k), §2.1"],
    ["Phase behaviour", "Coherent; ∝ reflector distance (Lemma 2.1)", "Corrupted by CPO; random per packet"],
    ["Sampling", "Dedicated ADC @ 10 MSps", "Bursty WiFi packet arrivals"],
]
table(s, rows, 0.62, 2.05, 12.1, [2.5, 4.6, 5.0], size=12.5)
tf = tb(s, 0.62, 4.85, 12.1, 2.2)
para(tf, "Why this matters", 15, INK, bold=True, first=True, space_after=6)
para(tf, "SiWiS's headline advantage — phase-coherent sensing — comes from hardware we do not have. The paper's own §2.2 "
         "shows that standard CSI phase is randomised packet-to-packet by carrier phase offset. So this work is best "
         "understood as a CSI-based reproduction of the paper's architecture and training recipe, closer in sensing "
         "modality to the CSI baselines SiWiS outperforms than to SiWiS itself.", 13.5, SECOND, space_after=0)

# ─────────────────────────── 4. Bugs found ───────────────────────────
s = slide()
header(s, "Phase 1 — Auditing the code against the paper", "Cross-referenced every equation and figure against the implementation")
rows = [
    ["Defect", "Impact", "Status"],
    ["Mask labels divided by 255 twice", "Ground-truth masks crushed to ~0.004; model would learn 'no person' everywhere", "Fixed"],
    ["Frames sorted lexicographically, not by timestamp", "Breaks the paper's chronological 80/20 split (§4.2); leaks test data into train", "Fixed"],
    ["Associative-embedding push loss counted self-pairs\nbut normalised as if it hadn't", "Biased the reported loss value (gradients unaffected)", "Fixed"],
    ["load_csi_pcap accepted num_antennas but ignored it", "Silently single-antenna regardless of the argument", "Flagged, then replaced in Phase 4"],
]
table(s, rows, 0.62, 2.05, 12.1, [4.6, 5.5, 2.0], size=11.5)
tf = tb(s, 0.62, 5.35, 12.1, 1.6)
para(tf, "The mask-scaling bug is the instructive one: Mask R-CNN already returns probabilities in [0, 1], "
         "but the code treated them as 0–255 pixel values. Training would have run, loss would have decreased, "
         "and the model would have learned nothing — a silent failure, not a crash.", 13.5, SECOND, first=True)

# ─────────────────────────── 5. nexmon verification ───────────────────────────
s = slide()
header(s, "Phase 2 — Verifying the CSI wire format", "Checked against the nexmon_csi source, not assumed")
bullets(s, [
    ("Header length: 18 bytes — confirmed", GOOD, True),
    "The README's field table sums to 19 (a typo: FrameControl listed as 2 bytes next to type uint8). "
    "The authoritative check is their own MATLAB reference parser: HOFFSET = 16 words = byte 60 of the frame, "
    "minus 42 bytes of Ethernet + IP + UDP headers = 18. Matches the code's existing value.",
    ("Byte order: was wrong — found and fixed", CRIT, True),
    "The code unpacked CSI samples as big-endian. The reference parser reads them in native order on a "
    "little-endian machine, so the wire format is little-endian. Every I/Q sample had its two bytes swapped — "
    "no crash, just silently corrupted values, which is worse.",
    ("CSI encoding: plain int16 — correct for this chip", GOOD, True),
    "bcm4358 / bcm4366c0 need a compressed float decoder; the Pi's bcm43455c0 uses straightforward "
    "interleaved int16 pairs, which is what the code already assumed.",
], size=14.5, gap=11)

# ─────────────────────────── 6. Prior findings ───────────────────────────
s = slide()
header(s, "Phase 3 — Finding hard-won prior evidence", "A sibling project had already measured what works on this exact rig")
tf = tb(s, 0.62, 1.95, 12.1, 0.8)
para(tf, "PROJECT_NOTES.md from the Collision-detection project documented empirical results on the same Pi, "
         "same router, same firmware — and several contradicted assumptions already baked into the reproduction.",
     14, SECOND, first=True)
rows = [
    ["Measured finding", "Consequence for our pipeline"],
    ["The 2×2 router alternates TX chains; each presents a different channel.\nMixing them: per-subcarrier std 0.285 → split: 0.162", "Must separate chains before use.\nn_tot = 2, not 1"],
    ["Guard/pilot/DC subcarriers hold ~94% of total power and are constant", "Must delete them before averaging"],
    ["Amplitude coherence 0.826 vs complex/phase 0.258", "Use amplitude; discard phase"],
    ["Packet jitter: 2.85 ms median, gaps to 35 ms", "Must resample to a uniform grid"],
    ["SHARP's phase sanitization degraded coherence 0.933 → 0.363 here", "Do not use it on this hardware"],
]
table(s, rows, 0.62, 2.85, 12.1, [7.1, 5.0], size=11.5)
tf = tb(s, 0.62, 5.75, 12.1, 1.2)
para(tf, "None of these steps existed in the reproduction's loader. Running our data through it as-written would have "
         "reproduced exactly the failure modes this document warns about.", 13.5, SECOND, first=True)

# ─────────────────────────── 7. The port ───────────────────────────
s = slide()
header(s, "Phase 4 — Rebuilding the CSI front end", "Ported the proven recipe into data/csi_pcap_loader.py")
rows = [
    ["#", "Step", "Why"],
    ["1", "Parse with csiread.Nexmon", "Replaces hand-rolled byte parsing with a tested library"],
    ["2", "Filter to one source MAC", "Otherwise CSI from every device on the channel is mixed together"],
    ["3", "Drop all-zero packets", "Corrupt frames would poison a window"],
    ["4", "Split TX chains (KMeans on amplitude profile)", "Chain identity isn't in the metadata; must be recovered"],
    ["5", "Delete guard/pilot/DC subcarriers", "They hold ~94% of power and are constant over time"],
    ["6", "Per-packet amplitude normalisation, discard phase", "Removes AGC steps; phase is CPO-corrupted"],
    ["7", "Resample all streams to one shared uniform grid", "Windowing and Conv1d both assume even spacing"],
]
table(s, rows, 0.62, 2.0, 12.1, [0.5, 5.6, 6.0], size=11.5)
tf = tb(s, 0.62, 5.55, 12.1, 1.4)
para(tf, "Also updated: config.py (2 streams, amplitude-only channels, derived window length), dataset.py, "
         "infer.py, train.py, check_sync.py, requirements.txt.", 13.5, SECOND, first=True)

# ─────────────────────────── 8. The zero-variance bug ───────────────────────────
s = slide()
header(s, "A bug that only testing could catch", "Code that compiles, runs, and produces perfectly useless numbers")
tf = tb(s, 0.62, 2.0, 5.7, 4.6)
para(tf, "The symptom", 16, INK, bold=True, first=True, space_after=8)
para(tf, "First run of the new loader on real data:", 13.5, SECOND, space_after=8)
para(tf, "stream 0:  mean=1.0000  std=0.0000\nstream 1:  mean=1.0000  std=0.0000",
     13, CRIT, space_after=12)
para(tf, "Both streams were exactly constant. No error, no warning — just a signal with zero information.",
     13.5, SECOND, space_after=0)

tf = tb(s, 6.7, 2.0, 6.0, 4.6)
para(tf, "The cause", 16, INK, bold=True, first=True, space_after=8)
para(tf, "Each packet was normalised by its own mean across subcarriers, then collapsed to a scalar by taking "
         "the mean across subcarriers again. The mean of x / mean(x) is exactly 1, always.", 13.5, SECOND, space_after=12)
para(tf, "The fix", 16, INK, bold=True, space_after=8)
para(tf, "After normalising away packet power, the information left is the shape of the frequency response — "
         "so collapse with the standard deviation across subcarriers, not the mean.", 13.5, SECOND, space_after=12)
para(tf, "stream 0:  mean=1.9516  std=0.0796\nstream 1:  mean=1.9671  std=0.0891", 13, GOOD)

# ─────────────────────────── 9. Pipeline evidence ───────────────────────────
s = slide()
header(s, "Validating the front end on real data", "poses.pcap — 100,000 packets over 78.6 s")
metrics = [
    ("90.6%", "packets kept after\nMAC filtering", S1),
    ("0.93", "correlation between the\ntwo recovered TX chains", GOOD),
    ("300.0 Hz", "uniform resampled rate\n(23,566 samples)", S1),
    ("0", "NaN or Inf values\nin the output tensor", GOOD),
]
for i, (big, lab, col) in enumerate(metrics):
    x = 0.62 + i * 3.07
    tf = tb(s, x, 2.35, 2.9, 1.9)
    para(tf, big, 40, col, bold=True, first=True, space_after=4)
    para(tf, lab, 12.5, MUTED, space_after=0)
tf = tb(s, 0.62, 4.55, 12.1, 2.2)
para(tf, "Why the 0.93 matters", 15.5, INK, bold=True, first=True, space_after=8)
para(tf, "The two streams are recovered by unsupervised clustering — nothing guarantees the split is meaningful. "
         "But both chains observe the same room, so a correct split should produce two highly correlated signals. "
         "A near-zero correlation would have meant the clustering was splitting noise. 0.93 is strong evidence the "
         "TX-chain separation is doing what it claims.", 13.5, SECOND, space_after=10)
para(tf, "Windowed variance also fluctuates across the recording (0.010–0.059), consistent with a session that "
         "alternates between movement and stillness — not a flat, dead signal.", 13.5, SECOND)

# ─────────────────────────── 10. Video → labels ───────────────────────────
s = slide()
header(s, "Phase 5 — Video, timestamps, and labels", "The cross-modal supervision path (Fig. 13, 'Vision Processing')")
bullets(s, [
    ("Frame extraction — capture/extract_frames_from_video.py", INK, True),
    "OpenCV reported 2,472 frames for poses.mkv; only 2,468 actually decode — matching the original ffmpeg encode log. "
    "The script indexes off decoded frames, not the unreliable container estimate, so the tail of the recording isn't "
    "silently mislabelled. 823 frames written at 10 fps.",
    ("Sync verification — check_sync.py", INK, True),
    "CSI spans 78.6 s, video spans 82.2 s; tcpdump and ffmpeg started ~3.5 s apart. "
    "786 of 823 frames (95.5%) fall inside CSI coverage; the other 37 get a zero-padded window, handled gracefully.",
    ("Label generation — data/build_labels.py", INK, True),
    "Mask R-CNN + HRNet-w32 over all 823 frames on CPU (~1 hour). Every frame produced a valid person detection; "
    "13.75 of 17 keypoints visible on average.",
], size=14, gap=11)

# ─────────────────────────── 11. Environment ───────────────────────────
s = slide()
header(s, "Phase 6 — Getting the stack to install", "Five interlocking dependency conflicts, resolved in order")
rows = [
    ["Problem", "Resolution"],
    ["No Python env had torch / mmpose / mmcv at all", "Created a dedicated venv"],
    ["mim broke: setuptools ≥ 77 removed pkg_resources", "Pinned setuptools 70.0.0"],
    ["No prebuilt mmcv wheel for torch 2.14", "Pinned torch 2.1.0 + torchvision 0.16.0 (CPU)"],
    ["torch 2.1 built against NumPy 1.x; NumPy 2.4 installed", "Pinned numpy < 2 (verified torch.from_numpy works)"],
    ["chumpy (mmpose dep) can't build — legacy setup.py", "Installed wheel + --no-build-isolation"],
    ["mmdet 3.3 needs mmcv < 2.2; mmpose 1.3 needs mmdet < 3.3", "Pinned mmcv 2.1.0 + mmdet 3.2.0"],
]
table(s, rows, 0.62, 2.05, 12.1, [6.3, 5.8], size=12)
tf = tb(s, 0.62, 5.65, 12.1, 1.3)
para(tf, "Verified end state: every package imports together, mmpose.apis loads, and HRNet-w32 COCO weights "
         "were fetched through mim rather than a guessed checkpoint URL.", 13.5, SECOND, first=True)

# ─────────────────────────── 12. Training ───────────────────────────
s = slide()
header(s, "Phase 7 — Training", "50 epochs, CPU only, 658 train / 165 held-out test frames")
picture(s, "loss_curves.png", 0.62, 1.85, 7.6, 5.1)
tf = tb(s, 8.5, 2.1, 4.3, 4.6)
para(tf, "Reading the curves", 15.5, INK, bold=True, first=True, space_after=10)
para(tf, "Both losses show the same healthy shape: a steep drop in the first few epochs, then a slow decline "
         "into a noisy plateau.", 13, SECOND, space_after=10)
para(tf, "Mask loss is large in absolute terms because Eqn. 12 is an unnormalised sum over pixels, not an "
         "average — the trend matters, not the magnitude.", 13, SECOND, space_after=10)
para(tf, "The plateau from roughly epoch 20 onward says the model has extracted most of what this "
         "single session supports at this capacity and learning rate.", 13, SECOND, space_after=0)

# ─────────────────────────── 13. Qualitative: pose ───────────────────────────
s = slide()
header(s, "Result — pose, before and after training", "Predicted skeleton in red, vision-branch ground truth in green")
tf = tb(s, 0.62, 1.64, 12.1, 0.4)
para(tf, "2-epoch baseline — the same generic skeleton in every frame, regardless of the actual pose", 13.5, SECOND, bold=True, first=True)
picture(s, "predictions_viz/epoch2_baseline/skeleton_overlay.png", 0.47, 2.0, 12.4, 2.2)
tf = tb(s, 0.62, 4.24, 12.1, 0.4)
para(tf, "After 50 epochs — arms now track the real configuration: wide, together, raised, reaching", 13.5, GOOD, bold=True, first=True)
picture(s, "predictions_viz/full_trained/skeleton_overlay.png", 0.47, 4.6, 12.4, 2.2)
tf = tb(s, 0.62, 6.86, 12.1, 0.5)
para(tf, "Legs and ankles stay weak throughout — they are partly out of frame and move less informatively in this session.",
     12, MUTED, first=True)

# ─────────────────────────── 14. Qualitative: mask ───────────────────────────
s = slide()
header(s, "Result — mask, before and after training", "Same three frames, ground truth beside both checkpoints")
picture(s, "mask_before_after.png", 1.4, 1.8, 10.5, 4.5)
tf = tb(s, 0.62, 6.45, 12.1, 0.8)
para(tf, "At 2 epochs the mask is a fixed blob, identical in every frame. After training it reproduces the raised arm "
         "and the extended arm — the silhouette now responds to the input signal, though it stays softer than the "
         "ground truth.", 13, SECOND, first=True)

# ─────────────────────────── 15. Quantitative ───────────────────────────
s = slide()
header(s, "Phase 8 — Measured performance", "evaluate.py — the paper's Eqn. 15 mask mAP and Eqn. 17 single-person PCK")
picture(s, "metrics_comparison.png", 0.62, 1.8, 8.2, 4.9)
tf = tb(s, 9.1, 2.1, 3.7, 4.6)
para(tf, "Mask improved most", 15, INK, bold=True, first=True, space_after=8)
para(tf, "mAP 0.143 → 0.250 (+75%), and gains at every IoU threshold.", 12.5, SECOND, space_after=12)
para(tf, "Loose PCK was already high", 15, INK, bold=True, space_after=8)
para(tf, "PCK@0.3–0.5 barely moved: face, torso and hips hardly shift between poses, so a near-average "
         "guess already passed a loose threshold.", 12.5, SECOND, space_after=12)
para(tf, "PCK@0.1 is the real signal", 15, INK, bold=True, space_after=8)
para(tf, "The tight threshold rose 41% — this is the metric that punishes a frozen skeleton, and it moved most.",
     12.5, SECOND, space_after=0)

# ─────────────────────────── 16. Honesty ───────────────────────────
s = slide()
header(s, "What these numbers do and don't say", "Stated plainly, so the results aren't over-read")
bullets(s, [
    ("They are held-out, not fitted", GOOD, True),
    "Evaluated on the chronologically last 20% of the session, never seen during training.",
    ("They are one session, one person, one room", CRIT, True),
    "No claim of generalisation. The paper used 12 scenes and 510,894 frames; this is 823 frames from a single recording. "
    "Cross-session and cross-room behaviour is untested.",
    ("Mask AP@0.80 is exactly 0.000 — before and after", CRIT, True),
    "The model never produced a near-pixel-perfect mask on any test frame. 'Improved' is not 'accurate'.",
    ("Pose uses single-person PCK, not the paper's multi-person OKS mAP", MUTED, True),
    "OKS mAP needs associative-embedding decoding to separate individuals, which this codebase never implements — "
    "it computes L_group as a training loss but never decodes it back into people.",
    ("Not comparable to the paper's Table 2", MUTED, True),
    "Different sensing hardware, different scale, different metric for pose. These numbers measure our pipeline "
    "against itself, not against SiWiS.",
], size=13.5, gap=9)

# ─────────────────────────── 17. Next steps ───────────────────────────
s = slide()
header(s, "Where this goes next", "Ordered by expected value per unit of effort")
rows = [
    ["Priority", "Action", "Rationale"],
    ["1", "Collect more sessions — vary person, clothing,\nroom, distance, approach angle", "823 frames from one session is the binding constraint,\nnot model capacity"],
    ["2", "Evaluate across sessions, not just within one", "The sibling project measured a large drop from\nwithin-session to leave-one-session-out"],
    ["3", "Longer / scheduled training, mask-branch capacity", "Loss plateaued but mask precision is clearly the\nweaker of the two heads"],
    ["4", "Preserve per-subcarrier structure (e.g. PCA)\ninstead of one scalar per packet", "The current front end discards frequency-selective\ninformation that may carry pose detail"],
    ["5", "Implement associative-embedding decoding", "Required for genuine multi-person support and\nthe paper's OKS mAP metric"],
]
table(s, rows, 0.62, 2.05, 12.1, [1.0, 5.5, 5.6], size=11.5)

# ─────────────────────────── 18. Artifacts ───────────────────────────
s = slide()
header(s, "What was built", "New and substantially rewritten components")
left = [
    ("Rewritten", INK, True),
    "data/csi_pcap_loader.py — csiread parsing, MAC filter, TX-chain split, guard-subcarrier removal, amplitude extraction, resampling",
    "config.py — stream count, amplitude channels, derived window length",
    "data/dataset.py, train.py, infer.py, check_sync.py — real-valued signal path and new CLI",
]
right = [
    ("New", INK, True),
    "capture/extract_frames_from_video.py — timestamped frame extraction from recorded video",
    "evaluate.py — mask mAP (Eqn. 15) and single-person PCK (Eqn. 17) on the held-out split",
    "visualize_predictions.py — mask/pose grids, skeleton overlays, per-joint channel maps",
    "make_deck.py — regenerates this deck",
]
tf = tb(s, 0.62, 2.0, 5.9, 4.8)
for i, it in enumerate(left):
    text, color, bold = (it if isinstance(it, tuple) else (it, SECOND, False))
    para(tf, text, 13, color, bold=bold, first=(i == 0), space_after=10)
tf = tb(s, 6.9, 2.0, 5.8, 4.8)
for i, it in enumerate(right):
    text, color, bold = (it if isinstance(it, tuple) else (it, SECOND, False))
    para(tf, text, 13, color, bold=bold, first=(i == 0), space_after=10)
tf = tb(s, 0.62, 6.15, 12.1, 0.9)
para(tf, "Fixed defects: mask label double-scaling · chronological frame ordering · associative-embedding push-loss "
         "normalisation · CSI byte order · zero-variance feature collapse", 12.5, MUTED, first=True)

# ─────────────────────────── 19. Closing ───────────────────────────
s = slide()
header(s, "Summary", None)
bullets(s, [
    ("The pipeline works end to end.", INK, True),
    "WiFi CSI from a commodity Raspberry Pi goes in; a human body mask and a 17-joint skeleton come out. "
    "Every stage was validated on real captured data, not assumed.",
    ("Five real defects were found and fixed", INK, True),
    "— three by reading the paper against the code, one by checking the firmware's own reference parser, "
    "and one that only appeared when the code was actually run on real data.",
    ("The model demonstrably learns from the signal", INK, True),
    "Predictions moved from a fixed average pose to input-conditioned masks and skeletons, confirmed both visually "
    "and by held-out metrics (mask mAP +75%, tight-threshold PCK +41%).",
    ("The limits are known and stated", INK, True),
    "One session, one person, no phase coherence, no cross-room evidence, and no near-perfect masks. "
    "More data is the next binding constraint — not architecture.",
], size=14.5, gap=12)

out = "SiWiS_Reproduction_Summary.pptx"
prs.save(out)
print(f"Saved {out} — {len(prs.slides.__iter__.__self__._sldIdLst)} slides")
