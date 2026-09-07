"""
Configuration for the SiWiS reproduction pipeline.

Every value below is either:
  (a) taken directly from the paper, with the section/table it comes
      from cited in the comment, or
  (b) a placeholder for a value the paper describes structurally but
      never publishes a number for (explicitly marked "NOT specified
      in paper"). Group (b) values must be tuned by you; they are not
      derived from the paper and should not be treated as such.
"""

# ---- Hardware / signal geometry ----
# Paper's hardware (§4.1.1): RX1-RX4, "four patch antennas (RX1-4)",
# sampled via a custom ADC at 10 MSps. NOT applicable to this
# reproduction's actual hardware: a Raspberry Pi 4B (bcm43455c0) running
# nexmon_csi against a TP-Link Archer AX10 in 2x2 mode. There are no
# patch antennas and no self-mixing RF front end here.
#
# NUM_SENSING_ANTENNAS below does NOT mean "RX antennas" as in the
# paper -- the Pi has exactly one physical antenna. It means "recovered
# transmit-chain streams": this router alternates between two transmit
# chains per packet, each presenting a different channel to the
# receiver, and they must be split apart in software (KMeans on the
# per-subcarrier amplitude profile -- see data/csi_pcap_loader.py)
# before they can be used as independent input channels. Confirmed
# empirically load-bearing on this hardware
# (Collision-detection/PROJECT_NOTES.md §4.3): averaging the two chains
# together was "a large artificial noise source."
NUM_SENSING_ANTENNAS = 2          # recovered TX-chain streams (n_tot), not RX antennas -- see note above
ADC_SAMPLE_RATE_HZ = 10_000_000   # descriptive only, from the paper's hardware; not read anywhere in this reproduction's code path
WIFI_CHANNEL = 36                 # matches this project's working router config (80MHz, chanspec 0xe02a); set to whatever channel you actually capture on

# ---- Heatmap output resolution (paper-specified) ----
HEATMAP_H = 48                    # "resolution of 48 x 64" (§4.1.2)
HEATMAP_W = 64

# ---- Dataset composition (paper-specified) ----
MAX_PEOPLE = 4                    # Table 1: "Person #" column spans 1-4
TRAIN_FRACTION = 0.8              # "first 80% of the data for training and remaining 20% for testing" (§4.2)

# ---- Loss weighting (paper-specified where given) ----
ASSOC_EMBED_LOSS_WEIGHT = 1e-3    # "associative embedding loss is weighted by a factor of 1e-3
                                   #  relative to the MSE loss" (§4.1.2)
LEARNING_RATE = 1e-3              # "learning rate of 1e-3" (§4.1.2)

# alpha1, alpha2 (Eqn. 12, BCE vs Dice weight) and lambda (Eqn. 14,
# mask vs pose weight) are referenced only as symbols in the paper;
# no numeric values are published anywhere in the text.
MASK_BCE_WEIGHT_ALPHA1 = 1.0      # NOT specified numerically in paper (Eqn. 12, alpha_1)
MASK_DICE_WEIGHT_ALPHA2 = 1.0     # NOT specified numerically in paper (Eqn. 12, alpha_2)
POSE_LOSS_LAMBDA = 1.0            # NOT specified numerically in paper (Eqn. 14, lambda)

# ---- Signal duration (paper-specified operating point) ----
# Fig. 20: precision improves with signal duration; "the rate of
# improvement gradually slows down when the signal duration exceeds
# 1.7 seconds" -- used here as the sliding-window length.
SIGNAL_DURATION_SEC = 1.7
VIDEO_FPS = 10                    # "web camera ... capture video frames at 10 FPS" (§4.2)

# Target rate (Hz) each recovered TX-chain stream is resampled onto a
# uniform time grid at (data/csi_pcap_loader.py). Real packet arrival is
# bursty (jitter measured at 2.85ms median, 35ms gaps, on this
# hardware), and everything downstream (windowing, Conv1d) implicitly
# assumes even spacing. 300Hz is this project's tuned value
# (Collision-detection/PROJECT_NOTES.md §4.5): enough Nyquist headroom
# for walking-speed motion, without sacrificing temporal resolution.
# NOT from the SiWiS paper -- its hardware samples continuously at a
# fixed 10 MSps and never needs to resample.
RESAMPLE_RATE_HZ = 300

# Signal samples per training window, derived from the two values above
# instead of guessed -- with uniform resampling, window length is
# deterministic rather than an arbitrary tunable.
SIGNAL_SEQ_LEN_SAMPLES = round(SIGNAL_DURATION_SEC * RESAMPLE_RATE_HZ)

# ---- Signal encoder / decoder capacity (NOT specified in paper) ----
# The paper describes the *topology* only (conv -> maxpool -> N x
# self-attention blocks -> conv -> maxpool -> linear, Fig. 13, §3.2.1)
# It never publishes channel widths, block count N, head count, or the
# bottleneck reduction scale X ("With a reduction scale of X ...",
# §3.2.1 -- X itself is left symbolic in the text).
SIGNAL_IN_CHANNELS = NUM_SENSING_ANTENNAS   # amplitude only (real-valued), 1 channel per recovered TX-chain stream -- phase dropped, see data/csi_pcap_loader.py point 6
CONV_CHANNELS = 64                # NOT specified in paper
EMBED_DIM_C = 256                 # "C" in §3.2.2 ("C x 1 x 1"); value not specified
NUM_SELF_ATTN_BLOCKS = 2          # "N" in Fig. 13 signal encoder; value not specified
NUM_CROSS_ATTN_BLOCKS = 2         # "N" in Fig. 13 decoders; value not specified
ATTN_HEADS = 8                    # NOT specified in paper
BOTTLENECK_REDUCTION_X = 4        # "X" in §3.2.1; value not specified

# ---- Decoder pixel-embedding grid (NOT specified in paper) ----
# "we initialize h x w trainable pixel embeddings" then upsample to
# h' x w' = HEATMAP_H x HEATMAP_W (§3.2.2). The paper never gives h, w.
PIXEL_EMBED_H = 12
PIXEL_EMBED_W = 16

# ---- Pose estimation keypoints ----
NUM_KEYPOINTS = 17                # COCO keypoint format, produced by HRNet [42] (§4.2)

# ---- Mask segmentation channels ----
NUM_MASK_CHANNELS = 1              # single foreground/background body mask (Fig. 14)