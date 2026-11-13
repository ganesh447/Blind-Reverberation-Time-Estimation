"""File and folder names that the pipeline steps share on disk, and the sample rate they all use."""

FS = 16000  # Hz, every RIR, utterance and feature
RIR_LABELS_CSV = "T30_ground_truth.csv"  # in an RIR folder: columns filename, t30 (seconds)
INDEX_FILE = "dataset.pkl"  # in a feature folder: list of (sample_id, n_frames)
DECAY_RATES_DIR = "decayRates"  # in a feature folder: <sample_id>.npy, float32 (mels, frames)
LABELS_DIR = "T30"  # in a feature folder: <sample_id>.npy, the T30 label in seconds
LEGACY_LABELS_DIR = "T60"  # the same folder in feature folders made before the rename
