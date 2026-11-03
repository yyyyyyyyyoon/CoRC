"""Shared paths and fixed experimental settings for the reproduction code."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "wsdream"
OUTPUT_DIR = ROOT / "outputs"
# The small results the paper is built from, versioned; `outputs/` holds the
# per-entry predictions behind them and is not.
RESULTS_DIR = ROOT / "results" / "paper"

RT_PATH = DATASET_DIR / "rtMatrix.txt"
TP_PATH = DATASET_DIR / "tpMatrix.txt"
USERLIST_PATH = DATASET_DIR / "userlist.txt"
WSLIST_PATH = DATASET_DIR / "wslist.txt"

# The MEC side of the dataset, which is not redistributed here. Read only by
# `pipeline.build_selection_dataset`; see data/shanghai_telecom/README.md.
MEC_DIR = Path(os.environ.get(
    "CORC_MEC_DIR", ROOT / "data" / "shanghai_telecom" / "files"))
MEC_SAMPLES = MEC_DIR / "base_samples_candidates_scored.csv"
MEC_LOCATIONS = MEC_DIR / "server_locations.csv"
MEC_STATUS = MEC_DIR / "server_status.csv"

MISSING_VALUE = -1.0

SEED = 42

# Validation/test are held out before nested density masks are sampled.
SPLIT_VERSION = "cerp_qos_shared_v2"
SPLIT_SEEDS = [42, 2026, 3407]

# Use entries observed in both matrices so RT and TP share one partition.
SHARED_OBSERVATION_MASK = True
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10
META_FOLDS = 5

DENSITIES = [0.05, 0.10, 0.15, 0.20]
QOS_TYPES = ["rt", "tp"]

# Optional stratified sample used by diagnostic analyses.
TEST_SAMPLE_SIZE = 30_000
STRATA_QUANTILES = 4

# Collaborative-neighbour settings.
TOP_K_NEIGHBORS = 20
MIN_CO_RATED = 2
SIM_SHRINKAGE = 10.0

# Context-aware evidence weighting. Multipliers are applied to positive CF
# similarities before top-K selection.
GEO_SAME_COUNTRY_BONUS = 0.50
GEO_DISTANCE_BONUS = 0.50
GEO_DISTANCE_SCALE_KM = 3000.0
AS_SAME_BONUS = 1.00

# User-side AS weighting is excluded because its sparse groups rarely change
# the reachable neighbours; service-side AS weighting remains informative.
CONFLICT_PROBES = ("ucf", "scf", "geo_ucf", "geo_scf", "as_scf")
CONFLICT_DEFINITION_VERSION = "v2_no_as_ucf"
