from pathlib import Path
from collections import defaultdict
import argparse
import csv
import heapq

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from preprocessing import add_normalized_columns
from blocking import create_blocking_keys


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = ROOT / "dataset"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = ROOT / "output"

S1_FILE = TEST_DIR / "test_source1.tsv"
S2_FILE = TEST_DIR / "test_source2.tsv"
S3_FILE = TEST_DIR / "test_source3.tsv"

MODEL_PATH = OUTPUT_DIR / "matcher_model.joblib"
THRESHOLD_PATH = OUTPUT_DIR / "best_threshold.txt"

CANDIDATE_FILE = OUTPUT_DIR / "candidate_pairs.tsv"
RESULT_FILE = OUTPUT_DIR / "matching_results.tsv"


# ============================================================
# PERFORMANCE SETTINGS
# ============================================================

CHUNK_SIZE = 500

# Maximum number of candidates scored for one S2/S3 record.
#
# This is NOT a hard cutoff applied blindly.
# Strong candidates are retained first.
MAX_CANDIDATES_PER_RECORD = 150

# Number of weaker address-only candidates allowed
# when they are competing with other candidates.
MAX_WEAK_CANDIDATES = 50


# ============================================================
# EXACT FEATURES USED DURING TRAINING
# ============================================================

FEATURE_COLUMNS = [
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort",
    "name_token_set",
    "name_jaccard",
    "name_exact",
    "name_length_difference",

    "address_ratio",
    "address_partial_ratio",
    "address_token_sort",
    "address_token_set",
    "address_jaccard",
    "address_exact",
    "address_number_overlap",
    "address_token_overlap",
    "address_length_difference",

    "country_exact",
    "s1_address_missing",
    "candidate_address_missing",
]


# ============================================================
# BLOCKING KEY WEIGHTS
# ============================================================

# Stronger keys receive larger evidence scores.
#
# The key names correspond to blocking.py.

KEY_WEIGHTS = {
    "exact_name": 10,

    "name_tokens": 9,
    "name_tokens_13": 8,
    "name_long_token": 7,

    "name_number": 8,
    "name_address": 8,
    "name_token_address": 7,

    "number_address_pair": 7,
    "number_address": 6,

    "address_pair": 5,
    "address_pair_unordered": 4,
}


# ============================================================
# TEXT / FUZZ HELPERS
# ============================================================

def safe_text(value):
    if value is None:
        return ""

    return str(value)


def ratio(a, b):
    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b) / 100.0


def partial_ratio(a, b):
    if not a or not b:
        return 0.0

    return fuzz.partial_ratio(a, b) / 100.0


def token_sort(a, b):
    if not a or not b:
        return 0.0

    return fuzz.token_sort_ratio(a, b) / 100.0


def token_set(a, b):
    if not a or not b:
        return 0.0

    return fuzz.token_set_ratio(a, b) / 100.0


def jaccard_sets(set_a, set_b):

    if not set_a or not set_b:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b

    if not union:
        return 0.0

    return len(intersection) / len(union)


def overlap_sets(set_a, set_b):

    if not set_a or not set_b:
        return 0.0

    intersection = set_a & set_b

    if not intersection:
        return 0.0

    return len(intersection) / min(
        len(set_a),
        len(set_b),
    )


def exact(a, b):

    if not a or not b:
        return 0

    return int(a == b)


def length_difference(a, b):

    return abs(
        len(a) - len(b)
    )


# ============================================================
# S1 STORE
# ============================================================

class S1Store:

    def __init__(self, s1):

        print()
        print("Preparing optimized S1 store...")

        self.entity_ids = (
            s1["entity_id"]
            .to_numpy(dtype=object)
        )

        self.names = (
            s1["name_matching"]
            .to_numpy(dtype=object)
        )

        self.addresses = (
            s1["address_normalized"]
            .to_numpy(dtype=object)
        )

        self.countries = (
            s1["country_normalized"]
            .to_numpy(dtype=object)
        )

        self.address_numbers = (
            s1["address_numbers"]
            .to_numpy(dtype=object)
        )

        self.address_tokens = (
            s1["address_tokens"]
            .to_numpy(dtype=object)
        )

        # Lazy caches.
        self.name_sets = {}
        self.address_sets = {}
        self.address_token_sets = {}
        self.number_sets = {}

        print(
            "S1 optimized store ready."
        )

    def get_name_set(self, pos):

        result = self.name_sets.get(pos)

        if result is not None:
            return result

        value = safe_text(
            self.names[pos]
        )

        if value:
            result = frozenset(
                value.split()
            )
        else:
            result = frozenset()

        self.name_sets[pos] = result

        return result

    def get_address_set(self, pos):

        result = self.address_sets.get(pos)

        if result is not None:
            return result

        value = safe_text(
            self.addresses[pos]
        )

        if value:
            result = frozenset(
                value.split()
            )
        else:
            result = frozenset()

        self.address_sets[pos] = result

        return result

    def get_address_token_set(self, pos):

        result = self.address_token_sets.get(pos)

        if result is not None:
            return result

        tokens = self.address_tokens[pos]

        if not isinstance(tokens, list):
            tokens = []

        result = frozenset(tokens)

        self.address_token_sets[pos] = result

        return result

    def get_number_set(self, pos):

        result = self.number_sets.get(pos)

        if result is not None:
            return result

        numbers = self.address_numbers[pos]

        if not isinstance(numbers, list):
            numbers = []

        result = frozenset(
            str(x)
            for x in numbers
        )

        self.number_sets[pos] = result

        return result


# ============================================================
# S1 BLOCKING INDEX
# ============================================================

MAX_BLOCK_SIZE = 500


def build_s1_index(s1):

    print()
    print(
        "Building memory-safe S1 blocking index..."
    )

    print(
        f"Maximum block size: "
        f"{MAX_BLOCK_SIZE:,}"
    )

    index = {}

    total_keys_generated = 0
    oversized_blocks = 0

    total_rows = len(s1)

    for pos, (_, row) in enumerate(
        s1.iterrows()
    ):

        if pos % 100_000 == 0:

            print(
                f"  Indexed "
                f"{pos:,}/"
                f"{total_rows:,} S1 rows | "
                f"unique keys = "
                f"{len(index):,}"
            )

        keys = create_blocking_keys(
            row
        )

        for key in keys:

            total_keys_generated += 1

            if key not in index:

                index[key] = [pos]

                continue

            values = index[key]

            # Already oversized.
            if values is None:
                continue

            if len(values) >= MAX_BLOCK_SIZE:

                index[key] = None

                oversized_blocks += 1

                continue

            values.append(pos)

    print()
    print(
        "Filtering oversized blocks..."
    )

    filtered_index = {
        key: values
        for key, values in index.items()
        if values is not None
    }

    print()
    print(
        "S1 blocking index ready."
    )

    print(
        f"Total keys generated: "
        f"{total_keys_generated:,}"
    )

    print(
        f"Raw unique keys:      "
        f"{len(index):,}"
    )

    print(
        f"Usable blocks:        "
        f"{len(filtered_index):,}"
    )

    print(
        f"Oversized blocks:     "
        f"{oversized_blocks:,}"
    )

    return filtered_index


# ============================================================
# CANDIDATE EVIDENCE
# ============================================================

def get_ranked_candidates(
    source_row,
    s1_index,
):
    """
    Generate candidates and rank them according to
    how many / how strong blocking keys they share.

    This prevents expensive fuzzy matching from being
    applied to thousands of weak candidates.

    Returns:
        list of S1 positions
    """

    candidate_scores = {}

    keys = create_blocking_keys(
        source_row
    )

    for key in keys:

        values = s1_index.get(key)

        if not values:
            continue

        key_type = key[0]

        weight = KEY_WEIGHTS.get(
            key_type,
            1,
        )

        for pos in values:

            old = candidate_scores.get(
                pos
            )

            if old is None:

                candidate_scores[pos] = (
                    weight,
                    1,
                )

            else:

                old_score, old_count = old

                candidate_scores[pos] = (
                    old_score + weight,
                    old_count + 1,
                )

    if not candidate_scores:
        return []

    # --------------------------------------------------------
    # Separate strong and weak candidates.
    # --------------------------------------------------------

    strong = []
    weak = []

    for pos, (
        score,
        count,
    ) in candidate_scores.items():

        # Multiple independent blocking signals
        # are strong evidence.

        if (
            score >= 7
            or count >= 2
        ):

            strong.append(
                (
                    score,
                    count,
                    pos,
                )
            )

        else:

            weak.append(
                (
                    score,
                    count,
                    pos,
                )
            )

    # --------------------------------------------------------
    # Sort strong candidates.
    # --------------------------------------------------------

    strong.sort(
        key=lambda x: (
            x[0],
            x[1],
        ),
        reverse=True,
    )

    # --------------------------------------------------------
    # Keep strong candidates.
    # --------------------------------------------------------

    selected = strong[
        :MAX_CANDIDATES_PER_RECORD
    ]

    # --------------------------------------------------------
    # Fill remaining slots with weak candidates.
    # --------------------------------------------------------

    remaining = (
        MAX_CANDIDATES_PER_RECORD
        - len(selected)
    )

    if remaining > 0 and weak:

        weak.sort(
            key=lambda x: (
                x[0],
                x[1],
            ),
            reverse=True,
        )

        weak_limit = min(
            remaining,
            MAX_WEAK_CANDIDATES,
        )

        selected.extend(
            weak[:weak_limit]
        )

    return [
        item[2]
        for item in selected
    ]


# ============================================================
# FAST FEATURE CALCULATION
# ============================================================

def build_fast_feature_row(
    s1_store,
    s1_pos,
    source_row,
):

    # --------------------------------------------------------
    # S1 values
    # --------------------------------------------------------

    s1_name = safe_text(
        s1_store.names[s1_pos]
    )

    s1_address = safe_text(
        s1_store.addresses[s1_pos]
    )

    s1_country = safe_text(
        s1_store.countries[s1_pos]
    )

    # --------------------------------------------------------
    # Source values
    # --------------------------------------------------------

    candidate_name = safe_text(
        source_row.get(
            "name_matching",
            "",
        )
    )

    candidate_address = safe_text(
        source_row.get(
            "address_normalized",
            "",
        )
    )

    candidate_country = safe_text(
        source_row.get(
            "country_normalized",
            "",
        )
    )

    # --------------------------------------------------------
    # Source sets already computed.
    # --------------------------------------------------------

    source_name_set = (
        source_row["_name_set"]
    )

    source_address_set = (
        source_row["_address_set"]
    )

    source_address_token_set = (
        source_row["_address_token_set"]
    )

    source_number_set = (
        source_row["_number_set"]
    )

    # --------------------------------------------------------
    # S1 sets
    # --------------------------------------------------------

    s1_name_set = (
        s1_store.get_name_set(
            s1_pos
        )
    )

    s1_address_set = (
        s1_store.get_address_set(
            s1_pos
        )
    )

    s1_address_token_set = (
        s1_store.get_address_token_set(
            s1_pos
        )
    )

    s1_number_set = (
        s1_store.get_number_set(
            s1_pos
        )
    )

    # --------------------------------------------------------
    # RETURN EXACT 19 FEATURES
    # --------------------------------------------------------

    return [
        ratio(
            s1_name,
            candidate_name,
        ),

        partial_ratio(
            s1_name,
            candidate_name,
        ),

        token_sort(
            s1_name,
            candidate_name,
        ),

        token_set(
            s1_name,
            candidate_name,
        ),

        jaccard_sets(
            s1_name_set,
            source_name_set,
        ),

        exact(
            s1_name,
            candidate_name,
        ),

        length_difference(
            s1_name,
            candidate_name,
        ),

        ratio(
            s1_address,
            candidate_address,
        ),

        partial_ratio(
            s1_address,
            candidate_address,
        ),

        token_sort(
            s1_address,
            candidate_address,
        ),

        token_set(
            s1_address,
            candidate_address,
        ),

        jaccard_sets(
            s1_address_set,
            source_address_set,
        ),

        exact(
            s1_address,
            candidate_address,
        ),

        overlap_sets(
            s1_number_set,
            source_number_set,
        ),

        overlap_sets(
            s1_address_token_set,
            source_address_token_set,
        ),

        length_difference(
            s1_address,
            candidate_address,
        ),

        int(
            bool(s1_country)
            and
            s1_country
            == candidate_country
        ),

        int(
            not bool(s1_address)
        ),

        int(
            not bool(candidate_address)
        ),
    ]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print()
    print(
        "Loading model..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        threshold = float(
            f.read().strip()
        )

    print(
        "Model loaded successfully."
    )

    print(
        f"Threshold = "
        f"{threshold}"
    )

    return (
        model,
        threshold,
    )


# ============================================================
# PREPARE SOURCE CHUNK
# ============================================================

def prepare_source_records(
    source_chunk,
):

    records = source_chunk.to_dict(
        orient="records"
    )

    for row in records:

        name = safe_text(
            row.get(
                "name_matching",
                "",
            )
        )

        address = safe_text(
            row.get(
                "address_normalized",
                "",
            )
        )

        row["_name_set"] = (
            frozenset(
                name.split()
            )
            if name
            else frozenset()
        )

        row["_address_set"] = (
            frozenset(
                address.split()
            )
            if address
            else frozenset()
        )

        tokens = row.get(
            "address_tokens",
            [],
        )

        if not isinstance(
            tokens,
            list,
        ):
            tokens = []

        row["_address_token_set"] = (
            frozenset(tokens)
        )

        numbers = row.get(
            "address_numbers",
            [],
        )

        if not isinstance(
            numbers,
            list,
        ):
            numbers = []

        row["_number_set"] = (
            frozenset(
                str(x)
                for x in numbers
            )
        )

    return records


# ============================================================
# PROCESS CHUNK
# ============================================================

def process_chunk(
    source_chunk,
    s1_store,
    s1_index,
    model,
    threshold,
):

    source_records = (
        prepare_source_records(
            source_chunk
        )
    )

    candidate_rows = []
    feature_rows = []

    # --------------------------------------------------------
    # Candidate generation + feature creation
    # --------------------------------------------------------

    for row_number, row in enumerate(
        source_records,
        start=1,
    ):

        candidate_positions = (
            get_ranked_candidates(
                row,
                s1_index,
            )
        )

        if not candidate_positions:
            continue

        candidate_country = safe_text(
            row.get(
                "country_normalized",
                "",
            )
        )

        source_id = row[
            "entity_id"
        ]

        for s1_pos in candidate_positions:

            # Country safety check.
            if (
                s1_store.countries[
                    s1_pos
                ]
                != candidate_country
            ):
                continue

            candidate_rows.append(
                (
                    s1_store.entity_ids[
                        s1_pos
                    ],
                    source_id,
                )
            )

            feature_rows.append(
                build_fast_feature_row(
                    s1_store,
                    s1_pos,
                    row,
                )
            )

    if not feature_rows:

        return (
            [],
            [],
        )

    # --------------------------------------------------------
    # NumPy array directly.
    #
    # Avoids constructing a pandas DataFrame for every chunk.
    # --------------------------------------------------------

    feature_array = np.asarray(
        feature_rows,
        dtype=np.float32,
    )

    probabilities = (
        model.predict_proba(
            feature_array
        )[:, 1]
    )

    # --------------------------------------------------------
    # Final matches.
    # --------------------------------------------------------

    match_rows = []

    for index, probability in enumerate(
        probabilities
    ):

        if probability >= threshold:

            match_rows.append(
                candidate_rows[index]
            )

    return (
        candidate_rows,
        match_rows,
    )


# ============================================================
# PROCESS SOURCE
# ============================================================

def process_source(
    source_path,
    source_name,
    s1_store,
    s1_index,
    model,
    threshold,
    candidate_map,
    matches_by_s1,
    sample_rows=None,
):

    print()
    print("=" * 60)
    print(
        f"Processing {source_name}"
    )
    print("=" * 60)

    total_rows = 0
    total_candidates = 0
    total_matches = 0

    reader = pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):

        # ----------------------------------------------------
        # SAMPLE MODE
        # ----------------------------------------------------

        if sample_rows is not None:

            remaining = (
                sample_rows
                - total_rows
            )

            if remaining <= 0:
                break

            if len(chunk) > remaining:

                chunk = chunk.iloc[
                    :remaining
                ]

        total_rows += len(chunk)

        print()
        print(
            f"{source_name}: "
            f"starting chunk "
            f"{chunk_number}"
        )

        print(
            f"{source_name}: "
            f"rows = "
            f"{total_rows:,}"
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        chunk = add_normalized_columns(
            chunk
        )

        # ----------------------------------------------------
        # Candidate generation + model
        # ----------------------------------------------------

        candidates, matches = (
            process_chunk(
                chunk,
                s1_store,
                s1_index,
                model,
                threshold,
            )
        )

        # ----------------------------------------------------
        # Store candidate IDs by S1.
        # ----------------------------------------------------

        for (
            s1_id,
            source_id,
        ) in candidates:

            candidate_map[
                s1_id
            ].add(
                source_id
            )

        # ----------------------------------------------------
        # Store matches.
        # ----------------------------------------------------

        for (
            s1_id,
            source_id,
        ) in matches:

            matches_by_s1[
                s1_id
            ].add(
                source_id
            )

        total_candidates += len(
            candidates
        )

        total_matches += len(
            matches
        )

        print(
            f"{source_name}: "
            f"finished chunk "
            f"{chunk_number}"
        )

        print(
            f"    candidates = "
            f"{len(candidates):,}"
        )

        print(
            f"    matches = "
            f"{len(matches):,}"
        )

        print(
            f"    total rows = "
            f"{total_rows:,}"
        )

        # ----------------------------------------------------
        # Sample complete
        # ----------------------------------------------------

        if (
            sample_rows is not None
            and
            total_rows >= sample_rows
        ):

            break

    print()
    print(
        f"{source_name} finished."
    )

    print(
        f"Rows processed: "
        f"{total_rows:,}"
    )

    print(
        f"Candidate pairs: "
        f"{total_candidates:,}"
    )

    print(
        f"Predicted matches: "
        f"{total_matches:,}"
    )

    return (
        total_candidates,
        total_matches,
    )


# ============================================================
# WRITE CANDIDATE FILE
# ============================================================

def write_candidate_file(
    s1,
    candidate_map,
):

    print()
    print(
        "Writing candidate_pairs.tsv..."
    )

    with open(
        CANDIDATE_FILE,
        "w",
        newline="",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as file:

        writer = csv.writer(
            file,
            delimiter="\t",
        )

        writer.writerow(
            [
                "source1_entity_id",
                "candidate_entity_ids",
            ]
        )

        for s1_id in s1[
            "entity_id"
        ]:

            candidate_ids = (
                candidate_map.get(
                    s1_id,
                    set(),
                )
            )

            ordered_ids = sorted(
                candidate_ids
            )

            writer.writerow(
                [
                    s1_id,
                    ",".join(
                        ordered_ids
                    ),
                ]
            )


# ============================================================
# WRITE MATCHING RESULTS
# ============================================================

def write_matching_results(
    s1,
    matches_by_s1,
):

    print()
    print(
        "Writing matching_results.tsv..."
    )

    with open(
        RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as file:

        writer = csv.writer(
            file,
            delimiter="\t",
        )

        writer.writerow(
            [
                "source1_entity_id",
                "matched_entity_ids",
            ]
        )

        for s1_id in s1[
            "entity_id"
        ]:

            matched_ids = (
                matches_by_s1.get(
                    s1_id,
                    set(),
                )
            )

            ordered_ids = sorted(
                matched_ids
            )

            writer.writerow(
                [
                    s1_id,
                    ",".join(
                        ordered_ids
                    ),
                ]
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Number of rows to process "
            "from each S2/S3 source."
        ),
    )

    parser.add_argument(
        "--s1-sample",
        type=int,
        default=None,
        help=(
            "Number of S1 rows to use."
        ),
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print(
        "AMAZON ML 2026"
    )
    print(
        "OPTIMIZED TEST PREDICTION V2"
    )
    print("=" * 60)

    # ========================================================
    # LOAD S1
    # ========================================================

    print()
    print(
        "Loading S1..."
    )

    s1 = pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"S1 rows: "
        f"{len(s1):,}"
    )

    if args.s1_sample is not None:

        s1 = s1.iloc[
            :args.s1_sample
        ].copy()

        print(
            f"S1 sample selected: "
            f"{len(s1):,}"
        )

    # ========================================================
    # NORMALIZE S1
    # ========================================================

    print()
    print(
        "Normalizing S1..."
    )

    s1 = add_normalized_columns(
        s1
    )

    # ========================================================
    # BLOCK INDEX
    # ========================================================

    s1_index = build_s1_index(
        s1
    )

    # ========================================================
    # S1 STORE
    # ========================================================

    s1_store = S1Store(
        s1
    )

    # ========================================================
    # MODEL
    # ========================================================

    model, threshold = (
        load_model()
    )

    # ========================================================
    # OUTPUT STORAGE
    # ========================================================

    candidate_map = defaultdict(
        set
    )

    matches_by_s1 = defaultdict(
        set
    )

    # ========================================================
    # S2
    # ========================================================

    s2_candidates, s2_matches = (
        process_source(
            S2_FILE,
            "S2",
            s1_store,
            s1_index,
            model,
            threshold,
            candidate_map,
            matches_by_s1,
            args.sample,
        )
    )

    # ========================================================
    # S3
    # ========================================================

    s3_candidates, s3_matches = (
        process_source(
            S3_FILE,
            "S3",
            s1_store,
            s1_index,
            model,
            threshold,
            candidate_map,
            matches_by_s1,
            args.sample,
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    total_candidates = (
        s2_candidates
        + s3_candidates
    )

    total_matches = (
        s2_matches
        + s3_matches
    )

    print()
    print("=" * 60)
    print(
        "PREDICTION SUMMARY"
    )
    print("=" * 60)

    print(
        f"Total candidate pairs: "
        f"{total_candidates:,}"
    )

    print(
        f"Total predicted matches: "
        f"{total_matches:,}"
    )

    print(
        f"S1 records with candidates: "
        f"{len(candidate_map):,}"
    )

    print(
        f"S1 records with matches: "
        f"{len(matches_by_s1):,}"
    )

    # ========================================================
    # WRITE OUTPUTS
    # ========================================================

    write_candidate_file(
        s1,
        candidate_map,
    )

    write_matching_results(
        s1,
        matches_by_s1,
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 60)
    print(
        "DONE"
    )
    print("=" * 60)

    print(
        "Candidate file:"
    )

    print(
        CANDIDATE_FILE
    )

    print(
        "Result file:"
    )

    print(
        RESULT_FILE
    )


if __name__ == "__main__":
    main()