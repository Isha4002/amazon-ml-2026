from collections import defaultdict
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

MAX_BLOCK_SIZE = 500


# ============================================================
# NAME HELPERS
# ============================================================

def first_meaningful_tokens(text: str, n: int = 3):
    if not text:
        return []

    tokens = [
        token
        for token in text.split()
        if len(token) >= 3
    ]

    return tokens[:n]


def name_prefix(text: str, length: int = 4) -> str:
    if not text:
        return ""

    return text.replace(" ", "")[:length]


# ============================================================
# ADDRESS HELPERS
# ============================================================

def useful_address_tokens(address_tokens):
    if not isinstance(address_tokens, list):
        return []

    return [
        token
        for token in address_tokens
        if len(token) >= 4
        and not token.isdigit()
    ]


def get_address_numbers(address_numbers):
    if not isinstance(address_numbers, list):
        return []

    return [
        str(number)
        for number in address_numbers
        if str(number).strip()
    ]


# ============================================================
# BLOCKING KEYS
# ============================================================

def create_blocking_keys(row):
    """
    Precision-oriented multi-strategy blocking.

    Blocks:

    1. Exact normalized business name
    2. First two meaningful name tokens
    3. Long single name token
    4. Name prefix + address number
    5. Name prefix + address token
    6. Name token + address token
    7. Address token pair
    8. Address number + address token
    9. Strong address token combinations

    Country is included in every key.
    """

    country = row.get(
        "country_normalized",
        ""
    )

    name = row.get(
        "name_matching",
        ""
    )

    address_numbers = get_address_numbers(
        row.get(
            "address_numbers",
            []
        )
    )

    address_tokens = useful_address_tokens(
        row.get(
            "address_tokens",
            []
        )
    )

    if not country:
        return []

    keys = []

    # ========================================================
    # 1. EXACT NAME
    # ========================================================

    if name:
        keys.append(
            (
                "exact_name",
                country,
                name,
            )
        )

    # ========================================================
    # 2. NAME TOKEN COMBINATION
    # ========================================================

    name_tokens = first_meaningful_tokens(
        name,
        3,
    )

    if len(name_tokens) >= 2:

        # First two tokens
        keys.append(
            (
                "name_tokens",
                country,
                name_tokens[0],
                name_tokens[1],
            )
        )

        # First + third token
        if len(name_tokens) >= 3:
            keys.append(
                (
                    "name_tokens_13",
                    country,
                    name_tokens[0],
                    name_tokens[2],
                )
            )

    elif len(name_tokens) == 1:

        token = name_tokens[0]

        # Only use long tokens because short/common
        # tokens create huge blocks.

        if len(token) >= 6:
            keys.append(
                (
                    "name_long_token",
                    country,
                    token,
                )
            )

    # ========================================================
    # NAME PREFIX
    # ========================================================

    prefix = name_prefix(
        name,
        4,
    )

    # ========================================================
    # 3. NAME + ADDRESS NUMBER
    # ========================================================

    if prefix and address_numbers:

        for number in address_numbers[:3]:

            keys.append(
                (
                    "name_number",
                    country,
                    prefix,
                    number,
                )
            )

    # ========================================================
    # 4. NAME + ADDRESS TOKEN
    # ========================================================

    if prefix and address_tokens:

        for token in address_tokens[:6]:

            keys.append(
                (
                    "name_address",
                    country,
                    prefix,
                    token,
                )
            )

    # ========================================================
    # 5. NAME TOKEN + ADDRESS TOKEN
    # ========================================================

    if name_tokens and address_tokens:

        # Use only the strongest name tokens.

        strong_name_tokens = [
            token
            for token in name_tokens
            if len(token) >= 4
        ][:2]

        for n_token in strong_name_tokens:

            for a_token in address_tokens[:4]:

                keys.append(
                    (
                        "name_token_address",
                        country,
                        n_token,
                        a_token,
                    )
                )

    # ========================================================
    # 6. ADDRESS TOKEN PAIRS
    # ========================================================

    # Adjacent pairs
    for i in range(
        len(address_tokens) - 1
    ):

        token1 = address_tokens[i]
        token2 = address_tokens[i + 1]

        keys.append(
            (
                "address_pair",
                country,
                token1,
                token2,
            )
        )

    # ========================================================
    # 7. NON-ADJACENT ADDRESS TOKEN PAIRS
    # ========================================================

    # Useful when address word ordering changes.

    limited_tokens = address_tokens[:6]

    for i in range(
        len(limited_tokens)
    ):

        for j in range(
            i + 1,
            len(limited_tokens)
        ):

            token1 = limited_tokens[i]
            token2 = limited_tokens[j]

            # Canonical ordering makes the block
            # insensitive to token order.

            if token1 > token2:
                token1, token2 = token2, token1

            keys.append(
                (
                    "address_pair_unordered",
                    country,
                    token1,
                    token2,
                )
            )

    # ========================================================
    # 8. ADDRESS NUMBER + ADDRESS TOKEN
    # ========================================================

    # This is important for cases such as:
    #
    # 26 Stone Street
    # 0026 STONE ST
    #
    # where the business names can be completely different.

    if address_numbers and address_tokens:

        for number in address_numbers[:3]:

            for token in address_tokens[:4]:

                keys.append(
                    (
                        "number_address",
                        country,
                        number,
                        token,
                    )
                )

    # ========================================================
    # 9. ADDRESS NUMBER + TWO TOKENS
    # ========================================================

    if (
        address_numbers
        and len(address_tokens) >= 2
    ):

        for number in address_numbers[:2]:

            for i in range(
                min(len(address_tokens) - 1, 4)
            ):

                token1 = address_tokens[i]
                token2 = address_tokens[i + 1]

                keys.append(
                    (
                        "number_address_pair",
                        country,
                        number,
                        token1,
                        token2,
                    )
                )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    # The same key can occasionally be produced by
    # multiple strategies.

    return list(
        dict.fromkeys(keys)
    )


# ============================================================
# BUILD BLOCK INDEX
# ============================================================

def build_block_index(
    df: pd.DataFrame,
    source_name: str,
    max_block_size: int = MAX_BLOCK_SIZE,
):

    index = defaultdict(list)

    total_rows = len(df)

    print(
        f"Building {source_name} blocking index "
        f"for {total_rows:,} rows..."
    )

    for idx, row in df.iterrows():

        keys = create_blocking_keys(
            row
        )

        for key in keys:

            index[key].append(
                idx
            )

    raw_blocks = len(index)

    # --------------------------------------------------------
    # Remove oversized blocks
    # --------------------------------------------------------

    filtered_index = {
        key: indices
        for key, indices in index.items()
        if len(indices) <= max_block_size
    }

    print(
        f"{source_name}: "
        f"{raw_blocks:,} raw blocks -> "
        f"{len(filtered_index):,} usable blocks"
    )

    return filtered_index


# ============================================================
# GENERATE CANDIDATES FOR ONE RECORD
# ============================================================

def generate_candidates_for_record(
    row,
    source2_index,
    source3_index,
):

    candidate_s2 = set()
    candidate_s3 = set()

    keys = create_blocking_keys(
        row
    )

    for key in keys:

        s2_matches = source2_index.get(
            key,
            []
        )

        s3_matches = source3_index.get(
            key,
            []
        )

        candidate_s2.update(
            s2_matches
        )

        candidate_s3.update(
            s3_matches
        )

    return (
        candidate_s2,
        candidate_s3,
    )


# ============================================================
# COMPLETE CANDIDATE GENERATION
# ============================================================

def generate_candidate_pairs(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    max_block_size: int = MAX_BLOCK_SIZE,
):

    print()
    print("=" * 70)
    print("BUILDING BLOCKING INDEXES")
    print("=" * 70)

    # --------------------------------------------------------
    # S2 INDEX
    # --------------------------------------------------------

    print()
    print("Building S2 blocking index...")

    source2_index = build_block_index(
        source2,
        "S2",
        max_block_size,
    )

    # --------------------------------------------------------
    # S3 INDEX
    # --------------------------------------------------------

    print()
    print("Building S3 blocking index...")

    source3_index = build_block_index(
        source3,
        "S3",
        max_block_size,
    )

    # --------------------------------------------------------
    # CANDIDATE GENERATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATING CANDIDATE PAIRS")
    print("=" * 70)

    candidates = []

    total_s1 = len(source1)

    for count, (idx, row) in enumerate(
        source1.iterrows(),
        start=1,
    ):

        candidate_s2, candidate_s3 = (
            generate_candidates_for_record(
                row,
                source2_index,
                source3_index,
            )
        )

        s1_id = row["entity_id"]

        # ----------------------------------------------------
        # S2
        # ----------------------------------------------------

        for s2_idx in candidate_s2:

            candidates.append(
                (
                    s1_id,
                    source2.iloc[
                        s2_idx
                    ]["entity_id"],
                )
            )

        # ----------------------------------------------------
        # S3
        # ----------------------------------------------------

        for s3_idx in candidate_s3:

            candidates.append(
                (
                    s1_id,
                    source3.iloc[
                        s3_idx
                    ]["entity_id"],
                )
            )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if count % 10_000 == 0:

            print(
                f"Processed "
                f"{count:,}/{total_s1:,} S1 rows | "
                f"Candidates: {len(candidates):,}"
            )

    print()
    print(
        f"Total candidate pairs: "
        f"{len(candidates):,}"
    )

    return candidates