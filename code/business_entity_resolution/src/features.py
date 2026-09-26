from rapidfuzz import fuzz


def safe_text(value):
    """
    Convert missing/empty values into a safe string.
    """
    if value is None:
        return ""

    return str(value)


def token_set_similarity(a, b):
    """
    Token-set similarity using RapidFuzz.
    """
    a = safe_text(a)
    b = safe_text(b)

    if not a or not b:
        return 0.0

    return fuzz.token_set_ratio(a, b) / 100.0


def token_sort_similarity(a, b):
    """
    Token-sort similarity using RapidFuzz.
    """
    a = safe_text(a)
    b = safe_text(b)

    if not a or not b:
        return 0.0

    return fuzz.token_sort_ratio(a, b) / 100.0


def ratio_similarity(a, b):
    """
    Standard fuzzy string similarity.
    """
    a = safe_text(a)
    b = safe_text(b)

    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b) / 100.0


def partial_similarity(a, b):
    """
    Partial fuzzy similarity.
    """
    a = safe_text(a)
    b = safe_text(b)

    if not a or not b:
        return 0.0

    return fuzz.partial_ratio(a, b) / 100.0


def jaccard_similarity(a, b):
    """
    Token-level Jaccard similarity.

    Example:

        "green logistics company"
        "green logistics"

    Shared tokens = green, logistics
    """

    a = safe_text(a)
    b = safe_text(b)

    if not a or not b:
        return 0.0

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union)


def number_overlap(numbers_a, numbers_b):
    """
    Compare all numeric address components.

    Example:

        [26, 3]
        [26, 5]

    overlap = 1 / 2 = 0.5
    """

    if not isinstance(numbers_a, list):
        numbers_a = []

    if not isinstance(numbers_b, list):
        numbers_b = []

    if not numbers_a or not numbers_b:
        return 0.0

    set_a = set(
        str(x)
        for x in numbers_a
    )

    set_b = set(
        str(x)
        for x in numbers_b
    )

    intersection = set_a & set_b

    if not intersection:
        return 0.0

    return len(intersection) / min(
        len(set_a),
        len(set_b)
    )


def address_token_overlap(tokens_a, tokens_b):
    """
    Calculate overlap between address tokens.
    """

    if not isinstance(tokens_a, list):
        tokens_a = []

    if not isinstance(tokens_b, list):
        tokens_b = []

    if not tokens_a or not tokens_b:
        return 0.0

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    intersection = set_a & set_b

    if not intersection:
        return 0.0

    return len(intersection) / min(
        len(set_a),
        len(set_b)
    )


def exact_match(a, b):
    """
    Exact normalized-value match.
    """
    a = safe_text(a)
    b = safe_text(b)

    if not a or not b:
        return 0

    return int(a == b)


def length_difference(a, b):
    """
    Absolute difference in string lengths.
    """
    a = safe_text(a)
    b = safe_text(b)

    return abs(
        len(a) - len(b)
    )


def build_pair_features(
    s1_row,
    candidate_row
):
    """
    Build all matching features for one S1/S2 or S1/S3 pair.
    """

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    s1_name = safe_text(
        s1_row.get(
            "name_matching",
            ""
        )
    )

    candidate_name = safe_text(
        candidate_row.get(
            "name_matching",
            ""
        )
    )

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    s1_address = safe_text(
        s1_row.get(
            "address_normalized",
            ""
        )
    )

    candidate_address = safe_text(
        candidate_row.get(
            "address_normalized",
            ""
        )
    )

    # --------------------------------------------------------
    # COUNTRY
    # --------------------------------------------------------

    s1_country = safe_text(
        s1_row.get(
            "country_normalized",
            ""
        )
    )

    candidate_country = safe_text(
        candidate_row.get(
            "country_normalized",
            ""
        )
    )

    # --------------------------------------------------------
    # ADDRESS NUMBERS
    # --------------------------------------------------------

    s1_numbers = s1_row.get(
        "address_numbers",
        []
    )

    candidate_numbers = candidate_row.get(
        "address_numbers",
        []
    )

    # --------------------------------------------------------
    # ADDRESS TOKENS
    # --------------------------------------------------------

    s1_address_tokens = s1_row.get(
        "address_tokens",
        []
    )

    candidate_address_tokens = candidate_row.get(
        "address_tokens",
        []
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = {

        # ============================
        # NAME FEATURES
        # ============================

        "name_ratio":
            ratio_similarity(
                s1_name,
                candidate_name
            ),

        "name_partial_ratio":
            partial_similarity(
                s1_name,
                candidate_name
            ),

        "name_token_sort":
            token_sort_similarity(
                s1_name,
                candidate_name
            ),

        "name_token_set":
            token_set_similarity(
                s1_name,
                candidate_name
            ),

        "name_jaccard":
            jaccard_similarity(
                s1_name,
                candidate_name
            ),

        "name_exact":
            exact_match(
                s1_name,
                candidate_name
            ),

        "name_length_difference":
            length_difference(
                s1_name,
                candidate_name
            ),

        # ============================
        # ADDRESS FEATURES
        # ============================

        "address_ratio":
            ratio_similarity(
                s1_address,
                candidate_address
            ),

        "address_partial_ratio":
            partial_similarity(
                s1_address,
                candidate_address
            ),

        "address_token_sort":
            token_sort_similarity(
                s1_address,
                candidate_address
            ),

        "address_token_set":
            token_set_similarity(
                s1_address,
                candidate_address
            ),

        "address_jaccard":
            jaccard_similarity(
                s1_address,
                candidate_address
            ),

        "address_exact":
            exact_match(
                s1_address,
                candidate_address
            ),

        "address_number_overlap":
            number_overlap(
                s1_numbers,
                candidate_numbers
            ),

        "address_token_overlap":
            address_token_overlap(
                s1_address_tokens,
                candidate_address_tokens
            ),

        "address_length_difference":
            length_difference(
                s1_address,
                candidate_address
            ),

        # ============================
        # COUNTRY
        # ============================

        "country_exact":
            int(
                bool(s1_country)
                and
                s1_country == candidate_country
            ),

        # ============================
        # MISSINGNESS
        # ============================

        "s1_address_missing":
            int(not bool(s1_address)),

        "candidate_address_missing":
            int(not bool(candidate_address)),
    }

    return features