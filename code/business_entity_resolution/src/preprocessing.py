import re
import unicodedata


def normalize_text(value: str) -> str:
    """
    General Unicode-aware normalization.
    """

    if value is None:
        return ""

    value = str(value).strip().lower()

    # Unicode normalization.
    value = unicodedata.normalize("NFKC", value)

    # Replace punctuation/symbols with spaces.
    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    # Collapse whitespace.
    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def normalize_business_name(value: str) -> str:
    return normalize_text(value)


def normalize_address(value: str) -> str:
    return normalize_text(value)


def normalize_country(value: str) -> str:
    return normalize_text(value)


def normalize_for_matching(value: str) -> str:
    """
    Remove common legal/business suffixes.
    """

    value = normalize_text(value)

    if not value:
        return ""

    legal_terms = {
        "llc",
        "inc",
        "incorporated",
        "ltd",
        "limited",
        "private",
        "pvt",
        "plc",
        "corp",
        "corporation",
        "co",
        "company",
    }

    tokens = value.split()

    tokens = [
        token
        for token in tokens
        if token not in legal_terms
    ]

    return " ".join(tokens)


def extract_address_numbers(value: str):
    """
    Extract ALL numeric components from an address.

    Numeric strings are converted to integers first so:

        00702 -> 702
        0702  -> 702
        702   -> 702

    can be treated as the same numeric component.
    """

    if not value:
        return []

    matches = re.findall(
        r"\d+",
        str(value)
    )

    numbers = set()

    for match in matches:

        try:
            number = str(
                int(match)
            )
        except ValueError:
            continue

        if number:
            numbers.add(number)

    return sorted(numbers)


def extract_address_number(value: str) -> str:
    """
    Backward-compatible helper.

    Returns the first numeric component.
    """

    numbers = extract_address_numbers(value)

    if numbers:
        return numbers[0]

    return ""


def extract_postal_code(value: str) -> str:
    """
    Extract 5- or 6-digit postal/PIN-like values.
    """

    if not value:
        return ""

    matches = re.findall(
        r"\b\d{5,6}\b",
        str(value)
    )

    normalized = []

    for match in matches:

        try:
            normalized.append(
                str(int(match))
            )
        except ValueError:
            pass

    if normalized:
        return normalized[-1]

    return ""


def extract_address_tokens(value: str):
    """
    Return useful address tokens.

    Very short tokens are ignored because values such as
    'a', 'of', 'no' create huge blocks.
    """

    if not value:
        return []

    tokens = normalize_address(value).split()

    useful = []

    for token in tokens:

        # Ignore extremely short tokens.
        if len(token) < 3:
            continue

        # Ignore purely numeric tokens.
        if token.isdigit():
            continue

        useful.append(token)

    return useful


def add_normalized_columns(df):

    df = df.copy()

    if "business_name" in df.columns:

        df["name_normalized"] = (
            df["business_name"]
            .apply(normalize_business_name)
        )

        df["name_matching"] = (
            df["business_name"]
            .apply(normalize_for_matching)
        )

    if "business_address" in df.columns:

        df["address_normalized"] = (
            df["business_address"]
            .apply(normalize_address)
        )

        df["address_number"] = (
            df["business_address"]
            .apply(extract_address_number)
        )

        df["address_numbers"] = (
            df["business_address"]
            .apply(extract_address_numbers)
        )

        df["postal_code"] = (
            df["business_address"]
            .apply(extract_postal_code)
        )

        df["address_tokens"] = (
            df["business_address"]
            .apply(extract_address_tokens)
        )

    if "country" in df.columns:

        df["country_normalized"] = (
            df["country"]
            .apply(normalize_country)
        )

    return df