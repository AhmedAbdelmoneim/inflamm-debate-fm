"""Gene name utilities."""


def deduplicate_names(names: list[str]) -> list[str]:
    """Deduplicate gene names by appending numbers.

    Args:
        names: List of gene names (may contain duplicates).

    Returns:
        List of deduplicated names.
    """
    seen = {}
    deduped = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            deduped.append(n)
        else:
            seen[n] += 1
            deduped.append(f"{n}.{seen[n]}")
    return deduped
