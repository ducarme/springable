def is_indexable(x):
    try:
        _ = x[0]
        return True
    except (TypeError, IndexError, KeyError):
        return False