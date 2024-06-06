
class ReadOnlyDict:

    def __init__(self, _dict: dict):
        self._dict = _dict

    def __getitem__(self, item):
        return self._dict[item]

    def keys(self):
        return self._dict.keys()

    def items(self):
        return self._dict.items()


class KeywordMapping:
    def __init__(self, _dict: dict):
        self._dict = ReadOnlyDict(_dict)
        inverted_dict = {}
        for k, v in _dict.items():
            if v in inverted_dict:
                raise ValueError(f"Some keys share the same value")
            inverted_dict[v] = k
        self._inverted_dict = ReadOnlyDict(inverted_dict)

    def keys(self):
        return self._dict.keys()

    @property
    def name_to_type(self):
        return self._inverted_dict

    @property
    def type_to_name(self):
        return self._dict
