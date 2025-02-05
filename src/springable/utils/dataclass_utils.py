import re


def _camel_to_text(cml_text):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', cml_text).lower()


class Updatable:
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                expected_type = self.__dataclass_fields__[key].type
                if (expected_type is float and type(value) is int) or isinstance(value, expected_type):
                    setattr(self, key, value)
                else:
                    print(f'Value of {_camel_to_text(type(self).__name__).rstrip('s')}'
                          f' "{key}" is ignored.'
                          f' It should be a {expected_type.__name__}, not a {type(value).__name__} ({value})')
            else:
                print(f'Unknown {_camel_to_text(type(self).__name__).rstrip('s')} "{key}". Probably check spelling...')