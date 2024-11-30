import os

import yaml


class DeserializationError(Exception):
    """Custom exception for YAML deserialization errors."""
    pass


class YamlDict(dict):
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__

    def __getattr__(self, attr):
        return self.get(attr)

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self


def read_yaml(file_path: str):
    """
    Deserialize YAML file to a python object

    Args:
        file_path (str): Path to the YAML file

    Returns:
        T: Deserialized dataclass instance

    Raises:
        DeserializationError: For various deserialization issues
    """
    try:
        # Validate file existence
        if not os.path.exists(file_path):
            raise DeserializationError(f"YAML file not found: {file_path}")

        # Read YAML file
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Validate YAML content
        if data is None:
            raise DeserializationError("Empty YAML file")

        return YamlDict(data)

    except yaml.YAMLError as e:
        raise DeserializationError(f"YAML parsing error: {e}")
    except IOError as e:
        raise DeserializationError(f"File read error: {e}")
