from dataclasses import dataclass
from typing import Optional, Type, TypeVar, Generic
import yaml
import os
from enum import Enum


class DeserializationError(Exception):
    """Custom exception for YAML deserialization errors."""
    pass


T = TypeVar('T')


@dataclass
class YAMLDeserializer(Generic[T]):
    """
    A generic YAML deserializer with advanced features for converting YAML to dataclasses.

    Supports:
    - Type conversion
    - Optional fields
    - Nested dataclasses
    - Enum support
    - Validation
    """

    @classmethod
    def from_yaml_file(cls, file_path: str, dataclass_type: Type[T]) -> T:
        """
        Deserialize YAML file to a specific dataclass type.

        Args:
            file_path (str): Path to the YAML file
            dataclass_type (Type[T]): Target dataclass type for deserialization

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
                yaml_data = yaml.safe_load(file)

            # Validate YAML content
            if yaml_data is None:
                raise DeserializationError("Empty YAML file")

            return cls.from_dict(yaml_data, dataclass_type)

        except yaml.YAMLError as e:
            raise DeserializationError(f"YAML parsing error: {e}")
        except IOError as e:
            raise DeserializationError(f"File read error: {e}")

    @classmethod
    def from_dict(cls, data: dict, dataclass_type: Type[T]) -> T:
        """
        Convert a dictionary to a specific dataclass type.

        Args:
            data (dict): Source dictionary
            dataclass_type (Type[T]): Target dataclass type

        Returns:
            T: Deserialized dataclass instance
        """
        try:
            # Prepare kwargs for dataclass initialization
            kwargs = {}

            # Iterate through dataclass fields
            for field_name, field_type in dataclass_type.__annotations__.items():
                # Handle optional fields
                if hasattr(field_type, '__origin__') and field_type.__origin__ is Optional:
                    field_type = field_type.__args__[0]
                    is_optional = True
                else:
                    is_optional = False

                # Check if field exists in source data
                if field_name in data:
                    value = data[field_name]

                    # Handle nested dataclasses
                    if hasattr(field_type, '__origin__') and field_type.__origin__ is list:
                        # Handle list of dataclasses or other types
                        item_type = field_type.__args__[0]
                        value = [
                            cls.from_dict(item, item_type) if isinstance(item, dict) else item
                            for item in value
                        ]
                    elif hasattr(field_type, '__dataclass_fields__'):
                        # Nested dataclass
                        value = cls.from_dict(value, field_type)
                    elif issubclass(field_type, Enum):
                        # Enum conversion
                        value = field_type(value)

                    kwargs[field_name] = value
                elif not is_optional:
                    # Raise error for required fields missing in source
                    raise DeserializationError(f"Missing required field: {field_name}")

            # Create and return dataclass instance
            return dataclass_type(**kwargs)

        except (TypeError, ValueError) as e:
            raise DeserializationError(f"Deserialization error: {e}")