import pickle


class Pickle:
    """
    A utility class for serializing and deserializing Python objects using pickle.

    This class provides static methods for saving objects to files and loading them back,
    which is particularly useful for persisting tree architectures.
    """

    @staticmethod
    def load(path):
        """
        Load a Python object from a pickle file.

        Args:
            path: File path to load the object from

        Returns:
            The deserialized Python object
        """
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def save(path, obj):
        """
        Save a Python object to a pickle file.

        Args:
            path: File path where the object will be saved
            obj: The Python object to serialize and save
        """
        with open(path, "wb") as file:
            pickle.dump(obj, file)


def first_uniques_mask(arr):
    """
    Create a boolean mask that identifies the first occurrence of each unique item in an array.

    This function is useful for filtering duplicates from an array while preserving the order
    of first appearances.

    Args:
        arr: An array-like object to analyze

    Returns:
        A list of booleans where True indicates the first occurrence of a value and
        False indicates a duplicate of a previously seen value
    """
    mask = []
    for index, item in enumerate(arr):
        if item not in arr[:index]:
            mask.append(True)
        else:
            mask.append(False)

    return mask
