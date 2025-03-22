import pickle


class Pickle:
    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def save(path, obj):
        with open(path, "wb") as file:
            pickle.dump(obj, file)


def first_uniques_mask(arr):
    mask = []
    for index, item in enumerate(arr):
        if item not in arr[:index]:
            mask.append(True)
        else:
            mask.append(False)

    return mask
