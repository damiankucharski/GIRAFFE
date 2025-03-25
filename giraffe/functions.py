from giraffe.globals import BACKEND as B
from giraffe.lib_types import Tensor


def scale_vector_to_sum_1(tensor: Tensor):
  return tensor / B.unsqueeze(B.sum(tensor, axis=-1), -1)
