from giraffe.globals import BACKEND as B, set_postprocessing_function
from giraffe.lib_types import Tensor


def scale_vector_to_sum_1(tensor: Tensor):
  return tensor / B.unsqueeze(B.sum(tensor, axis=-1), -1)


def set_multiclass_postprocessing():
    set_postprocessing_function(scale_vector_to_sum_1)
