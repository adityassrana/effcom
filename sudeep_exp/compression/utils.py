import tensorflow.compat.v1 as tf
import numpy as np 

class PackedTensors(object):
  """Packed representation of compressed tensors."""

  def __init__(self, string=None):
    self._example = tf.train.Example()
    if string:
      self.string = string

  @property
  def model(self):
    """Model identifier."""
    buf = self._example.features.feature["MD"].bytes_list.value[0]
    return buf.decode("ascii")

  @model.setter
  def model(self, value):
    self._example.features.feature["MD"].bytes_list.value[:] = [
        value.encode("ascii")]

  @model.deleter
  def model(self):
    del self._example.features.feature["MD"]

  @property
  def string(self):
    """A string representation of this object."""
    return self._example.SerializeToString()

  @string.setter
  def string(self, value):
    self._example.ParseFromString(value)

  def pack(self, tensors, arrays):
    """Packs Tensor values into this object."""
    if len(tensors) != len(arrays):
      raise ValueError("`tensors` and `arrays` must have same length.")
    i = 1
    for tensor, array in zip(tensors, arrays):
      feature = self._example.features.feature[chr(i)]
      feature.Clear()
      if array.ndim != 1:
        raise RuntimeError("Unexpected tensor rank: {}.".format(array.ndim))
      if tensor.dtype.is_integer:
        feature.int64_list.value[:] = array
      elif tensor.dtype == tf.string:
        feature.bytes_list.value[:] = array
      else:
        raise RuntimeError(
            "Unexpected tensor dtype: '{}'.".format(tensor.dtype))
      i += 1
    # Delete any remaining, previously set arrays.
    while chr(i) in self._example.features.feature:
      del self._example.features.feature[chr(i)]
      i += 1

  def unpack(self, tensors):
    """Unpacks Tensor values from this object."""
    arrays = []
    for i, tensor in enumerate(tensors):
      feature = self._example.features.feature[chr(i + 1)]
      np_dtype = tensor.dtype.as_numpy_dtype
      if tensor.dtype.is_integer:
        arrays.append(np.array(feature.int64_list.value, dtype=np_dtype))
      elif tensor.dtype == tf.string:
        arrays.append(np.array(feature.bytes_list.value, dtype=np_dtype))
      else:
        raise RuntimeError(
            "Unexpected tensor dtype: '{}'.".format(tensor.dtype))
    return arrays

def read_png(filename):
  """Loads a PNG image file."""
  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  return image

def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image

def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)
