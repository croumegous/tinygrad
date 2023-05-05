#!/usr/bin/env python
import unittest
import numpy as np
from tinygrad.tensor import Tensor, Device
from tinygrad.jit import TinyJit, SpecializedJit

@unittest.skipUnless(Device.DEFAULT == "GPU", "JIT is only for GPU")
class TestJit(unittest.TestCase):
  def test_simple_jit(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    for _ in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c = add(a, b)
      np.testing.assert_equal(c.numpy(), a.numpy()+b.numpy())

  def test_jit_shape_mismatch(self):
    @TinyJit
    def add(a, b): return (a+b).realize()
    for _ in range(3):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c = add(a, b)
    bad = Tensor.randn(20, 20)
    with self.assertRaises(AssertionError):
      add(a, bad)

  def test_jit_duplicate_fail(self):
    # the jit doesn't support duplicate arguments
    @TinyJit
    def add(a, b): return (a+b).realize()
    a = Tensor.randn(10, 10)
    with self.assertRaises(AssertionError):
      add(a, a)

  def test_kwargs_jit(self):
    @TinyJit
    def add_kwargs(first, second): return (first+second).realize()
    for _ in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      c = add_kwargs(first=a, second=b)
      np.testing.assert_equal(c.numpy(), a.numpy()+b.numpy())

  def test_array_jit(self):
    @TinyJit
    def add_array(a, arr): return (a+arr[0]).realize()
    for i in range(5):
      a = Tensor.randn(10, 10)
      b = Tensor.randn(10, 10)
      a.realize(), b.realize()
      c = add_array(a, [b])
      if i >= 2:
        # should fail once jitted since jit can't handle arrays
        np.testing.assert_equal(np.any(np.not_equal(c.numpy(),a.numpy()+b.numpy())), True)
      else:
        np.testing.assert_equal(c.numpy(), a.numpy()+b.numpy())

  def test_method_jit(self):
    class Fun:
      def __init__(self):
        self.a = Tensor.randn(10, 10)
      @TinyJit
      def __call__(self, b:Tensor) -> Tensor:
        return (self.a+b).realize()
    fun = Fun()
    for _ in range(5):
      b = Tensor.randn(10, 10)
      c = fun(b)
      np.testing.assert_equal(c.numpy(), fun.a.numpy()+b.numpy())
  
  ### Specialized JIT tests
  def test_specialized_jit(self):
    class Fun:
      def __init__(self):
        self.a = Tensor.randn(10, 10)

      @SpecializedJit
      def specialized_call(self, b: Tensor) -> Tensor:
        return (self.a + b).realize()

    fun = Fun()

    for _ in range(5):
      b = Tensor.randn(10, 10)
      c = fun.specialized_call(b)
      np.testing.assert_equal(c.numpy(), fun.a.numpy() + b.numpy())

    # Test with different shapes
    for _ in range(5):
      b = Tensor.randn(5, 5)
      c = fun.specialized_call(b)
      np.testing.assert_equal(c.numpy(), fun.a.numpy()[:5, :5] + b.numpy())

    # Test with a second instance
    fun2 = Fun()
    for _ in range(5):
      b = Tensor.randn(10, 10)
      c = fun2.specialized_call(b)
      np.testing.assert_equal(c.numpy(), fun2.a.numpy() + b.numpy())

if __name__ == '__main__':
  unittest.main()