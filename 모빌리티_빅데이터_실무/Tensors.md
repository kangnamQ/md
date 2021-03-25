Tensors
==

- 21.03.17
- Tensors란 무엇일까

---

참조
---

Reference : [TensorFlow_Guide][TensorFlow_link] (TensorFlow Guide Link)

[TensorFlow_link]: https://www.tensorflow.org/guide/tensor "TensorFlow_Guide"

---



Introduction to Tensors
--

```python
import tensorflow as tf
import numpy as np
```

Tensors are multi-dimensional arrays with a uniform type (called a `dtype`). You can see all supported `dtypes` at [`tf.dtypes.DType`](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType).

If you're familiar with [NumPy](https://numpy.org/devdocs/user/quickstart.html), tensors are (kind of) like `np.arrays`.

All tensors are immutable like Python numbers and strings: you can never update the contents of a tensor, only create a new one.



텐서는 균일한 형식의 다차원 배열입니다. (dtype으로 불림) 
tf.dtypes.DType에서 지원되는 dtypes를 확인할 수 있습니다.

Tensors는 np.arrays 형식과 유사합니다.

모든 Tensor는 파이썬의 숫자나 문자처럼 불변합니다. Tensor를 업데이트를 하는 유일한 방법은 새로 만드는 것입니다.



Basics
--

Here is a "scalar" or "rank-0" tensor . A scalar contains a single value, and no "axes".

스칼라 또는 rank-0 인 텐서, 스칼라에는 단일 값이 포함되며 "축"은 없다.

```python
# This will be an int32 tensor by default; see "dtypes" below.
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

-----
tf.Tensor(4, shape=(), dtype=int32)
```



A "vector" or "rank-1" tensor is like a list of values. A vector has one axis:

벡터 또는 rank-1인 텐서, 벡터에는 축이 하나 있다.

```python
# Let's make this a float tensor.
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)

-----
tf.Tensor([2. 3. 4.], shape=(3,), dtype=float32)
```



A "matrix" or "rank-2" tensor has two axes:

행렬 또는 rank-2인 텐서, 행렬에는 두개의 축이 있다.

```python
# If you want to be specific, you can set the dtype (see below) at creation time
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)

-----
tf.Tensor(
[[1. 2.]
 [3. 4.]
 [5. 6.]], shape=(3, 2), dtype=float16)
```

| 스칼라, 모양 : `[]`                                          | 벡터, 모양 : `[3]`                                           | 행렬, 모양 : `[3, 2]`                                        |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| ![스칼라, 숫자 4](https://www.tensorflow.org/guide/images/tensor/scalar.png) | ![각 섹션에 숫자가 포함 된 3 개의 섹션이있는 줄입니다.](https://www.tensorflow.org/guide/images/tensor/vector.png) | ![각 셀에 숫자가 포함 된 3x2 그리드.](https://www.tensorflow.org/guide/images/tensor/matrix.png) |

rank가 [], 축을 의미하는 것인듯.



Tensors may have more axes; here is a tensor with three axes:

텐서는 더많은 축을 가질 수 있으며 3개의 축을 가진 텐서는 다음과 같다.

```python
# There can be an arbitrary number of
# axes (sometimes called "dimensions")
rank_3_tensor = tf.constant([
  [[0, 1, 2, 3, 4],
   [5, 6, 7, 8, 9]],
  [[10, 11, 12, 13, 14],
   [15, 16, 17, 18, 19]],
  [[20, 21, 22, 23, 24],
   [25, 26, 27, 28, 29]],])

print(rank_3_tensor)

-----
tf.Tensor(
[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
  [15 16 17 18 19]]

 [[20 21 22 23 24]
  [25 26 27 28 29]]], shape=(3, 2, 5), dtype=int32)
```



There are many ways you might visualize a tensor with more than two axes.

축이 두개 이상인 텐서를 시각화 하는 방법은 여러가지가 있다.

| A 3-axis tensor, shape: `[3, 2, 5]`                          |                                                              |                                                              |
| :----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|                                                              |                                                              |                                                              |
| ![img](https://www.tensorflow.org/guide/images/tensor/3-axis_numpy.png) | ![img](https://www.tensorflow.org/guide/images/tensor/3-axis_front.png) | ![img](https://www.tensorflow.org/guide/images/tensor/3-axis_block.png) |



You can convert a tensor to a NumPy array either using `np.array` or the `tensor.numpy` method:

np.array나 tensor.numpy의 방법을 통하여 탠서를 NumPy의 배열로 사용할 수 있다. (변환할 수 있다.)

```python
np.array(rank_2_tensor)

-----
array([[1., 2.],
       [3., 4.],
       [5., 6.]], dtype=float16)
```

```python
rank_2_tensor.numpy()

-----
array([[1., 2.],
       [3., 4.],
       [5., 6.]], dtype=float16)

```



Tensors often contain floats and ints, but have many other types, including:

텐서는 보통(종종) floats와 ints를 포함하지만 더 많은 타입들을 가지고있다(사용할 수 있다).

- complex numbers (복소수)
- strings (문자열)



The base [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) class requires tensors to be "rectangular"---that is, along each axis, every element is the same size. However, there are specialized types of tensors that can handle different shapes:

기본 tf.Tensor클래스는 텐서가 "직사각형"이어야 한다. 즉 이것은 각 축에 따라서 요소들의 크기가 같아야 한다.
그러나 다양한(다른?) 모양의 타입을 사용(조정)할 수 있는 특별한 텐서타입이 있다.

- Ragged tensors (see [RaggedTensor](https://www.tensorflow.org/guide/tensor#ragged_tensors) below)  (비정형 텐서)
- Sparse tensors (see [SparseTensor](https://www.tensorflow.org/guide/tensor#sparse_tensors) below)  (희소 텐서)



You can do basic math on tensors, including addition, element-wise multiplication, and matrix multiplication.

 더하기, 요소별 곱셉, 행렬간의 곱을 텐서에서 기본적 수학(method)으로 포함하고 있어 사용할 수 있다.

```python
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")

-----
tf.Tensor(
[[2 3]
 [4 5]], shape=(2, 2), dtype=int32) 

tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32) 

tf.Tensor(
[[3 3]
 [7 7]], shape=(2, 2), dtype=int32) 
```



```python
print(a + b, "\n") # element-wise addition (요소간 더하기)
print(a * b, "\n") # element-wise multiplication (요소간 곱)
print(a @ b, "\n") # matrix multiplication (행렬 곱)

-----
tf.Tensor(
[[2 3]
 [4 5]], shape=(2, 2), dtype=int32) 

tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32) 

tf.Tensor(
[[3 3]
 [7 7]], shape=(2, 2), dtype=int32) 
```

굳이 add나 multiply, matmul 같은 것을 안써도 계산이 동일하게 나오는 것을 확인할 수 있다. 



Tensors are used in all kinds of operations (ops).

텐서 모든 종류의 작업(작전)에서 사용된다.

```python
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# Find the largest value
print(tf.reduce_max(c))
# Find the index of the largest value
print(tf.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))

-----
tf.Tensor(10.0, shape=(), dtype=float32)
tf.Tensor([1 0], shape=(2,), dtype=int64)
tf.Tensor(
[[2.6894143e-01 7.3105854e-01]
 [9.9987662e-01 1.2339458e-04]], shape=(2, 2), dtype=float32)
```





About shapes
---

Tensors have shapes. 

- **Shape**: The length (number of elements) of each of the axes of a tensor.

- **모양** : 텐서의 각 축 길이 (요소 수)

  

- **Rank**: Number of tensor axes. A scalar has rank 0, a vector has rank 1, a matrix is rank 2.

- **순위** : 텐서 축의 수, 스칼라는 rank0, 벡터는 rank1, 행렬은 rank2이다.



- **Axis** or **Dimension**: A particular dimension of a tensor.
- **축** 또는 **차원** : 텐서의 특정 차원.



- **Size**: The total number of items in the tensor, the product shape vector.
- **크기** : 해당 모양의 벡터형식의 텐서의 총 갯수(항목 수)



**Note:** Although you may see reference to a "tensor of two dimensions", a rank-2 tensor does not usually describe a 2D space.

2차원의 Tensor와 rank2의 Tensor은 같지 않다.
즉 여기서 말하는 것은 rank 와 차원은 같은 의미가 아니다는 것이다.



Tensors and [`tf.TensorShape`](https://www.tensorflow.org/api_docs/python/tf/TensorShape) objects have convenient properties for accessing these:

Tensor와 tf.TensorShape의 객체는 접근하기 위한 편리한 속성이 있다.



```python
rank_4_tensor = tf.zeros([3, 2, 4, 5])
```

| A rank-4 tensor, shape: `[3, 2, 4, 5]`                       |                                                              |
| :----------------------------------------------------------- | ------------------------------------------------------------ |
| ![A tensor shape is like a vector.](https://www.tensorflow.org/guide/images/tensor/shape.png) | ![A 4-axis tensor](https://www.tensorflow.org/guide/images/tensor/4-axis_block.png) |

```python
print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

-----
Type of every element: <dtype: 'float32'>
Number of axes: 4
Shape of tensor: (3, 2, 4, 5)
Elements along axis 0 of tensor: 3
Elements along the last axis of tensor: 5
Total number of elements (3*2*4*5):  120
```



While axes are often referred to by their indices, you should always keep track of the meaning of each. Often axes are ordered from global to local: The batch axis first, followed by spatial dimensions, and features for each location last. This way feature vectors are contiguous regions of memory.

축은 종종 인덱스로 참조되지만 항상 각각의 의미를 추적해야 한다. (알아봐야 한다.) 종종 축은 전역에서 로컬로 정렬된다. 
배치축이 먼저오고 공간 차원이오고 각 위치에 대한 기능이 마지막으로 온다. 
이런 식으로 특징 벡터는 연속적인 메모리 영역을 보인다.



| Typical axis order                                           |
| :----------------------------------------------------------- |
| ![Keep track of what each axis is. A 4-axis tensor might be: Batch, Width, Height, Features](https://www.tensorflow.org/guide/images/tensor/shape2.png) |





Indexing
--

#### Single-axis indexing

TensorFlow follows standard Python indexing rules, similar to [indexing a list or a string in Python](https://docs.python.org/3/tutorial/introduction.html#strings), and the basic rules for NumPy indexing.

TensorFlow는 기본적인 Python에서 리스트나 문자열의 인덱스를 생성하는 것처럼 기본적인 Python indexing 규칙과 NumPy의 기본 inexing 규칙을 따른다.



- indexes start at `0` (0부터 인덱스를 시작)
- negative indices count backwards from the end (음수 인덱스는 뒤에서 부터 계산 (-1하면 맨 끝에 계산되는 것))
- colons, `:`, are used for slices: `start:stop:step ` " : " 콜론은 슬라이스에 사용된다.

```python
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print(rank_1_tensor.numpy())

-----
[ 0  1  1  2  3  5  8 13 21 34]
```



Indexing with a scalar removes the axis:

스칼라로 인덱싱을 하면 축이 사라진다.

```python
print("First:", rank_1_tensor[0].numpy())
print("Second:", rank_1_tensor[1].numpy())
print("Last:", rank_1_tensor[-1].numpy())

-----
First: 0
Second: 1
Last: 34
```



Indexing with a `:` slice keeps the axis:

:을 사용한 슬라이스로 인덱싱하면 축이 유지된다.

```python
print("Everything:", rank_1_tensor[:].numpy())
print("Before 4:", rank_1_tensor[:4].numpy())
print("From 4 to the end:", rank_1_tensor[4:].numpy())
print("From 2, before 7:", rank_1_tensor[2:7].numpy())
print("Every other item:", rank_1_tensor[::2].numpy())
print("Reversed:", rank_1_tensor[::-1].numpy())

-----
Everything: [ 0  1  1  2  3  5  8 13 21 34]
Before 4: [0 1 1 2]
From 4 to the end: [ 3  5  8 13 21 34]
From 2, before 7: [1 2 3 5 8]
Every other item: [ 0  1  3  8 21]
Reversed: [34 21 13  8  5  3  2  1  1  0]
```





Multi-axis indexing
---

Higher rank tensors are indexed by passing multiple indices.

상위 rank에서는 여러 인덱스를 전달하여 인덱싱한다.

The exact same rules as in the single-axis case apply to each axis independently.

단일 축과 같은 규칙으로 각 축마다 독립적으로 적용된다.

```python
print(rank_2_tensor.numpy())

-----
[[1. 2.]
 [3. 4.]
 [5. 6.]]
```



Passing an integer for each index, the result is a scalar.

각 인덱스에 정수를 더하면 스칼라가 나온다.

```python
# Pull out a single value from a 2-rank tensor
print(rank_2_tensor[1, 1].numpy())

-----
4.0
```



You can index using any combination of integers and slices:

정수와 슬라이스를 같이 사용해서 인덱싱 할 수 있다.

```python
# Get row and column tensors
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")

-----
Second row: [3. 4.]
Second column: [2. 4. 6.]
Last row: [5. 6.]
First item in last column: 2.0
Skip the first row:
[[3. 4.]
 [5. 6.]] 
```



Here is an example with a 3-axis tensor (3축 tensor):

```python
print(rank_3_tensor[:, :, 4])

-----
tf.Tensor(
[[ 4  9]
 [14 19]
 [24 29]], shape=(3, 2), dtype=int32)
```

| Selecting the last feature across all locations in each example in the batch |                                                              |
| :----------------------------------------------------------- | ------------------------------------------------------------ |
| ![A 3x2x5 tensor with all the values at the index-4 of the last axis selected.](https://www.tensorflow.org/guide/images/tensor/index1.png) | ![The selected values packed into a 2-axis tensor.](https://www.tensorflow.org/guide/images/tensor/index2.png) |

Read the [tensor slicing guide](https://tensorflow.org/guide/tensor_slicing) to learn how you can apply indexing to manipulate individual elements in your tensors.

tensor slicing을 할 때 잘 모르겠으면 guide를 읽고 인덱싱을 적용해서 텐서의 개별 요소를 조작하는 방법을 알 수 있다.





Manipulating Shapes
--

Reshaping a tensor is of great utility.

텐서를 재구성(reshaping)하는 것은 좋은 방법이다(유용하다).

```python
# Shape returns a `TensorShape` object that shows the size along each axis
x = tf.constant([[1], [2], [3]])
print(x.shape)

-----
(3, 1)
```

```python
# You can convert this object into a Python list, too
print(x.shape.as_list())

-----
[3, 1]
```



You can reshape a tensor into a new shape. The [`tf.reshape`](https://www.tensorflow.org/api_docs/python/tf/reshape) operation is fast and cheap as the underlying data does not need to be duplicated.

텐서를 새로운 모양으로 바꿀 수 있다. tf.reshape는 복사같은 작업이 없어 작업을 빠르게 끝낼 수 있다.

```python
# You can reshape a tensor to a new shape.
# Note that you're passing in a list
reshaped = tf.reshape(x, [1, 3])

print(x.shape)
print(reshaped.shape)

-----
(3, 1)
(1, 3)
```



The data maintains its layout in memory and a new tensor is created, with the requested shape, pointing to the same data. TensorFlow uses C-style "row-major" memory ordering, where incrementing the rightmost index corresponds to a single step in memory.

데이터는 메모리에서 레이아웃을 유지하고 동일한 데이터를 가리키는 요청된 모양으로 새 텐서가 생성된다.
TensorFlow는 C스타일 "row-major"메모리 순서를 사용한다.
여기서 맨 오른쪽 인덱스를 증가시키는 것은 메모리의 단일 단계에 해단한다. 
(뭔소리지...)

```python
print(rank_3_tensor)

-----
tf.Tensor(
[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]]

 [[10 11 12 13 14]
  [15 16 17 18 19]]

 [[20 21 22 23 24]
  [25 26 27 28 29]]], shape=(3, 2, 5), dtype=int32)
```



If you flatten a tensor you can see what order it is laid out in memory.

텐서를 평평하게 하면 메모리에 배치 된 순서를 볼 수 있다.

```python
# A `-1` passed in the `shape` argument says "Whatever fits".
print(tf.reshape(rank_3_tensor, [-1]))

-----
tf.Tensor(
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29], shape=(30,), dtype=int32)
```



Typically the only reasonable use of [`tf.reshape`](https://www.tensorflow.org/api_docs/python/tf/reshape) is to combine or split adjacent axes (or add/remove `1`s).

일반적으로 tf.resheape의 유일한 합리적 사용은 인접한 축을 결합하거나 분리(분할)하는 것이다.

For this 3x2x5 tensor, reshaping to (3x2)x5 or 3x(2x5) are both reasonable things to do, as the slices do not mix:

이 3 * 2 * 5 센터의 경우 슬라이스가 혼합되지 않으므로 (3x2)x5나 3x(2x5) 같이 모양을 변경하는 것이 합리적이다.

```python
print(tf.reshape(rank_3_tensor, [3*2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))

-----
tf.Tensor(
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]
 [15 16 17 18 19]
 [20 21 22 23 24]
 [25 26 27 28 29]], shape=(6, 5), dtype=int32) 

tf.Tensor(
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]], shape=(3, 10), dtype=int32)
```

| Some good reshapes.                                          |                                                              |                                                              |
| :----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![A 3x2x5 tensor](https://www.tensorflow.org/guide/images/tensor/reshape-before.png) | ![The same data reshaped to (3x2)x5](https://www.tensorflow.org/guide/images/tensor/reshape-good1.png) | ![The same data reshaped to 3x(2x5)](https://www.tensorflow.org/guide/images/tensor/reshape-good2.png) |



Reshaping will "work" for any new shape with the same total number of elements, but it will not do anything useful if you do not respect the order of the axes.

reshaping은 동일한 총 요수의 수를 가진 새 모양에 대해서 "작동(일)"하지만 축의 순서를 따르지 않으면 유용하게 사용할 수 없다.

Swapping axes in [`tf.reshape`](https://www.tensorflow.org/api_docs/python/tf/reshape) does not work; you need [`tf.transpose`](https://www.tensorflow.org/api_docs/python/tf/transpose) for that.

축 교체는 tf.reshape가 작동하지 않기 때문에 축 교체를 하려면 tf.transpose를 해야한다.

```python
# Bad examples: don't do this

# You can't reorder axes with reshape.
print(tf.reshape(rank_3_tensor, [2, 3, 5]), "\n") 

# This is a mess
print(tf.reshape(rank_3_tensor, [5, 6]), "\n")

# This doesn't work at all
try:
  tf.reshape(rank_3_tensor, [7, -1])
except Exception as e:
  print(f"{type(e).__name__}: {e}")

-----
tf.Tensor(
[[[ 0  1  2  3  4]
  [ 5  6  7  8  9]
  [10 11 12 13 14]]

 [[15 16 17 18 19]
  [20 21 22 23 24]
  [25 26 27 28 29]]], shape=(2, 3, 5), dtype=int32) 

tf.Tensor(
[[ 0  1  2  3  4  5]
 [ 6  7  8  9 10 11]
 [12 13 14 15 16 17]
 [18 19 20 21 22 23]
 [24 25 26 27 28 29]], shape=(5, 6), dtype=int32) 

InvalidArgumentError: Input to reshape is a tensor with 30 values, but the requested shape requires a multiple of 7 [Op:Reshape]
```



| Some bad reshapes.                                           |                                                              |                                                              |
| :----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![You can't reorder axes, use tf.transpose for that](https://www.tensorflow.org/guide/images/tensor/reshape-bad.png) | ![Anything that mixes the slices of data together is probably wrong.](https://www.tensorflow.org/guide/images/tensor/reshape-bad4.png) | ![The new shape must fit exactly.](https://www.tensorflow.org/guide/images/tensor/reshape-bad2.png) |



You may run across not-fully-specified shapes. Either the shape contains a `None` (an axis-length is unknown) or the whole shape is `None` (the rank of the tensor is unknown).

완전히 지정되지 않은 shape를 가로질러 실행할 수 있다. 모양에 축의 길이를 모르거나 텐서의 rank를 모르는 None의 shape가 나올 것이다.



Except for [tf.RaggedTensor](https://www.tensorflow.org/guide/tensor#ragged_tensors), such shapes will only occur in the context of TensorFlow's symbolic, graph-building APIs:

tf.RaggedTensor를 제외하고 이런 모양은 TensorFlow의 상징적인 그레프에서만 나온다.

- [tf.function](https://www.tensorflow.org/guide/function)
- The [keras functional API](https://www.tensorflow.org/guide/keras/functional).





More on DTypes
--

To inspect a [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor)'s data type use the [`Tensor.dtype`](https://www.tensorflow.org/api_docs/python/tf/Tensor#dtype) property.

tf.Tensor의 데이터 타입을 알고 싶다면 Tensor.dtype를 사용하면 된다.



When creating a [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) from a Python object you may optionally specify the datatype.

tf.Tensor 에서 Python 객체를 만들 때 선택적으로 데이터 타입을 지정할 수 있다.



If you don't, TensorFlow chooses a datatype that can represent your data. TensorFlow converts Python integers to [`tf.int32`](https://www.tensorflow.org/api_docs/python/tf#int32) and Python floating point numbers to [`tf.float32`](https://www.tensorflow.org/api_docs/python/tf#float32). Otherwise TensorFlow uses the same rules NumPy uses when converting to arrays.

그렇지 않은 경우(지정해주지 않을 경우) TensorFlow는 데이터를 나타낼수 있는 데이터 유형을 선택한다.
TensorFlow는 Python에 정수 tf.int32나 tf.float32로 변환할 것이다. 그렇지 않다면 NumPy의 array로 변활 할  때 사용하는 것과 동일한 규칙을 사용한다.



You can cast from type to type.

타입에서 타입으로 캐스트할 수 있다.

```python
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
# Now, cast to an uint8 and lose the decimal precision
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)
print(the_u8_tensor)

-----
tf.Tensor([2 3 4], shape=(3,), dtype=uint8)
```





Broadcasting
--

Broadcasting is a concept borrowed from the [equivalent feature in NumPy](https://numpy.org/doc/stable/user/basics.html). In short, under certain conditions, smaller tensors are "stretched" automatically to fit larger tensors when running combined operations on them.

Broadcasting이란 넘파이의 중요한 기능에서 빌려온(가져온) 기능이다. 요약하자면 특정 조전에서 작은 텐서는 결합된 연산을 실행할 때 큰 텐서에 맞게 자동으로 "확장"된다.



The simplest and most common case is when you attempt to multiply or add a tensor to a scalar. In that case, the scalar is broadcast to be the same shape as the other argument.

가장 간단하고 인반적인 경우는 스칼라에 텐서를 곱하거나 더하려고 할때인데, 이 경우 스칼라는 다른 인수와 동일한 모양(shape)으로 브로드캐스트된다. 

```python
x = tf.constant([1, 2, 3])

y = tf.constant(2)
z = tf.constant([2, 2, 2])
# All of these are the same computation
print(tf.multiply(x, 2))
print(x * y)
print(x * z)

-----
tf.Tensor([2 4 6], shape=(3,), dtype=int32)
tf.Tensor([2 4 6], shape=(3,), dtype=int32)
tf.Tensor([2 4 6], shape=(3,), dtype=int32)
```



Likewise, axes with length 1 can be stretched out to match the other arguments. Both arguments can be stretched in the same computation.

마찬가지로 길이가 1인 축은 다른 인수와 일치하도록 늘릴 수 있다. 두 인수는 동일한 크기로 확장 되어 계산할 수 있다.



In this case a 3x1 matrix is element-wise multiplied by a 1x4 matrix to produce a 3x4 matrix. 
Note how the leading 1 is optional: The shape of y is `[4]`.

3x1행렬은 요소별로 1x4행렬을 곱하여 3x4행렬을 생성한다.  
선행 1이 선택사항인 방법을 참고해라, y의 모양은 [4]이다. (???) 

(3x1이 1x4가 되서 그런가, 행인가 열인가를 잘 보라는 말인가...)

```python
# These are the same computations
x = tf.reshape(x,[3,1])
y = tf.range(1, 5)
print(x, "\n")
print(y, "\n")
print(tf.multiply(x, y))

-----
tf.Tensor(
[[1]
 [2]
 [3]], shape=(3, 1), dtype=int32) 

tf.Tensor([1 2 3 4], shape=(4,), dtype=int32) 

tf.Tensor(
[[ 1  2  3  4]
 [ 2  4  6  8]
 [ 3  6  9 12]], shape=(3, 4), dtype=int32)
```

| A broadcasted add: a `[3, 1]` times a `[1, 4]` gives a `[3,4]` |
| :----------------------------------------------------------- |
| ![Adding a 3x1 matrix to a 4x1 matrix results in a 3x4 matrix](https://www.tensorflow.org/guide/images/tensor/broadcasting.png) |



Here is the same operation without broadcasting:

브로드 캐스팅없이 작업하는 내용이다.

```python
x_stretch = tf.constant([[1, 1, 1, 1],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3]])

y_stretch = tf.constant([[1, 2, 3, 4],
                         [1, 2, 3, 4],
                         [1, 2, 3, 4]])

print(x_stretch * y_stretch)  # Again, operator overloading


-----
tf.Tensor(
[[ 1  2  3  4]
 [ 2  4  6  8]
 [ 3  6  9 12]], shape=(3, 4), dtype=int32)
```

확실히 브로드 캐스팅이 편리한 것을 볼 수 있다.



Most of the time, broadcasting is both time and space efficient, as the broadcast operation never materializes the expanded tensors in memory.

대부분의 경우 브로드 캐스팅 작업은 확장 된 Tensor를 메모리에서 구현하지 않기 때문에 시간과 공간적으로 모두 효율적이다.



You see what broadcasting looks like using [`tf.broadcast_to`](https://www.tensorflow.org/api_docs/python/tf/broadcast_to).

tf.broadcast_to를 통해 브로드캐스팅(의 모습)을 볼 수 있다.

```python
print(tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))

-----
tf.Tensor(
[[1 2 3]
 [1 2 3]
 [1 2 3]], shape=(3, 3), dtype=int32)
```



Unlike a mathematical op, for example, `broadcast_to` does nothing special to save memory. Here, you are materializing the tensor.

예를 들어 수학적 연산과 다르게 broadcast_to 는 메모리를 절약하는데 특별한 작업이 없다.
여기에서 텐서를 구체화 하고 있다(?)



It can get even more complicated. [This section](https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html) of Jake VanderPlas's book *Python Data Science Handbook* shows more broadcasting tricks (again in NumPy).

그것은 더 복잡해 질 수 있다. Jake VanderPlas의 책 *Python Data Science Handbook* 에는 더 많은 브로드캐스팅 방법이 있다.





tf.convert_to_tensor
--

Most ops, like [`tf.matmul`](https://www.tensorflow.org/api_docs/python/tf/linalg/matmul) and [`tf.reshape`](https://www.tensorflow.org/api_docs/python/tf/reshape) take arguments of class [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor). However, you'll notice in the above case, Python objects shaped like tensors are accepted.

tf.matmul이나 tf.reshape 같은 대부분의 ops는 tf.Tensor 클레스 인수를 받는다. 하지만 이런 상황일 때 텐서 모양의 Python 객체는 허용된다.



Most, but not all, ops call `convert_to_tensor` on non-tensor arguments. There is a registry of conversions, and most object classes like NumPy's `ndarray`, `TensorShape`, Python lists, and [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) will all convert automatically.

전부는 아니지만 대부분의 ops는 convert_to_tensor가 아닌 non-tensor 인자를 호출한다.
변환 레지스트리와 넘파이 ndarray, TensorShape, Python lists, tf.Variabl와 같은 대부분의 객체 클레스는 자동적으로 변환된다.



See [`tf.register_tensor_conversion_function`](https://www.tensorflow.org/api_docs/python/tf/register_tensor_conversion_function) for more details, and if you have your own type you'd like to automatically convert to a tensor.

tf.register_tensor_conversion_function에 자세한 내용이 있으며 가지고 있는 것이 고유한 타입일 경우 자동으로 변환된다.





Ragged Tensors (비정형 텐서)
--

A tensor with variable numbers of elements along some axis is called "ragged". Use `tf.ragged.RaggedTensor` for ragged data.

축을 따라 가변 개수의 요소가 있는 텐서를 "비정형"이라고 부른다.



For example, This cannot be represented as a regular tensor:

예를들어 이건 일반적인 텐서라고 할 수 없다.

| A [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor), shape: `[4, None]` |
| :----------------------------------------------------------- |
| ![A 2-axis ragged tensor, each row can have a different length.](https://www.tensorflow.org/guide/images/tensor/ragged.png) |

```python
ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]

try:
  tensor = tf.constant(ragged_list)
except Exception as e:
  print(f"{type(e).__name__}: {e}")

-----
ValueError: Can't convert non-rectangular Python sequence to Tensor.
```



Instead create a [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor) using [`tf.ragged.constant`](https://www.tensorflow.org/api_docs/python/tf/ragged/constant):

대신 tf.ragged.constant를 이용하여 tf.RaggedTensor을 만들 수 있다.

```python
ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)

-----
<tf.RaggedTensor [[0, 1, 2, 3], [4, 5], [6, 7, 8], [9]]>
```



The shape of a [`tf.RaggedTensor`](https://www.tensorflow.org/api_docs/python/tf/RaggedTensor) will contain some axes with unknown lengths:

tf.RaggedTensor의 모양에는 축의 길이를 알수 없는 축이 포함된다. (None)

```python
print(ragged_tensor.shape)

-----
(4, None)
```





String tensors
--

[`tf.string`](https://www.tensorflow.org/api_docs/python/tf#string) is a `dtype`, which is to say you can represent data as strings (variable-length byte arrays) in tensors.

tf.string 은 dtype으로, 데이터를 텐서에서 문자열(가변길이의 바이트 배열)로 나타낼 수 있다.



The strings are atomic and cannot be indexed the way Python strings are. The length of the string is not one of the axes of the tensor. See [`tf.strings`](https://www.tensorflow.org/api_docs/python/tf/strings) for functions to manipulate them.

문자열은 원자(atomic)로 Python 문자열 처럼 인덱싱은 할 수 없다. 문자열의 길이는 텐서의 축 중 하나가 아니다.
tf.string 의 함수를 보고 manipulate 해라... (조작하려면 기능을 참조해라?)



Here is a scalar string tensor:

```python
# Tensors can be strings, too here is a scalar string.
scalar_string_tensor = tf.constant("Gray wolf")
print(scalar_string_tensor)

-----
tf.Tensor(b'Gray wolf', shape=(), dtype=string)
```

And a vector of strings:

| A vector of strings, shape: `[3,]`                           |
| :----------------------------------------------------------- |
| ![The string length is not one of the tensor's axes.](https://www.tensorflow.org/guide/images/tensor/strings.png) |

```python
# If you have three string tensors of different lengths, this is OK.
tensor_of_strings = tf.constant(["Gray wolf",
                                 "Quick brown fox",
                                 "Lazy dog"])
# Note that the shape is (3,). The string length is not included.
print(tensor_of_strings)

-----
tf.Tensor([b'Gray wolf' b'Quick brown fox' b'Lazy dog'], shape=(3,), dtype=string)
```



In the above printout the `b` prefix indicates that [`tf.string`](https://www.tensorflow.org/api_docs/python/tf#string) dtype is not a unicode string, but a byte-string. See the [Unicode Tutorial](https://www.tensorflow.org/tutorials/load_data/unicode) for more about working with unicode text in TensorFlow.

위의 출력에 b를 포함하는 이유는 tf.string dtype가 유니코드 문자열이 아닌 바이트 문자열임을 나타낸다.
TensorFlow에서 유니 코드 텍스트 작업을 하려면 Unicode Tutorial을 참조하해서 하면 좋다.



If you pass unicode characters they are utf-8 encoded.

유니코드 문자를 전달하면 utf-8로 인코딩된다.

```python
tf.constant("🥳👍")

-----
<tf.Tensor: shape=(), dtype=string, numpy=b'\xf0\x9f\xa5\xb3\xf0\x9f\x91\x8d'>
```



Some basic functions with strings can be found in [`tf.strings`](https://www.tensorflow.org/api_docs/python/tf/strings), including [`tf.strings.split`](https://www.tensorflow.org/api_docs/python/tf/strings/split).

tf.strings.split을 포함해서 tf.strings에서 몇개의 기본적인 함수를 찾을(사용할) 수 있다.

```python
# You can use split to split a string into a set of tensors
print(tf.strings.split(scalar_string_tensor, sep=" "))

-----
tf.Tensor([b'Gray' b'wolf'], shape=(2,), dtype=string)
```



```python
# ...but it turns into a `RaggedTensor` if you split up a tensor of strings,
# as each string might be split into a different number of parts.
print(tf.strings.split(tensor_of_strings))

-----
<tf.RaggedTensor [[b'Gray', b'wolf'], [b'Quick', b'brown', b'fox'], [b'Lazy', b'dog']]>
```

| Three strings split, shape: `[3, None]`                      |
| :----------------------------------------------------------- |
| ![Splitting multiple strings returns a tf.RaggedTensor](https://www.tensorflow.org/guide/images/tensor/string-split.png) |



And `tf.string.to_number`:

```python
text = tf.constant("1 10 100")
print(tf.strings.to_number(tf.strings.split(text, " ")))

-----
tf.Tensor([  1.  10. 100.], shape=(3,), dtype=float32)
```



Although you can't use [`tf.cast`](https://www.tensorflow.org/api_docs/python/tf/cast) to turn a string tensor into numbers, you can convert it into bytes, and then into numbers.

tf.cast는 문자열 텐서를 숫자로 변환할 수는 없지만 이것을 바이트로 변환할 수 있다. 그리고 변환한 바이트를 숫자로 바꿀 수 있다.

```python
byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)

-----
Byte strings: tf.Tensor([b'D' b'u' b'c' b'k'], shape=(4,), dtype=string)
Bytes: tf.Tensor([ 68 117  99 107], shape=(4,), dtype=uint8)
```



```python
# Or split it up as unicode and then decode it
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")

print("\nUnicode bytes:", unicode_bytes)
print("\nUnicode chars:", unicode_char_bytes)
print("\nUnicode values:", unicode_values)

-----
Unicode bytes: tf.Tensor(b'\xe3\x82\xa2\xe3\x83\x92\xe3\x83\xab \xf0\x9f\xa6\x86', shape=(), dtype=string)

Unicode chars: tf.Tensor([b'\xe3\x82\xa2' b'\xe3\x83\x92' b'\xe3\x83\xab' b' ' b'\xf0\x9f\xa6\x86'], shape=(5,), dtype=string)

Unicode values: tf.Tensor([ 12450  12498  12523     32 129414], shape=(5,), dtype=int32)
```



The [`tf.string`](https://www.tensorflow.org/api_docs/python/tf#string) dtype is used for all raw bytes data in TensorFlow. The [`tf.io`](https://www.tensorflow.org/api_docs/python/tf/io) module contains functions for converting data to and from bytes, including decoding images and parsing csv.

tf.string dtype는 텐서블로우의 모든 바이트데이터의 행에 사용된다. tf.io모듈에는 이미지 디코딩 및 csv구문 분석을 포함하여 데이터를 바이트로 또는 바이트에서 변환하는 기능이 있다.





Sparse tensors (희소 텐서)
--

Sometimes, your data is sparse, like a very wide embedding space. TensorFlow supports [`tf.sparse.SparseTensor`](https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor) and related operations to store sparse data efficiently.

때로는 매우 넓은 임베딩 공간처럼 데이터가 희소할 수 있다. 텐서플로우는  희소데이터를 효율적으로 저장하는 희소 텐서 및 관련작업을 지원하는 tf.sparse.SparseTensor기능을 지원한다.

| A [`tf.SparseTensor`](https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor), shape: `[3, 4]` |
| :----------------------------------------------------------- |
| ![An 3x4 grid, with values in only two of the cells.](https://www.tensorflow.org/guide/images/tensor/sparse.png) |

```python
# Sparse tensors store values by index in a memory-efficient manner
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")

# You can convert sparse tensors to dense
print(tf.sparse.to_dense(sparse_tensor))

-----
SparseTensor(indices=tf.Tensor(
[[0 0]
 [1 2]], shape=(2, 2), dtype=int64), values=tf.Tensor([1 2], shape=(2,), dtype=int32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64)) 

tf.Tensor(
[[1 0 0 0]
 [0 0 2 0]
 [0 0 0 0]], shape=(3, 4), dtype=int32)
```

-----

