

Better performance with tf.function
==



- 참고 : https://www.tensorflow.org/guide/function

---

In TensorFlow 2, eager execution is turned on by default. 
텐서플로 2에서는 즉시 실행(eager execution)이 기본적으로 활성화되어 있습니다. 

The user interface is intuitive and flexible (running one-off operations is much easier and faster), but this can come at the expense of performance and deployability.
직관적이고 유연한 사용자 인터페이스를 제공하지만 성능과 배포에 비용이 더 듭니다(하나의 연산을 실행할 때는 훨씬 간단하고 빠릅니다).



You can use [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) to make graphs out of your programs. It is a transformation tool that creates Python-independent dataflow graphs out of your Python code. This will help you create performant and portable models, and it is required to use `SavedModel`.
성능을 높이고 이식성이 좋은 모델을 만들려면 [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)을 사용해 그래프로 변환하세요. 하지만 조심해야 할 점이 있습니다. [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)은 무조건 속도를 높여주는 마법의 은총알이 아닙니다!



This guide will help you conceptualize how [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) works under the hood so you can use it effectively.
이 가이드는 [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)의 이면에 있는 개념을 이해하고 사용법을 완전히 터득할 수 있도록 도울 것입니다.



The main takeaways and recommendations are:
여기서 배울 주요 내용과 권고 사항은 다음과 같습니다:

- Debug in eager mode, then decorate with `@tf.function`.
  - 즉시 실행 모드에서 디버깅한 다음 `@tf.function`으로 데코레이팅하세요.
- Don't rely on Python side effects like object mutation or list appends.
  - 객체 변경(object mutation)이나 리스트 요소 추가 같은 파이썬의 부수 효과에 의존하지 마세요.
- [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) works best with TensorFlow ops; NumPy and Python calls are converted to constants.
  - [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)은 텐서플로 연산과 가장 잘 동작합니다: 넘파이와 파이썬 호출은 상수로 바뀝니다.

---



## Setup

```python
import tensorflow as tf
```

Define a helper function to demonstrate the kinds of errors you might encounter:
에러 출력을 위한 헬퍼 함수를 정의합니다:

```python
import traceback
import contextlib

# Some helper code to demonstrate the kinds of errors you might encounter.
# 에러 출력을 위한 헬퍼 함수
@contextlib.contextmanager
def assert_raises(error_class):
  try:
    yield
  except error_class as e:
    print('Caught expected exception \n  {}:'.format(error_class))
	# 기대하는 예외 발생
    traceback.print_exc(limit=2)
  except Exception as e:
    raise e
  else:
    raise Exception('Expected {} to be raised but no error was raised!'.format(
        error_class))
	# {}를 기대했지만 아무런 에러도 발생되지 않았습니다!
```

---



## Basics

### Usage

A `Function` you define (for example by applying the `@tf.function` decorator) is just like a core TensorFlow operation: You can execute it eagerly; you can compute gradients; and so on.
[`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)으로 정의한 함수는 기본 텐서플로 연산과 같습니다. 즉시 실행 모드로 실행하거나 그레이디언트를 계산할 수 있습니다.

```python
@tf.function  # The decorator converts `add` into a `Function`.
def add(a, b):
  return a + b

add(tf.ones([2, 2]), tf.ones([2, 2]))  #  [[2., 2.], [2., 2.]]
```

실행한 결과는 다음과 같다.

```python
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[2., 2.],
       [2., 2.]], dtype=float32)>
```



```python
v = tf.Variable(1.0)
with tf.GradientTape() as tape:
  result = add(v, 1.0)
tape.gradient(result, v)
```

실행한 결과는 다음과 같다.

```python
<tf.Tensor: shape=(), dtype=float32, numpy=1.0>
```



You can use `Function`s inside other `Function`s.
다른 함수 내부에 사용할 수 있습니다.

```python
@tf.function
def dense_layer(x, w, b):
  return add(tf.matmul(x, w), b)

dense_layer(tf.ones([3, 2]), tf.ones([2, 2]), tf.ones([2]))
```

실행한 결과는 다음과 같다.

```python
<tf.Tensor: shape=(3, 2), dtype=float32, numpy=
array([[3., 3.],
       [3., 3.],
       [3., 3.]], dtype=float32)>
```



`Function`s can be faster than eager code, especially for graphs with many small ops. But for graphs with a few expensive ops (like convolutions), you may not see much speedup.
[`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)은 즉시 실행 모드 보다 빠릅니다. 특히 그래프에 작은 연산이 많을 때 그렇습니다. 하지만 (합성곱처럼) 계산량이 많은 연산 몇 개로 이루어진 그래프는 속도 향상이 크지 않습니다.

```python
import timeit
conv_layer = tf.keras.layers.Conv2D(100, 3)

@tf.function
def conv_fn(image):
  return conv_layer(image)

image = tf.zeros([1, 200, 200, 100])
# warm up
conv_layer(image); conv_fn(image)
print("Eager conv:", timeit.timeit(lambda: conv_layer(image), number=10))
print("Function conv:", timeit.timeit(lambda: conv_fn(image), number=10))
print("Note how there's not much difference in performance for convolutions")
```

실행한 결과는 다음과 같다.

```python
Eager conv: 0.003693543000053978
Function conv: 0.004675635000012335
Note how there's not much difference in performance for convolutions
```

---



### Tracing

This section exposes how `Function` works under the hood, including implementation details *which may change in the future*. However, once you understand why and when tracing happens, it's much easier to use [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) effectively!
이 섹션에서는 향후 변경 될 수있는Function 구현 세부 정보 를 포함하여 내부에서 작동 하는 방식을 보여줍니다 . 그러나 추적이 발생하는 이유와시기를 이해하면 tf.function효과적으로 사용하는 것이 훨씬 쉽습니다 !

#### What is "tracing"?

A `Function` runs your program in a [TensorFlow Graph](https://www.tensorflow.org/guide/intro_to_graphs#what_are_graphs). However, a [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) cannot represent all the things that you'd write in an eager TensorFlow program. For instance, Python supports polymorphism, but [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) requires its inputs to have a specified data type and dimension. Or you may perform side tasks like reading command-line arguments, raising an error, or working with a more complex Python object; none of these things can run in a [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph).
A Function는 TensorFlow Graph 에서 프로그램을 실행합니다 . 그러나 tf.Graph열망 TensorFlow 프로그램에서 작성하는 모든 것을 나타낼 수는 없습니다. 예를 들어 Python은 다형성을 지원하지만 tf.Graph입력에 지정된 데이터 유형과 차원이 있어야합니다. 또는 명령 줄 인수 읽기, 오류 발생 또는 더 복잡한 Python 객체 작업과 같은 부수적 인 작업을 수행 할 수 있습니다. 이들 중 어느 것도 tf.Graph.



`Function` bridges this gap by separating your code in two stages:
Function 코드를 두 단계로 분리하여이 격차를 해소합니다.



1) In the first stage, referred to as "**tracing**", `Function` creates a new [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph). Python code runs normally, but all TensorFlow operations (like adding two Tensors) are *deferred*: they are captured by the [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) and not run.
1) " 추적 " 이라고하는 첫 번째 단계에서 Function새 tf.Graph. Python 코드는 정상적으로 실행되지만 모든 TensorFlow 작업 (예 : 두 개의 Tensor 추가)은 지연됩니다 . tf.Graph실행되지 않고에 의해 캡처됩니다 .



2) In the second stage, a [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) which contains everything that was deferred in the first stage is run. This stage is much faster than the tracing stage.

2) 두 번째 단계에서는 tf.Graph첫 번째 단계에서 연기 된 모든 것을 포함 하는 a 가 실행됩니다. 이 단계는 추적 단계보다 훨씬 빠릅니다.

Depending on its inputs, `Function` will not always run the first stage when it is called. See ["Rules of tracing"](https://www.tensorflow.org/guide/function#rules_of_tracing) below to get a better sense of how it makes that determination. Skipping the first stage and only executing the second stage is what gives you TensorFlow's high performance.
입력에 따라 Function호출 될 때 항상 첫 번째 단계를 실행하지는 않습니다. 그 결정을 내리는 방법을 더 잘 이해하려면 아래의 "추적 규칙"을 참조하십시오 . 첫 번째 단계를 건너 뛰고 두 번째 단계 만 실행하면 TensorFlow의 고성능을 얻을 수 있습니다.

When `Function` does decide to trace, the tracing stage is immediately followed by the second stage, so calling the `Function` both creates and runs the [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph). Later you will see how you can run only the tracing stage with [`get_concrete_function`](https://www.tensorflow.org/guide/function#obtaining_concrete_functions).
때 Function추적하기로 결정 않습니다는 추적 단계는 바로 이렇게 호출, 두 번째 단계 뒤에 Function모두 작성하고 실행 tf.Graph. 나중에를 사용하여 추적 단계 만 실행하는 방법을 볼 수 있습니다 get_concrete_function.



When we pass arguments of different types into a `Function`, both stages are run:
다른 종류의 매개변수를 함수를 호출할 때 무슨 일이 일어나는지 확인해 보죠.

```python
@tf.function
def double(a):
  print("Tracing with", a)
  return a + a

print(double(tf.constant(1)))
print()
print(double(tf.constant(1.1)))
print()
print(double(tf.constant("a")))
print()
```

실행한 결과는 다음과 같다.

```python
Tracing with Tensor("a:0", shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)

Tracing with Tensor("a:0", shape=(), dtype=float32)
tf.Tensor(2.2, shape=(), dtype=float32)

Tracing with Tensor("a:0", shape=(), dtype=string)
tf.Tensor(b'aa', shape=(), dtype=string)
```



Note that if you repeatedly call a `Function` with the same argument type, TensorFlow will skip the tracing stage and reuse a previously traced graph, as the generated graph would be identical.
Function동일한 인수 유형 으로를 반복적으로 호출 하면 생성 된 그래프가 동일하므로 TensorFlow가 추적 단계를 건너 뛰고 이전에 추적 한 그래프를 재사용합니다.

```python
# This doesn't print 'Tracing with ...'
print(double(tf.constant("b")))
```

실행한 결과는 다음과 같다.

```python
tf.Tensor(b'bb', shape=(), dtype=string)
```



You can use `pretty_printed_concrete_signatures()` to see all of the available traces:
를 사용 pretty_printed_concrete_signatures()하여 사용 가능한 모든 추적을 볼 수 있습니다 .

```python
print(double.pretty_printed_concrete_signatures())
```

실행한 결과는 다음과 같다.

```python
double(a)
  Args:
    a: float32 Tensor, shape=()
  Returns:
    float32 Tensor, shape=()

double(a)
  Args:
    a: string Tensor, shape=()
  Returns:
    string Tensor, shape=()

double(a)
  Args:
    a: int32 Tensor, shape=()
  Returns:
    int32 Tensor, shape=()
```



So far, you've seen that [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) creates a cached, dynamic dispatch layer over TensorFlow's graph tracing logic. To be more specific about the terminology:
지금까지 [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)TensorFlow의 그래프 추적 로직을 통해 캐시 된 동적 디스패치 레이어 를 생성하는 것을 확인했습니다 . 용어에 대해 더 구체적으로 설명하려면 :



- A [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) is the raw, language-agnostic, portable representation of a TensorFlow computation.
  - A [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph)는 TensorFlow 계산을 언어에 구애받지 않고 이식 가능한 원시 표현입니다.
- A `ConcreteFunction` wraps a [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph).
  - A `ConcreteFunction`는 [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph).
- A `Function` manages a cache of `ConcreteFunction`s and picks the right one for your inputs.
  - A `Function`는 `ConcreteFunction`s 의 캐시를 관리하고 입력에 적합한 캐시를 선택합니다.
- [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) wraps a Python function, returning a `Function` object.
  - [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)Python 함수를 래핑하여 `Function`객체를 반환 합니다.
- **Tracing** creates a [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) and wraps it in a `ConcreteFunction`, also known as a **trace.**
  - **추적은를** 만들고 **추적** 이라고도 [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph)하는으로 래핑합니다 **.**`ConcreteFunction`

---



#### Rules of tracing

A `Function` determines whether to reuse a traced `ConcreteFunction` by computing a **cache key** from an input's args and kwargs. A **cache key** is a key that identifies a `ConcreteFunction` based on the input args and kwargs of the `Function` call, according to the following rules (which may change):

A 는 입력의 args 및 kwargs에서 **캐시 키** 를 계산 `Function`하여 추적을 재사용할지 여부를 결정합니다 . **캐시 키** 식별이있는 키이며 , 입력 인수와의 kwargs로에 기초하여 다음과 같은 규칙 (변경 될 수 있음)에 따라, 통화 :`ConcreteFunction``ConcreteFunction``Function`



- The key generated for a [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) is its shape and dtype.
  - a [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor)에 대해 생성 된 키 는 모양과 dtype입니다.
- The key generated for a [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) is a unique variable id.
  - 에 대해 생성 된 키 [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable)는 고유 한 변수 ID입니다.
- The key generated for a Python primitive (like `int`, `float`, `str`) is its value.
  - Python 기본 요소 (예 `int`: `float`,, `str`)에 대해 생성 된 키 는 해당 값입니다.
- The key generated for nested `dict`s, `list`s, `tuple`s, `namedtuple`s, and [`attr`](https://www.attrs.org/en/stable/)s is the flattened tuple of leaf-keys (see [`nest.flatten`](https://www.tensorflow.org/api_docs/python/tf/nest/flatten)). (As a result of this flattening, calling a concrete function with a different nesting structure than the one used during tracing will result in a TypeError).
  - 중첩 된 `dict`s, `list`s, `tuple`s, `namedtuple`s 및 [`attr`](https://www.attrs.org/en/stable/)s에 대해 생성 된 키는 리프 키의 평면화 된 튜플입니다 (참조 [`nest.flatten`](https://www.tensorflow.org/api_docs/python/tf/nest/flatten)). (이 평탄화의 결과로 추적 중에 사용 된 것과 다른 중첩 구조로 구체적인 함수를 호출하면 TypeError가 발생합니다.)
- For all other Python types, the keys are based on the object `id()` so that methods are traced independently for each instance of a class.
  - 다른 모든 Python 유형의 경우 키는 객체를 기반으로 id()하므로 메서드가 클래스의 각 인스턴스에 대해 독립적으로 추적됩니다.

**Note:** Cache keys are based on the `Function` input parameters so changes to global and [free variables](https://docs.python.org/3/reference/executionmodel.html#binding-of-names) alone will not create a new trace. See [this section](https://www.tensorflow.org/guide/function#depending_on_python_global_and_free_variables) for recommended practices when dealing with Python global and free variables.

**참고 :** 캐시 키는 `Function`입력 매개 변수를 기반으로 하므로 전역 및 사용 [가능한 변수](https://docs.python.org/3/reference/executionmodel.html#binding-of-names) 만 변경해도 새 추적이 생성되지 않습니다. Python 전역 및 자유 변수를 처리 할 때 권장되는 방법 은 [이 섹션](https://www.tensorflow.org/guide/function#depending_on_python_global_and_free_variables) 을 참조하세요 .

---



#### Controlling retracing

Retracing, which is when your `Function` creates more than one trace, helps ensures that TensorFlow generates correct graphs for each set of inputs. However, tracing is an expensive operation! If your `Function` retraces a new graph for every call, you'll find that your code executes more slowly than if you didn't use [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function).
당신이 때 인 되돌아 `Function`하나 이상의 추적을 생성 TensorFlow는 입력의 각 세트에 대한 정확한 그래프를 생성하는 것을 보장하는 데 도움이됩니다. 그러나 추적은 비용이 많이 드는 작업입니다! 귀하의 경우 `Function`되짚어을 호출 할 때마다 새로운 그래프, 당신은 당신의 코드가 실행이 더 느리게 이상 사용하지 않은 경우 찾을 수 있습니다 [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function).



To control the tracing behavior, you can use the following techniques:

추적 동작을 제어하려면 다음 기술을 사용할 수 있습니다.



- Specify `input_signature` in [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) to limit tracing.
  - 추적을 제한하려면 `input_signature`in [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)을 지정하십시오 .

```python
@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def next_collatz(x):
  print("Tracing with", x)
  return tf.where(x % 2 == 0, x // 2, 3 * x + 1)

print(next_collatz(tf.constant([1, 2])))
# We specified a 1-D tensor in the input signature, so this should fail.
with assert_raises(ValueError):
  next_collatz(tf.constant([[1, 2], [3, 4]]))

# We specified an int32 dtype in the input signature, so this should fail.
with assert_raises(ValueError):
  next_collatz(tf.constant([1.0, 2.0]))
```

실행한 결과는 다음과 같다.

```python
Tracing with Tensor("x:0", shape=(None,), dtype=int32)
tf.Tensor([4 1], shape=(2,), dtype=int32)
Caught expected exception 
  <class 'ValueError'>:
Caught expected exception 
  <class 'ValueError'>:
Traceback (most recent call last):
  File "<ipython-input-1-73d0ca52e838>", line 8, in assert_raises
    yield
  File "<ipython-input-1-20f544b8adbf>", line 9, in <module>
    next_collatz(tf.constant([[1, 2], [3, 4]]))
ValueError: Python inputs incompatible with input_signature:
  inputs: (
    tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32))
  input_signature: (
    TensorSpec(shape=(None,), dtype=tf.int32, name=None))
Traceback (most recent call last):
  File "<ipython-input-1-73d0ca52e838>", line 8, in assert_raises
    yield
  File "<ipython-input-1-20f544b8adbf>", line 13, in <module>
    next_collatz(tf.constant([1.0, 2.0]))
ValueError: Python inputs incompatible with input_signature:
  inputs: (
    tf.Tensor([1. 2.], shape=(2,), dtype=float32))
  input_signature: (
    TensorSpec(shape=(None,), dtype=tf.int32, name=None))
```



- Specify a [None] dimension in [`tf.TensorSpec`](https://www.tensorflow.org/api_docs/python/tf/TensorSpec) to allow for flexibility in trace reuse.
[`tf.TensorSpec`](https://www.tensorflow.org/api_docs/python/tf/TensorSpec)추적 재사용의 유연성을 허용하려면 [None] 차원을 지정하십시오 .
  
  
  
  Since TensorFlow matches tensors based on their shape, using a `None` dimension as a wildcard will allow `Function`s to reuse traces for variably-sized input. Variably-sized input can occur if you have sequences of different length, or images of different sizes for each batch (See [Transformer](https://www.tensorflow.org/tutorials/text/transformer) and [Deep Dream](https://www.tensorflow.org/tutorials/generative/deepdream) tutorials for example).
  TensorFlow는 모양에 따라 텐서와 일치하므로 `None`차원을 와일드 카드로 사용하면 `Function`s가 다양한 크기의 입력에 대한 추적을 재사용 할 수 있습니다. 길이가 다른 시퀀스 또는 각 배치에 대해 다른 크기의 이미지가있는 경우 가변 크기 입력이 발생할 수 있습니다 (예 : [Transformer](https://www.tensorflow.org/tutorials/text/transformer) 및 [Deep Dream](https://www.tensorflow.org/tutorials/generative/deepdream) 자습서 참조 ).

```python
@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def g(x):
  print('Tracing with', x)
  return x

# No retrace!
print(g(tf.constant([1, 2, 3])))
print(g(tf.constant([1, 2, 3, 4, 5])))
```

실행한 결과는 다음과 같다.

```python
Tracing with Tensor("x:0", shape=(None,), dtype=int32)
tf.Tensor([1 2 3], shape=(3,), dtype=int32)
tf.Tensor([1 2 3 4 5], shape=(5,), dtype=int32)
```



- Cast Python arguments to Tensors to reduce retracing.
재 추적을 줄이기 위해 Python 인수를 Tensor로 캐스팅합니다.
  

  
  Often, Python arguments are used to control hyperparameters and graph constructions - for example, `num_layers=10` or `training=True` or `nonlinearity='relu'`. So if the Python argument changes, it makes sense that you'd have to retrace the graph.
  
  종종 Python 인수는 하이퍼 파라미터 및 그래프 구성을 제어하는 데 사용됩니다 (예 : `num_layers=10`또는 `training=True`또는) `nonlinearity='relu'`. 따라서 Python 인수가 변경되면 그래프를 다시 추적해야합니다.
  
  
  
  However, it's possible that a Python argument is not being used to control graph construction. In these cases, a change in the Python value can trigger needless retracing. Take, for example, this training loop, which AutoGraph will dynamically unroll. Despite the multiple traces, the generated graph is actually identical, so retracing is unnecessary.
  
  그러나 Python 인수가 그래프 구성을 제어하는 데 사용되지 않을 수 있습니다. 이러한 경우 Python 값이 변경되면 불필요한 재 추적이 트리거 될 수 있습니다. 예를 들어, AutoGraph가 동적으로 펼쳐지는 트레이닝 루프를 생각해보십시오. 여러 추적에도 불구하고 생성 된 그래프는 실제로 동일하므로 다시 추적 할 필요가 없습니다.

```python
def train_one_step():
  pass

@tf.function
def train(num_steps):
  print("Tracing with num_steps = ", num_steps)
  tf.print("Executing with num_steps = ", num_steps)
  for _ in tf.range(num_steps):
    train_one_step()

print("Retracing occurs for different Python arguments.")
train(num_steps=10)
train(num_steps=20)

print()
print("Traces are reused for Tensor arguments.")
train(num_steps=tf.constant(10))
train(num_steps=tf.constant(20))
```

실행한 결과는 다음과 같다.

```python
Retracing occurs for different Python arguments.
Tracing with num_steps =  10
Executing with num_steps =  10
Tracing with num_steps =  20
Executing with num_steps =  20

Traces are reused for Tensor arguments.
Tracing with num_steps =  Tensor("num_steps:0", shape=(), dtype=int32)
Executing with num_steps =  10
Executing with num_steps =  20
```



If you need to force retracing, create a new `Function`. Separate `Function` objects are guaranteed not to share traces.
강제로 다시 추적해야하는 경우 새 `Function`. 별도의 `Function`개체는 추적을 공유하지 않도록 보장됩니다.

```python
def f():
  print('Tracing!')
  tf.print('Executing')

tf.function(f)()
tf.function(f)()
```

실행한 결과는 다음과 같다.

```python
Tracing!
Executing
Tracing!
Executing
```

---



### Obtaining concrete functions

Every time a function is traced, a new concrete function is created. You can directly obtain a concrete function, by using `get_concrete_function`.
함수가 추적 될 때마다 새로운 구체적인 함수가 생성됩니다. 를 사용하여 구체적인 기능을 직접 얻을 수 있습니다 `get_concrete_function`.

```python
print("Obtaining concrete trace")
double_strings = double.get_concrete_function(tf.constant("a"))
print("Executing traced function")
print(double_strings(tf.constant("a")))
print(double_strings(a=tf.constant("b")))
```

실행한 결과는 다음과 같다.

```python
Obtaining concrete trace
Executing traced function
tf.Tensor(b'aa', shape=(), dtype=string)
tf.Tensor(b'bb', shape=(), dtype=string)
```



```python
# You can also call get_concrete_function on an InputSpec
double_strings_from_inputspec = double.get_concrete_function(tf.TensorSpec(shape=[], dtype=tf.string))
print(double_strings_from_inputspec(tf.constant("c")))
```

실행한 결과는 다음과 같다.

```
Tracing with Tensor("a:0", shape=(), dtype=string)
tf.Tensor(b'cc', shape=(), dtype=string)
```



Printing a `ConcreteFunction` displays a summary of its input arguments (with types) and its output type.
인쇄는 `ConcreteFunction`입력 인수 (유형 포함) 및 출력 유형의 요약을 표시합니다.

```python
print(double_strings)
```

실행한 결과는 다음과 같다.

```
ConcreteFunction double(a)
  Args:
    a: string Tensor, shape=()
  Returns:
    string Tensor, shape=()
```



You can also directly retrieve a concrete function's signature.
구체적인 함수의 서명을 직접 검색 할 수도 있습니다.

```python
print(double_strings.structured_input_signature)
print(double_strings.structured_outputs)
```

실행한 결과는 다음과 같다.

```
((TensorSpec(shape=(), dtype=tf.string, name='a'),), {})
Tensor("Identity:0", shape=(), dtype=string)
```



Using a concrete trace with incompatible types will throw an error
호환되지 않는 유형의 구체적인 추적을 사용하면 오류가 발생합니다.

```python
with assert_raises(tf.errors.InvalidArgumentError):
  double_strings(tf.constant(1))
```

실행한 결과는 다음과 같다.

```
Caught expected exception 
  <class 'tensorflow.python.framework.errors_impl.InvalidArgumentError'>:
Traceback (most recent call last):
  File "<ipython-input-1-73d0ca52e838>", line 8, in assert_raises
    yield
  File "<ipython-input-1-e4e2860a4364>", line 2, in <module>
    double_strings(tf.constant(1))
tensorflow.python.framework.errors_impl.InvalidArgumentError: cannot compute __inference_double_162 as input #0(zero-based) was expected to be a string tensor but is a int32 tensor [Op:__inference_double_162]
```



You may notice that Python arguments are given special treatment in a concrete function's input signature. Prior to TensorFlow 2.3, Python arguments were simply removed from the concrete function's signature. Starting with TensorFlow 2.3, Python arguments remain in the signature, but are constrained to take the value set during tracing.
구체적인 함수의 입력 시그니처에서 Python 인수가 특별하게 처리된다는 것을 알 수 있습니다. TensorFlow 2.3 이전에는 Python 인수가 콘크리트 함수의 서명에서 제거되었습니다. TensorFlow 2.3부터 Python 인수는 서명에 남아 있지만 추적 중에 설정된 값을 사용하도록 제한됩니다.

```python
@tf.function
def pow(a, b):
  return a ** b

square = pow.get_concrete_function(a=tf.TensorSpec(None, tf.float32), b=2)
print(square)
```

실행한 결과는 다음과 같다.

```
ConcreteFunction pow(a, b=2)
  Args:
    a: float32 Tensor, shape=<unknown>
  Returns:
    float32 Tensor, shape=<unknown>
```



```python
assert square(tf.constant(10.0)) == 100

with assert_raises(TypeError):
  square(tf.constant(10.0), b=3)
```

실행한 결과는 다음과 같다.

```
Caught expected exception 
  <class 'TypeError'>:
Traceback (most recent call last):
  File "/tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/eager/function.py", line 1683, in _call_impl
    cancellation_manager)
  File "/tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/eager/function.py", line 1728, in _call_with_flat_signature
    self._flat_signature_summary(), ", ".join(sorted(kwargs))))
TypeError: pow(a) got unexpected keyword arguments: b.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<ipython-input-1-73d0ca52e838>", line 8, in assert_raises
    yield
  File "<ipython-input-1-d163f3d206cb>", line 4, in <module>
    square(tf.constant(10.0), b=3)
TypeError: ConcreteFunction pow(a, b) was constructed with int value 2 in b, but was called with int value 3
```



---



### Obtaining graphs

### 그래프 얻기

Each concrete function is a callable wrapper around a [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph). Although retrieving the actual [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) object is not something you'll normally need to do, you can obtain it easily from any concrete function.
각 구체적인 함수는 [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph). 실제 [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph)객체를 검색하는 것은 일반적으로 수행해야 할 작업이 아니지만 구체적인 기능에서 쉽게 얻을 수 있습니다.

```python
graph = double_strings.graph
for node in graph.as_graph_def().node:
  print(f'{node.input} -> {node.name}')
```

실행한 결과는 다음과 같다.

```
[] -> a
['a', 'a'] -> add
['add'] -> Identity
```



---



### Debugging

In general, debugging code is easier in eager mode than inside [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function). 
일반적으로 [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) 보다 즉시 실행 모드가 디버깅하기 쉽습니다. 



You should ensure that your code executes error-free in eager mode before decorating with [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function). To assist in the debugging process, you can call [`tf.config.run_functions_eagerly(True)`](https://www.tensorflow.org/api_docs/python/tf/config/run_functions_eagerly) to globally disable and reenable [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function).
데코레이팅하기 전에 즉시 실행 모드에서 에러가 없는지 확인하세요. 디버깅 과정을 위해 [`tf.config.run_functions_eagerly(True)`](https://www.tensorflow.org/api_docs/python/tf/config/run_functions_eagerly)으로 전체 [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)을 비활성화하고 나중에 다시 활성화할 수 있습니다.



When tracking down issues that only appear within [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function), here are some tips:
[`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) 함수에서 버그를 추적할 때 다음 팁을 참고하세요:
에만 나타나는 문제를 추적 할 때 tf.function다음과 같은 몇 가지 팁을 얻을 수 있습니다.



- Plain old Python `print` calls only execute during tracing, helping you track down when your function gets (re)traced.
  - 파이썬 `print` 함수는 트레이싱(tracing)하는 동안에만 호출되므로 함수가 (재)트레이싱될 때 추적하는데 도움이 됩니다.
  - 평범한 이전 Python print호출은 추적 중에 만 실행되므로 함수가 (재) 추적 될 때 추적하는 데 도움이됩니다.
- [`tf.print`](https://www.tensorflow.org/api_docs/python/tf/print) calls will execute every time, and can help you track down intermediate values during execution.
  - [`tf.print`](https://www.tensorflow.org/api_docs/python/tf/print) 함수는 언제나 실행되므로 실행하는 동안 중간 값을 추적할 때 도움이 됩니다.
  - tf.print 호출은 매번 실행되며 실행 중에 중간 값을 추적하는 데 도움이 될 수 있습니다.
- [`tf.debugging.enable_check_numerics`](https://www.tensorflow.org/api_docs/python/tf/debugging/enable_check_numerics) is an easy way to track down where NaNs and Inf are created.
  - [`tf.debugging.enable_check_numerics`](https://www.tensorflow.org/api_docs/python/tf/debugging/enable_check_numerics)을 사용하면 쉽게 NaN과 Inf가 발생되는 곳을 추적할 수 있습니다.
  - tf.debugging.enable_check_numerics NaN과 Inf가 생성 된 위치를 쉽게 추적 할 수있는 방법입니다.
- `pdb` can help you understand what's going on during tracing. (Caveat: PDB will drop you into AutoGraph-transformed source code.)
  - `pdb`는 어떻게 트레이싱이 일어나는지 이해하는데 도움이 됩니다(주의: `pdb`는 오토그래프(AutoGraph)가 변환한 소스 코드를 보여줄 것입니다).
  - pdb추적하는 동안 무슨 일이 일어나고 있는지 이해하는 데 도움이 될 수 있습니다. (주의 : PDB는 AutoGraph로 변환 된 소스 코드에 빠져들게합니다.)

---



## AutoGraph Transformations



AutoGraph is a library that is on by default in [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function), and transforms a subset of Python eager code into graph-compatible TensorFlow ops. This includes control flow like `if`, `for`, `while`.
AutoGraph는에서 기본적으로 켜져있는 라이브러리 [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function)이며 Python eager 코드의 하위 집합을 그래프 호환 TensorFlow 작업으로 변환합니다. 이 같은 제어 흐름을 포함하는 `if`, `for`, `while`.

TensorFlow ops like [`tf.cond`](https://www.tensorflow.org/api_docs/python/tf/cond) and [`tf.while_loop`](https://www.tensorflow.org/api_docs/python/tf/while_loop) continue to work, but control flow is often easier to write and understand when written in Python.

TensorFlow ops는 좋아 tf.cond하고 tf.while_loop계속 작동하지만 제어 흐름은 Python으로 작성할 때 작성하고 이해하기가 더 쉽습니다.

```python
# Simple loop

@tf.function
def f(x):
  while tf.reduce_sum(x) > 1:
    tf.print(x)
    x = tf.tanh(x)
  return x

f(tf.random.uniform([5]))
```

실행한 결과는 다음과 같다.

```
[0.150836706 0.575092077 0.530339956 0.492499471 0.696070552]
[0.149703071 0.519089103 0.485640883 0.456198 0.601867676]
[0.148594663 0.476996601 0.450749785 0.426980466 0.538377285]
[0.147510558 0.44383505 0.422515154 0.402794749 0.491758645]
[0.146449849 0.416818231 0.399047196 0.382337689 0.45561108]
[0.145411745 0.394246608 0.379133403 0.364735901 0.42650038]
[0.144395441 0.375015408 0.361954659 0.349378705 0.402392507]
[0.143400162 0.358370811 0.346934587 0.3358244 0.381994188]
[0.142425224 0.343778163 0.333654165 0.323744506 0.364438057]
[0.141469926 0.330846161 0.321800381 0.312888771 0.34911716]
[0.140533626 0.319280863 0.311133921 0.303062797 0.335592359]
[0.139615685 0.308856487 0.301468313 0.294112951 0.323536754]
[0.13871555 0.299396455 0.292655736 0.285915941 0.312701344]
[0.137832627 0.290760159 0.28457731 0.278371543 0.302892596]
[0.136966363 0.282834291 0.277136147 0.271397203 0.293957472]
[0.136116222 0.275526226 0.270252436 0.264924437 0.285773188]
[0.135281757 0.268759459 0.263859689 0.258895695 0.278239816]
[0.134462461 0.262470126 0.257902056 0.253262341 0.271275192]
[0.133657902 0.256604493 0.252332211 0.24798286 0.264810979]
[0.132867619 0.251116842 0.247109696 0.243021563 0.258789837]
[0.132091209 0.245968238 0.242199793 0.238347709 0.253163248]
[0.131328285 0.241125017 0.237572461 0.233934492 0.247889832]
[0.130578429 0.236558065 0.233201504 0.229758516 0.242934018]
[0.129841298 0.232242063 0.229064062 0.225799173 0.238265097]
[0.129116505 0.228154778 0.22514002 0.222038344 0.233856365]
[0.128403738 0.224276677 0.221411601 0.218459949 0.229684487]
[0.127702653 0.220590457 0.217863053 0.215049684 0.225728899]
<tf.Tensor: shape=(5,), dtype=float32, numpy=
array([0.12701295, 0.21708074, 0.21448033, 0.21179478, 0.22197153],
      dtype=float32)>
```



If you're curious you can inspect the code autograph generates.

궁금하다면 사인이 생성하는 코드를 살펴볼 수 있습니다.

```python
print(tf.autograph.to_code(f.python_function))
```

실행한 결과는 다음과 같다.

```
def tf__f(x):
    with ag__.FunctionScope('f', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
        do_return = False
        retval_ = ag__.UndefinedReturnValue()

        def get_state():
            return (x,)

        def set_state(vars_):
            nonlocal x
            (x,) = vars_

        def loop_body():
            nonlocal x
            ag__.converted_call(ag__.ld(tf).print, (ag__.ld(x),), None, fscope)
            x = ag__.converted_call(ag__.ld(tf).tanh, (ag__.ld(x),), None, fscope)

        def loop_test():
            return (ag__.converted_call(ag__.ld(tf).reduce_sum, (ag__.ld(x),), None, fscope) > 1)
        ag__.while_stmt(loop_test, loop_body, get_state, set_state, ('x',), {})
        try:
            do_return = True
            retval_ = ag__.ld(x)
        except:
            do_return = False
            raise
        return fscope.ret(retval_, do_return)
```

---



### Conditionals

### 조건부

AutoGraph will convert some `if <condition>` statements into the equivalent `tf.cond` calls. This substitution is made if `<condition>` is a Tensor. Otherwise, the `if` statement is executed as a Python conditional.

AutoGraph는 일부 if <condition>문을 동등한 tf.cond호출 로 변환 합니다. 이 대체는 <condition>텐서 인 경우 이루어집니다 . 그렇지 않으면 if명령문은 Python 조건부로 실행됩니다.



A Python conditional executes during tracing, so exactly one branch of the conditional will be added to the graph. Without AutoGraph, this traced graph would be unable to take the alternate branch if there is data-dependent control flow.

Python 조건부는 추적 중에 실행되므로 정확히 조건부 분기 하나가 그래프에 추가됩니다. AutoGraph가 없으면이 추적 된 그래프는 데이터 종속 제어 흐름이있는 경우 대체 분기를 사용할 수 없습니다.



[`tf.cond`](https://www.tensorflow.org/api_docs/python/tf/cond) traces and adds both branches of the conditional to the graph, dynamically selecting a branch at execution time. Tracing can have unintended side effects; see [AutoGraph tracing effects](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/control_flow.md#effects-of-the-tracing-process) for more.

tf.cond조건부 분기를 추적하고 그래프에 추가하여 실행시 분기를 동적으로 선택합니다. 추적에는 의도하지 않은 부작용이있을 수 있습니다. 자세한 내용은 AutoGraph 추적 효과 를 참조하십시오 .

```python
@tf.function
def fizzbuzz(n):
  for i in tf.range(1, n + 1):
    print('Tracing for loop')
    if i % 15 == 0:
      print('Tracing fizzbuzz branch')
      tf.print('fizzbuzz')
    elif i % 3 == 0:
      print('Tracing fizz branch')
      tf.print('fizz')
    elif i % 5 == 0:
      print('Tracing buzz branch')
      tf.print('buzz')
    else:
      print('Tracing default branch')
      tf.print(i)

fizzbuzz(tf.constant(5))
fizzbuzz(tf.constant(20))
```

실행한 결과는 다음과 같다.

```
Tracing for loop
Tracing fizzbuzz branch
Tracing fizz branch
Tracing buzz branch
Tracing default branch
1
2
fizz
4
buzz
1
2
fizz
4
buzz
fizz
7
8
fizz
buzz
11
fizz
13
14
fizzbuzz
16
17
fizz
19
buzz
```

See the [reference documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/control_flow.md#if-statements) for additional restrictions on AutoGraph-converted if statements.

AutoGraph-converted if 문에 대한 추가 제한 사항 은 참조 문서 를 참조 하십시오 .

---



### Loops

AutoGraph will convert some `for` and `while` statements into the equivalent TensorFlow looping ops, like [`tf.while_loop`](https://www.tensorflow.org/api_docs/python/tf/while_loop). If not converted, the `for` or `while` loop is executed as a Python loop.

AutoGraph는 일부 for및 while문을 같은 텐서 플로우 루핑 연산으로 변환 tf.while_loop합니다. 변환되지 않으면 foror while루프가 Python 루프로 실행됩니다.



This substitution is made in the following situations:

이 대체는 다음 상황에서 이루어집니다.



- `for x in y`: if `y` is a Tensor, convert to [`tf.while_loop`](https://www.tensorflow.org/api_docs/python/tf/while_loop). In the special case where `y` is a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), a combination of [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) ops are generated.

  - for x in y: y텐서 인 경우 tf.while_loop. 특별한 경우 y인 tf.data.Dataset의 조합 tf.data.Dataset작전이 생성됩니다.

    

- `while <condition>`: if `<condition>` is a Tensor, convert to `tf.while_loop`.

  - while <condition>: <condition>텐서 인 경우 tf.while_loop.

    

A Python loop executes during tracing, adding additional ops to the [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) for every iteration of the loop.

Python 루프는 추적 중에 실행 tf.Graph되어 루프의 모든 반복에 대해에 추가 작업을 추가 합니다.



A TensorFlow loop traces the body of the loop, and dynamically selects how many iterations to run at execution time. The loop body only appears once in the generated [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph).

TensorFlow 루프는 루프의 본문을 추적하고 실행 시간에 실행할 반복 횟수를 동적으로 선택합니다. 루프 본문은 생성 된 tf.Graph.



See the [reference documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/control_flow.md#while-statements) for additional restrictions on AutoGraph-converted `for` and `while` statements.

AutoGraph-converted 및 문 에 대한 추가 제한 사항 은 참조 문서 를 참조 하십시오 .forwhile

---



#### Looping over Python data

Python 데이터 루핑



A common pitfall is to loop over Python/Numpy data within a [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function). This loop will execute during the tracing process, adding a copy of your model to the [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) for each iteration of the loop.

일반적인 함정은 .NET 파일 내에서 Python / Numpy 데이터를 반복하는 것 tf.function입니다. 이 루프는 추적 프로세스 중에 실행 tf.Graph되어 루프의 각 반복마다 모델 사본을에 추가합니다 .



If you want to wrap the entire training loop in [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function), the safest way to do this is to wrap your data as a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) so that AutoGraph will dynamically unroll the training loop.

전체 학습 루프를에서 래핑하려는 경우 tf.function가장 안전한 방법은 데이터를로 래핑하여 tf.data.DatasetAutoGraph가 학습 루프를 동적으로 풀도록하는 것입니다.



```python
def measure_graph_size(f, *args):
  g = f.get_concrete_function(*args).graph
  print("{}({}) contains {} nodes in its graph".format(
      f.__name__, ', '.join(map(str, args)), len(g.as_graph_def().node)))

@tf.function
def train(dataset):
  loss = tf.constant(0)
  for x, y in dataset:
    loss += tf.abs(y - x) # Some dummy computation.
  return loss

small_data = [(1, 1)] * 3
big_data = [(1, 1)] * 10
measure_graph_size(train, small_data)
measure_graph_size(train, big_data)

measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: small_data, (tf.int32, tf.int32)))
measure_graph_size(train, tf.data.Dataset.from_generator(
    lambda: big_data, (tf.int32, tf.int32)))
```

실행한 결과는 다음과 같다.

```
train([(1, 1), (1, 1), (1, 1)]) contains 11 nodes in its graph
train([(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]) contains 32 nodes in its graph
train(<FlatMapDataset shapes: (<unknown>, <unknown>), types: (tf.int32, tf.int32)>) contains 10 nodes in its graph
train(<FlatMapDataset shapes: (<unknown>, <unknown>), types: (tf.int32, tf.int32)>) contains 10 nodes in its graph
```

When wrapping Python/Numpy data in a Dataset, be mindful of [`tf.data.Dataset.from_generator`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator) versus `tf.data.Dataset.from_tensors`. The former will keep the data in Python and fetch it via [`tf.py_function`](https://www.tensorflow.org/api_docs/python/tf/py_function) which can have performance implications, whereas the latter will bundle a copy of the data as one large [`tf.constant()`](https://www.tensorflow.org/api_docs/python/tf/constant) node in the graph, which can have memory implications.

Python / Numpy 데이터를 Dataset에 래핑 할 때 tf.data.Dataset.from_generator대 tf.data.Dataset.from_tensors. 전자는 데이터를 Python에 보관하고이를 통해 tf.py_function성능에 영향을 미칠 수 있는 데이터를 가져 오는 반면 후자는 데이터 사본을 tf.constant()그래프에서 하나의 큰 노드 로 묶어 메모리에 영향을 줄 수 있습니다.



Reading data from files via TFRecordDataset/CsvDataset/etc. is the most effective way to consume data, as then TensorFlow itself can manage the asynchronous loading and prefetching of data, without having to involve Python. To learn more, see the [tf.data guide](https://www.tensorflow.org/guide/data).

TFRecordDataset / CsvDataset / etc를 통해 파일에서 데이터 읽기. 데이터를 소비하는 가장 효과적인 방법입니다. TensorFlow 자체는 Python을 사용하지 않고도 데이터의 비동기로드 및 프리 페치를 관리 할 수 있습니다. 자세한 내용은 tf.data 가이드를 참조하세요 .

---



#### Accumulating values in a loop

루프에서 값 누적



A common pattern is to accumulate intermediate values from a loop. Normally, this is accomplished by appending to a Python list or adding entries to a Python dictionary. However, as these are Python side effects, they will not work as expected in a dynamically unrolled loop. Use [`tf.TensorArray`](https://www.tensorflow.org/api_docs/python/tf/TensorArray) to accumulate results from a dynamically unrolled loop.

일반적인 패턴은 루프에서 중간 값을 축적하는 것입니다. 일반적으로 이것은 Python 목록에 추가하거나 Python 사전에 항목을 추가하여 수행됩니다. 그러나 이들은 Python 부작용이므로 동적으로 펼쳐진 루프에서 예상대로 작동하지 않습니다. tf.TensorArray동적으로 펼쳐진 루프의 결과를 누적하는 데 사용 합니다.

```python
batch_size = 2
seq_len = 3
feature_size = 4

def rnn_step(inp, state):
  return inp + state

@tf.function
def dynamic_rnn(rnn_step, input_data, initial_state):
  # [batch, time, features] -> [time, batch, features]
  input_data = tf.transpose(input_data, [1, 0, 2])
  max_seq_len = input_data.shape[0]

  states = tf.TensorArray(tf.float32, size=max_seq_len)
  state = initial_state
  for i in tf.range(max_seq_len):
    state = rnn_step(input_data[i], state)
    states = states.write(i, state)
  return tf.transpose(states.stack(), [1, 0, 2])

dynamic_rnn(rnn_step,
            tf.random.uniform([batch_size, seq_len, feature_size]),
            tf.zeros([batch_size, feature_size]))
```

실행한 결과는 다음과 같다.

```
<tf.Tensor: shape=(2, 3, 4), dtype=float32, numpy=
array([[[0.6345552 , 0.3862481 , 0.8089291 , 0.397156  ],
        [0.8435192 , 0.46313608, 1.2150625 , 1.2924602 ],
        [0.84762645, 0.9920107 , 1.6375196 , 1.8461396 ]],

       [[0.44864273, 0.8552673 , 0.36585057, 0.88470626],
        [0.6609589 , 1.4641674 , 0.79369175, 1.0593072 ],
        [0.8057562 , 2.4129076 , 1.5868478 , 1.8529567 ]]], dtype=float32)>
```

---



## Limitations

TensorFlow `Function` has a few limitations by design that you should be aware of when converting a Python function to a `Function`.

TensorFlow Function에는 Python 함수를 .NET Framework 로 변환 할 때 알아야 할 몇 가지 제한 사항이 있습니다 Function.

---



### Executing Python side effects

Python 부작용 실행



Side effects, like printing, appending to lists, and mutating globals, can behave unexpectedly inside a `Function`, sometimes executing twice or not all. They only happen the first time you call a `Function` with a set of inputs. Afterwards, the traced [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) is reexecuted, without executing the Python code.

인쇄, 목록에 추가 및 전역 변경과 같은 부작용은에서 예기치 않게 작동 할 수 Function있으며 때로는 두 번 또는 전부 실행되지 않을 수 있습니다. Function일련의 입력으로 a를 처음 호출 할 때만 발생합니다 . 그 후 추적 된 파일 tf.Graph은 Python 코드를 실행하지 않고 다시 실행됩니다.



The general rule of thumb is to avoid relying on Python side effects in your logic and only use them to debug your traces. Otherwise, TensorFlow APIs like [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data), [`tf.print`](https://www.tensorflow.org/api_docs/python/tf/print), [`tf.summary`](https://www.tensorflow.org/api_docs/python/tf/summary), [`tf.Variable.assign`](https://www.tensorflow.org/api_docs/python/tf/Variable#assign), and [`tf.TensorArray`](https://www.tensorflow.org/api_docs/python/tf/TensorArray) are the best way to ensure your code will be executed by the TensorFlow runtime with each call.

경험의 일반적인 규칙은 로직에서 Python 부작용에 의존하지 않고 추적을 디버그하는 데만 사용하는 것입니다. 그렇지 않으면, TensorFlow API는 좋아 tf.data, tf.print, tf.summary, tf.Variable.assign, 및 tf.TensorArray코드를 확인하는 가장 좋은 방법은 각 호출로 TensorFlow 런타임에 의해 실행됩니다된다.



```python
@tf.function
def f(x):
  print("Traced with", x)
  tf.print("Executed with", x)

f(1)
f(1)
f(2)
```

실행한 결과는 다음과 같다.

```
Traced with 1
Executed with 1
Executed with 1
Traced with 2
Executed with 2
```

If you would like to execute Python code during each invocation of a `Function`, [`tf.py_function`](https://www.tensorflow.org/api_docs/python/tf/py_function) is an exit hatch. The drawback of [`tf.py_function`](https://www.tensorflow.org/api_docs/python/tf/py_function) is that it's not portable or particularly performant, cannot be saved with SavedModel, and does not work well in distributed (multi-GPU, TPU) setups. Also, since [`tf.py_function`](https://www.tensorflow.org/api_docs/python/tf/py_function) has to be wired into the graph, it casts all inputs/outputs to tensors.

당신은 각각의 호출 중에 파이썬 코드를 실행하려는 경우 Function, tf.py_function출구 해치입니다. 단점은 tf.py_function이식성이 없거나 특히 성능이 뛰어나고 SavedModel로 저장할 수 없으며 분산 (다중 GPU, TPU) 설정에서 잘 작동하지 않는다는 것입니다. 또한 tf.py_function그래프에 연결되어야하므로 모든 입력 / 출력을 텐서로 캐스팅합니다.

---



#### Changing Python global and free variables

Python 전역 및 자유 변수 변경



Changing Python global and [free variables](https://docs.python.org/3/reference/executionmodel.html#binding-of-names) counts as a Python side effect, so it only happens during tracing.

Python 전역 및 자유 변수 변경 은 Python 부작용으로 간주되므로 추적 중에 만 발생합니다.

```python
external_list = []

@tf.function
def side_effect(x):
  print('Python side effect')
  external_list.append(x)

side_effect(1)
side_effect(1)
side_effect(1)
# The list append only happened once!
assert len(external_list) == 1
```

실행한 결과는 다음과 같다.

```
Python side effect
```

You should avoid mutating containers like lists, dicts, other objects that live outside the `Function`. Instead, use arguments and TF objects. For example, the section ["Accumulating values in a loop"](https://www.tensorflow.org/guide/function#accumulating_values_in_a_loop) has one example of how list-like operations can be implemented.

목록, 딕셔너리, .NET Framework 외부에있는 기타 개체와 같은 컨테이너 변경을 피해야합니다 Function. 대신 인수 및 TF 개체를 사용하십시오. 예를 들어, "루프에서 값 누적" 섹션 에는 목록과 유사한 작업을 구현할 수있는 방법에 대한 한 가지 예가 있습니다.



You can, in some cases, capture and manipulate state if it is a [`tf.Variable`](https://www.tensorflow.org/guide/variable). This is how the weights of Keras models are updated with repeated calls to the same `ConcreteFunction`.

경우에 따라 상태를 캡처하고 조작 할 수 있습니다 tf.Variable. 이것이 Keras 모델의 가중치가 동일한 ConcreteFunction.

---



#### Using Python iterators and generators

Python 반복기 및 생성기 사용



Many Python features, such as generators and iterators, rely on the Python runtime to keep track of state. In general, while these constructs work as expected in eager mode, they are examples of Python side effects and therefore only happen during tracing.

생성기 및 반복기와 같은 많은 Python 기능은 상태를 추적하기 위해 Python 런타임에 의존합니다. 일반적으로 이러한 구조는 eager 모드에서 예상대로 작동하지만 Python 부작용의 예이므로 추적 중에 만 발생합니다.

```python
@tf.function
def buggy_consume_next(iterator):
  tf.print("Value:", next(iterator))

iterator = iter([1, 2, 3])
buggy_consume_next(iterator)
# This reuses the first value from the iterator, rather than consuming the next value.
buggy_consume_next(iterator)
buggy_consume_next(iterator)
```

실행한 결과는 다음과 같다.

```
Value: 1
Value: 1
Value: 1
```



Just like how TensorFlow has a specialized [`tf.TensorArray`](https://www.tensorflow.org/api_docs/python/tf/TensorArray) for list constructs, it has a specialized [`tf.data.Iterator`](https://www.tensorflow.org/api_docs/python/tf/data/Iterator) for iteration constructs. See the section on [AutoGraph Transformations](https://www.tensorflow.org/guide/function#autograph_transformations) for an overview. Also, the [`tf.data`](https://www.tensorflow.org/guide/data) API can help implement generator patterns:

TensorFlow가 tf.TensorArray목록 구조에 특화된 것처럼 tf.data.Iterator반복 구조에 특화 되어 있습니다. 개요 는 자동 그래프 변환 섹션을 참조하십시오 . 또한 tf.dataAPI는 생성기 패턴을 구현하는 데 도움이 될 수 있습니다.



```python
@tf.function
def good_consume_next(iterator):
  # This is ok, iterator is a tf.data.Iterator
  tf.print("Value:", next(iterator))

ds = tf.data.Dataset.from_tensor_slices([1, 2, 3])
iterator = iter(ds)
good_consume_next(iterator)
good_consume_next(iterator)
good_consume_next(iterator)
```

실행한 결과는 다음과 같다.

```
Value: 1
Value: 2
Value: 3
```

---



### Deleting tf.Variables between `Function` calls

Function호출 간 tf.Variables 삭제



Another error you may encounter is a garbage-collected variable. `ConcreteFunction`s only retain [WeakRefs](https://docs.python.org/3/library/weakref.html) to the variables they close over, so you must retain a reference to any variables.

발생할 수있는 또 다른 오류는 가비지 수집 변수입니다. ConcreteFunctions는 닫히는 변수에 대한 WeakRefs 만 유지 하므로 모든 변수에 대한 참조를 유지해야합니다.

```python
external_var = tf.Variable(3)
@tf.function
def f(x):
  return x * external_var

traced_f = f.get_concrete_function(4)
print("Calling concrete function...")
print(traced_f(4))

# The original variable object gets garbage collected, since there are no more
# references to it.
external_var = tf.Variable(4)
print()
print("Calling concrete function after garbage collecting its closed Variable...")
with assert_raises(tf.errors.FailedPreconditionError):
  traced_f(4)
```

실행한 결과는 다음과 같다.

```
Calling concrete function...
tf.Tensor(12, shape=(), dtype=int32)

Calling concrete function after garbage collecting its closed Variable...
Caught expected exception 
  <class 'tensorflow.python.framework.errors_impl.FailedPreconditionError'>:
Traceback (most recent call last):
  File "<ipython-input-1-73d0ca52e838>", line 8, in assert_raises
    yield
  File "<ipython-input-1-9a93d2e07632>", line 16, in <module>
    traced_f(4)
tensorflow.python.framework.errors_impl.FailedPreconditionError: 2 root error(s) found.
  (0) Failed precondition:  Error while reading resource variable _AnonymousVar3 from Container: localhost. This could mean that the variable was uninitialized. Not found: Resource localhost/_AnonymousVar3/N10tensorflow3VarE does not exist.
     [[node ReadVariableOp (defined at <ipython-input-1-9a93d2e07632>:4) ]]
  (1) Failed precondition:  Error while reading resource variable _AnonymousVar3 from Container: localhost. This could mean that the variable was uninitialized. Not found: Resource localhost/_AnonymousVar3/N10tensorflow3VarE does not exist.
     [[node ReadVariableOp (defined at <ipython-input-1-9a93d2e07632>:4) ]]
     [[ReadVariableOp/_2]]
0 successful operations.
0 derived errors ignored. [Op:__inference_f_782]

Function call stack:
f -> f
```

---



## Known Issues

If your `Function` is not evaluating correctly, the error may be explained by these known issues which are planned to be fixed in the future.

Function올바르게 평가하지 않는 경우 오류는 향후 수정 될 예정인 알려진 문제로 설명 될 수 있습니다.

---



### Depending on Python global and free variables

Python 전역 및 자유 변수에 따라



`Function` creates a new `ConcreteFunction` when called with a new value of a Python argument. However, it does not do that for the Python closure, globals, or nonlocals of that `Function`. If their value changes in between calls to the `Function`, the `Function` will still use the values they had when it was traced. This is different from how regular Python functions work.

FunctionConcreteFunctionPython 인수의 새 값으로 호출 될 때 새를 만듭니다 . 그러나 Python 클로저, 전역 또는 비 로컬에 대해서는 그렇게하지 않습니다 Function. 자신의 값이 호출 사이에서 변경하는 경우 Function의는 Function여전히이 추적 될 때 보유했던 값을 사용합니다. 이것은 일반 Python 함수가 작동하는 방식과 다릅니다.



For that reason, we recommend a functional programming style that uses arguments instead of closing over outer names.

따라서 외부 이름을 닫는 대신 인수를 사용하는 함수형 프로그래밍 스타일을 권장합니다.

```python
@tf.function
def buggy_add():
  return 1 + foo

@tf.function
def recommended_add(foo):
  return 1 + foo

foo = 1
print("Buggy:", buggy_add())
print("Correct:", recommended_add(foo))
```

실행한 결과는 다음과 같다.

```
Buggy: tf.Tensor(2, shape=(), dtype=int32)
Correct: tf.Tensor(2, shape=(), dtype=int32)
```



```python
print("Updating the value of `foo` to 100!")
foo = 100
print("Buggy:", buggy_add())  # Did not change!
print("Correct:", recommended_add(foo))
```

실행한 결과는 다음과 같다.

```
Updating the value of `foo` to 100!
Buggy: tf.Tensor(2, shape=(), dtype=int32)
Correct: tf.Tensor(101, shape=(), dtype=int32)
```

You can close over outer names, as long as you don't update their values.

값을 업데이트하지 않는 한 외부 이름을 닫을 수 있습니다.

---



#### Depending on Python objects

Python 객체에 따라



The recommendation to pass Python objects as arguments into [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) has a number of known issues, that are expected to be fixed in the future. In general, you can rely on consistent tracing if you use a Python primitive or [`tf.nest`](https://www.tensorflow.org/api_docs/python/tf/nest)-compatible structure as an argument or pass in a *different* instance of an object into a `Function`. However, `Function` will *not* create a new trace when you pass **the same object and only change its attributes**.

Python 객체를 인수로 전달하라는 권장 사항 tf.function에는 향후 수정 될 것으로 예상되는 여러 알려진 문제가 있습니다. 일반적으로 Python 기본 또는 tf.nest호환 가능한 구조를 인수로 사용하거나 객체 의 다른 인스턴스를 .NET Framework에 전달 하는 경우 일관된 추적에 의존 할 수 있습니다 Function. 그러나 Function것 없는 당신이 통과 할 때 새 추적을 만들 **동일한 개체를 만의 속성을 변경 .**

```python
class SimpleModel(tf.Module):
  def __init__(self):
    # These values are *not* tf.Variables.
    self.bias = 0.
    self.weight = 2.

@tf.function
def evaluate(model, x):
  return model.weight * x + model.bias

simple_model = SimpleModel()
x = tf.constant(10.)
print(evaluate(simple_model, x))
```

실행한 결과는 다음과 같다.

```
tf.Tensor(20.0, shape=(), dtype=float32)
```



```python
print("Adding bias!")
simple_model.bias += 5.0
print(evaluate(simple_model, x))  # Didn't change :(
```

실행한 결과는 다음과 같다.

```
Adding bias!
tf.Tensor(20.0, shape=(), dtype=float32)
```



Using the same `Function` to evaluate the updated instance of the model will be buggy since the updated model has the [same cache key](https://www.tensorflow.org/guide/function#rules_of_tracing) as the original model.

동일한 Function모델을 사용하여 업데이트 된 모델 인스턴스를 평가하는 것은 업데이트 된 모델이 원래 모델과 동일한 캐시 키 를 갖기 때문에 버그가 있습니다 .



For that reason, we recommend that you write your `Function` to avoid depending on mutable object attributes or create new objects.

Function따라서 변경 가능한 객체 속성에 의존하지 않도록 작성하거나 새 객체를 생성 하는 것이 좋습니다 .



If that is not possible, one workaround is to make new `Function`s each time you modify your object to force retracing:

가능하지 않은 경우 한 가지 해결 방법은 Function개체를 수정하여 강제로 다시 추적 할 때마다 새를 만드는 것입니다 .

```python
def evaluate(model, x):
  return model.weight * x + model.bias

new_model = SimpleModel()
evaluate_no_bias = tf.function(evaluate).get_concrete_function(new_model, x)
# Don't pass in `new_model`, `Function` already captured its state during tracing.
print(evaluate_no_bias(x))
```

실행한 결과는 다음과 같다.

```
tf.Tensor(20.0, shape=(), dtype=float32)
```



```python
print("Adding bias!")
new_model.bias += 5.0
# Create new Function and ConcreteFunction since you modified new_model.
evaluate_with_bias = tf.function(evaluate).get_concrete_function(new_model, x)
print(evaluate_with_bias(x)) # Don't pass in `new_model`.
```

실행한 결과는 다음과 같다.

```
Adding bias!
tf.Tensor(25.0, shape=(), dtype=float32)
```



As [retracing can be expensive](https://www.tensorflow.org/guide/intro_to_graphs#tracing_and_performance), you can use [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable)s as object attributes, which can be mutated (but not changed, careful!) for a similar effect without needing a retrace.

으로 되돌아가 비쌀 수 있습니다 , 당신은 사용할 수 있습니다 tf.Variable귀선을하지 않고도 비슷한 효과 돌연변이 될 수 개체 특성, (그러나 변경하지 않도록주의!)로들.



```python
class BetterModel:

  def __init__(self):
    self.bias = tf.Variable(0.)
    self.weight = tf.Variable(2.)

@tf.function
def evaluate(model, x):
  return model.weight * x + model.bias

better_model = BetterModel()
print(evaluate(better_model, x))
```

실행한 결과는 다음과 같다.

```
tf.Tensor(20.0, shape=(), dtype=float32)
```



```python
print("Adding bias!")
better_model.bias.assign_add(5.0)  # Note: instead of better_model.bias += 5
print(evaluate(better_model, x))  # This works!
```

실행한 결과는 다음과 같다.

```
Adding bias!
tf.Tensor(25.0, shape=(), dtype=float32)
```

---



### Creating tf.Variables

tf.Variables 만들기



`Function` only supports creating variables once, when first called, and then reusing them. You cannot create `tf.Variables` in new traces. Creating new variables in subsequent calls is currently not allowed, but will be in the future.

Function변수를 처음 호출 할 때 한 번만 생성 한 다음 재사용 할 수 있습니다. tf.Variables새 추적에서 만들 수 없습니다 . 후속 호출에서 새 변수를 만드는 것은 현재 허용되지 않지만 앞으로는 가능합니다.



Example:

```python
@tf.function
def f(x):
  v = tf.Variable(1.0)
  return v

with assert_raises(ValueError):
  f(1.0)
```

실행한 결과는 다음과 같다.

```
Caught expected exception 
  <class 'ValueError'>:
Traceback (most recent call last):
  File "<ipython-input-1-73d0ca52e838>", line 8, in assert_raises
    yield
  File "<ipython-input-1-8a0913e250e0>", line 7, in <module>
    f(1.0)
ValueError: in user code:

    <ipython-input-1-8a0913e250e0>:3 f  *
        v = tf.Variable(1.0)
    /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/ops/variables.py:262 __call__  **
        return cls._variable_v2_call(*args, **kwargs)
    /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/ops/variables.py:256 _variable_v2_call
        shape=shape)
    /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/ops/variables.py:67 getter
        return captured_getter(captured_previous, **kwargs)
    /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/eager/def_function.py:731 invalid_creator_scope
        "tf.function-decorated function tried to create "

    ValueError: tf.function-decorated function tried to create variables on non-first call.
```



You can create variables inside a `Function` as long as those variables are only created the first time the function is executed.

Function함수가 처음 실행될 때만 해당 변수가 생성되는 한 내부에 변수를 생성 할 수 있습니다 .

```python
class Count(tf.Module):
  def __init__(self):
    self.count = None

  @tf.function
  def __call__(self):
    if self.count is None:
      self.count = tf.Variable(0)
    return self.count.assign_add(1)

c = Count()
print(c())
print(c())
```

실행한 결과는 다음과 같다.

```
tf.Tensor(1, shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)
```

---



#### Using with multiple Keras optimizers

여러 Keras 최적화 프로그램과 함께 사용



You may encounter `ValueError: tf.function-decorated function tried to create variables on non-first call.` when using more than one Keras optimizer with a `tf.function`. This error occurs because optimizers internally create `tf.Variables` when they apply gradients for the first time.

ValueError: tf.function-decorated function tried to create variables on non-first call..NET Framework와 함께 둘 이상의 Keras 최적화 프로그램을 사용할 때 발생할 수 있습니다 tf.function. 이 오류는 최적화 프로그램 tf.Variables이 처음으로 그라디언트를 적용 할 때 내부적으로 생성하기 때문에 발생합니다 .

```python
opt1 = tf.keras.optimizers.Adam(learning_rate = 1e-2)
opt2 = tf.keras.optimizers.Adam(learning_rate = 1e-3)

@tf.function
def train_step(w, x, y, optimizer):
   with tf.GradientTape() as tape:
       L = tf.reduce_sum(tf.square(w*x - y))
   gradients = tape.gradient(L, [w])
   optimizer.apply_gradients(zip(gradients, [w]))

w = tf.Variable(2.)
x = tf.constant([-1.])
y = tf.constant([2.])

train_step(w, x, y, opt1)
print("Calling `train_step` with different optimizer...")
with assert_raises(ValueError):
  train_step(w, x, y, opt2)
```

실행한 결과는 다음과 같다.

```
Calling `train_step` with different optimizer...
Caught expected exception 
  <class 'ValueError'>:
Traceback (most recent call last):
  File "<ipython-input-1-73d0ca52e838>", line 8, in assert_raises
    yield
  File "<ipython-input-1-d3d3937dbf1a>", line 18, in <module>
    train_step(w, x, y, opt2)
ValueError: in user code:

    <ipython-input-1-d3d3937dbf1a>:9 train_step  *
        optimizer.apply_gradients(zip(gradients, [w]))
    /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/keras/optimizer_v2/optimizer_v2.py:604 apply_gradients  **
        self._create_all_weights(var_list)
    /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/keras/optimizer_v2/optimizer_v2.py:781 _create_all_weights
        _ = self.iterations
    /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/keras/optimizer_v2/optimizer_v2.py:788 __getattribute__
        return super(OptimizerV2, self).__getattribute__(name)
    /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/keras/optimizer_v2/optimizer_v2.py:926 iterations
        aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA)
    /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/keras/optimizer_v2/optimizer_v2.py:1132 add_weight
        aggregation=aggregation)
    /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/training/tracking/base.py:810 _add_variable_with_custom_getter
        **kwargs_for_getter)
    /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/keras/engine/base_layer_utils.py:142 make_variable
        shape=variable_shape if variable_shape else None)
    /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/ops/variables.py:260 __call__
        return cls._variable_v1_call(*args, **kwargs)
    /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/ops/variables.py:221 _variable_v1_call
        shape=shape)
    /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/ops/variables.py:67 getter
        return captured_getter(captured_previous, **kwargs)
    /tmpfs/src/tf_docs_env/lib/python3.6/site-packages/tensorflow/python/eager/def_function.py:731 invalid_creator_scope
        "tf.function-decorated function tried to create "

    ValueError: tf.function-decorated function tried to create variables on non-first call.
```



If you need to change the optimizer during training, a workaround is to create a new `Function` for each optimizer, calling the [`ConcreteFunction`](https://www.tensorflow.org/guide/function#obtaining_concrete_functions) directly.

학습 중에 최적화 프로그램을 변경해야하는 경우 해결 방법은 Function각 최적화 프로그램에 대해 새로 생성 하여를 ConcreteFunction직접 호출하는 것 입니다.

```python
opt1 = tf.keras.optimizers.Adam(learning_rate = 1e-2)
opt2 = tf.keras.optimizers.Adam(learning_rate = 1e-3)

# Not a tf.function.
def train_step(w, x, y, optimizer):
   with tf.GradientTape() as tape:
       L = tf.reduce_sum(tf.square(w*x - y))
   gradients = tape.gradient(L, [w])
   optimizer.apply_gradients(zip(gradients, [w]))

w = tf.Variable(2.)
x = tf.constant([-1.])
y = tf.constant([2.])

# Make a new Function and ConcreteFunction for each optimizer.
train_step_1 = tf.function(train_step).get_concrete_function(w, x, y, opt1)
train_step_2 = tf.function(train_step).get_concrete_function(w, x, y, opt2)
for i in range(10):
  if i % 2 == 0:
    train_step_1(w, x, y) # `opt1` is not used as a parameter. 
  else:
    train_step_2(w, x, y) # `opt2` is not used as a parameter.
```

---



#### Using with multiple Keras models

여러 Keras 모델과 함께 사용



You may also encounter `ValueError: tf.function-decorated function tried to create variables on non-first call.` when passing different model instances to the same `Function`.

ValueError: tf.function-decorated function tried to create variables on non-first call.다른 모델 인스턴스를 동일한에 전달할 때도 발생할 수 있습니다 Function.



This error occurs because Keras models (which [do not have their input shape defined](https://www.tensorflow.org/guide/keras/custom_layers_and_models#best_practice_deferring_weight_creation_until_the_shape_of_the_inputs_is_known)) and Keras layers create `tf.Variables`s when they are first called. You may be attempting to initialize those variables inside a `Function`, which has already been called. To avoid this error, try calling `model.build(input_shape)` to initialize all the weights before training the model.

이 오류는 Keras 모델 ( 입력 모양이 정의되지 않음 )과 Keras 레이어 tf.Variables가 처음 호출 될 때를 생성 하기 때문에 발생합니다 . Function이미 호출 된 에서 해당 변수를 초기화하려고 할 수 있습니다 . 이 오류를 방지하려면 model.build(input_shape)모델을 훈련하기 전에 모든 가중치를 초기화하도록 호출 하십시오.

---



## Further reading

To learn about how to export and load a `Function`, see the [SavedModel guide](https://www.tensorflow.org/guide/saved_model). To learn more about graph optimizations that are performed after tracing, see the [Grappler guide](https://www.tensorflow.org/guide/graph_optimization). To learn how to optimize your data pipeline and profile your model, see the [Profiler guide](https://www.tensorflow.org/guide/profiler).

를 내보내고로드하는 방법에 대한 자세한 Function내용은 저장된 모델 가이드를 참조하십시오 . 추적 후 수행되는 그래프 최적화에 대해 자세히 알아 보려면 Grappler 가이드를 참조하세요 . 데이터 파이프 라인을 최적화하고 모델을 프로파일 링하는 방법을 알아 보려면 프로파일 러 가이드를 참조하십시오 .

---

