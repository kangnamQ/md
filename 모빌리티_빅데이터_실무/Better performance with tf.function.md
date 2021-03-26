

Better performance with tf.function
==



- 참고 : https://www.tensorflow.org/guide/function

---

In TensorFlow 2, eager execution is turned on by default. The user interface is intuitive and flexible (running one-off operations is much easier and faster), but this can come at the expense of performance and deployability.

You can use [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) to make graphs out of your programs. It is a transformation tool that creates Python-independent dataflow graphs out of your Python code. This will help you create performant and portable models, and it is required to use `SavedModel`.

This guide will help you conceptualize how [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) works under the hood so you can use it effectively.

The main takeaways and recommendations are:

- Debug in eager mode, then decorate with `@tf.function`.
- Don't rely on Python side effects like object mutation or list appends.
- [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) works best with TensorFlow ops; NumPy and Python calls are converted to constants.

---

## Setup

```python
import tensorflow as tf
```

Define a helper function to demonstrate the kinds of errors you might encounter:

```python
import traceback
import contextlib

# Some helper code to demonstrate the kinds of errors you might encounter.
@contextlib.contextmanager
def assert_raises(error_class):
  try:
    yield
  except error_class as e:
    print('Caught expected exception \n  {}:'.format(error_class))
    traceback.print_exc(limit=2)
  except Exception as e:
    raise e
  else:
    raise Exception('Expected {} to be raised but no error was raised!'.format(
        error_class))
```

---

## Basics

### Usage

A `Function` you define (for example by applying the `@tf.function` decorator) is just like a core TensorFlow operation: You can execute it eagerly; you can compute gradients; and so on.

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

#### What is "tracing"?

A `Function` runs your program in a [TensorFlow Graph](https://www.tensorflow.org/guide/intro_to_graphs#what_are_graphs). However, a [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) cannot represent all the things that you'd write in an eager TensorFlow program. For instance, Python supports polymorphism, but [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) requires its inputs to have a specified data type and dimension. Or you may perform side tasks like reading command-line arguments, raising an error, or working with a more complex Python object; none of these things can run in a [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph).

`Function` bridges this gap by separating your code in two stages:

1) In the first stage, referred to as "**tracing**", `Function` creates a new [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph). Python code runs normally, but all TensorFlow operations (like adding two Tensors) are *deferred*: they are captured by the [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) and not run.

2) In the second stage, a [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) which contains everything that was deferred in the first stage is run. This stage is much faster than the tracing stage.

Depending on its inputs, `Function` will not always run the first stage when it is called. See ["Rules of tracing"](https://www.tensorflow.org/guide/function#rules_of_tracing) below to get a better sense of how it makes that determination. Skipping the first stage and only executing the second stage is what gives you TensorFlow's high performance.

When `Function` does decide to trace, the tracing stage is immediately followed by the second stage, so calling the `Function` both creates and runs the [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph). Later you will see how you can run only the tracing stage with [`get_concrete_function`](https://www.tensorflow.org/guide/function#obtaining_concrete_functions).

When we pass arguments of different types into a `Function`, both stages are run:

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

```python
# This doesn't print 'Tracing with ...'
print(double(tf.constant("b")))
```

실행한 결과는 다음과 같다.

```python
tf.Tensor(b'bb', shape=(), dtype=string)
```



You can use `pretty_printed_concrete_signatures()` to see all of the available traces:

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

- A [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) is the raw, language-agnostic, portable representation of a TensorFlow computation.
- A `ConcreteFunction` wraps a [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph).
- A `Function` manages a cache of `ConcreteFunction`s and picks the right one for your inputs.
- [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) wraps a Python function, returning a `Function` object.
- **Tracing** creates a [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) and wraps it in a `ConcreteFunction`, also known as a **trace.**

---



#### Rules of tracing

A `Function` determines whether to reuse a traced `ConcreteFunction` by computing a **cache key** from an input's args and kwargs. A **cache key** is a key that identifies a `ConcreteFunction` based on the input args and kwargs of the `Function` call, according to the following rules (which may change):

- The key generated for a [`tf.Tensor`](https://www.tensorflow.org/api_docs/python/tf/Tensor) is its shape and dtype.
- The key generated for a [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable) is a unique variable id.
- The key generated for a Python primitive (like `int`, `float`, `str`) is its value.
- The key generated for nested `dict`s, `list`s, `tuple`s, `namedtuple`s, and [`attr`](https://www.attrs.org/en/stable/)s is the flattened tuple of leaf-keys (see [`nest.flatten`](https://www.tensorflow.org/api_docs/python/tf/nest/flatten)). (As a result of this flattening, calling a concrete function with a different nesting structure than the one used during tracing will result in a TypeError).
- For all other Python types, the keys are based on the object `id()` so that methods are traced independently for each instance of a class.

**Note:** Cache keys are based on the `Function` input parameters so changes to global and [free variables](https://docs.python.org/3/reference/executionmodel.html#binding-of-names) alone will not create a new trace. See [this section](https://www.tensorflow.org/guide/function#depending_on_python_global_and_free_variables) for recommended practices when dealing with Python global and free variables.

---



#### Controlling retracing

Retracing, which is when your `Function` creates more than one trace, helps ensures that TensorFlow generates correct graphs for each set of inputs. However, tracing is an expensive operation! If your `Function` retraces a new graph for every call, you'll find that your code executes more slowly than if you didn't use [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function).

To control the tracing behavior, you can use the following techniques:

- Specify `input_signature` in [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) to limit tracing.

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

  Since TensorFlow matches tensors based on their shape, using a `None` dimension as a wildcard will allow `Function`s to reuse traces for variably-sized input. Variably-sized input can occur if you have sequences of different length, or images of different sizes for each batch (See [Transformer](https://www.tensorflow.org/tutorials/text/transformer) and [Deep Dream](https://www.tensorflow.org/tutorials/generative/deepdream) tutorials for example).

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

  Often, Python arguments are used to control hyperparameters and graph constructions - for example, `num_layers=10` or `training=True` or `nonlinearity='relu'`. So if the Python argument changes, it makes sense that you'd have to retrace the graph.

  However, it's possible that a Python argument is not being used to control graph construction. In these cases, a change in the Python value can trigger needless retracing. Take, for example, this training loop, which AutoGraph will dynamically unroll. Despite the multiple traces, the generated graph is actually identical, so retracing is unnecessary.

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

--durdkfafd

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

Each concrete function is a callable wrapper around a [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph). Although retrieving the actual [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) object is not something you'll normally need to do, you can obtain it easily from any concrete function.

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

In general, debugging code is easier in eager mode than inside [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function). You should ensure that your code executes error-free in eager mode before decorating with [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function). To assist in the debugging process, you can call [`tf.config.run_functions_eagerly(True)`](https://www.tensorflow.org/api_docs/python/tf/config/run_functions_eagerly) to globally disable and reenable [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function).

When tracking down issues that only appear within [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function), here are some tips:

- Plain old Python `print` calls only execute during tracing, helping you track down when your function gets (re)traced.
- [`tf.print`](https://www.tensorflow.org/api_docs/python/tf/print) calls will execute every time, and can help you track down intermediate values during execution.
- [`tf.debugging.enable_check_numerics`](https://www.tensorflow.org/api_docs/python/tf/debugging/enable_check_numerics) is an easy way to track down where NaNs and Inf are created.
- `pdb` can help you understand what's going on during tracing. (Caveat: PDB will drop you into AutoGraph-transformed source code.)

---



## AutoGraph Transformations

AutoGraph is a library that is on by default in [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function), and transforms a subset of Python eager code into graph-compatible TensorFlow ops. This includes control flow like `if`, `for`, `while`.

TensorFlow ops like [`tf.cond`](https://www.tensorflow.org/api_docs/python/tf/cond) and [`tf.while_loop`](https://www.tensorflow.org/api_docs/python/tf/while_loop) continue to work, but control flow is often easier to write and understand when written in Python.

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

AutoGraph will convert some `if <condition>` statements into the equivalent `tf.cond` calls. This substitution is made if `<condition>` is a Tensor. Otherwise, the `if` statement is executed as a Python conditional.

A Python conditional executes during tracing, so exactly one branch of the conditional will be added to the graph. Without AutoGraph, this traced graph would be unable to take the alternate branch if there is data-dependent control flow.

[`tf.cond`](https://www.tensorflow.org/api_docs/python/tf/cond) traces and adds both branches of the conditional to the graph, dynamically selecting a branch at execution time. Tracing can have unintended side effects; see [AutoGraph tracing effects](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/control_flow.md#effects-of-the-tracing-process) for more.

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

---



### Loops

AutoGraph will convert some `for` and `while` statements into the equivalent TensorFlow looping ops, like [`tf.while_loop`](https://www.tensorflow.org/api_docs/python/tf/while_loop). If not converted, the `for` or `while` loop is executed as a Python loop.

This substitution is made in the following situations:

- `for x in y`: if `y` is a Tensor, convert to [`tf.while_loop`](https://www.tensorflow.org/api_docs/python/tf/while_loop). In the special case where `y` is a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), a combination of [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) ops are generated.
- `while <condition>`: if `<condition>` is a Tensor, convert to `tf.while_loop`.

A Python loop executes during tracing, adding additional ops to the [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) for every iteration of the loop.

A TensorFlow loop traces the body of the loop, and dynamically selects how many iterations to run at execution time. The loop body only appears once in the generated [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph).

See the [reference documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/control_flow.md#while-statements) for additional restrictions on AutoGraph-converted `for` and `while` statements.

---



#### Looping over Python data

A common pitfall is to loop over Python/Numpy data within a [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function). This loop will execute during the tracing process, adding a copy of your model to the [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) for each iteration of the loop.

If you want to wrap the entire training loop in [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function), the safest way to do this is to wrap your data as a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) so that AutoGraph will dynamically unroll the training loop.

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

Reading data from files via TFRecordDataset/CsvDataset/etc. is the most effective way to consume data, as then TensorFlow itself can manage the asynchronous loading and prefetching of data, without having to involve Python. To learn more, see the [tf.data guide](https://www.tensorflow.org/guide/data).

---



#### Accumulating values in a loop

A common pattern is to accumulate intermediate values from a loop. Normally, this is accomplished by appending to a Python list or adding entries to a Python dictionary. However, as these are Python side effects, they will not work as expected in a dynamically unrolled loop. Use [`tf.TensorArray`](https://www.tensorflow.org/api_docs/python/tf/TensorArray) to accumulate results from a dynamically unrolled loop.

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

---

### Executing Python side effects

Side effects, like printing, appending to lists, and mutating globals, can behave unexpectedly inside a `Function`, sometimes executing twice or not all. They only happen the first time you call a `Function` with a set of inputs. Afterwards, the traced [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) is reexecuted, without executing the Python code.

The general rule of thumb is to avoid relying on Python side effects in your logic and only use them to debug your traces. Otherwise, TensorFlow APIs like [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data), [`tf.print`](https://www.tensorflow.org/api_docs/python/tf/print), [`tf.summary`](https://www.tensorflow.org/api_docs/python/tf/summary), [`tf.Variable.assign`](https://www.tensorflow.org/api_docs/python/tf/Variable#assign), and [`tf.TensorArray`](https://www.tensorflow.org/api_docs/python/tf/TensorArray) are the best way to ensure your code will be executed by the TensorFlow runtime with each call.

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

---



#### Changing Python global and free variables

Changing Python global and [free variables](https://docs.python.org/3/reference/executionmodel.html#binding-of-names) counts as a Python side effect, so it only happens during tracing.

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

You can, in some cases, capture and manipulate state if it is a [`tf.Variable`](https://www.tensorflow.org/guide/variable). This is how the weights of Keras models are updated with repeated calls to the same `ConcreteFunction`.

---



#### Using Python iterators and generators

Many Python features, such as generators and iterators, rely on the Python runtime to keep track of state. In general, while these constructs work as expected in eager mode, they are examples of Python side effects and therefore only happen during tracing.

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

Another error you may encounter is a garbage-collected variable. `ConcreteFunction`s only retain [WeakRefs](https://docs.python.org/3/library/weakref.html) to the variables they close over, so you must retain a reference to any variables.

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

---



### Depending on Python global and free variables

`Function` creates a new `ConcreteFunction` when called with a new value of a Python argument. However, it does not do that for the Python closure, globals, or nonlocals of that `Function`. If their value changes in between calls to the `Function`, the `Function` will still use the values they had when it was traced. This is different from how regular Python functions work.

For that reason, we recommend a functional programming style that uses arguments instead of closing over outer names.

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

---



#### Depending on Python objects

The recommendation to pass Python objects as arguments into [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) has a number of known issues, that are expected to be fixed in the future. In general, you can rely on consistent tracing if you use a Python primitive or [`tf.nest`](https://www.tensorflow.org/api_docs/python/tf/nest)-compatible structure as an argument or pass in a *different* instance of an object into a `Function`. However, `Function` will *not* create a new trace when you pass **the same object and only change its attributes**.

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

For that reason, we recommend that you write your `Function` to avoid depending on mutable object attributes or create new objects.

If that is not possible, one workaround is to make new `Function`s each time you modify your object to force retracing:

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

`Function` only supports creating variables once, when first called, and then reusing them. You cannot create `tf.Variables` in new traces. Creating new variables in subsequent calls is currently not allowed, but will be in the future.

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

You may encounter `ValueError: tf.function-decorated function tried to create variables on non-first call.` when using more than one Keras optimizer with a `tf.function`. This error occurs because optimizers internally create `tf.Variables` when they apply gradients for the first time.

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

You may also encounter `ValueError: tf.function-decorated function tried to create variables on non-first call.` when passing different model instances to the same `Function`.

This error occurs because Keras models (which [do not have their input shape defined](https://www.tensorflow.org/guide/keras/custom_layers_and_models#best_practice_deferring_weight_creation_until_the_shape_of_the_inputs_is_known)) and Keras layers create `tf.Variables`s when they are first called. You may be attempting to initialize those variables inside a `Function`, which has already been called. To avoid this error, try calling `model.build(input_shape)` to initialize all the weights before training the model.

---



## Further reading

To learn about how to export and load a `Function`, see the [SavedModel guide](https://www.tensorflow.org/guide/saved_model). To learn more about graph optimizations that are performed after tracing, see the [Grappler guide](https://www.tensorflow.org/guide/graph_optimization). To learn how to optimize your data pipeline and profile your model, see the [Profiler guide](https://www.tensorflow.org/guide/profiler).

---

