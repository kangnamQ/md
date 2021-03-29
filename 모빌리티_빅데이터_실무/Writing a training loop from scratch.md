Writing a training loop from scratch
==

### 처음부터 교육 루프 작성



- 참고 : https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

---

## 0. Basic Setup

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
```

---

## 1. Introduction

`Keras` provides default training and evaluation loops, `fit()` and `evaluate()`. Their usage is covered in the guide [Training & evaluation with the built-in methods](https://www.tensorflow.org/guide/keras/train_and_evaluate/).

`Keras `는 기본교육 및 평가루프를 담당하는 `fit()`, `evaluate()` 를 제공한다.
기본적인 사용법은 `train_and_evaluate` 라는 `keras` guide에서 확인할 수 있다.



If you want to customize the learning algorithm of your model while still leveraging the convenience of `fit()` (for instance, to train a GAN using `fit()`), 
you can subclass the `Model` class and implement your own `train_step()` method, which is called repeatedly during `fit()`. 
This is covered in the guide [Customizing what happens in `fit()`](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit/).

편의성을 위해 모델의 학습 알고리즘을 사용자 정의하기를 원한다면 (예를들어, `fit()`을 사용하여 GAN을 훈련 할 때)
`Model`클래스를 하위클래스화 하고 `fit()`하는 동안 반복적으로 호출되는 자체 메소드`train_step()`를 구현할 수 있다.
이 내용은 Customizing what happens in `fit()` 가이드에서 다루고 있다.



Now, if you want very low-level control over training & evaluation, you should write your own training & evaluation loops from scratch. This is what this guide is about.

이제 훈련&평가에서 매우 낮은 레벨의 제어를 원한다면, 처음부터 자신의 훈련 & 평가루프를 작성해야한다.
이것에 대한 가이드의 내용이다.



## Using the **GradientTape:** a first end - to - end example

Calling a model inside a `GradientTape` scope enables you to retrieve the gradients of the trainable weights of the layer with respect to a loss value. 
Using an optimizer instance, you can use these gradients to update these variables (which you can retrieve using `model.trainable_weights`).

GradientTape 범위 내에서 모델을 부르면(호출하면) 손실값과 관련하여 레이어의 학습 가능한 가중치의 기울기를 검색할 수 있습니다.

옵티마이저 인스턴스를 사용하면 이런 gradient를 사용하여 그것들의 변수를 업데이트 할 수 있다.
(`model.trainable_weights`를 사용하여 검색 할 수 있다.)



Let's consider a simple MNIST model:

간단한 MNIST 모델을 고려해 보자면 (보자면)

```python
inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu")(inputs)
x2 = layers.Dense(64, activation="relu")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)
```

Let's train it using mini-batch gradient with a custom training loop.

맞춤형(커스텀) 훈련  루프와 함께 미니-배치 gradient를 사용하여 훈련해 본다면



First, we're going to need an optimizer, a loss function, and a dataset:

먼저, optimizer, 손실함수, 데이타 셋이 필요하다.

```python
# Instantiate an optimizer.
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the training dataset.
batch_size = 64
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)
```

Here's our training loop:

- We open a `for` loop that iterates over epochs
  - 에폭수만큼 반복하는 for루프를 연다.

- For each epoch, we open a `for` loop that iterates over the dataset, in batches
  - 각 epoch에 대해, 데이터 셋에서 일괄적으로 반복되는 `for`루프를 연다.
- For each batch, we open a `GradientTape()` scope
  - 각 배치에 대해, `GradientTape()`범위를 연다.
- Inside this scope, we call the model (forward pass) and compute the loss
  - 이 범위 내에서, 모델(순방향 전달)을 호출하여 손실(loss)를 계산한다.
- Outside the scope, we retrieve the gradients of the weights of the model with regard to the loss
  - 범위 밖에서, 손실과 관련된 모델 가중치의 기술기(gradients)를 검색한다.
- Finally, we use the optimizer to update the weights of the model based on the gradients
  - 마지막으로, optimizer(최적기)를 이용하여 기울기 기반의 모델 가중치를 업데이트한다.

```python
epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    # 데이터 세트의 배치에 대해 반복
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        # GradientTape(그라디언트 테이프)를 열어 자동 분리를 활성화하는 
        # 전달 패스(Forward Pass) 동안 실행되는 작업을 기록합니다.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # 계층의 순방향으로 패스를 실행합니다.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.        
            # 레이어가 입력에 적용하는 연산은 Gradient Tape에 기록될 것이다.
            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            # 미니배치에 대해서 손실값을 계산한다.
            loss_value = loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        # 그레이디언트 테이프를 사용하여 손실과 관련하여 교육 가능한 변수의 그레이디언트를 자동으로 검색합니다.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        # 손실을 최소화하기 위해 변수 값을 업데이트하여 경사 하강을 한 단계 실행한다.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        # 200개 배치마다 기록한다.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * batch_size))
```

실행한 결과는 다음과 같다.

```python
Start of epoch 0
Training loss (for one batch) at step 0: 153.8545
Seen so far: 64 samples
Training loss (for one batch) at step 200: 1.4767
Seen so far: 12864 samples
Training loss (for one batch) at step 400: 1.4645
Seen so far: 25664 samples
Training loss (for one batch) at step 600: 0.7049
Seen so far: 38464 samples

Start of epoch 1
Training loss (for one batch) at step 0: 0.9202
Seen so far: 64 samples
Training loss (for one batch) at step 200: 0.8473
Seen so far: 12864 samples
Training loss (for one batch) at step 400: 0.6632
Seen so far: 25664 samples
Training loss (for one batch) at step 600: 0.8758
Seen so far: 38464 samples
```

---



## Low-level handling of metrics

낮은 수준의 매트릭스 처리



Let's add metrics monitoring to this basic loop.
이 기본 루프에 메트릭 모니터링을 추가해 보자

You can readily reuse the built-in metrics (or custom ones you wrote) in such training loops written from scratch. Here's the flow:
처음부터 작성된 훈련루프에서 기본 제공된 메트릭(or 직접 만든 매트릭)을  쉽게 사용할 수 있다

- Instantiate the metric at the start of the loop
  - 루프 시작시 메트릭 인스턴스화(예를들어 설명하다, 예시하다)
- Call `metric.update_state()` after each batch
  - 각 배치한 후 `metric.update_state()` 를 호출
- Call `metric.result()` when you need to display the current value of the metric
  - 메트릭의 현재 값을 표시해야할 경우 `metric.result()`를 호출
- Call `metric.reset_states()` when you need to clear the state of the metric (typically at the end of an epoch)
  - 매트릭의 상태를 초기화해야할 때 `metric.reset_states()`를 호출한다. (일반적으로 에폭 종료시)

Let's use this knowledge to compute `SparseCategoricalAccuracy` on validation data at the end of each epoch:
이 기술을 사용해서 각 에폭이 끝날때 검증 데이터에 대한 `SparseCategoricalAccuracy`를 계산할 수 있습니다.
`SparseCategoricalAccuracy` : 희박한 범주형 정확도

```python
# Get model
inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Instantiate an optimizer to train the model.
# optimizer을 사용한 훈련모델을 예시로 가지고 온다
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
```

Here's our training & evaluation loop:

트레이닝 & 평가 루프는 다음과 같다.

```python
import time

epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(y_batch_train, logits)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * batch_size))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))
```

실행한 결과는 다음과 같다.

```python
Start of epoch 0
Training loss (for one batch) at step 0: 114.3453
Seen so far: 64 samples
Training loss (for one batch) at step 200: 2.2635
Seen so far: 12864 samples
Training loss (for one batch) at step 400: 0.5206
Seen so far: 25664 samples
Training loss (for one batch) at step 600: 1.0906
Seen so far: 38464 samples
Training acc over epoch: 0.7022
Validation acc: 0.7853
Time taken: 5.38s

Start of epoch 1
Training loss (for one batch) at step 0: 0.5879
Seen so far: 64 samples
Training loss (for one batch) at step 200: 0.9477
Seen so far: 12864 samples
Training loss (for one batch) at step 400: 0.4649
Seen so far: 25664 samples
Training loss (for one batch) at step 600: 0.6874
Seen so far: 38464 samples
Training acc over epoch: 0.8114
Validation acc: 0.8291
Time taken: 5.46s
```



---

## Speeding-up your training step with 

## [tf.function](https://www.tensorflow.org/api_docs/python/tf/function)

The default runtime in TensorFlow 2.0 is [eager execution](https://www.tensorflow.org/guide/eager). As such, our training loop above executes eagerly.
TensorFlow 2.0의 기본 런타임은 즉시 실행(열성적인 실행) 입니다. 따라서 위의 교육 루프는 열심히 실행됩니다.



This is great for debugging, but graph compilation has a definite performance advantage. 
이것은 디버깅에 적합하지만 그래프 컴파일에는 확실한 성능 이점이 있습니다. 

Describing your computation as a static graph enables the framework to apply global performance optimizations. 
계산을 정적 그래프로 설명하면 프레임 워크가 전역 성능 최적화를 적용 할 수 있습니다. 

This is impossible when the framework is constrained to greedly execute one operation after another, with no knowledge of what comes next.
프레임 워크가 다음에 무슨 일이 일어날 지 알지 못한 채 탐욕스럽게 한 작업을 차례로 실행하도록 제한되어있는 경우 불가능합니다.

You can compile into a static graph any function that takes tensors as input. Just add a `@tf.function` decorator on it, like this:
텐서를 입력으로 사용하는 모든 함수를 정적 그래프로 컴파일 할 수 있습니다. 다음과 같이 `@tf.function` 데코레이터를 추가하면됩니다.

```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value
```



Let's do the same with the evaluation step:
평가 단계에서도 똑같이합시다.

```python
@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)
```

Now, let's re-run our training loop with this compiled training step:
이제이 컴파일 된 학습 단계를 사용하여 학습 루프를 다시 실행 해 보겠습니다.

```python
import time

epochs = 2
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        loss_value = train_step(x_batch_train, y_batch_train)

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * batch_size))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))
```

실행한 결과는 다음과 같다.

```python
Start of epoch 0
Training loss (for one batch) at step 0: 0.4854
Seen so far: 64 samples
Training loss (for one batch) at step 200: 0.5259
Seen so far: 12864 samples
Training loss (for one batch) at step 400: 0.5035
Seen so far: 25664 samples
Training loss (for one batch) at step 600: 0.2240
Seen so far: 38464 samples
Training acc over epoch: 0.8502
Validation acc: 0.8616
Time taken: 1.32s

Start of epoch 1
Training loss (for one batch) at step 0: 0.6278
Seen so far: 64 samples
Training loss (for one batch) at step 200: 0.3667
Seen so far: 12864 samples
Training loss (for one batch) at step 400: 0.3374
Seen so far: 25664 samples
Training loss (for one batch) at step 600: 0.5318
Seen so far: 38464 samples
Training acc over epoch: 0.8709
Validation acc: 0.8720
Time taken: 1.02s
```

Much faster, isn't it?

더 빠른 결과를 보여준다.

---



## Low-level handling of losses tracked by the model

## 모델에 의해 추적 된 손실의 저수준 처리



Layers & models recursively track any losses created during the forward pass by layers that call `self.add_loss(value)`. The resulting list of scalar loss values are available via the property `model.losses` at the end of the forward pass.
레이어 및 모델은 `self.add_loss(value)` 를 호출하는 레이어에 의해 순방향 전달 중에 생성 된 손실을 재귀 적으로 추적합니다. 스칼라 손실 값의 결과 목록은 순방향 패스가 끝날 때 `model.losses` 속성을 통해 사용할 수 있습니다.

If you want to be using these loss components, you should sum them and add them to the main loss in your training step.
이러한 손실 구성 요소를 사용하려면이를 합산하고 훈련 단계의 주요 손실에 추가해야합니다.

Consider this layer, that creates an activity regularization loss:
활동 정규화 손실을 생성하는 다음 계층을 고려하십시오.

```python
class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(1e-2 * tf.reduce_sum(inputs))
        return inputs
```

Let's build a really simple model that uses it:
그것을 사용하는 정말 간단한 모델을 만들어 봅시다 :

```python
inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu")(inputs)
# Insert activity regularization as a layer
x = ActivityRegularizationLayer()(x)
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

Here's what our training step should look like now:
이제 훈련 단계는 다음과 같습니다.

```python
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
        # Add any extra losses created during the forward pass.
        loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    train_acc_metric.update_state(y, logits)
    return loss_value
```

---



## Summary (요약)

Now you know everything there is to know about using built-in training loops and writing your own from scratch.
이제 기본 제공 교육 루프를 사용하고 처음부터 직접 작성하는 방법에 대해 알아야 할 모든 것을 알았습니다.

To conclude, here's a simple end-to-end example that ties together everything you've learned in this guide: a DCGAN trained on MNIST digits.
결론을 내리기 위해이 가이드에서 배운 모든 내용을 연결하는 간단한 종단 간 예가 있습니다. MNIST 숫자로 훈련 된 DCGAN입니다.

---



## End-to-end example: a GAN training loop from scratch

## 엔드-투-엔드 예제 : 처음부터 GAN 훈련 루프

You may be familiar with Generative Adversarial Networks (GANs). 
GAN (Generative Adversarial Network)에 익숙 할 수 있습니다. 

GANs can generate new images that look almost real, by learning the latent distribution of a training dataset of images (the "latent space" of the images).
GAN은 이미지 훈련 데이터 세트 (이미지의 "잠재 공간")의 잠재 분포를 학습하여 거의 실제처럼 보이는 새로운 이미지를 생성 할 수 있습니다.

A GAN is made of two parts: a "generator" model that maps points in the latent space to points in image space, a "discriminator" model, a classifier that can tell the difference between real images (from the training dataset) and fake images (the output of the generator network).
GAN은 잠복 공간의 점을 이미지 공간의 점으로 매핑하는 "생성자"모델, "분별 자"모델, 실제 이미지 (학습 데이터 세트의)와 가짜 이미지의 차이를 구분할 수있는 분류기의 두 부분으로 구성됩니다.  (생성기 네트워크의 출력).



A GAN training loop looks like this:
GAN 교육 루프는 다음과 같습니다.

1) Train the discriminator. 
판별자를 훈련하십시오. 

- Sample a batch of random points in the latent space. 
 - 잠재 공간에서 임의의 지점을 샘플링합니다. 

- Turn the points into fake images via the "generator" model. 
  -  "생성기"모델을 통해 포인트를 가짜 이미지로 전환합니다.
- Get a batch of real images and combine them with the generated images. 
  - 실제 이미지의 배치를 가져와 생성 된 이미지와 결합합니다.
-  Train the "discriminator" model to classify generated vs. real images.
  - 생성 된 이미지와 실제 이미지를 분류하기 위해 "분별 자"모델을 훈련시킵니다.



2) Train the generator. 
발전기를 훈련 시키십시오.

- Sample random points in the latent space.
  - 잠재 공간에서 무작위 포인트를 샘플링합니다.
- Turn the points into fake images via the "generator" network. 
  -  "생성기"네트워크를 통해 포인트를 가짜 이미지로 전환합니다.
- Get a batch of real images and combine them with the generated images. 
  - 실제 이미지의 배치를 가져와 생성 된 이미지와 결합합니다.
- Train the "generator" model to "fool" the discriminator and classify the fake images as real.
  -  "generator"모델을 훈련시켜 판별자를 "속이고"가짜 이미지를 실제 이미지로 분류합니다.



For a much more detailed overview of how GANs works, see [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python).
GAN의 작동 방식에 대한 훨씬 더 자세한 개요는 [Python을 사용한 Deep Learning 항목을](https://www.manning.com/books/deep-learning-with-python) 참조하십시오.

Let's implement this training loop. First, create the discriminator meant to classify fake vs real digits:
이 훈련 루프를 구현해 봅시다. 먼저 가짜와 실수를 구분하기위한 판별자를 만듭니다.

```python
discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)
discriminator.summary()
```

실행한 결과는 다음과 같다.

```python
Model: "discriminator"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 14, 14, 64)        640       
_________________________________________________________________
leaky_re_lu (LeakyReLU)      (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 7, 7, 128)         73856     
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 7, 7, 128)         0         
_________________________________________________________________
global_max_pooling2d (Global (None, 128)               0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 129       
=================================================================
Total params: 74,625
Trainable params: 74,625
Non-trainable params: 0
_________________________________________________________________
```



Then let's create a generator network, that turns latent vectors into outputs of shape `(28, 28, 1)` (representing MNIST digits):
그런 다음 잠재 벡터를 형태 `(28, 28, 1)` (MNIST 숫자를 나타냄 `(28, 28, 1)` 출력으로 변환하는 생성기 네트워크를 만들어 보겠습니다.

```python
latent_dim = 128

generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        # We want to generate 128 coefficients to reshape into a 7x7x128 map
        layers.Dense(7 * 7 * 128),
        layers.LeakyReLU(alpha=0.2),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)
```

Here's the key bit: the training loop. As you can see it is quite straightforward. The training step function only takes 17 lines.
핵심 부분은 훈련 루프입니다. 보시다시피 매우 간단합니다. 훈련 단계 기능은 17 줄만 사용합니다.

```python
# Instantiate one optimizer for the discriminator and another for the generator.
d_optimizer = keras.optimizers.Adam(learning_rate=0.0003)
g_optimizer = keras.optimizers.Adam(learning_rate=0.0004)

# Instantiate a loss function.
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function
def train_step(real_images):
    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    # Decode them to fake images
    generated_images = generator(random_latent_vectors)
    # Combine them with real images
    combined_images = tf.concat([generated_images, real_images], axis=0)

    # Assemble labels discriminating real from fake images
    labels = tf.concat(
        [tf.ones((batch_size, 1)), tf.zeros((real_images.shape[0], 1))], axis=0
    )
    # Add random noise to the labels - important trick!
    labels += 0.05 * tf.random.uniform(labels.shape)

    # Train the discriminator
    with tf.GradientTape() as tape:
        predictions = discriminator(combined_images)
        d_loss = loss_fn(labels, predictions)
    grads = tape.gradient(d_loss, discriminator.trainable_weights)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    # Assemble labels that say "all real images"
    misleading_labels = tf.zeros((batch_size, 1))

    # Train the generator (note that we should *not* update the weights
    # of the discriminator)!
    with tf.GradientTape() as tape:
        predictions = discriminator(generator(random_latent_vectors))
        g_loss = loss_fn(misleading_labels, predictions)
    grads = tape.gradient(g_loss, generator.trainable_weights)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
    return d_loss, g_loss, generated_images
```

Let's train our GAN, by repeatedly calling `train_step` on batches of images.
이미지 배치에 대해 `train_step` 을 반복적으로 호출하여 GAN을 훈련시켜 봅시다.

Since our discriminator and generator are convnets, you're going to want to run this code on a GPU.
우리의 판별 자와 생성기는 convnet이기 때문에이 코드를 GPU에서 실행하고 싶을 것입니다.

```python
import os

# Prepare the dataset. We use both the training & test MNIST digits.
batch_size = 64
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(all_digits)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

epochs = 1  # In practice you need at least 20 epochs to generate nice digits.
save_dir = "./"

for epoch in range(epochs):
    print("\nStart epoch", epoch)

    for step, real_images in enumerate(dataset):
        # Train the discriminator & generator on one batch of real images.
        d_loss, g_loss, generated_images = train_step(real_images)

        # Logging.
        if step % 200 == 0:
            # Print metrics
            print("discriminator loss at step %d: %.2f" % (step, d_loss))
            print("adversarial loss at step %d: %.2f" % (step, g_loss))

            # Save one generated image
            img = tf.keras.preprocessing.image.array_to_img(
                generated_images[0] * 255.0, scale=False
            )
            img.save(os.path.join(save_dir, "generated_img" + str(step) + ".png"))

        # To limit execution time we stop after 10 steps.
        # Remove the lines below to actually train the model!
        if step > 10:
            break
```

실행한 결과는 다음과 같다.

```python
Start epoch 0
discriminator loss at step 0: 0.68
adversarial loss at step 0: 0.67
```

That's it! You'll get nice-looking fake MNIST digits after just ~30s of training on the Colab GPU.
그게 다야! Colab GPU에서 30 초 정도 훈련 한 후에 멋진 가짜 MNIST 숫자를 얻을 수 있습니다.