#### Questions@:, Lecture 1 Questions
- Considering a neural network with $D$ input neurons, a single hidden layer
  with $H$ neurons, $K$ output neurons, hidden activation $f$ and output
  activation $a$, list its parameters (including their size) and write down how
  is the output computed. [5]

- List the definitions of frequently used MLP output layer activations (the ones
  producing parameters of a Bernoulli distribution and a categorical distribution).
  Then write down three commonly used hidden layer activations (sigmoid, tanh,
  ReLU). [5]

- Formulate the Universal approximation theorem. [5]

#### Questions@:, Lecture 2 Questions
- Describe maximum likelihood estimation, as minimizing NLL, cross-entropy and
  KL divergence. [10]

- Define mean squared error and show how it can be derived using MLE. [5]

- Describe gradient descent and compare it to stochastic (i.e., online) gradient
  descent and minibatch stochastic gradient descent. [5]

- Formulate conditions on the sequence of learning rates used in SGD to converge
  to optimum almost surely. [5]

- Write down the backpropagation algorithm. [5]

- Write down the mini-batch SGD algorithm with momentum. Then, formulate
  SGD with Nesterov momentum and show the difference between them. [5]

- Write down the AdaGrad algorithm and show that it tends to internally decay
  learning rate by a factor of $1/\sqrt{t}$ in step $t$. Then write down
  the RMSProp algorithm and explain how it solves the problem with the
  involuntary learning rate decay. [10]

- Write down the Adam algorithm. Then show why the bias-correction terms
  $(1-\beta^t)$ make the estimation of the first and second moment unbiased.
  [10]

#### Questions@:, Lecture 3 Questions
- Considering a neural network with $D$ input neurons, a single ReLU hidden
  layer with $H$ units and softmax output layer with $K$ units, write down the
  formulas of the gradient of all the MLP parameters (two weight matrices and
  two bias vectors), assuming input $\boldsymbol x$, target $g$ and negative log
  likelihood loss. [10]

- Assume a network with MSE loss generated a single output $o \in \mathbb{R}$,
  and the target output is $g$. What is the value of the loss function itself,
  and what is the gradient of the loss function with respect to $o$? [5]

- Assume a network with cross-entropy loss generated a single output
  $z \in \mathbb{R}$, which is passed through the sigmoid output activation
  function, producing $o = \sigma(z)$. If the target output is $g$, what is the value
  of the loss function itself, and what is the gradient of the loss function
  with respect to $z$? [5]

- Assume a network with cross-entropy loss generated a k-element output
  $\boldsymbol z \in \mathbb{R}^K$, which is passed through the softmax output
  activation function, producing $\boldsymbol o=\operatorname{softmax}(\boldsymbol z)$.
  If the target distribution is $\boldsymbol g$, what is the value of the loss
  function itself, and what is the gradient of the loss function with respect to
  $\boldsymbol z$? [5]

- Define $L_2$ regularization and describe its effect both on the value of the
  loss function and on the value of the loss function gradient. [5]

- Describe the dropout method and write down exactly how it is used during training and
  during inference. [5]

- Describe how label smoothing works for cross-entropy loss, both for sigmoid
  and softmax activations. [5]

- How are weights and biases initialized using the default Glorot
  initialization? [5]

#### Questions@:, Lecture 4 Questions
- Write down the equation of how convolution of a given image is computed.
  Assume the input is an image $I$ of size $H \times W$ with $C$ channels, the
  kernel $K$ has size $N \times M$, the stride is $T \times S$, the operation
  performed is in fact cross-correlation (as usual in convolutional neural
  networks) and that $O$ output channels are computed. [5]

- Explain both `SAME` and `VALID` padding schemes and write down the output
  size of a convolutional operation with an $N \times M$ kernel on image
  of size $H \times W$ for both these padding schemes (stride is 1). [5]

- Describe batch normalization and write down an algorithm how it is used during
  training and an algorithm how it is used during inference. Be sure to
  explicitly write over what is being normalized in case of fully connected
  layers, and in case of convolutional layers. [10]

- Describe overall architecture of VGG-19 (you do not need to remember the exact
  number of layers/filters, but you should describe which layers are used). [5]
