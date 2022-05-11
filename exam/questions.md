#### Questions@:, Lecture 1 Questions
- Considering a neural network with $D$ input neurons, a single hidden layer
  with $H$ neurons, $K$ output neurons, hidden activation $f$ and output
  activation $a$, list its parameters (including their size) and write down how
  the output is computed. [5]

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

#### Questions@:, Lecture 5 Questions
- Describe overall architecture of ResNet. You do not need to remember the exact
  number of layers/filters, but you should draw a bottleneck block (including
  the applications of BatchNorms and ReLUs) and state how residual connections
  work when the number of channels increases. [10]

- Draw the original ResNet block (including the exact positions of BatchNorms
  and ReLUs) and also the improved variant with full pre-activation. [5]

- Compare the bottleneck block of ResNet and ResNeXt architectures (draw the
  latter using convolutions only, i.e., do not use grouped convolutions). [5]

- Describe the CNN regularization method of networks with stochastic depth. [5]

- Compare Cutout and BlockDrop. [5]

- Describe Squeeze and Excitation applied to a ResNet block. [5]

- Draw the Mobile inverted bottleneck block (including explanation of separable
  convolutions, the expansion factor, exact positions of BatchNorms and ReLUs,
  but without describing Squeeze and excitation bocks). [5]

- Assume an input image $I$ of size $H \times W$ with $C$ channels, and
  a convolutional kernel $K$ with size $N \times M$, stride $S$ and $O$ output
  channels. Write down (or derive) the equation of transposed convolution
  (or equivalently backpropagation through a convolution to its inputs). [5]

#### Questions@:, Lecture 6 Questions
- Write down how $\mathit{AP}_{50}$ is computed. [5]

- Considering a Fast-RCNN architecture, draw overall network architecture,
  explain what a RoI-pooling layer is, show how the network parametrizes
  bounding boxes and write down the loss. Finally, describe non-maximum
  suppression and how the Fast-RCNN prediction is performed. [10]

- Considering a Faster-RCNN architecture, describe the region proposal network
  (what are anchors, architecture including both heads, how are the coordinates
  of proposals parametrized, what does the loss look like). [10]

- Considering Mask-RCNN architecture, describe the additions to a Faster-RCNN
  architecture (the RoI-Align layer, the new mask-producing head). [5]

- Write down the focal loss with class weighting, including the commonly used
  hyperparameter values. [5]

- Draw the overall architecture of a RetinaNet architecture (the FPN
  architecture including the block combining feature maps of different
  resolutions; the classification and bounding box generation heads, including
  their output size). [5]

- Draw the BiFPN block architecture, including the positions of all
  convolutions (and what kind of CNN is used), BatchNorms and ReLUs.
  Finally describe how downscaling and upscaling is performed. [5]

#### Questions@:, Lecture 7 Questions
- Write down how the Long Short-Term Memory (LSTM) cell operates, including
  the explicit formulas. Also mention the forget gate bias. [10]

- Write down how the Gated Recurrent Unit (GRU) operates, including
  the explicit formulas. [10]

- Describe Highway network computation. [5]

- Why the usual dropout cannot be used on recurrent state? Describe
  how the problem can be alleviated with variational dropout. [5]

- Describe layer normalization and write down an algorithm how it is used during
  training and an algorithm how it is used during inference. [5]

- Sketch a tagger architecture utilizing word embeddings, recurrent
  character-level word embeddings and two sentence-level bidirectional RNNs with
  a residual connection. [10]

#### Questions@:, Lecture 8 Questions
- Considering a linear-chain CRF, write down how a score of a label sequence
  $\boldsymbol y$ is defined, and how can a log probability be computed
  using the label sequence scores. [5]

- Write down the dynamic programming algorithm for computing log probability of
  a linear-chain CRF, including its asymptotic complexity. [10]

- Write down the dynamic programming algorithm for linear-chain CRF decoding,
  i.e., an algorithm computing the most probable label sequence $\boldsymbol y$.
  [10]

- In the context of CTC loss, describe regular and extended labelings and
  write down an algorithm for computing the log probability of a gold label
  sequence $\boldsymbol y$. [10]

- Describe how CTC predictions are performed using a beam-search. [5]

- Draw the CBOW architecture from `word2vec`, including the sizes of the inputs
  and the sizes of the outputs and used non-linearities. Also make sure to
  indicate where the embeddings are being trained. [5]

- Draw the SkipGram architecture from `word2vec`, including the sizes of the
  inputs and the sizes of the outputs and used non-linearities. Also make sure
  to indicate where the embeddings are being trained. [5]

- Describe the hierarchical softmax used in `word2vec`. [5]

- Describe the negative sampling proposed in `word2vec`, including
  the choice of distribution of negative samples. [5]

#### Questions@:, Lecture 9 Questions
- Draw a sequence-to-sequence architecture for machine translation, both during
  training and during inference (without attention). [5]

- Draw a sequence-to-sequence architecture for machine translation used during
  training, including the attention. Then write down how exactly is the attention
  computed. [10]

- Explain how can word embeddings tying be used in a sequence-to-sequence
  architecture. [5]

- Write down why are subword units used in text processing, and describe the BPE
  algorithm for constructing a subword dictionary from a large corpus. [5]

- Write down why are subword units used in text processing, and describe the
  WordPieces algorithm for constructing a subword dictionary from a large
  corpus. [5]

- Pinpoint the differences between the BPE and WordPieces algorithms, both
  during dictionary construction and during inference. [5]

#### Questions@:, Lecture 11 Questions
- Describe the Transformer encoder architecture, including the description of
  self-attention (but you do not need to describe multi-head attention), FFN
  and positions of LNs and dropouts. [10]

- Write down the formula of Transformer self-attention, and then describe
  multi-head self-attention in detail. [10]

- Describe the Transformer decoder architecture, including the description of
  self-attention and masked self-attention (but you do not need to describe
  multi-head attention), FFN and positions of LNs and dropouts. Also discuss the
  difference between training and prediction regimes. [10]

- Why are positional embeddings needed in Transformer architecture? Write down
  the sinusoidal positional embeddings used in the Transformer. [5]

- Compare RNN to Transformer – what are the strengths and weaknesses of these
  architectures? [5]

- Explain how are ELMo embeddings trained and how are they used in downstream
  applications. [5]

- Describe the BERT architecture (you do not need to describe the (multi-head)
  self-attention operation). Elaborate also on what positional embeddings
  are used and what are the GELU activations. [10]

- Describe the GELU activations and explain why are they a combination of ReLUs
  and Dropout. [5]

- Elaborate on BERT training process (what are the two objectives used and how
  exactly are the corresponding losses computed). [10]

#### Questions@:, Lecture 12 Questions
- Write down the variational lower bound (ELBO) in the form of a reconstruction
  error minus the KL divergence between the encoder and the prior. Then prove it
  is actually a lower bound on probability $\log P(\boldsymbol x)$ (you can use
  Jensen's inequality if you want). [10]

- Draw an architecture of a variational autoencoder (VAE). Pay attention to the
  parametrization of the distribution from the encoder (including the used
  activation functions), and show how to perform latent variable sampling so
  that it is differentiable with respect to the encoder parameters (the
  reparametrization trick). [10]

- Write down the min-max formulation of generative adversarial network (GAN)
  objective. Then describe what loss is actually used for training the generator
  in order to avoid vanishing gradients at the beginning of the training. [5]

- Write down the training algorithm of generative adversarial networks (GAN),
  including the losses minimized by the discriminator and the generator. Be sure
  to use the version of generator loss which avoids vanishing gradients at the
  beginning of the training. [10]

- Explain how the class label is used when training a conditional generative
  adversarial network (CGAN). [5]

- Illustrate that alternating SGD steps are not guaranteed to converge for
  a min-max problem. [5]

#### Questions@:, Lecture 13 Questions
- Show how to incrementally update a running average (how to compute
  an average of $N$ numbers using the average of the first $N-1$ numbers). [5]

- Describe multi-arm bandits and write down the $\epsilon$-greedy algorithm
  for solving it. [5]

- Define the Markov Decision Process, including the definition of the return. [5]

- Define the value function, such that all expectations are over simple random
  variables (actions, states, rewards), not trajectories. [5]

- Define the action-value function, such that all expectations are over simple
  random variables (actions, states, rewards), not trajectories. [5]

- Express the value function using the action-value function, and express the
  action-value function using the value function. [5]

- Define the optimal value function and the optimal action-value function. Then define
  optimal policy in such a way that its existence is guaranteed. [5]

- Write down the Monte-Carlo on-policy every-visit $\epsilon$-soft algorithm. [10]

- Formulate the policy gradient theorem. [5]

- Prove the part of the policy gradient theorem showing the value
  of $\nabla_{\boldsymbol\theta} v_\pi(s)$. [10]

- Assuming the policy gradient theorem, formulate the loss used by the REINFORCE
  algorithm and show how can its gradient be expressed as an expectation
  over states and actions. [5]

- Write down the REINFORCE algorithm. [10]

- Show that introducing baseline does not influence validity of the policy
  gradient theorem. [5]

- Write down the REINFORCE with baseline algorithm. [10]
