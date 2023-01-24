# miniconv

This library is designed to be a minimal implementation of a Long Short-Term
Memory Recurrent Neural Network (LSTM) in Python.

### Usage

- We use basic numpy for some matrix operations and random generation
- You should not use this library in production code. It will be (much!) slower
  than PyTorch/a more verbose implementation. It is purely for educational
  purposes and helping to understand the basic operation of an LSTM.

### Interpretation

- See my [miniconv page](https://github.com/HNx1/miniconv) for information on
  the GradFloat class and other aspects of this implementation.
- This is the classical LSTM as described by
  [Hochreiter & Schmidhuber](https://gwern.net/docs/ai/nn/rnn/1997-hochreiter.pdf)
- We have the simplest possible output here for minimality - just taking the sum
  of the hidden vector produced by the final layer of our LSTM network. It's
  easy to adapt this output for other purposes.
- A good exercise is to make the small adaptations required to this codebase to
  implement some minor LSTM variants such as peepholes, coupling or gating.
- You could also implement features like clipping or batching in the running process.
