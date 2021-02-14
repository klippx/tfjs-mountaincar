# TensorFlow.js Example: Reinforcement Learning with Mountain Car Simulation

## Overview

This is a modification of [this tensorflow.js subrepository](https://github.com/prouhard/tfjs-mountaincar/tree/master), by [@prouhard](https://github.com/prouhard)

It has been converted to TS to more easily understand what is going on.

### Features:

- Allows user to specify the architecture of the policy network, in particular,
  the number of the neural networks's layers and their sizes (# of units).
- Allows training of the policy network in the browser, optionally with
  simultaneous visualization of the cart-pole system.
- Allows testing in the browser, with visualization.
- Allows saving the policy network to the browser's IndexedDB. The saved policy
  network can later be loaded back for testing and/or further training.

## Usage

```sh
npm install
npm run serve
```
