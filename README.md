# Neural Network Training and Testing

This project implements a simple neural network from scratch in JavaScript. It
includes functionalities for initializing a neural network, training it with
provided data, and testing its performance. The neural network supports two
types of weight initialization distributions: uniform and Gaussian. The project
is designed for educational purposes, demonstrating the core concepts of neural
networks, including forward propagation, backpropagation, and the training
process.

## Features

- Initialize a neural network with random weights and biases.
- Train the neural network using backpropagation.
- Test the neural network's accuracy on provided test data.
- Support for different weight initialization distributions (uniform and
  Gaussian).
- Configurable training parameters such as epochs and learning rates.

## Installation

To get started with this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/travishorn/xor-nn
   cd xor-nn
   ```

2. Install dependencies (if you want to run linting and formatting):

   ```bash
   npm install
   ```

## Usage

To use the neural network, you can import the functions from the `index.js` file
in your JavaScript code. Here's a basic example of how to initialize, train, and
test the neural network:

```javascript
import { initializeNetwork, train, test } from "./index.js";

// Define your training data
const trainingData = [
  { inputs: [0, 0], targets: [0] },
  { inputs: [0, 1], targets: [1] },
  { inputs: [1, 0], targets: [1] },
  { inputs: [1, 1], targets: [0] },
];

// Initialize the network
const network = initializeNetwork(2, 2, 1, "gaussian");

// Train the network
const trainedNetwork = train(network, trainingData, 50000, 0.1, true);

// Test the network
const testResults = test(trainedNetwork, trainingData);
console.log(
  `Correct Predictions: ${testResults.correctPredictions}/${testResults.totalTests}`,
);
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or new
features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.
