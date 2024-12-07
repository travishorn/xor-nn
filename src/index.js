/**
 * @typedef {number[][]} WeightsMatrix - A matrix of weights.
 */

/**
 * @typedef {number[]} BiasArray - An array of biases.
 */

/**
 * @typedef {('uniform'|'gaussian')} Distribution - A random number distribution method
 */

/**
 * @typedef {Object} NeuralNetwork
 * @property {WeightsMatrix} weightsInputHidden - The weights connecting input layer to hidden layer.
 * @property {WeightsMatrix} weightsHiddenOutput - The weights connecting hidden layer to output layer.
 * @property {BiasArray} biasHidden - The biases for the hidden layer neurons.
 * @property {BiasArray} biasOutput - The biases for the output layer neurons.
 */

/**
 * @typedef {Object} ForwardResult
 * @property {number[]} hiddenOutputs - The outputs from the hidden layer neurons.
 * @property {number[]} finalOutputs - The outputs from the output layer neurons.
 */

/**
 * @typedef {Object} DataPoint
 * @property {number[]} inputs - The input values for the neural network.
 */

/**
 * @typedef {Object} TrainingDataPoint
 * @property {number[]} inputs - The input values for the neural network.
 * @property {number[]} targets - The expected output values for the neural network.
 */

/**
 * @typedef {Object} TestResult
 * @property {number[]} inputs - The input values for the test.
 * @property {number[]} predicted - The predicted output values.
 * @property {number[]} targets - The expected target values.
 * @property {boolean} correct - Indicates if the prediction was correct.
 */

/**
 * @typedef {Object} TestResults
 * @property {number} correctPredictions - The number of correct predictions.
 * @property {number} totalTests - The total number of tests.
 * @property {TestResult[]} results - The results of each test.
 */

/**
 * Computes the sigmoid activation function.
 * @param {number} x - The input value.
 * @returns {number} The output of the sigmoid function.
 */
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Computes the derivative of the sigmoid function.
 * @param {number} x - The output of the sigmoid function.
 * @returns {number} The derivative of the sigmoid function.
 */
function sigmoidDerivative(x) {
  return x * (1 - x);
}

/**
 * Generates a random weight using a Gaussian distribution.
 *
 * @param {Distribution} [distribution='uniform'] - The distribution method (`uniform` or `gaussian`)
 * @returns {number} A random weight.
 */
function randomWeight(distribution = "uniform") {
  if (!["uniform", "gaussian"].includes(distribution)) {
    throw new Error(
      "Random weight distribution must be specified as 'uniform' or 'gaussian'.",
    );
  }

  if (distribution === "gaussian") {
    const mean = 0;
    const stdDev = 0.5;
    return mean + stdDev * (Math.random() * 2 - 1);
  }

  // Fall back to uniform distribution
  return Math.random() * 2 - 1;
}

/**
 * Performs a forward pass through the neural network.
 * @param {number[]} inputs - The input values to the network.
 * @param {NeuralNetwork} network - The neural network structure containing weights and biases.
 * @returns {ForwardResult} The outputs from both the hidden and output layers.
 */
function forward(inputs, network) {
  const { weightsInputHidden, weightsHiddenOutput, biasHidden, biasOutput } =
    network;

  const hiddenInputs = weightsInputHidden.map((weights, i) =>
    weights.reduce((sum, weight, j) => sum + weight * inputs[j], biasHidden[i]),
  );
  const hiddenOutputs = hiddenInputs.map(sigmoid);

  const finalInputs = Array.from(
    { length: biasOutput.length },
    (_, outputIndex) =>
      hiddenOutputs.reduce(
        (sum, hiddenOutput, hiddenIndex) =>
          sum + hiddenOutput * weightsHiddenOutput[hiddenIndex][outputIndex],
        biasOutput[outputIndex],
      ),
  );
  const finalOutputs = finalInputs.map(sigmoid);

  return { hiddenOutputs, finalOutputs };
}

/**
 * Performs backpropagation to update the weights and biases of the network.
 * @param {number[]} inputs - The input values to the network.
 * @param {number[]} targets - The expected output values for the network.
 * @param {ForwardResult} results - The results from the forward pass.
 * @param {NeuralNetwork} network - The neural network structure containing weights and biases.
 * @param {number} learningRate - The learning rate for weight updates.
 * @returns {NeuralNetwork} The updated neural network.
 */
function backpropagate(
  inputs,
  targets,
  { hiddenOutputs, finalOutputs },
  network,
  learningRate,
) {
  const { weightsInputHidden, weightsHiddenOutput, biasHidden, biasOutput } =
    network;

  const outputErrors = targets.map((target, i) => target - finalOutputs[i]);
  const outputDeltas = outputErrors.map(
    (error, i) => error * sigmoidDerivative(finalOutputs[i]),
  );

  const hiddenErrors = hiddenOutputs.map((_, hiddenIndex) =>
    outputDeltas.reduce(
      (sum, delta, outputIndex) =>
        sum + delta * weightsHiddenOutput[hiddenIndex][outputIndex],
      0,
    ),
  );
  const hiddenDeltas = hiddenErrors.map(
    (error, i) => error * sigmoidDerivative(hiddenOutputs[i]),
  );

  const updatedWeightsHiddenOutput = weightsHiddenOutput.map(
    (hiddenWeights, hiddenIndex) =>
      hiddenWeights.map(
        (weight, outputIndex) =>
          weight +
          learningRate * outputDeltas[outputIndex] * hiddenOutputs[hiddenIndex],
      ),
  );
  const updatedBiasOutput = biasOutput.map(
    (bias, i) => bias + learningRate * outputDeltas[i],
  );

  const updatedWeightsInputHidden = weightsInputHidden.map((weights, i) =>
    weights.map(
      (weight, j) => weight + learningRate * hiddenDeltas[i] * inputs[j],
    ),
  );
  const updatedBiasHidden = biasHidden.map(
    (bias, i) => bias + learningRate * hiddenDeltas[i],
  );

  return {
    weightsInputHidden: updatedWeightsInputHidden,
    weightsHiddenOutput: updatedWeightsHiddenOutput,
    biasHidden: updatedBiasHidden,
    biasOutput: updatedBiasOutput,
  };
}

/**
 * Initializes a neural network with random weights and biases.
 * @param {number} inputSize - The number of input neurons.
 * @param {number} hiddenSize - The number of hidden neurons.
 * @param {number} outputSize - The number of output neurons.
 * @param {('uniform'|'gaussian')} [randomWeightDistribution='uniform'] - The distribution method (`uniform` or `gaussian`)
 * @returns {NeuralNetwork} The initialized neural network.
 */
export function initializeNetwork(
  inputSize,
  hiddenSize,
  outputSize,
  randomWeightDistribution = "uniform",
) {
  return {
    weightsInputHidden: Array.from({ length: inputSize }, () =>
      Array.from({ length: hiddenSize }, () =>
        randomWeight(randomWeightDistribution),
      ),
    ),
    weightsHiddenOutput: Array.from({ length: hiddenSize }, () =>
      Array.from({ length: outputSize }, () =>
        randomWeight(randomWeightDistribution),
      ),
    ),
    biasHidden: Array.from({ length: hiddenSize }, () =>
      randomWeight(randomWeightDistribution),
    ),
    biasOutput: Array.from({ length: outputSize }, () =>
      randomWeight(randomWeightDistribution),
    ),
  };
}

/**
 * Trains the neural network using the provided training data.
 * @param {NeuralNetwork} network - The neural network to train.
 * @param {TrainingDataPoint[]} trainingData - The data used for training.
 * @param {number} [epochs=10000] - The number of training iterations.
 * @param {number} [learningRate=0.1] - The learning rate for weight updates.
 * @param {boolean} [verbose=false] - Whether or not to print outputs every 1000 epochs
 * @returns {NeuralNetwork} The trained neural network.
 */
export function train(
  network,
  trainingData,
  epochs = 10000,
  learningRate = 0.1,
  verbose = false,
) {
  for (let epoch = 0; epoch < epochs; epoch++) {
    trainingData.forEach(({ inputs, targets }) => {
      const { hiddenOutputs, finalOutputs } = forward(inputs, network);

      network = backpropagate(
        inputs,
        targets,
        { hiddenOutputs, finalOutputs },
        network,
        learningRate,
      );
    });

    if (verbose && epoch % 1000 === 0) {
      console.log(`Epoch ${epoch}:`);
      trainingData.forEach(({ inputs, targets }) => {
        const { finalOutputs } = forward(inputs, network);
        console.log(
          `  Input: ${inputs} -> Output: ${finalOutputs.map((o) => o.toFixed(3))} (Target: ${targets})`,
        );
      });
    }
  }

  return network;
}

/**
 * Tests the neural network with the provided test data.
 *
 * @param {NeuralNetwork} network - The trained neural network.
 * @param {TrainingDataPoint[]} testData - The data used for testing.
 * @returns {TestResults} The test results.
 */
export function test(network, testData) {
  let correctPredictions = 0;
  const results = testData.map(({ inputs, targets }) => {
    const { finalOutputs } = forward(inputs, network);
    const predicted = finalOutputs.map((o) => Math.round(o));
    const result = {
      inputs,
      predicted,
      targets,
      correct: JSON.stringify(predicted) === JSON.stringify(targets),
    };
    if (result.correct) {
      correctPredictions++;
    }
    return result;
  });

  const totalTests = testData.length;
  return { correctPredictions, totalTests, results };
}
