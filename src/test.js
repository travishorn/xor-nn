import { initializeNetwork, train, test } from "./index.js";

const trainingData = [
  { inputs: [0, 0], targets: [0] },
  { inputs: [0, 1], targets: [1] },
  { inputs: [1, 0], targets: [1] },
  { inputs: [1, 1], targets: [0] },
];

/** @type {import('./index.js').Distribution[]} */
const randomWeightDistributions = ["gaussian"];
const epochs = [40000, 45000, 50000];
const learningRates = [0.09, 0.1, 0.11];
const testRuns = 100;
let configurationId = 0;
let highestAccuracy = 0;
/** @type {number[]} */
let highestAccuracyConfigurations = [];

randomWeightDistributions.forEach((randomWeightDistribution) => {
  epochs.forEach((epoch) => {
    learningRates.forEach((learningRate) => {
      let totalCorrectPredictions = 0;
      let totalTests = 0;
      for (let run = 0; run < testRuns; run++) {
        const initialNetwork = initializeNetwork(
          2,
          2,
          1,
          randomWeightDistribution,
        );
        const network = train(
          initialNetwork,
          trainingData,
          epoch,
          learningRate,
        );
        const testResults = test(network, trainingData);
        totalCorrectPredictions += testResults.correctPredictions;
        totalTests += testResults.totalTests;
      }
      const averageAccuracy = (totalCorrectPredictions / totalTests) * 100;
      console.log(
        `ID: ${`${configurationId},`.padEnd(3)} Epochs: ${epoch.toString().padStart(5)}, Learning Rate: ${learningRate.toFixed(5).padStart(7)}, Distribution: ${`${randomWeightDistribution},`.padEnd(9)} Accuracy: ${averageAccuracy.toFixed(2).padStart(6)}%`,
      );
      if (averageAccuracy > highestAccuracy) {
        highestAccuracy = averageAccuracy;
        highestAccuracyConfigurations = [configurationId];
      } else if (averageAccuracy === highestAccuracy) {
        highestAccuracyConfigurations.push(configurationId);
      }
      configurationId++;
    });
  });
});

console.log(
  `Highest Average Accuracy: ${highestAccuracy.toFixed(2)}% achieved by: ${highestAccuracyConfigurations.join(", ")}`,
);
