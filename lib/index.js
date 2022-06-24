const { NeuralNetwork } = require("./neural-network");
const weights = require("../config/weights");
// const { generateWeights } = require('./generate-weights');
// const weights = generateWeights(...networkConfig)

const { testCases } = require("./test-cases");

const networkConfig = [9, 2, 9, 9];

const net = new NeuralNetwork(...networkConfig);
net.createNetwork();
net.loadWeights(weights);

console.log("Begin Training");
for (let index = 0; index < 100_000; index++) {
  console.log(`Epoch: ${index}`);
  const { input, output } =
    testCases[Math.floor(testCases.length * Math.random())];
  net.training(input, output);
}
console.log("Training Finished.");

const fancytestcase = [1, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.5];
const fancyOutput = [0, 0, 0, 0, 0, 0, 0, 0, 1];

const testOutput = net.evaluate(fancytestcase);
console.log(testOutput);

console.log("Trained Weights", JSON.stringify(net.exportWeights()));
