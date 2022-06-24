const { NeuralNetwork } = require('./neural-network');
const weights = require('../config/weights');
// const { generateWeights } = require('./generate-weights');

const networkConfig = [9, 2, 9, 9];

// 0    <=> O
// 0.5  <=> _
// 1    <=> X

// input board
/**
 * _ | O | X
 * ---------
 * _ | O | X
 * ---------
 * _ | _ | _
 */
const inputs = [0.5, 0, 1, 0.5, 0, 1, 0.5, 0.5, 0.5];

// ideal output
/**
 * _ | _ | _
 * ---------
 * _ | _ | _
 * ---------
 * _ | _ | X
 */
const idealOutput = [0, 0, 0, 0, 0, 0, 0, 0, 1];

// const weights = generateWeights(...networkConfig)
// console.log('weights', JSON.stringify(weights));

const net = new NeuralNetwork(...networkConfig);
net.createNetwork();
net.loadWeights(weights);
const outputs = net.evaluate(inputs)

console.log(outputs)

