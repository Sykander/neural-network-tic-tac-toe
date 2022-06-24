function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function sigmoidPrime(z) {
  return z * (1 - z);
}

module.exports = { sigmoid, sigmoidPrime };
