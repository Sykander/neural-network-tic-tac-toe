/**
 * @param {Number} inputs
 * @param {Number} width
 * @param {Number} depth
 * @param {Number} outputs
 * @return {{ weights: Number[], bias: Number }[][]}
 */
function generateWeights(inputs, width, depth, outputs) {
  const weightsArray = [];

  for (let i = 1; i < width + 2; i++) {
    const countNodesInLeftLayer = i === 1 ? inputs : depth;
    const countNodesInCurrentLayer = i === width + 1 ? outputs : depth;
    const weightsForLayer = [];

    for (let j = 0; j < countNodesInCurrentLayer; j++) {
      const weights = [];
      const bias = Math.random();

      for (let k = 0; k < countNodesInLeftLayer; k++) {
        // generate a random weight between -1 and 1
        const weight = Math.random() * 2 - 1;
        weights.push(weight);
      }

      weightsForLayer.push({
        weights,
        bias,
      });
    }

    weightsArray.push(weightsForLayer);
  }

  return weightsArray;
}

module.exports = { generateWeights };
