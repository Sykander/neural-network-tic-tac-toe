const { sigmoid: transfer, sigmoidPrime: backTransfer } = require('../sigmoid');

class Node {
    /**
     * @param {Number[]} weights - an array of weights
     * @param {Number} bias - the importance of this Node
     * @param {Number} position - the index of this Node within its layer
     */
    constructor(weights, bias, position) {
        this.weights = weights;
        this.bias = bias;
        this.position = position;

        this.output = 0;
        this.delta = 0;
    }

    /**
     * Evaluate
     * @param {Node[]} inputs - the layer to the left
     */
    evaluate(inputs) {
        const weights = this.weights;
        const bias = this.bias;

        const weightedInputs = inputs.map(
            (input, index) => transfer(input.getOutput()) * weights[index]
        );

        this.output = bias + weightedInputs.reduce(
            (acc, value) => acc + value,
            0
        );
    }

    setWeights(weights) {
        this.weights = weights;
    }

    setBias(bias) {
        this.bias = bias;
    }

    getOutput() {
        return this.output;
    }
}

module.exports = { Node };
