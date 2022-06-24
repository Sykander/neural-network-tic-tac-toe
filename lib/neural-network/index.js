const { Node } = require('../node');
const { InputNode } = require('../node/input-node');
const { OutputNode } = require('../node/output-node');
const { sigmoid: transfer, sigmoidPrime: backTransfer } = require('../sigmoid');

class NeuralNetwork {
    /**
     * @param {Number} inputs
     * @param {Number} width
     * @param {Number} depth
     * @param {Number} outputs
     */
    constructor(inputs, width, depth, outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.depth = depth;
        this.width = width;

        this.network = null;
    }

    /**
     * @param {Number[]} inputs
     */
    evaluate(inputs) {
        const network = this.network;
        const inputLayer = network[0];

        
        // first feed inputs into input layer
        for (let i = 0; i < inputLayer.length; i++) {
            const idealOutput = inputs[i];
            const node = inputLayer[i];
            node.setOutput(idealOutput)
        }
        
        // call evaluate on each node on each layer in sequence
        for (let j = 1; j < network.length; j++) {
            const hiddenLayer = network[j];
            const leftLayer = network[j - 1];
            
            for (let k = 0; k < hiddenLayer.length; k++) {
                const node = hiddenLayer[k];
                
                node.evaluate(leftLayer)
            }
        }

        // take the output layer values into an array
        const outputs = network[network.length - 1].map(
            node => transfer(node.getOutput())
        )

        // return that array
        return outputs;
    }

    /**
     * @return {Node[][]}
     */
    createNetwork() {
        // hidden layers + input layer + output layer
        const layerCount = this.width + 2;
        const network = [];

        for (let h = 0; h < layerCount; h++) {
            network[h] = [];
        }

        for (let i = 0; i < this.inputs; i++) {
            network[0][i] = new InputNode(i);
        }

        for (let j = 0; j < this.width; j++) {
            // we get j + 1 here because input layer offsets our number
            const hiddenLayer = network[j + 1];
            const leftLayer = network[j];

            for (let k = 0; k < this.depth; k++) {
                const initialWeights = Array(leftLayer.length).fill(1);
                const bias = 1;

                hiddenLayer[k] = new Node(initialWeights, bias, k);
            }
        }


        // get the second to last layer
        for (let l = 0; l < this.outputs; l++) {
            const initialWeights = Array(this.depth).fill(1);
            const bias = 1;

            network[network.length - 1][l] = new OutputNode(initialWeights, bias, l);
        }

        this.network = network;
    }

    /**
     * @param {{ weights: Number[], bias: Number }[][]} weights
     */
    loadWeights(weights) {
        const network = this.network;
        
        for (let i = 0; i < weights.length; i++) {
            const layerWeights = weights[i];
            // we don't load weights into input layer
            // so start at layer i + 1
            const layer = network[i + 1];
            
            for (let j = 0; j < layerWeights.length; j++) {
                const { weights, bias } = layerWeights[j];
                

                const node = layer[j];

                node.setWeights(weights);
                node.setBias(bias);
            }
        }
    }

    backPropogation() {
        // calculate the error in the output

        // distribute that error across all the nodes (so how much each node contributed to that error)

        // adjust each node weight by a value corresponding to the error that node was responsible for

        // repeat until network is smart
    }
}

module.exports = { NeuralNetwork };
