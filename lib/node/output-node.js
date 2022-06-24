const { Node } = require(".");
const { cost } = require("../cost");

class OutputNode extends Node {
  evaluateDelta(idealOutput) {
    this.delta = cost(idealOutput, this.getOutput());
  }
}

module.exports = { OutputNode };
