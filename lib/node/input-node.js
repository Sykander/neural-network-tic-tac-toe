const { Node } = require(".");

class InputNode extends Node {
  constructor(position) {
    super([], 0, position);
  }

  setOutput(output) {
    this.output = output;
  }
}

module.exports = { InputNode };
