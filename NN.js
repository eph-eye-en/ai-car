class NN {
	constructor(network, learningRate) {
		this.network = network;
		this.learningRate = learningRate;
	}

	static random(layerSpecs, activationFunc, learningRate) {
		return new NN(NN.makeModel(layerSpecs, activationFunc), learningRate);
	}

	getResults(inputs) {
		let currentLayer = inputs;
		for(let i = 1; i < this.network.length; i++) {
			currentLayer = this.calculateLayer(this.network[i], currentLayer);
		}
		return currentLayer;
	}
	
	getAllResults(inputs) {
		let results = [inputs];
		let currentLayer = inputs;
		for(let i = 1; i < this.network.length; i++) {
			currentLayer = this.calculateLayer(this.network[i], currentLayer);
			results.push(currentLayer);
		}
		return results;
	}

	calculateLayer(layer, previousLayer) {
		let newValues = [];
		for(let i = 0; i < layer.length; i++) {
			newValues.push(layer[i].calculateOutput(previousLayer));
		}
		return newValues;
	}

	static calculateTotalCost(expected, actual) {
		let cost = 0;
		for(let i = 0; i < expected.length; i++)
			cost += NN.costOf(expected[i], actual[i]);
		return cost;
	}

	calculateLayerError(layer, output, target) {
		let error = [];
		for(let i = 0; i < layer.length; i++) {
			error.push(layer[i].calculateError(output[i], target[i]));
		}
		return error;
	}

	totalLayerWeight(layer, index) {
		let total = 0;
		for(let i = 0; i < layer[0].weights.length; i++)
			total += layer[i].weights[index];
		return total;
	}

	static costOf(expected, actual, signed=false) {
		return ((expected - actual) ** 2) * (expected < actual && signed ? -1 : 1);
	}

	train(count, data) {
		for(let i = 0; i < count; i++) {
			let batch = random(data);
			this.trainOnce(batch.inputs, batch.outputs);
		}
	}

	trainOnce(inputs, targets) {
		let results = this.getAllResults(inputs);
		let errors = results[results.length - 1].map((r, i) => targets[i] - r);

		for(let i = this.network.length - 1; i >= 1; i--) {
			let layer = this.network[i];
			let newErrors = Array(this.network[i - 1].length).fill(0);

			for(let j = 0; j < layer.length; j++) {
				let neuron = layer[j];
				let newErrorChanges = neuron.train(results[i-1],
									errors[j], this.learningRate);
				newErrors = newErrors.map((e, idx) => e + newErrorChanges[idx]);
			}
			errors = newErrors;
		}

		let outputs = results[results.length - 1];
		return outputs;
	}

	trainFromResults(results, targets) {
		let errors = results[results.length - 1].map((r, i) => targets[i] - r);

		for(let i = this.network.length - 1; i >= 1; i--) {
			let layer = this.network[i];
			let newErrors = Array(this.network[i - 1].length).fill(0);

			for(let j = 0; j < layer.length; j++) {
				let neuron = layer[j];
				let newErrorChanges = neuron.train(results[i-1],
									errors[j], this.learningRate);
				newErrors = newErrors.map((e, idx) => e + newErrorChanges[idx]);
			}
			errors = newErrors;
		}

		let outputs = results[results.length - 1];
		return outputs.indexOf(max(outputs)) == targets.indexOf(max(targets));
	}

	trainOnceOld(inputs, targets) {
		let results = this.getResults(inputs);
		for(let i = 1; i < this.network.length; i++) {
			let layer = this.network[i];
			for(let j = 0; j < layer.length; j++) {
				let neuron = layer[j];
				let error = neuron.calculateError(results[j], targets[j]);
				neuron.trainOld(inputs, error, this.learningRate);
			}
		}
	}

	static makeModel(layerSpecs, activationFunc) {
		let m = [NN.makeLayer(layerSpecs[0], activationFunc)];
		for(let i = 1; i < layerSpecs.length; i++)
			m.push(NN.makeLayer(
				layerSpecs[i], activationFunc, layerSpecs[i-1].count));
		return m;
	}

	static makeLayer(layerSpec, activationFunc, previousCount=0) {
		const { count, rand } = layerSpec;
		let layer = [];
		for(let i = 0; i < count; i++)
			layer.push(Neuron.getRandom(
				previousCount, activationFunc, rand));
		return layer;
	}

	exportModel() {
		let layers = this.network.map(this.exportLayer);
		let data = {
			layers,
			learningRate: this.learningRate
		};
		return JSON.stringify(data);
	}

	exportLayer(layer) {
		return layer.map(n => n.export());
	}

	importModelJson(str) {
		let data = JSON.parse(str);
		this.importModelData(data);
	}

	importModelData(data) {
		this.learningRate = data.learningRate;
		this.model = data.layers;
		this.model = this.model.map(
			l => l.map(n => new Neuron(n.weights, n.bias, n.activationFunc)));
	}
}

class Neuron {
	constructor(weights, bias, activationFunc) {
		this.weights = weights;
		this.bias = bias;
		this.activationFunc = activationFunc;
	}

	calculateOutput(previousLayer) {
		let total = this.bias;
		for(let i = 0; i < this.weights.length; i++) {
			total += this.weights[i] * previousLayer[i];
		}
		return this.activationFunc(total);
	}

	calculateError(output, target) {
		let diff = target - output;
		let error = {
			weights: [],
			bias: diff
		}
		for(let i = 0; i < this.weights.length; i++) {
			error.weights.push(diff);
		}
		return error;
	}

	train(inputs, error, learningRate) {
		let prevErrors = [];

		for(let i = 0; i < this.weights.length; i++) {
			let weightChange = inputs[i] * error * learningRate;
			this.weights[i] += weightChange;

			prevErrors.push(this.weights[i] * error * learningRate);
		}
		this.bias += error * learningRate;

		return prevErrors;
	}

	trainOld(inputs, costs, learningRate) {
		for(let i = 0; i < this.weights.length; i++) {
			let weightChange = inputs[i] * costs.weights[i] * learningRate;
			this.weights[i] += weightChange;
		}
		this.bias += costs.bias * learningRate;
	}

	static getRandom(count, activationFunc, rand) {
		let weights = [];
		for(let i = 0; i < count; i++)
			weights.push(rand());
		return new Neuron(weights, rand(), activationFunc);
	}

	export() {
		return {
			weights: this.weights,
			bias: this.bias
		};
	}
}

function sigmoid(n) {
	return 1/(1+Math.pow(Math.E, -n));
}

function tanh(n) {
	return Math.tanh(n);
}