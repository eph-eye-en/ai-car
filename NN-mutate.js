class NNMutate {
	constructor(model, mutateStrength) {
		this.model = model;

		this.mutateStrength = mutateStrength;
	}

	static fromNetwork(network, mutateStrength) {
		return new NNMutate(new NN(network), mutateStrength);
	}

	static random(layerSpecs, activationFunc, mutateStrength) {
		const m = NN.random(layerSpecs, activationFunc, 0);
		return new NNMutate(m, mutateStrength);
	}

	getResults(inputs) {
		return this.model.getResults(inputs);
	}

	mutate(count) {
		let children = [];
		for(let i = 0; i < count; i++) {
			let c = NNMutate.fromNetwork(
				NNMutate.mutateLayers(this.model.network, this.mutateStrength),
				this.mutateStrength);
			children.push(c);
		}
		return children;
	}

	static mutateLayers(layers, strength) {
		let mutated = [];
		for(let l of layers) {
			let ml = [];
			for(let n of l) {
				let mn = NNMutate.mutateNeuron(n, strength);
				ml.push(mn);
			}
			mutated.push(ml);
		}
		return mutated;
	}

	static mutateNeuron(n, strength) {
		return new Neuron(
			n.weights.map(w => w + (Math.random() * 2 - 1) * strength),
			n.bias + (Math.random() * 2 - 1) * strength,
			n.activationFunc
		);
	}
}