class CarSim {
	constructor(models) {
		this.models = models;

		this.generation = 0;
	}

	* trainGeneration(getCar, time, dt, map, keepRateRecip) {
		let scores = [];
		for(let m of this.models) {
			const s = CarSim.simulateModel(m, getCar, time, dt, map);
			scores.push({
				model: m,
				score: s
			});

			yield;
		}

		scores.sort((a, b) => b.score - a.score);
		let newModels = [];
		for(let i = 0; i < scores.length / keepRateRecip; i++) {
			let m = scores[i].model;
			newModels.push(...m.mutate(keepRateRecip));
		}

		this.models = newModels;
		this.generation++;

		return scores;
	}
	
	static simulateModel(model, getCar, time, dt, map) {
		const { width, height, walls } = map;
		let c = getCar();
		let steps = time / dt;
		let target = CarSim.randomTarget(width, height);
		let score = 0;
		let mem = Array(memSize).fill(0);
		for(let i = 0; i < steps; i++) {
			if(target.copy().sub(c.x, c.y).magSq() < 400) {
				score++;
				target = CarSim.randomTarget(width, height);
			}
			const ds = c.getVision(walls);
			({ mem } = CarSim.updateCar(c, model, dt, target, ds, mem));
			if(walls.some(w => c.intersectsWall(w))) {
				//Uncomment to adjust punishment for hitting a wall.
				//score--;
				//score /= 2;
				score -= 10;
				break;
			}
		}
		const maxDist = createVector(width, height).mag();
		score = score * maxDist + maxDist - target.copy().sub(c.x, c.y).mag();
		return score;
	}
	
	static updateCar(c, model, dt, target, ds, preMem) {
		const v = target.copy().sub(c.x, c.y).rotate(-c.angle);
		let inputs = [v.x * 10, v.y * 10, ...ds].map(Math.tanh);
		let [fd, rt, ...mem] = model.getResults(inputs.concat(preMem));

		c.update(fd, rt, dt);
	
		return { fd, rt, mem };
	}

	static randomTarget(width, height) {
		return createVector(
			random(-width / 2, width / 2),
			random(-height / 2, height / 2)
		);
	}
}