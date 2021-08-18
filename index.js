let car;

const A = 65;
const D = 68;
const S = 83;
const W = 87;

let sim;
let memSize = 20;
let layerSpecs = [
	{
		count: 10 + memSize,
		rand: () => 0
	},
	{
		count: 20,
		rand: () => random(-1, 1)
	},
	{
		count: 20,
		rand: () => random(-1, 1)
	},
	{
		count: 2 + memSize,
		rand: () => random(-1, 1)
	}];
let mutateStrength = 0.1;
let keepRateRecip = 5;

let simLen = 10;
let simStep = 1/30;

let simMap = {
	width: 1000,
	height: 1000,
	walls: []
};
let statMarginSize = 300;

let demoStep = 0;
let demoModel;
let demoMem = Array(memSize).fill(0);
let demoTarget;

let currGen = [];
let topScores = [];

let bestScore = 0;

let doTraining = false;
let doUserCar = false;
let trainingGen;

function setup() {
	createCanvas(simMap.width + statMarginSize, simMap.height);

	car = makeCar();

	for(let i = 0; i <= 5; i++) {
		simMap.walls.push({
			x1: simMap.width / 5 * i - simMap.width / 2,
			y1: (i % 2 == 0 ? 0 : 600) - simMap.height / 2,
			x2: simMap.width / 5 * i - simMap.width / 2,
			y2: simMap.height / 2 - (i % 2 == 0 ? 600 : 0)
		});
	}
	//simMap.walls.push({
	//	x1: -100,
	//	y1: -100,
	//	x2: -100,
	//	y2: 100
	//});

	let models = [];
	for(let i = 0; i < 300; i++)
		models.push(NNMutate.random(layerSpecs, Math.tanh, mutateStrength));
	sim = new CarSim(models);
	currGen = models.map(m => ({ model: m }));
	trainingGen = sim.trainGeneration(makeCar, simLen, simStep,
									  simMap, keepRateRecip);

	demoModel = models[0];
	demoTarget = CarSim.randomTarget(simMap.width, simMap.height);

	frameRate(30);
}

function draw() {
	background(51);

	if(doTraining) {
		let startMillis = millis();
		let count = 0;
		while(millis() < startMillis + 100) {
			let done;
			({ value, done } = trainingGen.next());
			count++;

			if(done) {
				currGen = value;
				topScores = currGen.slice(0, 20).map(s => int(s.score));
			
				if(topScores[0] > bestScore)
					bestScore = topScores[0];
				trainingGen = sim.trainGeneration(makeCar, simLen, simStep,
												  simMap, keepRateRecip);
			}
		}
	}

	drawStats(sim, topScores, bestScore, doTraining);
		
	translate(statMarginSize + simMap.width / 2, simMap.height / 2);

	drawWalls(simMap.walls);

	if(doUserCar)
		drawCar(car, updateUserCar(car), car.getVision(simMap.walls));
	else {
		const mv = CarSim.updateCar(car, demoModel, simStep,
									demoTarget, car.getVision(simMap.walls),
									demoMem);
		demoMem = mv.mem;
		drawCar(car, mv, car.getVision(simMap.walls));
		stroke(255, 0, 0);
		strokeWeight(15);
		point(demoTarget);
		demoStep++;

		if(simMap.walls.some(w => car.intersectsWall(w)))
			resetDemo(currGen[0].model);

		if(demoTarget.copy().sub(car.x, car.y).magSq() < 400)
			demoTarget = CarSim.randomTarget(simMap.width, simMap.height);

		if(demoStep * simStep > simLen)
			resetDemo(currGen[0].model);
	}
}

function resetDemo(model) {
	demoModel = model;
	car = makeCar();
	demoStep = 0;
	demoMem = Array(memSize).fill(0);
}

function keyPressed() {
	if(key === " ") {
		doTraining = !doTraining;
	}
}

function mousePressed() {
	if(statMarginSize < mouseX && mouseX < width
	&& 0 < mouseY && mouseY < height)
		demoTarget = createVector(mouseX - simMap.width / 2 - statMarginSize,
								mouseY - simMap.height / 2);
}

function makeCar() {
	return new Car(20, 40, PI/4, 30, PI * 7/4, 8);
}

function drawWalls(walls) {
	stroke(200);
	strokeWeight(3);
	for(let w of walls)
		line(w.x1, w.y1, w.x2, w.y2);
}

function drawStats(sim, scores, bestScore, isTraining) {
	noStroke();
	fill(40, 40, 60);
	rect(0, 0, statMarginSize, height);

	stroke(0);
	strokeWeight(3);
	fill(255);
	textSize(30);
	text("Generation: " + sim.generation, 15, 15, 300);
	
	textSize(20);
	if(isTraining)
		fill(100, 255, 100);
	else
		fill(255, 50, 50);
	text("Training: " + isTraining, 15, 50, 300);

	fill(255);
	text("Best score: "+ bestScore, 15, 75, 300);
	text("Top scores:\n" + scores.join(", "), 15, 100, 300);
}

function updateUserCar(c) {
	let fd = int(keyIsDown(W)) - int(keyIsDown(S));
	let rt = int(keyIsDown(D)) - int(keyIsDown(A));
	
	let dt = 1 / (frameRate() || 1);
	c.update(fd, rt, dt);

	return { fd, rt };
}

function drawCar(c, { fd, rt }, ds) {
	push();
	
	rectMode(CENTER);
	fill(0, 255, 0);
	noStroke();
	translate(c.x, c.y);
	rotate(c.angle - HALF_PI);
	rect(0, 0, c.width, c.height);

	drawLines(c, ds);
	
	translate(0, c.height / 4);
	rotate(c.turnAngle * rt);
	stroke(255, 0, 0);
	strokeWeight(3);
	line(0, 0, 0, c.height / 2 * fd);
	
	pop();
}

function drawLines(c, ds) {
	stroke(0, 0, 255);
	const n = c.eyeCount;
	
	let startAngle = PI/2;
	let angleStep = c.pov / (n - 1);
	if(n == 1) {
		startAngle = 0;
		angleStep = 0;
	}
	for(let i = 0; i < n; i++) {
		const a = startAngle + (i - (n-1) / 2) * angleStep;
		const x2 = Math.cos(a);
		const y2 = Math.sin(a);
		line(0, 0, cos(a) * ds[i], sin(a) * ds[i]);
	}
}