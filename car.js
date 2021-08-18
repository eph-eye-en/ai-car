class Car {
	constructor(w, h, t, p, pov, n) {
		this.width = w;
		this.height = h;
		this.turnAngle = t;
		this.power = p;
		this.pov = pov;
		this.eyeCount = n;
		
		this.x = 0;
		this.y = 0;
		this.angle = 0;
		this.speed = 0;
		this.drag = 0.1;
	}

	update(fd, rt, dt) {
		fd *= this.power;
		this.speed = (this.speed + fd * dt) * (1 - this.drag);
		this.angle += this.turnAngle * rt * dt * this.speed;

		this.x += this.speed * Math.cos(this.angle);
		this.y += this.speed * Math.sin(this.angle);
	}

	getVision(walls) {
		const n = this.eyeCount;		
		let startAngle = this.angle;
		let angleStep = this.pov / (n - 1);
		if(n == 1) {
			startAngle = 0;
			angleStep = 0;
		}
		let ds = [];
		for(let i = 0; i < n; i++) {
			const a = startAngle + (i - (n-1) / 2) * angleStep;
			const x2 = this.x + Math.cos(a);
			const y2 = this.y + Math.sin(a);
			let minD = 400;
			for(let w of walls) {
				const { x: ix, y: iy, parallel }
					= findIntersect(this.x, this.y, x2, y2, w.x1, w.y1, w.x2, w.y2);
				if(parallel)
					continue;
				if((ix - w.x1) * (ix - w.x2) <= 0
				&& (iy - w.y1) * (iy - w.y2) <= 0
				&& Math.abs(ix - x2) < Math.abs(ix - this.x)) {
					const dSq = (ix - this.x) ** 2 + (iy - this.y) ** 2;
					minD = Math.min(minD, Math.sqrt(dSq));
				}
			}
			ds.push(minD);
		}
		return ds;
	}

	getCorners() {
		const hh = this.height / 2;
		const hw = this.width / 2;
		const s = Math.sin(this.angle);
		const c = Math.cos(this.angle);
		return [
			{
				x: this.x + hh * c - hw * s,
				y: this.y + hh * s + hw * c
			},
			{
				x: this.x - hh * c - hw * s,
				y: this.y - hh * s + hw * c
			},
			{
				x: this.x - hh * c + hw * s,
				y: this.y - hh * s - hw * c
			},
			{
				x: this.x + hh * c + hw * s,
				y: this.y + hh * s - hw * c
			}
		];
	}

	intersectsWall({ x1, y1, x2, y2 }) {
		//Based on https://stackoverflow.com/a/37621050
		const [c1, ...cs] = this.getCorners();

		let dir = {
			x: x2 - x1,
			y: y2 - y1
		};
		const segLen = vectorMag(dir.x, dir.y);
		dir.x /= segLen;
		dir.y /= segLen;
		const lineDist = vectorDot(dir.x, dir.y, x1, y1);
		const perpDir = {
			x: -dir.y,
			y: dir.x
		};
		const perpLineDist = vectorDot(perpDir.x, perpDir.y, x1, y1);

		//Check if infinite line intersects rectangle
		let minDist = vectorDot(perpDir.x, perpDir.y, c1.x, c1.y) - perpLineDist;
		let maxDist = minDist;
		for(let c of cs) {
			const d = vectorDot(perpDir.x, perpDir.y, c.x, c.y) - perpLineDist;
			minDist = Math.min(minDist, d);
			maxDist = Math.max(maxDist, d);
		}

		if(minDist > 0 || maxDist < 0) //All corners are on same side of wall
			return false;
		
		//Check if rectangle falls within line segment
		minDist = vectorDot(dir.x, dir.y, c1.x, c1.y) - lineDist;
		maxDist = minDist;
		for(let c of cs) {
			const d = vectorDot(dir.x, dir.y, c.x, c.y) - lineDist;
			minDist = Math.min(minDist, d);
			maxDist = Math.max(maxDist, d);
		}

		if(maxDist < 0 || minDist > segLen)
			return false; //Some corner falls within segment
		
		//Rectangle lies on infinite line and within segment
		return true;
	}
}

function findIntersect(x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b, tol=0.01) {
	if(Math.abs(x1a - x2a) < tol && Math.abs(x1b - x2b) < tol)
		return { parallel: true };
	if(Math.abs(x1a - x2a) < tol) {
		const mb = (y1b - y2b) / (x1b - x2b);
		const x = x1a;
		const y = mb * (x - x1b) + y1b;
		return {
			parallel: false,
			x, y
		};
	}
	if(Math.abs(x1b - x2b) < tol) {
		const ma = (y1a - y2a) / (x1a - x2a);
		const x = x1b;
		const y = ma * (x - x1a) + y1a;
		return {
			parallel: false,
			x, y
		};
	}
	const ma = (y1a - y2a) / (x1a - x2a);
	const mb = (y1b - y2b) / (x1b - x2b);
	if(Math.abs(ma - mb) < tol)
		return { parallel: true };
	const x = (ma*x1a - mb*x1b - y1a + y1b) / (ma - mb);
	const y = ma * (x - x1a) + y1a;
	return { x, y, parallel: false };
}

function vectorDot(x1, y1, x2, y2) {
	return x1 * x2 + y1 * y2;
}

function vectorMag(x, y) {
	return Math.sqrt(x*x + y*y);
}