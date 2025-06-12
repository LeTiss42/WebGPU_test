function test(name) {
	console.log("utils test", name);
}

const degToRad = d => d / 180 * Math.PI;

export { test, degToRad};
