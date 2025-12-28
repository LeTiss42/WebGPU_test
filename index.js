import {test as toto, degToRad} from "./utils.js"

async function start() {

	if (!navigator.gpu) {
		throw new Error("WebGPU not supported on this browser.");
	}

	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		throw new Error("No appropriate GPUAdapter found.");
	}

	// the ? is an optional chaining operator
	// avoid error if nullish
	const device = await adapter?.requestDevice();
	device.lost.then((info) => {
		console.error(`WebGPU device was lost: ${info.message}`);
		// retry if the loss was unintentionnal
		if (info.reason !== 'destroyed') {
			start();
		}
	});

	main(device);
}

function createFVertices() {
	// Each vertex now has: position (x, y) + barycentric (a, b, c)
	const vertexData = new Float32Array([
		// left column - triangle 1
		0, 0,      1, 0, 0,
		30, 0,     0, 1, 0,
		0, 150,    0, 0, 1,
		// left column - triangle 2
		0, 150,    1, 0, 0,
		30, 0,     0, 1, 0,
		30, 150,   0, 0, 1,
	
		// top rung - triangle 1
		30, 0,     1, 0, 0,
		100, 0,    0, 1, 0,
		30, 30,    0, 0, 1,
		// top rung - triangle 2
		30, 30,    1, 0, 0,
		100, 0,    0, 1, 0,
		100, 30,   0, 0, 1,
	
		// middle rung - triangle 1
		30, 60,    1, 0, 0,
		70, 60,    0, 1, 0,
		30, 90,    0, 0, 1,
		// middle rung - triangle 2
		30, 90,    1, 0, 0,
		70, 60,    0, 1, 0,
		70, 90,    0, 0, 1,
	]);

	// Old indexData with shared vertices (more efficient but incompatible with barycentric coords):
	// const indexData = new Uint32Array([
	//     0,  1,  2,    2,  1,  3,  // left column
	//     4,  5,  6,    6,  5,  7,  // top rung
	//     8,  9, 10,   10,  9, 11,  // middle rung
	// ]);

	const indexData = new Uint32Array([
		0,  1,  2,
		3,  4,  5,
		6,  7,  8,
		9, 10, 11,
		12, 13, 14,
		15, 16, 17,
	]);

	return {
		vertexData,
		indexData,
		numVertices: indexData.length,
	};
}

//shader code
function myShaderCode() {
	return /* wgsl */`
		struct Uniforms {
			color: vec4f,
			resolution: vec2f,
			translation: vec2f,
			rotation: vec2f,
			scale: vec2f,
		};

		struct Vertex {
			@location(0) position: vec2f,
			@location(1) barycentric: vec3f,
		};

		struct VSOutput {
			@builtin(position) position: vec4f,
			@location(0) barycentric: vec3f,
		};

		@group(0) @binding(0) var<uniform> uni: Uniforms;

		@vertex fn vs(vert: Vertex) -> VSOutput {
			var vsOut: VSOutput;

			let scaledPosition = vert.position * uni.scale;
			let rotatedPosition = vec2f(
				scaledPosition.x * uni.rotation.x - scaledPosition.y * uni.rotation.y,
				scaledPosition.x * uni.rotation.y + scaledPosition.y * uni.rotation.x
			);
			//translating 2D data to required 4D vector on <-1, 1> clip space
			let position = rotatedPosition + uni.translation;
			let zeroToOne = position / uni.resolution;
			let zeroToTwo = zeroToOne * 2.0;
			// clip space is from -1 to 1
			let flippedClipSpace = zeroToTwo - 1;
			//flip Y component to keep first data = first pixel
			let clipSpace = flippedClipSpace * vec2f(1, -1);

			vsOut.position = vec4f(clipSpace, 0.0, 1.0);
			vsOut.barycentric = vert.barycentric;
			return vsOut;
		}

		@fragment fn fs(fsInput: VSOutput) -> @location(0) vec4f {
			// Edge detection using barycentric coordinates
			let edgeThickness = 0.03;
			let isEdge = min(min(fsInput.barycentric.x, fsInput.barycentric.y), fsInput.barycentric.z) < edgeThickness;
			
			if (isEdge) {
				return vec4f(0.054, 0.298, 0.165, 1.0);
			} else {
				return uni.color;
			}
		}
	`;
}

function main(device) {

	const settings = {
		translation: [0, 0],
		rotation: 0,
		scale: [1, 1],
	};

	//get canvas
	const canvas = document.querySelector("#gpuCanvas");
	const context = canvas.getContext("webgpu");
	const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
	context.configure({
		device: device,
		format: canvasFormat,
		alphaMode: "premultiplied",
	});

	const myShaderModule = device.createShaderModule({
		label: "myShader",
		code: myShaderCode(),
	});

	//pipeline
	const pipeline = device.createRenderPipeline({
		label: "just 2d position",
		layout: "auto",
		vertex: {
			module: myShaderModule,
			buffers: [{
				arrayStride: (2 + 3) * 4,
				attributes: [
					{
						format: "float32x2",
						offset: 0,
						shaderLocation: 0,
					},
					{
						format: "float32x3",
						offset: 2 * 4,
						shaderLocation: 1,
					}
				],
			}]
		},
		fragment: {
			module: myShaderModule,
			targets: [{
				format: canvasFormat
			}]
		}
	});


	//uniform buffer
	const uniformBufferSize = (4 + 2 + 2 + 2 + 2) * 4;
	const uniformBuffer = device.createBuffer({
		label: "uniforms buffer",
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	const uniformValues = new Float32Array(uniformBufferSize / 4);

	const kColorOffset = 0;
	const kResolutionOffset = 4;
	const kTranslationOffset = 6;
	const kRotationOffset = 8;
	const kScaleOffset = 10;

	// subarray() method is creating a new view on the existing buffer;
	// changes to the new object's contents will impact the original object and vice versa.
	const colorValue = uniformValues.subarray(kColorOffset, kColorOffset + 4);
	const resolutionValue = uniformValues.subarray(kResolutionOffset, kResolutionOffset + 2);
	const translationValue = uniformValues.subarray(kTranslationOffset, kTranslationOffset + 2);
	const rotationValue = uniformValues.subarray(kRotationOffset, kRotationOffset + 2);
	const scaleValue = uniformValues.subarray(kScaleOffset, kScaleOffset + 2);

	// colorValue.set([Math.random(), Math.random(), Math.random(), 1]);
	colorValue.set([0.141, 0.749, 0.412, 1]);

	// javascript destructuring syntax
	const { vertexData, indexData, numVertices} = createFVertices();

	const vertexBuffer = device.createBuffer({
		label: 'vertex buffer vertices',
		size: vertexData.byteLength,
		usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
	});
	device.queue.writeBuffer(vertexBuffer, 0, vertexData);

	const indexBuffer = device.createBuffer({
		label: 'index buffer',
		size: indexData.byteLength,
		usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
	});
	device.queue.writeBuffer(indexBuffer, 0, indexData);

	const bindGroup = device.createBindGroup({
		label: "my bindGroup",
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: { buffer: uniformBuffer }},
		],
	});

	const renderPassDescriptor = {
		label: 'our basic canvas renderPass',
		colorAttachments: [
			{
				// view: will be filled at render
				clearValue: [0, 0, 0, 0],
				loadOp: 'clear',
				storeOp: 'store',
			},
		],
	};

	function render() {

		renderPassDescriptor.colorAttachments[0].view =
			context.getCurrentTexture().createView();

		const encoder = device.createCommandEncoder({ label: "our encoder"});
		const pass = encoder.beginRenderPass(renderPassDescriptor);

		pass.setPipeline(pipeline);
		pass.setVertexBuffer(0, vertexBuffer);
		pass.setIndexBuffer(indexBuffer, 'uint32');

		// Set uniform values
		resolutionValue.set([canvas.width, canvas.height]);
		translationValue.set(settings.translation);
		rotationValue.set([
			Math.cos(degToRad(settings.rotation)),
			Math.sin(degToRad(settings.rotation)),
		]);
		scaleValue.set(settings.scale);
		device.queue.writeBuffer(uniformBuffer, 0, uniformValues);

		pass.setBindGroup(0, bindGroup);
		pass.drawIndexed(numVertices);
		pass.end();
		device.queue.submit([encoder.finish()]);
	}

	// resize canvas properly
	const observer = new ResizeObserver(entries => {
		for (const entry of entries) {
			const canvas = entry.target;
			const width = entry.contentBoxSize[0].inlineSize;
			const height = entry.contentBoxSize[0].blockSize;
			canvas.width = Math.max(1, Math.min(width, device.limits.maxTextureDimension2D));
			canvas.height = Math.max(1, Math.min(height, device.limits.maxTextureDimension2D));
		}
		render();
	});
	observer.observe(canvas);

	document.addEventListener("keydown", (event) => {
		const transStep = 5; // Déplacement en pixels
		const rotStep = 5; // Rotation en degres
		const scaleStep = 0.1;

		switch (event.key) {
			case "ArrowUp":
				settings.translation[1] -= transStep; // Déplacer vers le haut
				break;
			case "ArrowDown":
				settings.translation[1] += transStep; // Déplacer vers le bas
				break;
			case "ArrowLeft":
				settings.translation[0] -= transStep; // Déplacer à gauche
				break;
			case "ArrowRight":
				settings.translation[0] += transStep; // Déplacer à droite
				break;
			case "a":
				settings.rotation += rotStep; // Déplacer à gauche
				break;
			case "s":
				settings.rotation -= rotStep; // Déplacer à droite
				break;
			case "z":
				settings.scale[0] += scaleStep; // Déplacer à gauche
				break;
			case "x":
				settings.scale[0] -= scaleStep; // Déplacer à droite
				break;
			case "c":
				settings.scale[1] += scaleStep; // Déplacer à gauche
				break;
			case "v":
				settings.scale[1] -= scaleStep; // Déplacer à droite
				break;
		}
		render();
	});
}

start();
