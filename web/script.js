const CLASS_NAMES = ["cardboard", "paper", "metal", "plastic", "glass", "trash"];
const MODEL_INPUT_SIZE = 416;
const MODEL_URL = "./model/model.json";
const MAX_DETECTIONS = 20;
const IOU_THRESHOLD = 0.45;

let confidenceThreshold = 0.4;

async function main() {
  const status = document.getElementById("status");
  const video = document.getElementById("webcam");
  const canvas = document.getElementById("overlay");
  const ctx = canvas.getContext("2d");
  const confidenceInput = document.getElementById("confidence");
  const confidenceValue = document.getElementById("confidence-value");

  confidenceInput.addEventListener("input", () => {
    confidenceThreshold = parseFloat(confidenceInput.value);
    confidenceValue.textContent = confidenceThreshold.toFixed(2);
  });

  status.textContent = "모델을 불러오는 중...";
  const model = await tf.loadGraphModel(MODEL_URL);

  status.textContent = "웹캠 접근 권한을 요청하는 중...";
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await new Promise((resolve) => {
    video.onloadedmetadata = resolve;
  });

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  status.textContent = "실시간 분류 중";
  detectLoop(model, video, canvas, ctx);
}

async function detectLoop(model, video, canvas, ctx) {
  const detections = await runInference(model, video);
  drawDetections(ctx, canvas, detections);
  requestAnimationFrame(() => detectLoop(model, video, canvas, ctx));
}

async function runInference(model, video) {
  const input = tf.tidy(() =>
    tf.browser
      .fromPixels(video)
      .resizeBilinear([MODEL_INPUT_SIZE, MODEL_INPUT_SIZE])
      .div(255.0)
      .expandDims(0)
  );

  const output = await model.executeAsync(input);
  const predictions = Array.isArray(output) ? output[0] : output;
  const [boxesRaw] = await predictions.array();

  input.dispose();
  if (Array.isArray(output)) {
    output.forEach((t) => t.dispose());
  } else {
    output.dispose();
  }

  const boxes = [];
  const scores = [];
  const classIds = [];

  for (const row of boxesRaw) {
    const [cx, cy, w, h, objectness, ...classScores] = row;
    let bestClass = 0;
    let bestScore = classScores[0];
    for (let i = 1; i < classScores.length; i++) {
      if (classScores[i] > bestScore) {
        bestScore = classScores[i];
        bestClass = i;
      }
    }
    const confidence = objectness * bestScore;
    if (confidence < confidenceThreshold) continue;

    // tf.image.nonMaxSuppressionAsync expects normalized [y1, x1, y2, x2]
    boxes.push([
      (cy - h / 2) / MODEL_INPUT_SIZE,
      (cx - w / 2) / MODEL_INPUT_SIZE,
      (cy + h / 2) / MODEL_INPUT_SIZE,
      (cx + w / 2) / MODEL_INPUT_SIZE,
    ]);
    scores.push(confidence);
    classIds.push(bestClass);
  }

  if (boxes.length === 0) return [];

  const boxesTensor = tf.tensor2d(boxes);
  const scoresTensor = tf.tensor1d(scores);
  const nmsIndices = await tf.image.nonMaxSuppressionAsync(
    boxesTensor,
    scoresTensor,
    MAX_DETECTIONS,
    IOU_THRESHOLD,
    confidenceThreshold
  );
  const keep = await nmsIndices.array();
  boxesTensor.dispose();
  scoresTensor.dispose();
  nmsIndices.dispose();

  return keep.map((i) => ({
    box: boxes[i],
    score: scores[i],
    className: CLASS_NAMES[classIds[i]] ?? `class_${classIds[i]}`,
  }));
}

function drawDetections(ctx, canvas, detections) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2;
  ctx.font = "16px sans-serif";
  ctx.textBaseline = "top";

  for (const { box, score, className } of detections) {
    const [y1, x1, y2, x2] = box;
    const x = x1 * canvas.width;
    const y = y1 * canvas.height;
    const w = (x2 - x1) * canvas.width;
    const h = (y2 - y1) * canvas.height;

    ctx.strokeStyle = "#2ecc71";
    ctx.strokeRect(x, y, w, h);

    const label = `${className} ${(score * 100).toFixed(0)}%`;
    const textWidth = ctx.measureText(label).width;
    const labelY = Math.max(0, y - 20);
    ctx.fillStyle = "#2ecc71";
    ctx.fillRect(x, labelY, textWidth + 8, 20);
    ctx.fillStyle = "#111";
    ctx.fillText(label, x + 4, labelY);
  }
}

main().catch((err) => {
  console.error(err);
  document.getElementById("status").textContent = `오류: ${err.message}`;
});
