// renderer.js
//
// Phase 3 - debug-only pose skeleton overlay.
//
// Draws MediaPipe PoseLandmarker results onto a canvas stacked over
// the shared <video> element. This is visualization only: it does
// not feed exerciseAnalyzer.js, compute angles, or count reps.
//
// Coordinate mapping: the canvas is sized to the video element's
// full rendered box, same convention as roi.js's ROI overlay canvas,
// and landmarks are mapped into it using the same object-fit:
// contain letterbox math roi.js uses (duplicated here in miniature -
// both are small, stable formulas; keep them in sync if the video's
// CSS fit approach ever changes).
//
// Mirroring: the <video> is mirrored for display via CSS
// (transform: scaleX(-1)), but MediaPipe's detectForVideo() (see
// pose.js) reads the video element's actual decoded frame, which CSS
// transforms do NOT affect - MediaPipe always sees the raw,
// unmirrored camera image. So landmark x-coordinates arrive in
// *unmirrored* space, while this canvas (which has no CSS mirror of
// its own, unlike the video) is rendered in on-screen/mirrored space
// to line up with everything else. To make the skeleton visually
// track the mirrored video, landmark x is flipped here:
//   canvasX = offsetX + (1 - landmark.x) * videoWidth * scale
// This is a DISPLAY-ONLY correction for drawing. It does not change,
// swap, or interpret which MediaPipe landmark is "left" vs "right" -
// see pose.js's header comment and the Phase 3 report for that.
//
// Arm-only rendering: drawPoseOverlay() now takes the selected arm's
// landmark indices (armIndices, from armMapping.js's
// armLandmarkIndices()) and draws ONLY that shoulder/elbow/wrist plus
// the two connections between them - no face, no opposite arm, no
// hands/fingers, no legs/hips/torso, regardless of what MediaPipe
// detected. This is purely a rendering restriction: result.landmarks
// itself is never modified or filtered before reaching the analyzer -
// MediaPipe still detects and returns the full 33-point pose every
// frame, and app.js's armMapping.js-based selection (which is what
// actually reaches ExerciseAnalyzer.update()) is entirely unaffected
// by what this module chooses to draw.

function videoContentBox(videoEl) {
  const rect = videoEl.getBoundingClientRect();
  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;

  if (!vw || !vh || rect.width === 0 || rect.height === 0) {
    return null;
  }

  const scale = Math.min(rect.width / vw, rect.height / vh);
  const contentW = vw * scale;
  const contentH = vh * scale;

  return {
    scale,
    vw,
    vh,
    offsetX: (rect.width - contentW) / 2,
    offsetY: (rect.height - contentH) / 2
  };
}

export function resizePoseCanvas(canvasEl, videoEl) {
  const rect = videoEl.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));

  if (canvasEl.width !== w) canvasEl.width = w;
  if (canvasEl.height !== h) canvasEl.height = h;
}

export function clearPoseOverlay(canvasEl) {
  const ctx = canvasEl.getContext("2d");
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
}

/**
 * Draws ONLY the selected arm's shoulder/elbow/wrist (plus the two
 * connections between them) from a MediaPipe PoseLandmarkerResult
 * onto canvasEl, aligned to videoEl's current on-screen (letterboxed,
 * mirrored) box. Every other detected landmark - face, opposite arm,
 * hands/fingers, hips/legs/torso - is never drawn, regardless of what
 * MediaPipe returned.
 *
 * armIndices: {shoulder, elbow, wrist} MediaPipe landmark indices for
 *   the currently selected arm, from armMapping.js's
 *   armLandmarkIndices(). Required to draw anything - with no
 *   armIndices (or no detected pose), this only clears the canvas.
 */
export function drawPoseOverlay(canvasEl, videoEl, result, armIndices) {
  resizePoseCanvas(canvasEl, videoEl);
  clearPoseOverlay(canvasEl);

  if (!result || !result.landmarks || result.landmarks.length === 0) {
    return;
  }
  if (!armIndices) return;

  const box = videoContentBox(videoEl);
  if (!box) return;

  const ctx = canvasEl.getContext("2d");

  const toCanvas = (landmark) => ({
    x: box.offsetX + (1 - landmark.x) * box.vw * box.scale,
    y: box.offsetY + landmark.y * box.vh * box.scale
  });

  for (const landmarks of result.landmarks) {
    const shoulder = landmarks[armIndices.shoulder];
    const elbow = landmarks[armIndices.elbow];
    const wrist = landmarks[armIndices.wrist];
    if (!shoulder || !elbow || !wrist) continue;

    const pShoulder = toCanvas(shoulder);
    const pElbow = toCanvas(elbow);
    const pWrist = toCanvas(wrist);

    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pShoulder.x, pShoulder.y);
    ctx.lineTo(pElbow.x, pElbow.y);
    ctx.moveTo(pElbow.x, pElbow.y);
    ctx.lineTo(pWrist.x, pWrist.y);
    ctx.stroke();

    ctx.fillStyle = "#fbbf24";
    for (const p of [pShoulder, pElbow, pWrist]) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
