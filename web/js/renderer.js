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
// Face landmarks (display-only filter): MediaPipe's 33-point BlazePose
// topology reserves indices 0-10 for the face (nose, eyes, ears,
// mouth) and 11+ for the body. drawPoseOverlay() below skips drawing
// any point or connection touching indices < FIRST_BODY_LANDMARK_INDEX.
// This is purely a rendering choice - result.landmarks itself is never
// modified, so app.js's analyzer wiring (which reads shoulder/elbow/
// wrist at indices 11-16 via armMapping.js) is completely unaffected;
// MediaPipe detection, landmark indices, and visibility values are
// untouched.

import { PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

// MediaPipe Pose landmarks 0-10 are the face (nose, eyes x6, ears x2,
// mouth x2); 11+ are the body, starting with the shoulders.
const FIRST_BODY_LANDMARK_INDEX = 11;

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
 * Draws every detected pose's landmarks + skeleton connections from a
 * MediaPipe PoseLandmarkerResult onto canvasEl, aligned to videoEl's
 * current on-screen (letterboxed, mirrored) box. Safe to call with a
 * "no pose detected" result (empty/missing landmarks) - it just
 * resizes and clears the canvas.
 */
export function drawPoseOverlay(canvasEl, videoEl, result) {
  resizePoseCanvas(canvasEl, videoEl);
  clearPoseOverlay(canvasEl);

  if (!result || !result.landmarks || result.landmarks.length === 0) {
    return;
  }

  const box = videoContentBox(videoEl);
  if (!box) return;

  const ctx = canvasEl.getContext("2d");

  const toCanvas = (landmark) => ({
    x: box.offsetX + (1 - landmark.x) * box.vw * box.scale,
    y: box.offsetY + landmark.y * box.vh * box.scale
  });

  for (const landmarks of result.landmarks) {
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (const connection of PoseLandmarker.POSE_CONNECTIONS) {
      // Skip any connection touching a face landmark - render body/arm
      // connections only.
      if (connection.start < FIRST_BODY_LANDMARK_INDEX || connection.end < FIRST_BODY_LANDMARK_INDEX) {
        continue;
      }

      const start = landmarks[connection.start];
      const end = landmarks[connection.end];
      if (!start || !end) continue;

      const p1 = toCanvas(start);
      const p2 = toCanvas(end);
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
    }

    ctx.stroke();

    ctx.fillStyle = "#fbbf24";
    for (let i = FIRST_BODY_LANDMARK_INDEX; i < landmarks.length; i++) {
      // Starting the loop at FIRST_BODY_LANDMARK_INDEX (rather than
      // filtering inside a for-of) skips every face point (0-10)
      // without touching the landmarks array itself.
      const landmark = landmarks[i];
      if (!landmark) continue;

      const p = toCanvas(landmark);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}
