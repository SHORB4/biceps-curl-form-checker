// testPoseRoiTransform.js
//
// Deterministic coordinate tests for pose.js's ROI crop-boundary
// transform: roiToRawCropRect() (mirrored ROI -> raw crop rect) and
// landmarkToFullFrame() (crop-relative landmark -> full-frame
// normalized landmark). Pure math, no DOM/camera/MediaPipe needed.
//
// Run in a browser: open test-pose-roi-transform.html via a local
// server (ES module imports require http(s)://, not file://).

import { roiToRawCropRect, landmarkToFullFrame } from "./pose.js";

function makeRecorder() {
  const results = [];
  let current = null;

  return {
    start(name) {
      current = { name, passed: true, failures: [] };
      results.push(current);
    },
    assertClose(actual, expected, tol, message) {
      const ok = Math.abs(actual - expected) < tol;
      if (!ok) {
        current.passed = false;
        current.failures.push(
          `${message} (got ${actual}, expected ~${expected}, tol ${tol})`
        );
      }
    },
    results
  };
}

const EPS = 1e-9;

export function runTests() {
  const r = makeRecorder();
  const W = 640;
  const H = 480;

  // ===============================================================
  // roiToRawCropRect: mirrored-display ROI -> raw (unmirrored) crop
  // rect. Only X flips (rawX = W - mirroredX2, i.e. the box's right
  // edge in mirrored space becomes the left edge in raw space); Y is
  // untouched, since nothing in this app mirrors vertically.
  // ===============================================================

  r.start("roiToRawCropRect - centered ROI");
  {
    const roi = { x1: 100, y1: 50, x2: 300, y2: 250 };
    const crop = roiToRawCropRect(roi, W);
    r.assertClose(crop.x, 340, EPS, "crop.x");
    r.assertClose(crop.y, 50, EPS, "crop.y");
    r.assertClose(crop.width, 200, EPS, "crop.width");
    r.assertClose(crop.height, 200, EPS, "crop.height");
  }

  r.start("roiToRawCropRect - ROI at mirrored-frame left edge (position case)");
  {
    // Left edge in DISPLAYED/mirrored space -> right edge in raw space.
    const roi = { x1: 0, y1: 0, x2: 160, y2: 120 };
    const crop = roiToRawCropRect(roi, W);
    r.assertClose(crop.x, 480, EPS, "crop.x"); // 640 - 160
    r.assertClose(crop.y, 0, EPS, "crop.y");
    r.assertClose(crop.width, 160, EPS, "crop.width");
    r.assertClose(crop.height, 120, EPS, "crop.height");
  }

  r.start("roiToRawCropRect - wide aspect ratio ROI");
  {
    const roi = { x1: 50, y1: 100, x2: 590, y2: 300 };
    const crop = roiToRawCropRect(roi, W);
    r.assertClose(crop.x, 50, EPS, "crop.x"); // 640 - 590
    r.assertClose(crop.y, 100, EPS, "crop.y");
    r.assertClose(crop.width, 540, EPS, "crop.width");
    r.assertClose(crop.height, 200, EPS, "crop.height");
  }

  r.start("roiToRawCropRect - ROI near mirrored-frame right edge (position case)");
  {
    const roi = { x1: 500, y1: 200, x2: 639, y2: 400 };
    const crop = roiToRawCropRect(roi, W);
    r.assertClose(crop.x, 1, EPS, "crop.x"); // 640 - 639
    r.assertClose(crop.y, 200, EPS, "crop.y");
    r.assertClose(crop.width, 139, EPS, "crop.width");
    r.assertClose(crop.height, 200, EPS, "crop.height");
  }

  // ===============================================================
  // landmarkToFullFrame: crop-relative [0,1] landmark -> full-frame
  // normalized landmark. Tested directly in the crop's own reference
  // frame (crop-relative (0,0)/(0.5,0.5)/(1,1) ARE literally the
  // crop's top-left/center/bottom-right corners - the frame of
  // reference MediaPipe itself reports landmarks in after cropping).
  // ===============================================================

  r.start("1. landmarkToFullFrame - crop top-left corner");
  {
    const cropRect = { x: 340, y: 50, width: 200, height: 200 };
    const out = landmarkToFullFrame({ x: 0, y: 0, z: 0.1, visibility: 0.95 }, cropRect, W, H);
    r.assertClose(out.x, 0.53125, EPS, "fullX");
    r.assertClose(out.y, 50 / 480, EPS, "fullY");
    r.assertClose(out.z, 0.1, EPS, "z should be preserved unchanged");
    r.assertClose(out.visibility, 0.95, EPS, "visibility should be preserved unchanged");
  }

  r.start("2. landmarkToFullFrame - crop center");
  {
    const cropRect = { x: 340, y: 50, width: 200, height: 200 };
    const out = landmarkToFullFrame({ x: 0.5, y: 0.5, visibility: 0.8 }, cropRect, W, H);
    r.assertClose(out.x, 0.6875, EPS, "fullX");
    r.assertClose(out.y, 0.3125, EPS, "fullY");
  }

  r.start("3. landmarkToFullFrame - crop bottom-right corner");
  {
    const cropRect = { x: 340, y: 50, width: 200, height: 200 };
    const out = landmarkToFullFrame({ x: 1, y: 1, visibility: 0.7 }, cropRect, W, H);
    r.assertClose(out.x, 0.84375, EPS, "fullX");
    r.assertClose(out.y, 250 / 480, EPS, "fullY");
  }

  r.start("4. landmarkToFullFrame - different ROI aspect ratio (wide crop)");
  {
    const cropRect = { x: 50, y: 100, width: 540, height: 200 }; // from the wide-ROI case above
    const out = landmarkToFullFrame({ x: 0.5, y: 0.5, visibility: 0.9 }, cropRect, W, H);
    r.assertClose(out.x, 0.5, EPS, "fullX");
    r.assertClose(out.y, 200 / 480, EPS, "fullY");
  }

  r.start("5. landmarkToFullFrame - different ROI position");
  {
    const cropRect = { x: 1, y: 200, width: 139, height: 200 }; // from the right-edge ROI case above
    const topLeft = landmarkToFullFrame({ x: 0, y: 0, visibility: 0.9 }, cropRect, W, H);
    const bottomRight = landmarkToFullFrame({ x: 1, y: 1, visibility: 0.9 }, cropRect, W, H);
    r.assertClose(topLeft.x, 1 / 640, EPS, "topLeft.x");
    r.assertClose(topLeft.y, 200 / 480, EPS, "topLeft.y");
    r.assertClose(bottomRight.x, 140 / 640, EPS, "bottomRight.x");
    r.assertClose(bottomRight.y, 400 / 480, EPS, "bottomRight.y");
  }

  // ===============================================================
  // Round-trip identity: for ANY real raw-frame position inside the
  // crop, computing what MediaPipe would report for that position
  // (the inverse of landmarkToFullFrame) and feeding it back through
  // landmarkToFullFrame must reproduce the original position exactly.
  // This is a general correctness check independent of any specific
  // corner, decoupled from the separate mirroring-flip tests above.
  // ===============================================================

  r.start("landmarkToFullFrame - round-trip identity");
  {
    const cropRect = { x: 340, y: 50, width: 200, height: 200 };
    const rawNormX = 0.6;
    const rawNormY = 0.3;

    const cropRelX = (rawNormX * W - cropRect.x) / cropRect.width;
    const cropRelY = (rawNormY * H - cropRect.y) / cropRect.height;

    const out = landmarkToFullFrame({ x: cropRelX, y: cropRelY }, cropRect, W, H);
    r.assertClose(out.x, rawNormX, EPS, "round-trip fullX");
    r.assertClose(out.y, rawNormY, EPS, "round-trip fullY");
  }

  // ===============================================================
  // End-to-end: mirrored ROI -> raw crop rect -> landmark -> full
  // frame, composing BOTH functions together for a couple of
  // realistic scenarios (not just each function in isolation).
  // ===============================================================

  r.start("End-to-end - mirrored ROI through both transform steps");
  {
    const roi = { x1: 100, y1: 50, x2: 300, y2: 250 };
    const cropRect = roiToRawCropRect(roi, W);
    const out = landmarkToFullFrame({ x: 0.5, y: 0.5, visibility: 1 }, cropRect, W, H);
    // Center of the crop (which is also the center of the ROI, since
    // flipping a rect about the frame width doesn't change its own
    // center-relative geometry) should land at the crop's own center
    // in full-frame terms.
    r.assertClose(out.x, (cropRect.x + cropRect.width / 2) / W, EPS, "end-to-end center fullX");
    r.assertClose(out.y, (cropRect.y + cropRect.height / 2) / H, EPS, "end-to-end center fullY");
  }

  return r.results;
}

export function printReport(results) {
  let passCount = 0;

  for (const test of results) {
    if (test.passed) {
      passCount += 1;
      console.log(`[PASS] ${test.name}`);
    } else {
      console.log(`[FAIL] ${test.name}`);
      for (const failure of test.failures) {
        console.log(`        - ${failure}`);
      }
    }
  }

  console.log("");
  console.log(`${passCount}/${results.length} scenarios passed.`);

  return passCount === results.length;
}
