// testArmMapping.js
//
// Deterministic tests for armMapping.js's index selection and
// landmark extraction. This does NOT (and cannot, without a real
// camera/browser) verify that the mapping is physically correct for
// a real person - only that the code does what it says: selects the
// documented MediaPipe indices for each arm, and extracts exactly
// those landmarks from an array. See the Phase 4 report for the
// required manual left/right verification.

import { armLandmarkIndices, selectArmLandmarks, ARM_MAPPING_SWAPPED } from "./armMapping.js";

function makeRecorder() {
  const results = [];
  let current = null;
  return {
    start(name) {
      current = { name, passed: true, failures: [] };
      results.push(current);
    },
    assert(condition, message) {
      if (!condition) {
        current.passed = false;
        current.failures.push(message);
      }
    },
    results
  };
}

function buildLandmarks() {
  // 33 placeholder landmarks, each tagged with its own index so
  // extraction can be verified unambiguously.
  const landmarks = [];
  for (let i = 0; i < 33; i++) {
    landmarks.push({ x: i / 100, y: i / 200, z: 0, visibility: 0.9, index: i });
  }
  return landmarks;
}

export function runTests() {
  const r = makeRecorder();

  r.start("armLandmarkIndices - left arm (unswapped default)");
  {
    const ids = armLandmarkIndices("left");
    if (!ARM_MAPPING_SWAPPED) {
      r.assert(ids.shoulder === 11, `expected LEFT_SHOULDER=11, got ${ids.shoulder}`);
      r.assert(ids.elbow === 13, `expected LEFT_ELBOW=13, got ${ids.elbow}`);
      r.assert(ids.wrist === 15, `expected LEFT_WRIST=15, got ${ids.wrist}`);
    } else {
      r.assert(ids.shoulder === 12, `swapped: expected RIGHT_SHOULDER=12, got ${ids.shoulder}`);
      r.assert(ids.elbow === 14, `swapped: expected RIGHT_ELBOW=14, got ${ids.elbow}`);
      r.assert(ids.wrist === 16, `swapped: expected RIGHT_WRIST=16, got ${ids.wrist}`);
    }
  }

  r.start("armLandmarkIndices - right arm (unswapped default)");
  {
    const ids = armLandmarkIndices("right");
    if (!ARM_MAPPING_SWAPPED) {
      r.assert(ids.shoulder === 12, `expected RIGHT_SHOULDER=12, got ${ids.shoulder}`);
      r.assert(ids.elbow === 14, `expected RIGHT_ELBOW=14, got ${ids.elbow}`);
      r.assert(ids.wrist === 16, `expected RIGHT_WRIST=16, got ${ids.wrist}`);
    } else {
      r.assert(ids.shoulder === 11, `swapped: expected LEFT_SHOULDER=11, got ${ids.shoulder}`);
      r.assert(ids.elbow === 13, `swapped: expected LEFT_ELBOW=13, got ${ids.elbow}`);
      r.assert(ids.wrist === 15, `swapped: expected LEFT_WRIST=15, got ${ids.wrist}`);
    }
  }

  r.start("armLandmarkIndices - left and right always resolve to different indices");
  {
    const left = armLandmarkIndices("left");
    const right = armLandmarkIndices("right");
    r.assert(left.shoulder !== right.shoulder, "left/right shoulder must differ");
    r.assert(left.elbow !== right.elbow, "left/right elbow must differ");
    r.assert(left.wrist !== right.wrist, "left/right wrist must differ");
  }

  r.start("selectArmLandmarks - extracts the correct objects by index");
  {
    const landmarks = buildLandmarks();
    const ids = armLandmarkIndices("left");
    const selected = selectArmLandmarks(landmarks, "left");
    r.assert(selected.shoulder.index === ids.shoulder, "shoulder extraction mismatch");
    r.assert(selected.elbow.index === ids.elbow, "elbow extraction mismatch");
    r.assert(selected.wrist.index === ids.wrist, "wrist extraction mismatch");
  }

  r.start("selectArmLandmarks - preserves landmark properties unchanged");
  {
    const landmarks = buildLandmarks();
    const selected = selectArmLandmarks(landmarks, "right");
    r.assert(typeof selected.elbow.visibility === "number", "visibility should pass through");
    r.assert(selected.elbow.visibility === 0.9, "visibility value should be unchanged");
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
      for (const failure of test.failures) console.log(`        - ${failure}`);
    }
  }
  console.log("");
  console.log(`${passCount}/${results.length} scenarios passed.`);
  return passCount === results.length;
}
