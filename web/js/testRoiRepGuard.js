// testRoiRepGuard.js
//
// Regression test for the mobile ROI glitch: leaving the confirmed
// ROI mid-curl and returning to it must never complete a rep from
// stale analyzer state. Exercises the REAL ExerciseAnalyzer and the
// REAL isArmWithinRoi() (both imported, unmodified) through a small
// driveFrame() helper that mirrors app.js's onResult decision.
//
// IMPORTANT: driveFrame() below must be kept in sync with the actual
// onResult handler in app.js (the "if (result.landmarks...) { if
// (isArmWithinRoi(...)) { analyzer.update(...) } else {
// resetInProgressRep() } }" block) - it is a deliberate, disclosed
// copy for testability, not a reimplementation of separate logic.
// app.js's own resetInProgressRep() field list must match the one
// here if it ever changes.
//
// Run in a browser: open test-roi-rep-guard.html via a local server
// (ES module imports require http(s)://, not file://).

import { ExerciseAnalyzer } from "./exerciseAnalyzer.js";
import { isArmWithinRoi } from "./pose.js";

// ---------------------------------------------------------------
// Synthetic geometry: a virtual 2000x1000 video frame with a
// confirmed ROI whose raw (unmirrored) footprint is x:[100,900],
// y:[50,950] - roomy enough to hold every "inside" landmark position
// below regardless of curl angle, since calculateAngle() only cares
// about direction, not distance, so a small wrist-to-elbow radius
// keeps the whole swing safely inside those bounds.
// ---------------------------------------------------------------

const VIDEO_W = 2000;
const VIDEO_H = 1000;

// Mirrored/display-space ROI (as roi.js would store it) that produces
// the raw rect described above via pose.js's own roiToRawCropRect():
// raw.x = VIDEO_W - roi.x2 = 100 => roi.x2 = 1900; raw.width = 800 =>
// roi.x1 = 1100. y is untouched by mirroring.
const ROI = { x1: 1100, y1: 50, x2: 1900, y2: 950 };

const INSIDE_SHOULDER_XY = [0.25, 0.10]; // -> raw (500, 100), well inside [100,900]x[50,950]
const INSIDE_ELBOW_XY = [0.25, 0.30]; // -> raw (500, 300)
const INSIDE_RADIUS = 0.05; // wrist stays within (450-550, 250-350) of the raw frame - all inside the ROI

// Landmarks that are unambiguously outside the ROI (raw x ~1700,
// beyond the ROI's raw x-max of 900) - used for "arm left the ROI"
// frames. The exact angle they'd imply doesn't matter: driveFrame()
// never passes them to analyzer.update() in the first place.
const OUTSIDE_SHOULDER = lm(0.85, 0.10);
const OUTSIDE_ELBOW = lm(0.85, 0.30);
const OUTSIDE_WRIST = lm(0.85, 0.32);

function lm(x, y, visibility = 0.9) {
  return { x, y, visibility };
}

function wristForAngle(angleDeg, shoulder, elbow, radius) {
  const shoulderTheta = radToDeg(
    Math.atan2(shoulder[1] - elbow[1], shoulder[0] - elbow[0])
  );
  const theta = degToRad(shoulderTheta + angleDeg);
  return [elbow[0] + radius * Math.cos(theta), elbow[1] + radius * Math.sin(theta)];
}

function radToDeg(r) {
  return (r * 180) / Math.PI;
}

function degToRad(d) {
  return (d * Math.PI) / 180;
}

/** An "inside the ROI" landmark triple at the given elbow angle. */
function insideFrame(angleDeg) {
  const [wx, wy] = wristForAngle(angleDeg, INSIDE_SHOULDER_XY, INSIDE_ELBOW_XY, INSIDE_RADIUS);
  return {
    shoulder: lm(INSIDE_SHOULDER_XY[0], INSIDE_SHOULDER_XY[1]),
    elbow: lm(INSIDE_ELBOW_XY[0], INSIDE_ELBOW_XY[1]),
    wrist: lm(wx, wy)
  };
}

/** An "outside the ROI" landmark triple - angle is irrelevant (see above). */
function outsideFrame() {
  return { shoulder: OUTSIDE_SHOULDER, elbow: OUTSIDE_ELBOW, wrist: OUTSIDE_WRIST };
}

// ---------------------------------------------------------------
// driveFrame(): the exact decision app.js's onResult makes for one
// frame, given a landmark triple already selected for the arm. See
// the header comment - keep this in sync with app.js.
// ---------------------------------------------------------------

function resetInProgressRep(analyzer) {
  analyzer.stage = "down";
  analyzer.repStartTime = null;
  analyzer.currentRepMinAngle = 180;
  analyzer.currentRepElbowStartX = null;
}

function driveFrame(analyzer, frame, timestamp) {
  const { shoulder, elbow, wrist } = frame;

  if (isArmWithinRoi(shoulder, elbow, wrist, ROI, VIDEO_W, VIDEO_H)) {
    return analyzer.update(shoulder, elbow, wrist, timestamp);
  }

  resetInProgressRep(analyzer);
  return null;
}

// ---------------------------------------------------------------
// Minimal assertion helpers - same pattern as the project's other
// test files.
// ---------------------------------------------------------------

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

export function runTests() {
  const r = makeRecorder();

  // ===============================================================
  // Sanity: the geometry fixtures above actually land where intended.
  // ===============================================================
  r.start("0. Fixture sanity - inside/outside frames resolve as expected");
  {
    const inside = insideFrame(180);
    const outside = outsideFrame();
    r.assert(
      isArmWithinRoi(inside.shoulder, inside.elbow, inside.wrist, ROI, VIDEO_W, VIDEO_H) === true,
      "insideFrame() should be accepted by isArmWithinRoi()"
    );
    r.assert(
      isArmWithinRoi(outside.shoulder, outside.elbow, outside.wrist, ROI, VIDEO_W, VIDEO_H) === false,
      "outsideFrame() should be rejected by isArmWithinRoi()"
    );
  }

  // ===============================================================
  // 1/2. Normal reps entirely inside the ROI - both arms - must count
  // exactly as ExerciseAnalyzer already does on its own (no
  // interference from the ROI guard when the arm never leaves).
  // ===============================================================
  for (const arm of ["left", "right"]) {
    r.start(`1. Normal full curl entirely inside ROI (${arm} arm)`);
    {
      const analyzer = new ExerciseAnalyzer(arm);
      driveFrame(analyzer, insideFrame(180), 0.0); // start extended
      driveFrame(analyzer, insideFrame(30), 0.5); // curl
      const res = driveFrame(analyzer, insideFrame(180), 2.0); // extend -> completes

      r.assert(res !== null, "in-ROI frames should reach the analyzer");
      r.assert(analyzer.reps === 1, `reps should be 1, got ${analyzer.reps}`);
      r.assert(analyzer.goodReps === 1, `goodReps should be 1, got ${analyzer.goodReps}`);
    }
  }

  // ===============================================================
  // 3/4/5. THE REGRESSION CASE: valid curl begins inside ROI, arm
  // leaves ROI mid-curl, several frames occur outside ROI, arm
  // returns to ROI - must NOT produce a false rep. Both arms.
  // ===============================================================
  for (const arm of ["left", "right"]) {
    r.start(`2. Leave ROI mid-curl and return - no false rep (${arm} arm)`);
    {
      const analyzer = new ExerciseAnalyzer(arm);

      driveFrame(analyzer, insideFrame(180), 0.0); // 1. curl begins inside ROI (resting)
      const midCurl = driveFrame(analyzer, insideFrame(30), 0.5); // still inside - now mid-curl
      r.assert(midCurl.stage === "up", 'stage should be "up" right before leaving the ROI');

      // 2/3. arm leaves the ROI, several frames occur outside it -
      // none of these may reach analyzer.update().
      driveFrame(analyzer, outsideFrame(), 0.6);
      driveFrame(analyzer, outsideFrame(), 0.7);
      driveFrame(analyzer, outsideFrame(), 0.8);

      r.assert(analyzer.stage === "down", 'leaving the ROI should discard the in-progress rep (stage back to "down")');
      r.assert(analyzer.reps === 0, "reps must still be 0 while outside the ROI");

      // 4/5. arm returns to the ROI, already extended - this must NOT
      // be interpreted as completing the rep that was in progress
      // before leaving.
      const backInRoi = driveFrame(analyzer, insideFrame(180), 1.0);

      r.assert(backInRoi !== null, "the returning frame should reach the analyzer");
      r.assert(
        analyzer.reps === 0,
        `returning to the ROI must not create a false rep, got reps=${analyzer.reps}`
      );
    }
  }

  // ===============================================================
  // A genuine rep performed entirely AFTER the leave/return sequence
  // must still count normally - the guard should not permanently
  // break tracking, only discard the interrupted one.
  // ===============================================================
  r.start("3. A real rep after returning to the ROI still counts");
  {
    const analyzer = new ExerciseAnalyzer("left");

    driveFrame(analyzer, insideFrame(180), 0.0);
    driveFrame(analyzer, insideFrame(30), 0.5);
    driveFrame(analyzer, outsideFrame(), 0.6);
    driveFrame(analyzer, outsideFrame(), 0.7);
    driveFrame(analyzer, insideFrame(180), 1.0); // back in ROI, no false rep (as above)

    r.assert(analyzer.reps === 0, "sanity: still 0 right after returning");

    driveFrame(analyzer, insideFrame(30), 1.5); // a fresh, real curl
    const completed = driveFrame(analyzer, insideFrame(180), 3.0); // and extension

    r.assert(completed.reps === 1, `a genuine subsequent rep should still count, got ${completed.reps}`);
  }

  // ===============================================================
  // Desktop-equivalent path unaffected: a session that never leaves
  // the ROI never calls resetInProgressRep() at all, so its results
  // are identical to driving ExerciseAnalyzer directly (no ROI
  // involved) - covered already by testExerciseAnalyzer.js's own
  // suite; this just confirms the guard is a no-op on that path too.
  // ===============================================================
  r.start("4. Multiple reps entirely inside ROI (no ROI interaction at all)");
  {
    const analyzer = new ExerciseAnalyzer("right");

    driveFrame(analyzer, insideFrame(180), 0.0);
    driveFrame(analyzer, insideFrame(30), 0.5);
    driveFrame(analyzer, insideFrame(180), 2.0);
    driveFrame(analyzer, insideFrame(30), 2.5);
    const res = driveFrame(analyzer, insideFrame(180), 4.5);

    r.assert(res.reps === 2, `reps should be 2, got ${res.reps}`);
  }

  return r.results;
}

/**
 * Logs a pass/fail report to the console and returns whether every
 * scenario passed. Same shape as the project's other test files.
 */
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
