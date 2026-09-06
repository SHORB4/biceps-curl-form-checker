// testRoiRepGuard.js
//
// Regression test for the mobile ROI glitch and its ROI-exit debounce
// fix. Exercises the REAL ExerciseAnalyzer and the REAL
// isArmWithinRoi() (both imported, unmodified) through a small
// driveFrame() factory that mirrors app.js's onResult decision,
// including its elapsed-time debounce before resetInProgressRep().
//
// IMPORTANT: makeFrameDriver() below must be kept in sync with the
// actual onResult handler in app.js - it is a deliberate, disclosed
// copy for testability, not a reimplementation of separate logic.
// app.js's own resetInProgressRep() field list and ROI_EXIT_DEBOUNCE_MS
// value must match the ones here if they ever change.
//
// HISTORY (why this file keeps changing - each revision closed a gap
// the previous one's passing suite didn't cover):
//   1. First version only modeled "person detected, but outside the
//      ROI" - missed the "MediaPipe detects nobody at all" case, which
//      the deployed code (at the time) didn't reset on. Real bug,
//      passing test.
//   2. Second version added the "nobody detected" case and made
//      driveFrame() reset immediately on ANY non-qualifying frame -
//      matching the code at the time. That code turned out to be too
//      AGGRESSIVE: a single transient MediaPipe dropout (common on
//      mobile) could discard a genuine, entirely-in-ROI rep. Passing
//      test, but a different real bug (missed reps, not false reps).
//   3. This version models the ROI_EXIT_DEBOUNCE_MS elapsed-time
//      debounce that fixes (2): resetInProgressRep() now only fires
//      after ROI_EXIT_DEBOUNCE_MS of CONTINUOUS non-qualifying frames,
//      not on the first one. driveFrame() takes an explicit `nowMs`
//      (mirroring performance.now() in app.js) so debounce timing is
//      deterministic and fast to test, rather than relying on real
//      sleeps.
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

/**
 * "MediaPipe detected nobody this frame" - result.landmarks.length
 * === 0 in the real app. Passed to driveFrame() as `null`, distinct
 * from outsideFrame() (a person WAS detected, just outside the ROI).
 * Both must be rejected identically as far as analyzer.update() goes,
 * and both count toward the SAME debounce timer - see the header.
 */
function noDetectionFrame() {
  return null;
}

// ---------------------------------------------------------------
// Debounce constant - MUST match app.js's ROI_EXIT_DEBOUNCE_MS.
// ---------------------------------------------------------------
const ROI_EXIT_DEBOUNCE_MS = 120;

function resetInProgressRep(analyzer) {
  analyzer.stage = "down";
  analyzer.repStartTime = null;
  analyzer.currentRepMinAngle = 180;
  analyzer.currentRepElbowStartX = null;
}

/**
 * Creates a fresh driveFrame(analyzer, frame, timestamp, nowMs)
 * function with its own private roiExitStartedAt debounce state -
 * mirrors ONE Workout visit's worth of onResult closure state in
 * app.js (see enterPoseDetection()'s `let roiExitStartedAt = null;`).
 * Create one new driver per test scenario - never share one across
 * scenarios, or one scenario's exit timing would leak into the next.
 *
 * `frame` is either the return value of insideFrame()/outsideFrame()
 * (a {shoulder, elbow, wrist} triple - a person was detected) or
 * noDetectionFrame() (null - nobody was detected at all).
 * `timestamp` (seconds) is the analyzer's own clock, used for
 * MIN_REP_TIME/MAX_REP_TIME classification - unrelated to the
 * debounce.
 * `nowMs` (milliseconds) mirrors performance.now() in app.js - the
 * wall-clock time this frame arrived, used ONLY for the debounce's
 * elapsed-time measurement.
 *
 * Keep this in sync with app.js's actual onResult handler.
 */
function makeFrameDriver() {
  let roiExitStartedAt = null; // null = currently qualifying (or not yet started)

  return function driveFrame(analyzer, frame, timestamp, nowMs) {
    let armInsideRoi = false;
    let shoulder = null;
    let elbow = null;
    let wrist = null;

    if (frame) {
      ({ shoulder, elbow, wrist } = frame);
      armInsideRoi = isArmWithinRoi(shoulder, elbow, wrist, ROI, VIDEO_W, VIDEO_H);
    }

    if (armInsideRoi) {
      roiExitStartedAt = null; // clear the debounce - back to a qualifying frame
      return analyzer.update(shoulder, elbow, wrist, timestamp);
    }

    // Invalid/outside frame: NEVER reaches analyzer.update() - the
    // debounce below only decides WHEN the in-progress state gets
    // invalidated, never whether THIS frame can advance it.
    if (roiExitStartedAt === null) {
      roiExitStartedAt = nowMs;
    }
    const exitDurationMs = nowMs - roiExitStartedAt;
    if (exitDurationMs >= ROI_EXIT_DEBOUNCE_MS) {
      resetInProgressRep(analyzer);
    }
    return null;
  };
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
  // TEST 1: One invalid frame during an otherwise valid rep - no
  // reset, no false rep. The debounce's core purpose: a single
  // transient dropout must not disturb an in-progress rep at all.
  // Both arms.
  // ===============================================================
  for (const arm of ["left", "right"]) {
    r.start(`1. One transient invalid frame mid-curl - no reset, real rep still completes (${arm} arm)`);
    {
      const drive = makeFrameDriver();
      const analyzer = new ExerciseAnalyzer(arm);

      drive(analyzer, insideFrame(180), 0.0, 0); // resting, inside
      const midCurl = drive(analyzer, insideFrame(30), 0.5, 500); // mid-curl, inside
      r.assert(midCurl.stage === "up", 'stage should be "up" mid-curl');

      // ONE transient blip - e.g. a single dropped detection - then
      // immediately back inside on the very next frame.
      drive(analyzer, noDetectionFrame(), 0.55, 510);

      r.assert(analyzer.stage === "up", "a single transient blip must NOT reset the in-progress rep");
      r.assert(analyzer.reps === 0, "no false rep from the blip itself");

      // The SAME rep completes normally afterward - proving the blip
      // didn't corrupt anything.
      const completed = drive(analyzer, insideFrame(180), 1.0, 1000);
      r.assert(completed.reps === 1, `the real rep should still complete despite the earlier blip, got ${completed.reps}`);
    }
  }

  // ===============================================================
  // TEST 2: Two consecutive invalid frames, still well within the
  // debounce window - no reset. (Requested as "3-frame approach"
  // language; this project uses the elapsed-time approach instead per
  // the fix spec, so this models the time-based equivalent: a short
  // burst that totals far less than ROI_EXIT_DEBOUNCE_MS.)
  // ===============================================================
  for (const arm of ["left", "right"]) {
    r.start(`2. Two consecutive invalid frames within the debounce window - no reset (${arm} arm)`);
    {
      const drive = makeFrameDriver();
      const analyzer = new ExerciseAnalyzer(arm);

      drive(analyzer, insideFrame(180), 0.0, 0);
      const midCurl = drive(analyzer, insideFrame(30), 0.5, 500);
      r.assert(midCurl.stage === "up", 'stage should be "up" mid-curl');

      // Two invalid frames 30ms apart, well under ROI_EXIT_DEBOUNCE_MS.
      drive(analyzer, outsideFrame(), 0.51, 510);
      drive(analyzer, noDetectionFrame(), 0.54, 540);

      r.assert(analyzer.stage === "up", "two invalid frames within the debounce window must not reset yet");
      r.assert(analyzer.reps === 0, "reps must still be 0");

      // Returning inside completes the ORIGINAL rep (state was never
      // reset), not a false one.
      const completed = drive(analyzer, insideFrame(180), 1.0, 1000);
      r.assert(completed.reps === 1, `original rep should complete normally, got ${completed.reps}`);
    }
  }

  // ===============================================================
  // TEST 3: Sustained ROI exit - reset DOES occur once the debounce
  // threshold is exceeded, and the previously COMPLETED rep count is
  // preserved (only the in-progress second rep is discarded).
  // ===============================================================
  for (const arm of ["left", "right"]) {
    r.start(`3. Sustained ROI exit resets in-progress rep but preserves completed reps (${arm} arm)`);
    {
      const drive = makeFrameDriver();
      const analyzer = new ExerciseAnalyzer(arm);

      // First, a genuine completed rep.
      drive(analyzer, insideFrame(180), 0.0, 0);
      drive(analyzer, insideFrame(30), 0.5, 500);
      const firstRep = drive(analyzer, insideFrame(180), 2.0, 2000);
      r.assert(firstRep.reps === 1, "sanity: first rep should complete");

      // Start a second rep, then leave the ROI for LONGER than the
      // debounce threshold (frames spaced well beyond
      // ROI_EXIT_DEBOUNCE_MS apart in total elapsed time).
      const secondMidCurl = drive(analyzer, insideFrame(30), 2.5, 2500);
      r.assert(secondMidCurl.stage === "up", 'stage should be "up" for the second rep');

      drive(analyzer, outsideFrame(), 2.6, 2600); // exit starts (elapsed 0ms)
      drive(analyzer, outsideFrame(), 2.65, 2650); // elapsed 50ms - still under threshold
      const afterSustained = drive(analyzer, outsideFrame(), 2.75, 2750); // elapsed 150ms - crosses 120ms

      r.assert(afterSustained === null, "an outside frame should never reach the analyzer");
      r.assert(analyzer.stage === "down", "sustained exit (>120ms) should reset the in-progress second rep");
      r.assert(
        analyzer.reps === 1,
        `the completed FIRST rep must be preserved through the reset, got reps=${analyzer.reps}`
      );
    }
  }

  // ===============================================================
  // TEST 4: Leave ROI -> several invalid frames -> return: zero false
  // reps (the original reported bug, re-verified through the debounce
  // path). Both arms.
  // ===============================================================
  for (const arm of ["left", "right"]) {
    r.start(`4. Leave ROI, several invalid frames, return - zero false reps (${arm} arm)`);
    {
      const drive = makeFrameDriver();
      const analyzer = new ExerciseAnalyzer(arm);

      drive(analyzer, insideFrame(180), 0.0, 0);
      const midCurl = drive(analyzer, insideFrame(30), 0.5, 500);
      r.assert(midCurl.stage === "up", 'stage should be "up" right before leaving the ROI');

      // Several invalid frames, comfortably spanning past the
      // debounce threshold (mix of detected-but-outside and
      // undetected, as a real phone would actually produce).
      drive(analyzer, outsideFrame(), 0.6, 600);
      drive(analyzer, noDetectionFrame(), 0.7, 700);
      drive(analyzer, noDetectionFrame(), 0.8, 800);
      drive(analyzer, outsideFrame(), 0.9, 900);

      r.assert(analyzer.stage === "down", "sustained multi-frame exit should have reset by now");
      r.assert(analyzer.reps === 0, "reps must still be 0 while outside the ROI");

      // Arm returns to the ROI, already extended - must NOT be
      // interpreted as completing the rep that was in progress before
      // leaving.
      const backInRoi = drive(analyzer, insideFrame(180), 1.0, 1000);

      r.assert(backInRoi !== null, "the returning frame should reach the analyzer");
      r.assert(
        analyzer.reps === 0,
        `returning to the ROI must not create a false rep, got reps=${analyzer.reps}`
      );
    }
  }

  // ===============================================================
  // TEST 5: Genuine rep entirely inside the ROI still counts exactly
  // once - the debounce must never interfere when the arm never
  // leaves. Both arms.
  // ===============================================================
  for (const arm of ["left", "right"]) {
    r.start(`5. Genuine rep entirely inside ROI counts exactly once (${arm} arm)`);
    {
      const drive = makeFrameDriver();
      const analyzer = new ExerciseAnalyzer(arm);

      drive(analyzer, insideFrame(180), 0.0, 0);
      drive(analyzer, insideFrame(30), 0.5, 500);
      const res = drive(analyzer, insideFrame(180), 2.0, 2000);

      r.assert(res !== null, "in-ROI frames should reach the analyzer");
      r.assert(analyzer.reps === 1, `reps should be 1, got ${analyzer.reps}`);
      r.assert(analyzer.goodReps === 1, `goodReps should be 1, got ${analyzer.goodReps}`);
    }
  }

  // ===============================================================
  // A real rep performed AFTER a sustained leave/return sequence must
  // still count normally - the debounce/reset should not permanently
  // break tracking, only discard the interrupted rep.
  // ===============================================================
  r.start("5b. A real rep after a sustained leave/return still counts");
  {
    const drive = makeFrameDriver();
    const analyzer = new ExerciseAnalyzer("left");

    drive(analyzer, insideFrame(180), 0.0, 0);
    drive(analyzer, insideFrame(30), 0.5, 500);
    drive(analyzer, outsideFrame(), 0.6, 600);
    drive(analyzer, outsideFrame(), 0.7, 750); // elapsed 150ms - crosses threshold, resets
    drive(analyzer, insideFrame(180), 1.0, 1000); // back in ROI, no false rep (as TEST 4)

    r.assert(analyzer.reps === 0, "sanity: still 0 right after returning");

    const midCurl2 = drive(analyzer, insideFrame(30), 1.5, 1500); // a fresh, real curl
    r.assert(midCurl2.stage === "up", 'stage should be "up" for the fresh curl');
    const completed = drive(analyzer, insideFrame(180), 3.0, 3000); // and extension

    r.assert(completed.reps === 1, `a genuine subsequent rep should still count, got ${completed.reps}`);
  }

  // ===============================================================
  // TEST 6 is covered by the (arm) loops above (TESTS 1-5 each run
  // for both "left" and "right").
  // ===============================================================

  // ===============================================================
  // TEST 7: Normal multi-rep session with NO ROI interaction at all -
  // the debounce path is never even touched (roiExitStartedAt stays
  // null throughout), so results are identical to driving
  // ExerciseAnalyzer directly - covered already by
  // testExerciseAnalyzer.js's own suite; this just confirms the guard
  // is a total no-op on that path too.
  // ===============================================================
  r.start("7. Multiple reps entirely inside ROI (no ROI interaction at all)");
  {
    const drive = makeFrameDriver();
    const analyzer = new ExerciseAnalyzer("right");

    drive(analyzer, insideFrame(180), 0.0, 0);
    drive(analyzer, insideFrame(30), 0.5, 500);
    drive(analyzer, insideFrame(180), 2.0, 2000);
    drive(analyzer, insideFrame(30), 2.5, 2500);
    const res = drive(analyzer, insideFrame(180), 4.5, 4500);

    r.assert(res.reps === 2, `reps should be 2, got ${res.reps}`);
  }

  // ===============================================================
  // Bonus: a realistic mixed sequence (detected-but-outside AND
  // no-detection frames interleaved while leaving) must still produce
  // zero false reps once the sustained exit crosses the debounce.
  // ===============================================================
  r.start("Bonus. Mixed detected-outside / no-detection frames while leaving - no false rep");
  {
    const drive = makeFrameDriver();
    const analyzer = new ExerciseAnalyzer("left");

    drive(analyzer, insideFrame(180), 0.0, 0);
    drive(analyzer, insideFrame(30), 0.5, 500);
    drive(analyzer, outsideFrame(), 0.6, 600); // detected, just outside (elapsed 0ms)
    drive(analyzer, noDetectionFrame(), 0.7, 700); // lost entirely (elapsed 100ms)
    drive(analyzer, noDetectionFrame(), 0.8, 800); // still lost (elapsed 200ms - crosses threshold)
    drive(analyzer, outsideFrame(), 0.9, 900); // detected again, still outside
    const backInRoi = drive(analyzer, insideFrame(180), 1.0, 1000);

    r.assert(backInRoi !== null, "the returning frame should reach the analyzer");
    r.assert(analyzer.reps === 0, `mixed leave sequence must not create a false rep, got reps=${analyzer.reps}`);
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
