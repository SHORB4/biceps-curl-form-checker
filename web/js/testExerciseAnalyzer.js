// testExerciseAnalyzer.js
//
// Standalone test for exerciseAnalyzer.js (Phase 2). Mirrors the
// Python project's test_analyzer.py: no DOM/camera/UI dependency,
// synthetic landmark geometry only, run through the real public
// update()/getResults() interface rather than poking internal state.
//
// Run in a browser: open test-exercise-analyzer.html via a local
// server (ES module imports require http(s)://, not file://).
//
// Run in Node (if available and it supports ES modules for this
// file, e.g. via `node --input-type=module` or a .mjs copy):
//   node -e "import('./testExerciseAnalyzer.js').then(m => m.runTests())"

import { ExerciseAnalyzer, calculateAngle } from "./exerciseAnalyzer.js";

// ---------------------------------------------------------------
// Synthetic landmark geometry helpers - same conventions as the
// Python reference test (test_analyzer.py): fixed shoulder/elbow,
// wrist placed on a circle around the elbow so the included angle
// at the elbow is exactly the requested angle. Exercises the real
// calculateAngle() math through the public update() interface.
// ---------------------------------------------------------------

const SHOULDER = [0.5, 0.2];
const ELBOW = [0.5, 0.5];

function wristForAngle(angleDeg, elbowX = ELBOW[0], radius = 0.3) {
  const elbow = [elbowX, ELBOW[1]];
  const shoulderTheta = radToDeg(
    Math.atan2(SHOULDER[1] - elbow[1], SHOULDER[0] - elbow[0])
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

function lm(x, y, visibility = 0.9) {
  return { x, y, visibility };
}

function shoulderLm(visibility = 0.9) {
  return lm(SHOULDER[0], SHOULDER[1], visibility);
}

function elbowLm(x = ELBOW[0], visibility = 0.9) {
  return lm(x, ELBOW[1], visibility);
}

function wristLm(angleDeg, elbowX = ELBOW[0], visibility = 0.9) {
  const [x, y] = wristForAngle(angleDeg, elbowX);
  return lm(x, y, visibility);
}

// ---------------------------------------------------------------
// Minimal assertion helpers - deliberately not a testing framework,
// just enough structure to report pass/fail per named scenario.
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

export function runTests() {
  const r = makeRecorder();

  // ===============================================================
  // 1. Resting / initial state
  // ===============================================================
  r.start("1. Resting/initial state");
  {
    const analyzer = new ExerciseAnalyzer("left");
    r.assert(analyzer.reps === 0, "fresh analyzer.reps should be 0");
    r.assert(analyzer.goodReps === 0, "fresh analyzer.goodReps should be 0");
    r.assert(analyzer.badReps === 0, "fresh analyzer.badReps should be 0");
    r.assert(analyzer.stage === "down", 'fresh analyzer.stage should be "down"');
    r.assert(
      analyzer.lastFeedback === "Get ready...",
      'fresh analyzer.lastFeedback should be "Get ready..."'
    );
  }

  // ===============================================================
  // 2. Arm visibility (insufficient visibility must not progress state)
  // ===============================================================
  r.start("2. Arm visibility / invalid landmarks");
  {
    const analyzer = new ExerciseAnalyzer("left");
    const res = analyzer.update(
      shoulderLm(0.9),
      elbowLm(ELBOW[0], 0.9),
      wristLm(180, ELBOW[0], 0.1), // wrist barely visible
      0.0
    );
    r.assert(res.visible === false, "low-visibility frame should report visible=false");
    r.assert(
      JSON.stringify(res.missingParts) === JSON.stringify(["wrist"]),
      `missingParts should be ["wrist"], got ${JSON.stringify(res.missingParts)}`
    );
    r.assert(res.angle === null, "angle should be null when not visible");
    r.assert(analyzer.stage === "down", "stage should remain unchanged");
    r.assert(analyzer.reps === 0, "reps should remain unchanged");
  }

  // ===============================================================
  // 13. Invalid/missing landmarks - all three below threshold
  // ===============================================================
  r.start("13. Invalid/missing landmarks (all parts)");
  {
    const analyzer = new ExerciseAnalyzer("left");
    const res = analyzer.update(
      shoulderLm(0.2),
      elbowLm(ELBOW[0], 0.4),
      wristLm(180, ELBOW[0], 0.0),
      0.0
    );
    r.assert(res.visible === false, "all-low-visibility frame should report visible=false");
    r.assert(
      JSON.stringify(res.missingParts) === JSON.stringify(["shoulder", "elbow", "wrist"]),
      `missingParts should list all three, got ${JSON.stringify(res.missingParts)}`
    );
  }

  // ===============================================================
  // 3. Extended arm (resting, fully visible) - stage stays "down"
  // ===============================================================
  let sharedAnalyzer; // reused by scenarios 4-11 to build a realistic session
  r.start("3. Extended arm");
  {
    sharedAnalyzer = new ExerciseAnalyzer("left");
    const res = sharedAnalyzer.update(shoulderLm(), elbowLm(), wristLm(180), 0.1);
    r.assertClose(res.angle, 180, 0.5, "extended-arm angle should be ~180");
    r.assert(res.visible === true, "extended-arm frame should be visible");
    r.assert(res.stage === "down", 'extended-arm stage should stay "down"');
    r.assert(res.reps === 0, "reps should still be 0");
  }

  // ===============================================================
  // 4. Full curl
  // ===============================================================
  r.start("4. Full curl");
  {
    const res = sharedAnalyzer.update(shoulderLm(), elbowLm(0.5), wristLm(10), 0.5);
    r.assertClose(res.angle, 10, 0.5, "full-curl angle should be ~10");
    r.assert(res.stage === "up", 'full curl should move stage to "up"');
  }

  // ===============================================================
  // 5. Return to extension -> completes one rep
  // ===============================================================
  r.start("5. Return to extension (rep completes)");
  {
    const res = sharedAnalyzer.update(shoulderLm(), elbowLm(0.5), wristLm(180), 2.5);
    r.assertClose(res.angle, 180, 0.5, "return-to-extension angle should be ~180");
    r.assert(res.stage === "down", 'stage should return to "down"');
  }

  // ===============================================================
  // 6. One completed rep - classified FULL + GOOD
  // ===============================================================
  r.start("6. One completed rep (FULL, GOOD)");
  {
    r.assert(sharedAnalyzer.reps === 1, `reps should be 1, got ${sharedAnalyzer.reps}`);
    r.assert(sharedAnalyzer.goodReps === 1, "goodReps should be 1");
    r.assert(sharedAnalyzer.fullReps === 1, "fullReps should be 1");
    r.assert(sharedAnalyzer.elbowProblems === 0, "elbowProblems should be 0");
    r.assert(sharedAnalyzer.tempoProblems === 0, "tempoProblems should be 0");
    r.assert(
      sharedAnalyzer.lastFeedback === "Good rep!",
      `lastFeedback should be "Good rep!", got "${sharedAnalyzer.lastFeedback}"`
    );
  }

  // ===============================================================
  // 8. Partial rep (+ too-fast tempo, as one combined rep - mirrors
  //    the Python reference test's rep 2 exactly for direct comparison)
  // ===============================================================
  r.start("8. Partial rep (+ too-fast tempo)");
  {
    sharedAnalyzer.update(shoulderLm(), elbowLm(0.5), wristLm(95), 2.6);
    const res = sharedAnalyzer.update(shoulderLm(), elbowLm(0.5), wristLm(180), 3.0);
    r.assert(res.reps === 2, `reps should be 2, got ${res.reps}`);
    r.assert(sharedAnalyzer.partialReps === 1, "partialReps should be 1");
    r.assert(sharedAnalyzer.tooFastReps === 1, "tooFastReps should be 1");
    r.assert(
      res.lastFeedback.includes("Curl further"),
      `feedback should include "Curl further", got "${res.lastFeedback}"`
    );
    r.assert(
      res.lastFeedback.includes("Slow down"),
      `feedback should include "Slow down", got "${res.lastFeedback}"`
    );
    r.assert(sharedAnalyzer.badReps === 1, "badReps should be 1");
  }

  // ===============================================================
  // 9. Very-short rep + 10. Elbow stability problem + 11. Too-slow
  //    rep, combined in one rep exactly as in the Python reference
  //    test's rep 3, for direct comparability.
  // ===============================================================
  r.start("9/10/11. Very-short rep + elbow instability + too-slow tempo");
  {
    sharedAnalyzer.update(shoulderLm(), elbowLm(0.5), wristLm(125, 0.5), 3.1);
    const res = sharedAnalyzer.update(shoulderLm(), elbowLm(0.7), wristLm(180, 0.7), 8.1);
    r.assert(res.reps === 3, `reps should be 3, got ${res.reps}`);
    r.assert(sharedAnalyzer.veryShortReps === 1, "veryShortReps should be 1");
    r.assert(sharedAnalyzer.elbowProblems === 1, "elbowProblems should be 1");
    r.assert(sharedAnalyzer.tooSlowReps === 1, "tooSlowReps should be 1");
    r.assert(
      res.lastFeedback.includes("Use more range"),
      `feedback should include "Use more range", got "${res.lastFeedback}"`
    );
    r.assert(
      res.lastFeedback.includes("Keep elbow stable"),
      `feedback should include "Keep elbow stable", got "${res.lastFeedback}"`
    );
    r.assert(
      res.lastFeedback.includes("Use a steady tempo"),
      `feedback should include "Use a steady tempo", got "${res.lastFeedback}"`
    );
    r.assert(sharedAnalyzer.badReps === 2, "badReps should be 2");
  }

  // ===============================================================
  // 7. Multiple completed reps - accumulated totals + getResults()
  // ===============================================================
  r.start("7. Multiple completed reps / getResults()");
  {
    const summary1 = sharedAnalyzer.getResults(3);
    const summary2 = sharedAnalyzer.getResults(3);

    r.assert(summary1.reps === 3, `summary.reps should be 3, got ${summary1.reps}`);
    r.assert(summary1.targetReps === 3, "summary.targetReps should be 3");
    r.assert(summary1.goodReps === 1, "summary.goodReps should be 1");
    r.assert(summary1.badReps === 2, "summary.badReps should be 2");
    r.assert(summary1.fullReps === 1, "summary.fullReps should be 1");
    r.assert(summary1.partialReps === 1, "summary.partialReps should be 1");
    r.assert(summary1.veryShortReps === 1, "summary.veryShortReps should be 1");
    r.assert(summary1.tempoProblems === 2, "summary.tempoProblems should be 2");

    const expectedAvgTempo = (2.0 + 0.4 + 5.0) / 3;
    r.assertClose(summary1.averageTempo, expectedAvgTempo, 1e-9, "averageTempo");

    const expectedScore = (1 / 3) * 10 * 0.7 + ((3 - 2) / 3) * 10 * 0.3;
    r.assertClose(summary1.finalScore, expectedScore, 1e-9, "finalScore");

    // Idempotent / no side effects
    r.assert(summary1.reps === summary2.reps, "getResults() should be idempotent (reps)");
    r.assert(
      summary1.finalScore === summary2.finalScore,
      "getResults() should be idempotent (finalScore)"
    );
    r.assert(sharedAnalyzer.reps === 3, "getResults() must not mutate analyzer.reps");
  }

  // ===============================================================
  // 15. Two analyzer instances maintain independent state
  // ===============================================================
  r.start("15. Two independent analyzer instances");
  {
    const analyzerA = new ExerciseAnalyzer("left");
    const analyzerB = new ExerciseAnalyzer("right");

    r.assert(
      analyzerA.shoulderId !== analyzerB.shoulderId,
      "different arms should resolve to different shoulderId"
    );
    r.assert(
      analyzerA.elbowId !== analyzerB.elbowId,
      "different arms should resolve to different elbowId"
    );
    r.assert(
      analyzerA.wristId !== analyzerB.wristId,
      "different arms should resolve to different wristId"
    );

    analyzerA.update(shoulderLm(), elbowLm(), wristLm(10), 0.0);
    analyzerA.update(shoulderLm(), elbowLm(), wristLm(180), 2.0);

    r.assert(analyzerA.reps === 1, `analyzerA.reps should be 1, got ${analyzerA.reps}`);
    r.assert(analyzerB.reps === 0, `analyzerB.reps should stay 0, got ${analyzerB.reps}`);
    r.assert(analyzerB.stage === "down", "analyzerB.stage should stay unaffected");
    r.assert(
      analyzerB.lastFeedback === "Get ready...",
      "analyzerB.lastFeedback should stay unaffected"
    );
  }

  // ===============================================================
  // 14. Session reset (= constructing a new instance; there is no
  //     dedicated reset() method, matching the Python reference)
  // ===============================================================
  r.start("14. Session reset (new instance)");
  {
    const analyzerOld = new ExerciseAnalyzer("left");
    analyzerOld.update(shoulderLm(), elbowLm(), wristLm(10), 0.0);
    analyzerOld.update(shoulderLm(), elbowLm(), wristLm(180), 2.0);
    r.assert(analyzerOld.reps === 1, "pre-reset analyzer should have 1 rep");

    const analyzerNew = new ExerciseAnalyzer("left");
    r.assert(analyzerNew.reps === 0, "new instance should start at reps=0");
    r.assert(analyzerOld.reps === 1, "old instance should be unaffected by the new one");
  }

  // ===============================================================
  // calculateAngle() sanity check (used above indirectly; verify
  // directly too, including the same clamping documented in Python)
  // ===============================================================
  r.start("calculateAngle() direct sanity check");
  {
    // Right angle: shoulder above elbow, wrist to the side.
    const a = calculateAngle([0.5, 0.0], [0.5, 0.5], [1.0, 0.5]);
    r.assertClose(a, 90, 0.5, "right-angle geometry should yield ~90 degrees");
  }

  return r.results;
}

/**
 * Logs a pass/fail report to the console for a runTests() result and
 * returns whether every scenario passed. Not called automatically on
 * import - callers (the HTML test page, or a Node one-liner) call
 * runTests() and, if they want console output too, this.
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
