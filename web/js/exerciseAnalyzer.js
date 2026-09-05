// exerciseAnalyzer.js
//
// JavaScript port of exercise_analyzer.py (the reference implementation).
// This is Phase 2: the algorithm only. It is NOT wired to camera.js,
// pose.js, roi.js, or the Workout UI yet - that happens in a later
// phase, once real pose landmarks exist to feed it.
//
// Port policy: every threshold, formula, state transition, and piece
// of feedback text below is copied from exercise_analyzer.py exactly
// as written, in the same operation order, including anything that
// might look odd in isolation (e.g. the "elif angle > EXTEND_ANGLE"
// gap - a frame reporting angle === EXTEND_ANGLE exactly falls
// through untouched in both languages, on purpose). Nothing here is
// simplified, "improved", or redesigned relative to the Python
// original. Only identifier naming was adapted from snake_case to
// camelCase for JS convention - that is a spelling change, not a
// behavioral one.
//
// No DOM, HTML, CSS, camera, MediaPipe, Canvas, browser UI, or
// OpenCV dependency. Pure per-instance state + math, safe to run in
// Node, a browser, or any other JS environment, and safe to
// instantiate multiple independent times.

// =========================================================
// SETTINGS (unchanged from exercise_analyzer.py)
// =========================================================

export const FULL_CURL_ANGLE = 80;
export const HALF_CURL_ANGLE = 110;
export const EXTEND_ANGLE = 140;

export const VISIBILITY_THRESHOLD = 0.5;

export const MIN_REP_TIME = 1.0;
export const MAX_REP_TIME = 4.0;

export const MAX_ELBOW_MOVEMENT = 0.12;

// MediaPipe Pose landmark indices. These are the numeric equivalent
// of exercise_analyzer.py's mp_pose.PoseLandmark.*_SHOULDER/ELBOW/
// WRIST members - the landmark schema is the same fixed 33-point
// model whether accessed through MediaPipe's Python Solutions API or
// its JS/WASM Tasks Vision API used in a later phase, so these
// indices are the correct equivalent, not a stand-in.
const POSE_LANDMARK = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16
};

// =========================================================
// ANGLE CALCULATION
// =========================================================

export function calculateAngle(a, b, c) {
  let angle = radToDeg(
    Math.atan2(c[1] - b[1], c[0] - b[0]) -
    Math.atan2(a[1] - b[1], a[0] - b[0])
  );

  angle = Math.abs(angle);

  if (angle > 180) {
    angle = 360 - angle;
  }

  return angle;
}

function radToDeg(radians) {
  return (radians * 180) / Math.PI;
}

// =========================================================
// LANDMARK COORDINATE CONVERSION
// =========================================================

export function landmarkToFullFrame(landmark, roi, frameW, frameH) {
  const [roiX1, roiY1, roiX2, roiY2] = roi;

  const roiW = roiX2 - roiX1;
  const roiH = roiY2 - roiY1;

  const fullX = (roiX1 + landmark.x * roiW) / frameW;
  const fullY = (roiY1 + landmark.y * roiH) / frameH;

  return {
    x: fullX,
    y: fullY,
    visibility: landmark.visibility
  };
}

// =========================================================
// SELECTED-ARM LANDMARK SELECTION
// =========================================================

export function landmarkIdsForArm(arm) {
  // Mirrors exercise_analyzer.py's landmark_ids_for_arm(): a mirrored
  // camera feed inverts MediaPipe's anatomical LEFT_*/RIGHT_*
  // landmarks relative to what's on screen, so the selected arm maps
  // to the OPPOSITE MediaPipe landmark set here, matching the
  // Python reference's mirror-corrected mapping exactly.
  if (arm === "left") {
    return [
      POSE_LANDMARK.RIGHT_SHOULDER,
      POSE_LANDMARK.RIGHT_ELBOW,
      POSE_LANDMARK.RIGHT_WRIST
    ];
  }

  return [
    POSE_LANDMARK.LEFT_SHOULDER,
    POSE_LANDMARK.LEFT_ELBOW,
    POSE_LANDMARK.LEFT_WRIST
  ];
}

// =========================================================
// EXERCISE ANALYZER
// =========================================================

/**
 * Holds the state and logic for one biceps-curl workout session.
 *
 * Consumes already-normalized shoulder/elbow/wrist landmarks (plain
 * objects with .x, .y, .visibility, in full-frame-normalized
 * coordinates) plus a timestamp, and returns per-frame analysis
 * results. Has no dependency on any camera, DOM, or GUI library.
 */
export class ExerciseAnalyzer {
  constructor(arm) {
    this.arm = arm;

    const [shoulderId, elbowId, wristId] = landmarkIdsForArm(arm);
    this.shoulderId = shoulderId;
    this.elbowId = elbowId;
    this.wristId = wristId;

    this.stage = "down";

    this.repStartTime = null;
    this.currentRepMinAngle = 180;
    this.currentRepElbowStartX = null;

    this.reps = 0;
    this.goodReps = 0;
    this.badReps = 0;

    this.fullReps = 0;
    this.partialReps = 0;
    this.veryShortReps = 0;

    this.elbowProblems = 0;
    this.tempoProblems = 0;

    this.tooFastReps = 0;
    this.tooSlowReps = 0;

    this.repTempos = [];

    this.lastFeedback = "Get ready...";
  }

  /**
   * shoulder/elbow/wrist: objects with .x, .y, .visibility in
   * full-frame-normalized coordinates.
   * timestamp: a number in seconds (e.g. performance.now() / 1000),
   * supplied by the caller.
   *
   * Returns a plain object describing this frame's result: visible,
   * missingParts, angle, stage, reps, goodReps, badReps, lastFeedback.
   */
  update(shoulder, elbow, wrist, timestamp) {
    const visible =
      shoulder.visibility > VISIBILITY_THRESHOLD &&
      elbow.visibility > VISIBILITY_THRESHOLD &&
      wrist.visibility > VISIBILITY_THRESHOLD;

    if (!visible) {
      const missingParts = [];

      if (shoulder.visibility <= VISIBILITY_THRESHOLD) missingParts.push("shoulder");
      if (elbow.visibility <= VISIBILITY_THRESHOLD) missingParts.push("elbow");
      if (wrist.visibility <= VISIBILITY_THRESHOLD) missingParts.push("wrist");

      return {
        visible: false,
        missingParts,
        angle: null,
        stage: this.stage,
        reps: this.reps,
        goodReps: this.goodReps,
        badReps: this.badReps,
        lastFeedback: this.lastFeedback
      };
    }

    const shoulderPoint = [shoulder.x, shoulder.y];
    const elbowPoint = [elbow.x, elbow.y];
    const wristPoint = [wrist.x, wrist.y];

    const angle = calculateAngle(shoulderPoint, elbowPoint, wristPoint);

    if (angle < EXTEND_ANGLE) {
      if (this.stage === "down") {
        this.stage = "up";
        this.repStartTime = timestamp;
        this.currentRepMinAngle = angle;
        this.currentRepElbowStartX = elbow.x;
      } else {
        if (angle < this.currentRepMinAngle) {
          this.currentRepMinAngle = angle;
        }
      }
    } else if (angle > EXTEND_ANGLE) {
      if (this.stage === "up") {
        this._completeRep(elbow, timestamp);
      }
    }

    return {
      visible: true,
      missingParts: [],
      angle,
      stage: this.stage,
      reps: this.reps,
      goodReps: this.goodReps,
      badReps: this.badReps,
      lastFeedback: this.lastFeedback
    };
  }

  _completeRep(elbow, timestamp) {
    this.reps += 1;
    this.stage = "down";

    let repTime;
    if (this.repStartTime !== null) {
      repTime = timestamp - this.repStartTime;
    } else {
      repTime = 0;
    }

    this.repTempos.push(repTime);

    let repType;
    if (this.currentRepMinAngle <= FULL_CURL_ANGLE) {
      repType = "FULL";
      this.fullReps += 1;
    } else if (this.currentRepMinAngle <= HALF_CURL_ANGLE) {
      repType = "PARTIAL";
      this.partialReps += 1;
    } else {
      repType = "VERY SHORT";
      this.veryShortReps += 1;
    }

    let elbowGood = true;
    if (this.currentRepElbowStartX !== null) {
      const elbowMovement = Math.abs(elbow.x - this.currentRepElbowStartX);
      if (elbowMovement > MAX_ELBOW_MOVEMENT) {
        elbowGood = false;
        this.elbowProblems += 1;
      }
    }

    let tempoGood = true;
    if (repTime < MIN_REP_TIME) {
      tempoGood = false;
      this.tempoProblems += 1;
      this.tooFastReps += 1;
    } else if (repTime > MAX_REP_TIME) {
      tempoGood = false;
      this.tempoProblems += 1;
      this.tooSlowReps += 1;
    }

    const repFeedback = [];

    if (repType === "PARTIAL") {
      repFeedback.push("Curl further");
    } else if (repType === "VERY SHORT") {
      repFeedback.push("Use more range");
    }

    if (!elbowGood) {
      repFeedback.push("Keep elbow stable");
    }

    if (!tempoGood) {
      if (repTime < MIN_REP_TIME) {
        repFeedback.push("Slow down");
      } else {
        repFeedback.push("Use a steady tempo");
      }
    }

    if (repType === "FULL" && elbowGood && tempoGood) {
      this.goodReps += 1;
    } else {
      this.badReps += 1;
    }

    if (repFeedback.length === 0) {
      repFeedback.push("Good rep!");
    }

    this.lastFeedback = repFeedback.join(" | ");

    this.repStartTime = null;
    this.currentRepMinAngle = 180;
    this.currentRepElbowStartX = null;
  }

  /**
   * Computes the final score and qualitative feedback lines,
   * mirroring the original RESULTS-screen computation exactly.
   */
  getResults(targetReps) {
    const reps = this.reps;

    let averageTempo;
    let finalScore;

    if (reps > 0) {
      const goodRepRatio = this.goodReps / reps;
      const baseScore = goodRepRatio * 10;

      if (this.repTempos.length > 0) {
        let sum = 0;
        for (const t of this.repTempos) sum += t;
        averageTempo = sum / this.repTempos.length;
      } else {
        averageTempo = 0;
      }

      const goodTempoReps = reps - this.tempoProblems;
      const tempoRatio = goodTempoReps / reps;

      finalScore = baseScore * 0.7 + tempoRatio * 10 * 0.3;
    } else {
      averageTempo = 0;
      finalScore = 0;
    }

    const feedbackLines = [];

    // Range
    if (this.partialReps > 0) {
      feedbackLines.push(`- ${this.partialReps} partial rep(s): curl further`);
    }

    if (this.veryShortReps > 0) {
      feedbackLines.push(`- ${this.veryShortReps} very short rep(s): use more range`);
    }

    if (this.partialReps === 0 && this.veryShortReps === 0) {
      feedbackLines.push("- Excellent range of motion");
    }

    // Elbow
    if (this.elbowProblems > 0) {
      feedbackLines.push(`- Elbow moved too much in ${this.elbowProblems} rep(s)`);
    } else {
      feedbackLines.push("- Good elbow stability");
    }

    // Tempo
    if (this.tooFastReps > 0) {
      feedbackLines.push(`- ${this.tooFastReps} rep(s) were too fast: slow down`);
    }

    if (this.tooSlowReps > 0) {
      feedbackLines.push(`- ${this.tooSlowReps} rep(s) were too slow`);
    }

    if (this.tempoProblems === 0) {
      feedbackLines.push("- Tempo was consistent");
    }

    // Overall
    if (finalScore >= 9) {
      feedbackLines.push("- Excellent overall technique!");
    } else if (finalScore >= 8) {
      feedbackLines.push("- Great job! Your technique was mostly consistent.");
    } else if (finalScore >= 7) {
      feedbackLines.push("- Good workout. A few areas can be improved.");
    } else if (finalScore >= 5) {
      feedbackLines.push("- Decent effort. Focus on the feedback above.");
    } else {
      feedbackLines.push("- Focus on controlled movement and full range.");
    }

    return {
      reps,
      targetReps,
      goodReps: this.goodReps,
      badReps: this.badReps,
      fullReps: this.fullReps,
      partialReps: this.partialReps,
      veryShortReps: this.veryShortReps,
      tempoProblems: this.tempoProblems,
      averageTempo,
      finalScore,
      feedbackLines
    };
  }
}
