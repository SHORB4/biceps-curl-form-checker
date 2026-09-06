// app.js
//
// Screen orchestration, plus browser camera access (Phase 1), ROI
// selection (Phase 1), pose detection (Phase 3), and exercise
// analysis (Phase 4).
//
// This is the browser equivalent of form.py's `screen` state machine:
// it owns which of the six screens is visible and wires up navigation
// between them. It also plays form.py's role of gluing everything
// together each frame: extract this arm's landmarks (armMapping.js)
// -> feed them to the analyzer (exerciseAnalyzer.js, untouched) ->
// reflect the result in the HUD/Results screen. No exercise-analysis
// math lives in this file - only data plumbing.

import { INSTRUCTIONS_SECTIONS } from "./instructionsContent.js";
import { startCamera, stopCamera, isCameraSupported } from "./camera.js";
import { createRoiController } from "./roi.js";
import { PoseDetector, isPoseSupported, DEBUG_TIMING, isArmWithinRoi, roiToRawCropRect } from "./pose.js";
import { drawPoseOverlay, clearPoseOverlay } from "./renderer.js";
import { ExerciseAnalyzer } from "./exerciseAnalyzer.js";
import { selectArmLandmarks, armLandmarkIndices } from "./armMapping.js";

// Same six screens, same order, as the desktop app's flow:
// INSTRUCTIONS -> ARM_SELECT -> REP_INPUT -> ROI_SELECT -> WORKOUT -> RESULTS
const SCREEN_ORDER = [
  { id: "instructions", label: "Instructions" },
  { id: "arm-select", label: "Arm Selection" },
  { id: "rep-count", label: "Rep Count" },
  { id: "roi-select", label: "ROI Selection" },
  { id: "workout", label: "Workout" },
  { id: "results", label: "Results" }
];

// Minimal placeholder state - just enough to demonstrate navigation
// and basic UI feedback. No workout/analysis logic lives here.
const state = {
  selectedArm: null,
  targetReps: 10
};

const stepIndicatorEl = document.getElementById("step-indicator");
const progressFillEl = document.getElementById("progress-fill");

let currentScreenId = null;

function getScreenSection(screenId) {
  return document.querySelector(`.screen[data-screen="${screenId}"]`);
}

function showScreen(screenId) {
  const index = SCREEN_ORDER.findIndex((s) => s.id === screenId);

  if (index === -1) {
    console.error(`Unknown screen id: "${screenId}"`);
    return;
  }

  for (const screen of SCREEN_ORDER) {
    const section = getScreenSection(screen.id);
    if (!section) continue;
    section.hidden = screen.id !== screenId;
  }

  const { label } = SCREEN_ORDER[index];
  stepIndicatorEl.textContent = `Step ${index + 1} of ${SCREEN_ORDER.length} — ${label}`;
  progressFillEl.style.width = `${((index + 1) / SCREEN_ORDER.length) * 100}%`;

  window.scrollTo(0, 0);

  // Camera lifecycle: the ROI Selection and Workout screens share one
  // live stream. Entering either of them from outside this pair starts
  // the camera; leaving both stops it. Moving between the two (in
  // either direction) just relocates the existing <video>/state UI —
  // no new getUserMedia() call, no second stream.
  const wasCameraScreen = CAMERA_SCREENS.has(currentScreenId);
  const willBeCameraScreen = CAMERA_SCREENS.has(screenId);

  if (willBeCameraScreen) {
    moveCameraUnitInto(screenId === "roi-select" ? roiCameraStageEl : workoutCameraStageEl);
  }

  if (!wasCameraScreen && willBeCameraScreen) {
    enterCameraFlow();
  } else if (wasCameraScreen && !willBeCameraScreen) {
    leaveCameraFlow();
  } else if (willBeCameraScreen) {
    // Moving between ROI Selection and Workout with the stream already
    // running: the video just moved into a differently-sized container,
    // so whichever canvas is now visible needs to re-measure and redraw.
    if (screenId === "roi-select") {
      roiController.resizeCanvas();
    } else {
      roiController.refreshOverlay(workoutRoiCanvasEl);
    }
  }

  // Pose detection (Phase 3) is scoped to the Workout screen only -
  // not ROI Selection, even though the camera is active on both.
  if (screenId === "workout" && currentScreenId !== "workout") {
    enterPoseDetection();
  } else if (currentScreenId === "workout" && screenId !== "workout") {
    leavePoseDetection();
  }

  // Prefetch (Phase 3 optimization): start loading MediaPipe in the
  // background as soon as ROI Selection is reached, well before
  // Workout needs it. init() is idempotent - _ensureLandmarker()
  // only ever starts one load per PoseDetector instance, so calling
  // this on every ROI Selection visit is safe and does not trigger a
  // second download/initialization. This never starts detectForVideo()
  // or the requestAnimationFrame loop - only start() does that.
  if (screenId === "roi-select") {
    if (DEBUG_TIMING && poseDetector.getState() === "idle") {
      console.log("[pose timing] Prefetch triggered from ROI Selection entry.");
    }
    poseDetector.init().catch(() => {
      // Already recorded via the state/onStateChange mechanism below
      // (state -> "error", getError() set) - nothing further to do
      // here, just avoid an unhandled-rejection console warning.
    });

    // Phase 5 readability: name the selected arm and remind the user
    // what needs to stay inside the box - display text only, no ROI
    // logic touched.
    const armLabel = state.selectedArm === "left" ? "left" : "right";
    roiSubtitleEl.textContent =
      `Drag a box around your ${armLabel} arm — make sure your shoulder, elbow, and wrist are all inside it.`;
  }

  currentScreenId = screenId;
}

// -----------------------------------------------------------
// Shared camera stream (ROI Selection + Workout screens)
// -----------------------------------------------------------

const CAMERA_SCREENS = new Set(["roi-select", "workout"]);

const cameraVideoEl = document.getElementById("camera-video");
const cameraStateEl = document.getElementById("camera-state");
const cameraStateIconEl = document.getElementById("camera-state-icon");
const cameraStateTextEl = document.getElementById("camera-state-text");
const cameraRetryBtn = document.getElementById("camera-retry-btn");

const roiSubtitleEl = document.getElementById("roi-subtitle");
const roiCameraStageEl = document.getElementById("roi-camera-stage");
const workoutCameraStageEl = document.getElementById("camera-stage");
const roiCanvasEl = document.getElementById("roi-canvas");
const roiStatusEl = document.getElementById("roi-status");
const roiResetBtn = document.getElementById("roi-reset-btn");
const roiConfirmBtn = document.getElementById("roi-confirm-btn");
const workoutRoiCanvasEl = document.getElementById("workout-roi-canvas");

const roiController = createRoiController({
  videoEl: cameraVideoEl,
  canvasEl: roiCanvasEl,
  statusEl: roiStatusEl
});

// Read-only mirror of the confirmed ROI on the Workout screen — same
// controller, same data, same coordinate mapping, no second system.
roiController.attachOverlay(workoutRoiCanvasEl);

function moveCameraUnitInto(containerEl) {
  if (cameraVideoEl.parentElement !== containerEl) {
    containerEl.appendChild(cameraVideoEl);
  }
  if (cameraStateEl.parentElement !== containerEl) {
    containerEl.appendChild(cameraStateEl);
  }
}

function setCameraState(mode, message) {
  const isLive = mode === "live";
  cameraVideoEl.hidden = !isLive;
  cameraStateEl.hidden = isLive;

  if (isLive) {
    roiController.resizeCanvas();
    roiController.refreshOverlay(workoutRoiCanvasEl);
    return;
  }

  cameraStateEl.classList.toggle("camera-state-error", mode === "error");
  cameraStateIconEl.textContent = mode === "error" ? "!" : "◻";
  cameraStateTextEl.textContent = message || "";
  cameraRetryBtn.hidden = mode !== "error";
}

async function enterCameraFlow() {
  if (!isCameraSupported()) {
    setCameraState(
      "error",
      "This browser does not support camera access. Try the latest Chrome, Edge, Firefox, or Safari."
    );
    return;
  }

  setCameraState("starting", "Starting camera…");

  try {
    await startCamera(cameraVideoEl);
    setCameraState("live");
  } catch (err) {
    console.error("Camera start failed:", err);
    setCameraState("error", err.message);
  }
}

function leaveCameraFlow() {
  stopCamera(cameraVideoEl);
  setCameraState("idle", "Camera preview will appear here");
}

cameraRetryBtn.addEventListener("click", () => {
  enterCameraFlow();
});

roiResetBtn.addEventListener("click", () => {
  roiController.reset();
});

roiConfirmBtn.addEventListener("click", () => {
  if (roiController.confirm()) {
    showScreen("workout");
  }
});

// Defensive cleanup: also release the camera if the tab is closed or
// navigated away from entirely while ROI Selection or Workout is active.
window.addEventListener("pagehide", () => {
  if (CAMERA_SCREENS.has(currentScreenId)) {
    stopCamera(cameraVideoEl);
  }
  poseDetector.stop();
});

// -----------------------------------------------------------
// Pose detection (Phase 3) + exercise analysis (Phase 4).
//
// One PoseDetector instance for the whole page session (see pose.js -
// constructing a new one per Workout visit would reload the MediaPipe
// model every time). start()/stop() just toggle its detection loop;
// the underlying model, once loaded, stays loaded across visits.
//
// One ExerciseAnalyzer instance per Workout VISIT (not per page
// session, unlike PoseDetector) - created fresh in
// enterPoseDetection(), discarded in leavePoseDetection(), so a new
// workout never inherits a previous one's rep count. See the Phase 4
// report for why every fresh Workout entry (not just Restart) resets
// it, and how that was tested.
// -----------------------------------------------------------

const workoutSkeletonCanvasEl = document.getElementById("workout-skeleton-canvas");
const poseStatusEl = document.getElementById("pose-status");
const poseStatusTextEl = document.getElementById("pose-status-text");

const hudArmEl = document.getElementById("hud-arm");
const hudRepsEl = document.getElementById("hud-reps");
const hudAngleEl = document.getElementById("hud-angle");
const hudStageEl = document.getElementById("hud-stage");
const hudGoodEl = document.getElementById("hud-good");
const hudBadEl = document.getElementById("hud-bad");
const hudFeedbackEl = document.getElementById("hud-feedback");
const hudFeedbackTextEl = document.getElementById("hud-feedback-text");

const poseDetector = new PoseDetector();

// The single ExerciseAnalyzer for the CURRENT Workout visit only -
// null whenever Workout is not active. Never instantiated more than
// once per visit, never updated from anywhere but the onResult
// callback below (one frame -> at most one analyzer.update() call).
let analyzer = null;

// Guards automatic completion (analyzer's confirmed reps >= target)
// so it can only ever fire once per Workout visit - see
// finishWorkout() below. Reset alongside `analyzer` on each fresh
// Workout entry.
let workoutFinished = false;

let hasLoggedPoseReady = false;
let hasLoggedFirstDetection = false;
let lastLoggedPoseErrorCode = null;

// Reflects PoseDetector's state (the single source of truth - see
// pose.js) onto the Workout screen's loading/error banner. Subscribed
// once for the whole page session, so it stays in sync even when the
// state changes while the user isn't looking at Workout yet (e.g. the
// ROI Selection prefetch finishing, or failing, before Workout is
// ever entered).
function updatePoseStatusUI(state, error) {
  if (state === "loading") {
    poseStatusEl.hidden = false;
    poseStatusEl.classList.remove("pose-status-error");
    poseStatusTextEl.textContent = "Preparing pose detection…";
  } else if (state === "error") {
    poseStatusEl.hidden = false;
    poseStatusEl.classList.add("pose-status-error");
    poseStatusTextEl.textContent =
      "Pose detection could not be loaded. Please refresh and try again.";
    if (error) {
      console.error("pose.js initialization error:", error.code, error.message, error.cause || "");
    }
  } else {
    // "idle" or "ready" - nothing to show over the camera.
    poseStatusEl.hidden = true;
  }
}

poseDetector.onStateChange(updatePoseStatusUI);

// TEMPORARY (loading-performance audit): marks when the Workout
// screen was entered, so the console can report a real end-to-end
// "screen entry -> first detected pose" time - the closest proxy to
// what a user actually experiences as "MediaPipe load time" (it also
// includes camera/video readiness and however long the person takes
// to step into frame, which is disclosed in the audit report rather
// than hidden). Still not removed - untouched this phase; still
// gated behind pose.js's DEBUG_TIMING flag, still safe to delete
// later whenever it's no longer needed.
let workoutEnteredAt = null;

/**
 * Discards whatever rep the analyzer currently has in progress,
 * without touching its rep count/history - the same fields
 * ExerciseAnalyzer._completeRep() itself resets to after a rep
 * finishes, set directly since exerciseAnalyzer.js is not to be
 * modified for this fix (see the mobile ROI glitch reports this
 * shipped with).
 *
 * Root cause this fixes: when the selected arm's landmarks are
 * outside the confirmed ROI, this file already skips calling
 * analyzer.update() for that frame (see isArmWithinRoi() below) -
 * but skipping update() alone leaves the analyzer's in-progress rep
 * state (stage/repStartTime/currentRepMinAngle/currentRepElbowStartX)
 * completely frozen while the arm is outside the ROI. On a handheld
 * phone, it's common for the arm to drift out of a fixed ROI box
 * mid-curl and come back; without this reset, the FIRST frame back
 * inside the ROI resumes from that stale frozen state - if the arm
 * happened to already be extended again by the time it returns, that
 * single frame alone completes a "rep" that was never actually
 * tracked continuously through the ROI.
 *
 * Called only after ROI_EXIT_DEBOUNCE_MS of CONTINUOUS non-qualifying
 * frames (see the debounce logic in onResult below) - not on the
 * first such frame - so a single transient MediaPipe dropout (common
 * on mobile: weaker/variable delegate performance, motion blur, lower
 * frame rate) can't wipe out a genuine, entirely-in-ROI rep just
 * because one frame happened to miss detection.
 */
function resetInProgressRep() {
  if (!analyzer) return;
  analyzer.stage = "down";
  analyzer.repStartTime = null;
  analyzer.currentRepMinAngle = 180;
  analyzer.currentRepElbowStartX = null;
}

// How long the selected arm must be CONTINUOUSLY outside the ROI (or
// undetected) before resetInProgressRep() fires - not on the very
// first non-qualifying frame. Elapsed-time based (not a frame count)
// so it behaves consistently regardless of the device's actual detect
// rate: a single transient dropout is always tolerated, while a real,
// sustained exit is always caught, at any frame rate. Chosen from the
// requested ~100-150ms range.
const ROI_EXIT_DEBOUNCE_MS = 120;

// ---------------------------------------------------------------
// TEMPORARY diagnostic instrumentation (mobile ROI false-rep
// re-investigation - see the report this shipped with). Logs ONLY on
// an ROI-membership transition, an analyzer.update() call, or a
// reset triggered by the ROI guard - not every raf tick. Gated behind
// DEBUG_ROI_GUARD so it can be silenced (or this whole block deleted)
// once the real-device investigation is done; not meant to ship on
// indefinitely.
// ---------------------------------------------------------------
const DEBUG_ROI_GUARD = true;

/**
 * Diagnostic-only duplicate of isArmWithinRoi()'s per-point check
 * (pose.js), used solely to log which INDIVIDUAL landmark(s) are
 * outside the ROI - never used for any actual accept/reject decision.
 */
function debugLandmarkInsideRoi(landmark, roi, videoWidth, videoHeight) {
  if (!roi || !landmark) return null;
  const rect = roiToRawCropRect(roi, videoWidth);
  const px = landmark.x * videoWidth;
  const py = landmark.y * videoHeight;
  return px >= rect.x && px <= rect.x + rect.width && py >= rect.y && py <= rect.y + rect.height;
}

/**
 * Diagnostic-only: logs the full video/canvas/CSS geometry picture on
 * demand (window resize/orientationchange, or the first Workout
 * frame) - lets a real-device session confirm or rule out mobile
 * layout changes silently altering the coordinate system the
 * confirmed ROI was captured in. Never used for any decision.
 */
function debugLogGeometrySnapshot(label) {
  if (!DEBUG_ROI_GUARD) return;
  const rect = cameraVideoEl.getBoundingClientRect();
  console.log(`[geometry-debug] ${label}`, {
    videoWidth: cameraVideoEl.videoWidth,
    videoHeight: cameraVideoEl.videoHeight,
    cssRect: { width: rect.width, height: rect.height },
    devicePixelRatio: window.devicePixelRatio,
    orientation: window.screen && window.screen.orientation ? window.screen.orientation.type : "unknown",
    innerWidth: window.innerWidth,
    innerHeight: window.innerHeight,
    screen: currentScreenId
  });
}

if (DEBUG_ROI_GUARD) {
  window.addEventListener("resize", () => debugLogGeometrySnapshot("window resize event"));
  window.addEventListener("orientationchange", () => debugLogGeometrySnapshot("orientationchange event"));
}

function enterPoseDetection() {
  if (!isPoseSupported()) {
    console.error("pose.js: WebAssembly is not supported in this browser - pose detection unavailable.");
    return;
  }

  // Immediate sync in addition to the persistent onStateChange
  // subscription above - covers the case where state already changed
  // (e.g. ROI prefetch finished or failed) before this listener call
  // happens to fire again.
  updatePoseStatusUI(poseDetector.getState(), poseDetector.getError());

  debugLogGeometrySnapshot("Workout entry (before first detection)");

  workoutEnteredAt = performance.now();

  // Fresh analyzer for this Workout visit - see the section header
  // comment above for why this happens on every entry, not just
  // Restart. Uses the arm chosen on the Arm Selection screen.
  analyzer = new ExerciseAnalyzer(state.selectedArm);
  workoutFinished = false;

  // Computed once per visit (the selected arm never changes mid-
  // workout) and reused every frame below, both to select the
  // analyzer's landmarks and to restrict what the renderer draws -
  // same arm, same indices, same armMapping.js call, no separate
  // logic path that could drift from the analyzer's selection.
  const selectedArmIndices = armLandmarkIndices(state.selectedArm);

  // Captured once for this Workout visit, matching the existing
  // "ROI is locked in at Confirm time" design (see the poseDetector.
  // start() call below) - also reused by isArmWithinRoi() so both the
  // crop and the ROI-membership gate agree on the exact same rect.
  const confirmedRoi = roiController.getConfirmed();

  // ROI-exit debounce state for this visit only (see
  // ROI_EXIT_DEBOUNCE_MS above) - null whenever the arm is currently
  // qualifying (detected + inside ROI); set to the timestamp of the
  // FIRST non-qualifying frame in the current exit streak otherwise,
  // so elapsed time can be measured against it every subsequent frame.
  let roiExitStartedAt = null;

  // TEMPORARY diagnostic instrumentation state for this visit only -
  // see DEBUG_ROI_GUARD above.
  let debugFrameCounter = 0;
  let debugPrevArmInsideRoi = null; // null = no frame processed yet this visit
  let debugLastDetectionTimestamp = -Infinity;

  hudArmEl.textContent = state.selectedArm === "left" ? "Left" : "Right";
  hudRepsEl.textContent = `0 / ${state.targetReps}`;
  hudAngleEl.textContent = "—";
  hudStageEl.textContent = "—";
  hudGoodEl.textContent = "0";
  hudBadEl.textContent = "0";
  hudFeedbackTextEl.textContent = "—";
  hudFeedbackEl.classList.remove("hud-feedback-good", "hud-feedback-attention");

  // ROI crop boundary: the confirmed rect is captured once, here, at
  // Workout entry - matching the existing app's design where ROI is
  // locked in at Confirm time (see roi.js) and cannot change without
  // leaving and re-entering Workout, which stops and restarts
  // detection via enter/leavePoseDetection() anyway.
  poseDetector.start(cameraVideoEl, {
    roi: confirmedRoi,
    onResult(result) {
      if (!hasLoggedPoseReady) {
        hasLoggedPoseReady = true;
        console.log("Pose detection ready.");
      }
      if (!hasLoggedFirstDetection && result.landmarks && result.landmarks.length > 0) {
        hasLoggedFirstDetection = true;
        if (DEBUG_TIMING) {
          const elapsed = performance.now() - workoutEnteredAt;
          console.log(
            `[pose timing] Workout entry -> first detected pose: ${elapsed.toFixed(0)}ms ` +
              `(includes model load + camera readiness + time to step into frame)`
          );
        }
      }

      drawPoseOverlay(workoutSkeletonCanvasEl, cameraVideoEl, result, selectedArmIndices);

      // Exercise analysis: the hard boundary is "selected arm
      // confirmed DETECTED and INSIDE the user's ROI this frame" -
      // anything else must not advance the analyzer's state at all.
      //
      // BUG FOUND DURING THE MOBILE ROI RE-INVESTIGATION: the
      // previous version of this gate only reset the analyzer's
      // in-progress rep when a person WAS detected but outside the
      // ROI. When MediaPipe detected NOBODY at all this frame
      // (result.landmarks.length === 0 - a normal, expected outcome,
      // e.g. the arm moved far enough that even the padded detection
      // crop no longer contains it, or a momentary tracking dropout),
      // BOTH branches were skipped: no update(), but also no reset.
      // On a real phone, "the arm leaves the ROI" very often ALSO
      // means MediaPipe stops detecting the person entirely (not just
      // "detected but outside the tight box"), so the in-progress rep
      // state stayed frozen through that gap and could still complete
      // a false rep the moment detection resumed - the same bug the
      // previous fix targeted, surviving through a path its own
      // regression test never modeled (it only ever fed the analyzer
      // detected-but-outside-ROI landmarks, never a true "nobody
      // detected" frame). See the report this shipped with.
      //
      // Fixed by computing one armInsideRoi boolean that is only ever
      // true when BOTH a person was detected AND isArmWithinRoi()
      // passes, and resetting in every other case - not just the
      // "detected but outside" one.
      let armInsideRoi = false;
      let shoulder = null;
      let elbow = null;
      let wrist = null;

      if (result.landmarks && result.landmarks.length > 0) {
        ({ shoulder, elbow, wrist } = selectArmLandmarks(result.landmarks[0], state.selectedArm));
        armInsideRoi = isArmWithinRoi(shoulder, elbow, wrist, confirmedRoi, cameraVideoEl.videoWidth, cameraVideoEl.videoHeight);
      }

      if (DEBUG_ROI_GUARD) {
        debugFrameCounter += 1;
        const detectionTimestampMs = performance.now();
        const detectionTimestampIsNewer = detectionTimestampMs > debugLastDetectionTimestamp;
        debugLastDetectionTimestamp = detectionTimestampMs;

        const transitioned = debugPrevArmInsideRoi !== null && debugPrevArmInsideRoi !== armInsideRoi;
        if (transitioned) {
          console.log(
            `[roi-guard] frame ${debugFrameCounter} t=${detectionTimestampMs.toFixed(1)}ms ` +
              `TRANSITION ${debugPrevArmInsideRoi ? "inside->outside" : "outside->inside"}`,
            {
              screen: currentScreenId,
              roi: confirmedRoi,
              videoWidth: cameraVideoEl.videoWidth,
              videoHeight: cameraVideoEl.videoHeight,
              landmarksDetected: Boolean(shoulder),
              shoulder: shoulder && { x: shoulder.x, y: shoulder.y },
              elbow: elbow && { x: elbow.x, y: elbow.y },
              wrist: wrist && { x: wrist.x, y: wrist.y },
              shoulderInsideRoi: debugLandmarkInsideRoi(shoulder, confirmedRoi, cameraVideoEl.videoWidth, cameraVideoEl.videoHeight),
              elbowInsideRoi: debugLandmarkInsideRoi(elbow, confirmedRoi, cameraVideoEl.videoWidth, cameraVideoEl.videoHeight),
              wristInsideRoi: debugLandmarkInsideRoi(wrist, confirmedRoi, cameraVideoEl.videoWidth, cameraVideoEl.videoHeight),
              armInsideRoi,
              fromPaddedInferenceRoi: Boolean(confirmedRoi),
              detectionTimestampIsNewerThanPrevious: detectionTimestampIsNewer
            }
          );
        }
        debugPrevArmInsideRoi = armInsideRoi;
      }

      const repsBeforeThisFrame = analyzer.reps;
      const stageBeforeThisFrame = analyzer.stage;

      if (armInsideRoi) {
        // Back inside the ROI (or was never outside it this visit) -
        // clear any in-progress exit debounce so the NEXT exit, if
        // any, starts counting from zero rather than inheriting stale
        // elapsed time from a previous, already-recovered-from one.
        roiExitStartedAt = null;

        const analyzerResult = analyzer.update(shoulder, elbow, wrist, performance.now() / 1000);
        updateWorkoutHud(analyzerResult);

        if (DEBUG_ROI_GUARD) {
          // Distinguishes the analyzer's OWN existing "soft skip" path
          // (landmarks geometrically inside the ROI, but below
          // VISIBILITY_THRESHOLD - update() itself preserves state,
          // no reset ever happens) from the "hard reset" path logged
          // below in the else branch (armInsideRoi === false). Both
          // are legitimate, but they have very different failure
          // signatures if either fires too often - see the audit
          // report for how this distinction maps to categories A/D.
          console.log(
            `[roi-guard] frame ${debugFrameCounter} analyzer.update() called` +
              (analyzerResult.visible ? "" : " (SOFT SKIP - low landmark visibility, state PRESERVED)"),
            {
              screen: currentScreenId,
              roi: confirmedRoi,
              videoWidth: cameraVideoEl.videoWidth,
              videoHeight: cameraVideoEl.videoHeight,
              shoulder: { x: shoulder.x, y: shoulder.y },
              elbow: { x: elbow.x, y: elbow.y },
              wrist: { x: wrist.x, y: wrist.y },
              armInsideRoi: true,
              landmarkVisible: analyzerResult.visible,
              missingParts: analyzerResult.missingParts,
              fromPaddedInferenceRoi: Boolean(confirmedRoi),
              stageBefore: stageBeforeThisFrame,
              stageAfter: analyzer.stage,
              repsBefore: repsBeforeThisFrame,
              repsAfter: analyzer.reps,
              repCountChanged: analyzer.reps !== repsBeforeThisFrame
            }
          );
        }

        // Auto-completion: only ever driven by the analyzer's own
        // confirmed rep count (analyzerResult.reps, incremented
        // solely inside ExerciseAnalyzer._completeRep()) - never by
        // angle, stage, or any other proxy. finishWorkout()
        // self-guards via workoutFinished, so this can fire on every
        // frame at/above target without any risk of running twice.
        if (analyzerResult.reps >= state.targetReps) {
          finishWorkout();
        }
      } else {
        // Invalid/outside frame: analyzer.update() is NEVER called
        // here - the debounce below only ever decides WHEN the
        // existing in-progress state gets invalidated, never whether
        // this frame itself can advance it. The ROI boundary stays
        // strict: an outside-ROI (or undetected) frame can never reach
        // the analyzer as a valid exercise frame, debounce or not.
        const now = performance.now();
        if (roiExitStartedAt === null) {
          roiExitStartedAt = now; // first non-qualifying frame of a new exit streak
        }
        const exitDurationMs = now - roiExitStartedAt;
        const shouldReset = exitDurationMs >= ROI_EXIT_DEBOUNCE_MS;

        if (shouldReset) {
          resetInProgressRep();
        }

        if (DEBUG_ROI_GUARD && (stageBeforeThisFrame !== analyzer.stage || repsBeforeThisFrame !== analyzer.reps || debugFrameCounter === 1)) {
          // reason distinguishes the two ways a frame can fail the
          // hard boundary: "no-detection" (MediaPipe found nobody at
          // all this frame) vs "detected-but-outside-roi" (a person
          // WAS found, just not positioned inside the drawn box). A
          // HARD RESET DISCARDING A MID-CURL "up" STAGE VIA
          // "no-detection" WHILE THE ARM NEVER ACTUALLY LEFT THE ROI
          // (i.e. this fires mid-rep, repeatedly, without the user
          // having moved) is the signature of category D/A from the
          // audit report - a transient MediaPipe dropout being treated
          // as a genuine ROI exit. Only reachable now once the
          // ROI_EXIT_DEBOUNCE_MS grace period has actually elapsed.
          console.log(`[roi-guard] frame ${debugFrameCounter} HARD RESET (reason: ${shoulder ? "detected-but-outside-roi" : "no-detection"}, after ${exitDurationMs.toFixed(0)}ms continuous exit)`, {
            screen: currentScreenId,
            roi: confirmedRoi,
            videoWidth: cameraVideoEl.videoWidth,
            videoHeight: cameraVideoEl.videoHeight,
            landmarksDetected: Boolean(shoulder),
            shoulder: shoulder && { x: shoulder.x, y: shoulder.y },
            elbow: elbow && { x: elbow.x, y: elbow.y },
            wrist: wrist && { x: wrist.x, y: wrist.y },
            armInsideRoi: false,
            fromPaddedInferenceRoi: Boolean(confirmedRoi),
            exitDurationMs,
            roiExitDebounceMs: ROI_EXIT_DEBOUNCE_MS,
            stageBefore: stageBeforeThisFrame,
            stageAfter: analyzer.stage,
            repsBefore: repsBeforeThisFrame,
            repsAfter: analyzer.reps,
            discardedInProgressRep: stageBeforeThisFrame === "up" && analyzer.stage === "down"
          });
        }
      }
    },
    onError(err) {
      // Avoid re-logging the same failure every frame (e.g. a
      // persistent per-frame detection error) - only log when the
      // failure kind actually changes.
      if (err.code !== lastLoggedPoseErrorCode) {
        lastLoggedPoseErrorCode = err.code;
        console.error(`pose.js [${err.code}]:`, err.message, err.cause || "");
      }
    }
  });
}

function leavePoseDetection() {
  poseDetector.stop();
  clearPoseOverlay(workoutSkeletonCanvasEl);
  lastLoggedPoseErrorCode = null;
  // Reset per-visit so returning to Workout logs its own fresh timing
  // (expected to be much faster than the first visit, since the
  // model itself should already be loaded - see the audit report).
  hasLoggedFirstDetection = false;
  workoutEnteredAt = null;
  // Stop using this Workout visit's analyzer session - the next
  // Workout entry creates a brand new one (see enterPoseDetection()).
  analyzer = null;
  workoutFinished = false;
}

/** Reflects one analyzer.update() result onto the Workout HUD. */
function updateWorkoutHud(result) {
  hudRepsEl.textContent = `${result.reps} / ${state.targetReps}`;
  hudAngleEl.textContent = result.angle !== null ? `${Math.round(result.angle)}°` : "—";
  hudStageEl.textContent = result.visible ? result.stage : `${result.stage} (arm not visible)`;
  hudGoodEl.textContent = result.goodReps;
  hudBadEl.textContent = result.badReps;
  hudFeedbackTextEl.textContent = result.lastFeedback;

  // Display-only color cue layered on top of the analyzer's own,
  // unmodified feedback text (exact string "Good rep!" for a clean
  // rep, per exerciseAnalyzer.js - never invented or recomputed here).
  // The wording itself already differs between outcomes, so color is
  // a supplement, not the only signal.
  hudFeedbackEl.classList.remove("hud-feedback-good", "hud-feedback-attention");
  if (result.lastFeedback === "Good rep!") {
    hudFeedbackEl.classList.add("hud-feedback-good");
  } else if (result.lastFeedback !== "Get ready...") {
    hudFeedbackEl.classList.add("hud-feedback-attention");
  }
}

/** Reflects analyzer.getResults() onto the Results screen - the analyzer's own values, not a new scoring system. */
function renderResults(summary) {
  resultsArmLabelEl.textContent = state.selectedArm === "left" ? "Left arm" : "Right arm";
  resultsScoreEl.textContent = `${summary.finalScore.toFixed(1)} / 10`;
  resultsCompletedEl.textContent = `${summary.reps} / ${summary.targetReps}`;
  resultsGoodEl.textContent = summary.goodReps;
  resultsBadEl.textContent = summary.badReps;
  resultsPartialEl.textContent = summary.partialReps;
  resultsFullEl.textContent = summary.fullReps;
  resultsVeryShortEl.textContent = summary.veryShortReps;
  resultsAvgTempoEl.textContent = `${summary.averageTempo.toFixed(2)} s`;
  resultsTempoIssuesEl.textContent = summary.tempoProblems;

  resultsFeedbackListEl.innerHTML = "";
  for (const line of summary.feedbackLines) {
    const li = document.createElement("li");
    li.textContent = line;
    resultsFeedbackListEl.appendChild(li);
  }
}

const resultsArmLabelEl = document.getElementById("results-arm-label");
const resultsScoreEl = document.getElementById("results-score");
const resultsCompletedEl = document.getElementById("results-completed");
const resultsGoodEl = document.getElementById("results-good");
const resultsBadEl = document.getElementById("results-bad");
const resultsPartialEl = document.getElementById("results-partial");
const resultsFullEl = document.getElementById("results-full");
const resultsVeryShortEl = document.getElementById("results-very-short");
const resultsAvgTempoEl = document.getElementById("results-avg-tempo");
const resultsTempoIssuesEl = document.getElementById("results-tempo-issues");
const resultsFeedbackListEl = document.getElementById("results-feedback-list");

/**
 * Ends the current Workout visit and shows its results - shared by
 * the manual "Finish Workout" button and automatic target-reps
 * completion (see the onResult callback above), so there is exactly
 * one place that captures results and navigates.
 *
 * Self-guarded via workoutFinished: a second call (e.g. the button
 * clicked in the same instant automatic completion fires - JS is
 * single-threaded, so these can never truly overlap, but the guard
 * makes that guarantee explicit rather than implicit) is a no-op -
 * no second analyzer read, no duplicate navigation.
 */
function finishWorkout() {
  if (workoutFinished) return;
  workoutFinished = true;

  // Capture results BEFORE navigating - showScreen("workout" -> "results")
  // triggers leavePoseDetection(), which stops detection and discards
  // this visit's analyzer. Reading getResults() here, on the exact
  // analyzer instance that produced the workout, is what "preserve
  // the analyzer session" means - no new analyzer is created first.
  if (analyzer) {
    renderResults(analyzer.getResults(state.targetReps));
  }
  showScreen("results");
}

document.getElementById("finish-workout-btn").addEventListener("click", () => {
  finishWorkout();
});

// -----------------------------------------------------------
// Instructions content rendering
// -----------------------------------------------------------

function renderInstructions() {
  const container = document.getElementById("instructions-content");
  container.innerHTML = "";

  for (const section of INSTRUCTIONS_SECTIONS) {
    const sectionEl = document.createElement("div");
    sectionEl.className = "instruction-section";

    const heading = document.createElement("h2");
    heading.textContent = section.heading;
    sectionEl.appendChild(heading);

    if (section.intro) {
      const intro = document.createElement("p");
      intro.className = "intro";
      intro.textContent = section.intro;
      sectionEl.appendChild(intro);
    }

    if (section.items) {
      const list = document.createElement("ul");
      for (const item of section.items) {
        const li = document.createElement("li");
        li.textContent = item;
        list.appendChild(li);
      }
      sectionEl.appendChild(list);
    }

    if (section.checklist) {
      const list = document.createElement("ul");
      list.className = "checklist";
      for (const item of section.checklist) {
        const li = document.createElement("li");
        li.textContent = item;
        list.appendChild(li);
      }
      sectionEl.appendChild(list);
    }

    if (section.note) {
      const note = document.createElement("p");
      note.className = "note";
      note.textContent = section.note;
      sectionEl.appendChild(note);
    }

    container.appendChild(sectionEl);
  }
}

// -----------------------------------------------------------
// Arm selection
// -----------------------------------------------------------

function setupArmSelection() {
  const cards = document.querySelectorAll(".arm-card");
  const continueBtn = document.getElementById("arm-continue-btn");

  cards.forEach((card) => {
    card.addEventListener("click", () => {
      cards.forEach((c) => c.setAttribute("aria-pressed", "false"));
      card.setAttribute("aria-pressed", "true");

      state.selectedArm = card.dataset.arm;
      continueBtn.disabled = false;
    });
  });
}

function resetArmSelection() {
  state.selectedArm = null;

  document.querySelectorAll(".arm-card").forEach((c) => {
    c.setAttribute("aria-pressed", "false");
  });

  document.getElementById("arm-continue-btn").disabled = true;
}

// -----------------------------------------------------------
// Rep count stepper
// -----------------------------------------------------------

function setupRepCount() {
  const input = document.getElementById("rep-count-input");
  const decBtn = document.getElementById("rep-decrement");
  const incBtn = document.getElementById("rep-increment");

  const clamp = (value) => Math.min(100, Math.max(1, value));

  const commit = (value) => {
    const clamped = clamp(value);
    input.value = clamped;
    state.targetReps = clamped;
  };

  decBtn.addEventListener("click", () => {
    commit((parseInt(input.value, 10) || 1) - 1);
  });

  incBtn.addEventListener("click", () => {
    commit((parseInt(input.value, 10) || 0) + 1);
  });

  input.addEventListener("change", () => {
    commit(parseInt(input.value, 10) || 1);
  });
}

// -----------------------------------------------------------
// Generic navigation + restart
// -----------------------------------------------------------

function setupNavigation() {
  document.querySelectorAll("[data-nav]").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.id === "restart-btn") {
        resetArmSelection();
        document.getElementById("rep-count-input").value = 10;
        state.targetReps = 10;
        roiController.reset();
      }

      showScreen(btn.dataset.nav);
    });
  });
}

// -----------------------------------------------------------
// Init
// -----------------------------------------------------------

function init() {
  renderInstructions();
  setupArmSelection();
  setupRepCount();
  setupNavigation();
  showScreen("instructions");
}

document.addEventListener("DOMContentLoaded", init);
