// pose.js
//
// MediaPipe Tasks Vision PoseLandmarker wrapper (Phase 3, + the ROI
// crop-boundary correction that followed it).
//
// Runs pose detection cropped to the confirmed ROI (see the crop
// pipeline below) and hands the caller the MediaPipe
// PoseLandmarkerResult with `landmarks` transformed back into
// full-frame-normalized coordinates - this module does not filter to
// specific landmarks, compute angles, or touch exerciseAnalyzer.js.
// That is later-phase work.
//
// ---------------------------------------------------------------
// Mirroring / left-right note (see also renderer.js's header):
//
// MediaPipe's detectForVideo() reads decoded video/canvas pixels
// directly. CSS's `transform: scaleX(-1)` (used elsewhere in this app
// purely to give the preview a natural "look in a mirror" feel) is a
// rendering/compositing-only effect - it does not alter the pixels
// MediaPipe reads, or the pixels a canvas drawImage(videoEl, ...)
// call copies. So MediaPipe here always operates on the RAW,
// unmirrored camera frame - both before and after the ROI crop.
//
// The confirmed ROI rect, however, is captured from pointer
// positions over the MIRRORED/displayed video (see roi.js) and is
// stored in that mirrored orientation - by design, unchanged by this
// correction. roiToRawCropRect() below converts that mirrored-space
// rect into the raw-space rect the canvas crop and MediaPipe both
// need, by flipping X once (`rawX = videoWidth - mirroredX`); Y is
// never flipped, since nothing in this app mirrors vertically.
//
// This module's own convention for the OUTPUT of landmarkToFullFrame
// (and therefore of drawPoseOverlay's input) is RAW/unmirrored,
// full-frame-normalized space - the same convention MediaPipe would
// already produce on an uncropped frame, and the same one
// renderer.js's existing display flip already assumes. That keeps a
// single coordinate convention end to end: only two places ever
// flip anything - roiToRawCropRect (mirrored ROI -> raw crop rect,
// for cropping) and renderer.js's drawPoseOverlay (raw landmarks ->
// mirrored screen pixels, for display only).
//
// That is architecturally different from the desktop app, which
// explicitly mirrors the frame (cv2.flip) BEFORE calling MediaPipe,
// and therefore swaps LEFT_*/RIGHT_* in exercise_analyzer.py's
// landmark_ids_for_arm() to compensate. Whether this web pipeline
// will need that same swap, the opposite swap, or no swap at all
// depends on a future phase's arm-selection wiring - explicitly out
// of scope here. This module does not swap, filter, or otherwise
// interpret LEFT_*/RIGHT_* in any way; it only passes through what
// MediaPipe returns, repositioned (not reinterpreted) into full-frame
// coordinates.
// ---------------------------------------------------------------

import {
  FilesetResolver,
  PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const WASM_BASE_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";

const MODEL_ASSET_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const HAVE_CURRENT_DATA = 2; // HTMLMediaElement.HAVE_CURRENT_DATA

// ---------------------------------------------------------------
// TEMPORARY diagnostic instrumentation (loading-performance audit).
// Sole purpose: let real load-time numbers be captured from an
// actual browser console, since this environment has none. Set to
// false (or delete this block and its call sites below) before
// Phase 4 - this is not meant to ship as permanent logging.
// ---------------------------------------------------------------
export const DEBUG_TIMING = true;
let _initCallCount = 0;

export function isPoseSupported() {
  return typeof WebAssembly !== "undefined";
}

// ---------------------------------------------------------------
// ROI crop-boundary coordinate transform
//
// roi.js's confirmed ROI ({x1,y1,x2,y2}) is in mirrored/display
// pixel space (see the header note above). Pure, DOM-free, and
// individually unit-testable.
// ---------------------------------------------------------------

/**
 * Converts a confirmed ROI rect (mirrored/display pixel space) into
 * the equivalent rect in the video's RAW (unmirrored) pixel space -
 * what canvas drawImage(videoEl, ...) actually reads from, and what
 * MediaPipe therefore needs to crop to. Only X is flipped; nothing
 * in this app mirrors vertically.
 */
export function roiToRawCropRect(roi, videoWidth) {
  return {
    x: videoWidth - roi.x2,
    y: roi.y1,
    width: roi.x2 - roi.x1,
    height: roi.y2 - roi.y1
  };
}

/**
 * Transforms one landmark's normalized [0,1] coordinates from
 * "relative to a raw-space crop rect" into "relative to the raw,
 * unmirrored, full video frame" - exactly the formula
 * exercise_analyzer.py's landmark_to_full_frame() uses:
 *   fullX = (crop.x + landmark.x * crop.width)  / videoWidth
 *   fullY = (crop.y + landmark.y * crop.height) / videoHeight
 * Every other landmark property (z, visibility, presence, ...) is
 * preserved unchanged.
 */
export function landmarkToFullFrame(landmark, cropRect, videoWidth, videoHeight) {
  return {
    ...landmark,
    x: (cropRect.x + landmark.x * cropRect.width) / videoWidth,
    y: (cropRect.y + landmark.y * cropRect.height) / videoHeight
  };
}

function wrapError(code, message, cause) {
  const err = new Error(message);
  err.code = code;
  if (cause) err.cause = cause;
  return err;
}

async function initPoseLandmarker() {
  _initCallCount += 1;
  const callNumber = _initCallCount;
  const tStart = DEBUG_TIMING ? performance.now() : 0;

  if (DEBUG_TIMING) {
    console.log(
      `[pose timing] initPoseLandmarker() call #${callNumber}` +
        (callNumber > 1
          ? " - WARNING: this is not the first call, the model may be loading more than once"
          : "")
    );
  }

  let vision;

  try {
    vision = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
  } catch (err) {
    throw wrapError(
      "fileset-load-failed",
      "Could not load the pose detection runtime. Check your connection and try again.",
      err
    );
  }

  const tFileset = DEBUG_TIMING ? performance.now() : 0;
  if (DEBUG_TIMING) {
    console.log(
      `[pose timing] #${callNumber} WASM fileset resolved: ${(tFileset - tStart).toFixed(0)}ms`
    );
  }

  const baseOptions = {
    modelAssetPath: MODEL_ASSET_URL,
    delegate: "GPU"
  };

  let landmarker;

  try {
    landmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions,
      runningMode: "VIDEO",
      numPoses: 1
    });
  } catch (gpuErr) {
    // Not every device/browser combination has a working GPU delegate
    // for WASM SIMD tasks. One CPU retry is a reasonable, well-scoped
    // fallback - not a redesign of the initialization approach.
    console.warn(
      "pose.js: GPU delegate failed to initialize, retrying with CPU.",
      gpuErr
    );

    try {
      landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { ...baseOptions, delegate: "CPU" },
        runningMode: "VIDEO",
        numPoses: 1
      });
    } catch (cpuErr) {
      throw wrapError(
        "model-load-failed",
        "Could not load the pose detection model. Check your connection and try again.",
        cpuErr
      );
    }
  }

  if (DEBUG_TIMING) {
    const tCreated = performance.now();
    console.log(
      `[pose timing] #${callNumber} PoseLandmarker created ` +
        `(model download + WASM init - these happen inside the single ` +
        `createFromOptions() call, so they can't be split further ` +
        `without a separate manual prefetch): ${(tCreated - tFileset).toFixed(0)}ms`
    );
    console.log(
      `[pose timing] #${callNumber} TOTAL initPoseLandmarker() time: ${(tCreated - tStart).toFixed(0)}ms`
    );
  }

  return landmarker;
}

/**
 * Owns one MediaPipe PoseLandmarker instance and its
 * requestAnimationFrame detection loop.
 *
 * When start() is given a roi, every frame is cropped (via a reused
 * offscreen canvas) to that rect BEFORE being handed to MediaPipe -
 * detectForVideo() only ever sees the cropped image, so a person
 * entirely outside the ROI cannot be detected at all. Resulting
 * landmarks are transformed back into full-frame-normalized
 * coordinates before reaching callers, so nothing downstream needs
 * to know cropping happened.
 *
 * Create exactly one PoseDetector per page session and reuse it for
 * every Workout screen visit - do not construct a new one per visit,
 * and never call start() twice without an intervening stop() (start()
 * is a no-op while already running, as a guard against accidentally
 * creating a second loop).
 */
export class PoseDetector {
  constructor() {
    this._landmarkerPromise = null;
    this._loopRunning = false;
    this._rafHandle = null;
    this._lastVideoTime = -1;
    // Bumped on every start() call. Lets a stale in-flight load from
    // an earlier start() (still pending when start()/stop()/start()
    // happens in quick succession) recognize it's no longer current
    // and skip scheduling a second, redundant tick loop - see the
    // start() fix below, found during the Phase 3 loading audit.
    this._generation = 0;

    // Single source of truth for initialization progress, so any
    // screen (ROI Selection prefetching, Workout displaying a loading
    // state) reads the same state instead of each tracking its own.
    // "idle" -> "loading" -> "ready" | "error".
    this._state = "idle";
    this._lastError = null;
    this._stateListeners = new Set();

    // Reused offscreen ROI-crop canvas (created lazily, resized only
    // when dimensions change, never recreated per frame) - see
    // _getCroppedFrame().
    this._cropCanvas = null;
    this._cropCtx = null;
  }

  _setState(state) {
    if (this._state === state) return;
    this._state = state;
    for (const listener of this._stateListeners) {
      listener(this._state, this._lastError);
    }
  }

  /** Current initialization state: "idle" | "loading" | "ready" | "error". */
  getState() {
    return this._state;
  }

  /** The error from the most recent failed initialization, or null. */
  getError() {
    return this._lastError;
  }

  /**
   * Subscribes to state changes (fired for every idle/loading/ready/
   * error transition, regardless of which caller - ROI prefetch or
   * Workout - triggered the load). Returns an unsubscribe function.
   */
  onStateChange(listener) {
    this._stateListeners.add(listener);
    return () => this._stateListeners.delete(listener);
  }

  _ensureLandmarker() {
    if (!this._landmarkerPromise) {
      this._setState("loading");
      this._lastError = null;

      this._landmarkerPromise = initPoseLandmarker()
        .then((landmarker) => {
          this._setState("ready");
          return landmarker;
        })
        .catch((err) => {
          // Deliberately NOT nulling _landmarkerPromise here: the
          // failure is stored (this._lastError / state "error") so
          // every caller - ROI's prefetch, Workout entered before or
          // after the failure - consistently sees and can display the
          // same stored error, rather than silently retrying and
          // re-downloading ~18MB on every subsequent screen visit.
          this._lastError = err;
          this._setState("error");
          throw err;
        });
    }
    return this._landmarkerPromise;
  }

  /**
   * Loads (or reuses) the shared PoseLandmarker instance ahead of
   * time, without starting the detection loop. Safe to call
   * repeatedly (e.g. every time the ROI Selection screen is entered)
   * - _ensureLandmarker() only starts one load, ever, per instance.
   */
  init() {
    return this._ensureLandmarker();
  }

  /**
   * Starts the detection loop against videoEl. Safe to call while
   * already running (no-op) - this never creates a second loop or a
   * second MediaPipe instance.
   *
   * callbacks.roi: the confirmed ROI rect ({x1,y1,x2,y2}, mirrored/
   *   display pixel space - see roi.js's getConfirmed()), or null/
   *   undefined to detect on the full frame. Captured once for this
   *   detection session (matches the existing app: ROI is locked in
   *   at Confirm time and cannot change without leaving and
   *   re-entering Workout, which stops and restarts the loop).
   * callbacks.onResult(result): called for every frame that was
   *   actually run through the detector, with the MediaPipe
   *   PoseLandmarkerResult - `landmarks` already transformed back
   *   into full-frame-normalized coordinates when a roi was given.
   *   result.landmarks may legitimately be an empty array - that is
   *   a normal "no pose detected" outcome (e.g. nobody in the ROI at
   *   all), not an error.
   * callbacks.onError(err): called for initialization or per-frame
   *   detection failures. err.code identifies the failure kind
   *   ("fileset-load-failed" | "model-load-failed" | "detection-failed").
   */
  start(videoEl, callbacks = {}) {
    if (this._loopRunning) return;

    const { roi, onResult, onError } = callbacks;

    this._loopRunning = true;
    this._lastVideoTime = -1;
    this._generation += 1;
    const generation = this._generation;

    // BUG FOUND DURING THE PHASE 3 LOADING AUDIT (fixed here, since
    // it is a duplicate-detection-loop bug, same category as the
    // "trivial duplicate initialization" fixes this audit allows):
    // start() -> stop() -> start() in quick succession, before the
    // FIRST model load has resolved, used to schedule TWO tick loops.
    // _ensureLandmarker() correctly reuses the same in-flight promise
    // (no duplicate download/model-load), but that promise then had
    // two .then() handlers attached - one per start() call - and the
    // old `if (!this._loopRunning) return;` guard alone couldn't tell
    // the stale (first) handler apart from the current (second) one,
    // since a later start() had already flipped _loopRunning back to
    // true. Comparing the captured `generation` against the current
    // one closes that gap.
    this._ensureLandmarker()
      .then((landmarker) => {
        if (!this._loopRunning || generation !== this._generation) return;
        this._scheduleTick(videoEl, landmarker, roi, onResult, onError);
      })
      .catch((err) => {
        if (generation !== this._generation) return; // superseded by a newer start()
        this._loopRunning = false;
        if (onError) onError(err);
      });
  }

  _scheduleTick(videoEl, landmarker, roi, onResult, onError) {
    this._rafHandle = requestAnimationFrame(() =>
      this._tick(videoEl, landmarker, roi, onResult, onError)
    );
  }

  /**
   * Draws the given raw-space crop rect from videoEl onto a reused,
   * detached (never appended to the DOM) <canvas>, resized only when
   * its dimensions actually change - not recreated every frame.
   * Returns that canvas, ready to hand to detectForVideo().
   */
  _getCroppedFrame(videoEl, cropRect) {
    const w = Math.max(1, Math.round(cropRect.width));
    const h = Math.max(1, Math.round(cropRect.height));

    if (!this._cropCanvas) {
      this._cropCanvas = document.createElement("canvas");
      this._cropCtx = this._cropCanvas.getContext("2d");
    }
    if (this._cropCanvas.width !== w) this._cropCanvas.width = w;
    if (this._cropCanvas.height !== h) this._cropCanvas.height = h;

    this._cropCtx.drawImage(
      videoEl,
      cropRect.x, cropRect.y, cropRect.width, cropRect.height,
      0, 0, w, h
    );

    return this._cropCanvas;
  }

  _tick(videoEl, landmarker, roi, onResult, onError) {
    if (!this._loopRunning) return;

    const videoReady =
      videoEl.readyState >= HAVE_CURRENT_DATA &&
      videoEl.videoWidth > 0 &&
      videoEl.currentTime !== this._lastVideoTime;

    if (videoReady) {
      // Guards against running detectForVideo twice on the same
      // decoded frame if requestAnimationFrame fires faster than the
      // video actually advances.
      this._lastVideoTime = videoEl.currentTime;

      try {
        // ROI crop boundary: MediaPipe receives ONLY the cropped ROI
        // image when a roi is set, so a person entirely outside it
        // cannot influence detection at all - not a post-hoc filter
        // on landmarks from a full-frame detection.
        let source = videoEl;
        let cropRect = null;

        if (roi) {
          cropRect = roiToRawCropRect(roi, videoEl.videoWidth);
          source = this._getCroppedFrame(videoEl, cropRect);
        }

        const result = landmarker.detectForVideo(source, performance.now());

        if (cropRect) {
          result.landmarks = result.landmarks.map((personLandmarks) =>
            personLandmarks.map((landmark) =>
              landmarkToFullFrame(landmark, cropRect, videoEl.videoWidth, videoEl.videoHeight)
            )
          );
        }

        if (onResult) onResult(result);
      } catch (err) {
        // A single bad frame should not end the whole detection
        // session - report it and keep the loop running.
        if (onError) {
          onError(
            wrapError("detection-failed", "Pose detection failed on a frame.", err)
          );
        }
      }
    }

    this._scheduleTick(videoEl, landmarker, roi, onResult, onError);
  }

  /**
   * Stops the detection loop. Does not release the MediaPipe model -
   * safe to start() again immediately (e.g. re-entering the Workout
   * screen), which resumes without reloading anything.
   */
  stop() {
    this._loopRunning = false;
    if (this._rafHandle !== null) {
      cancelAnimationFrame(this._rafHandle);
      this._rafHandle = null;
    }
    this._lastVideoTime = -1;
  }

  isRunning() {
    return this._loopRunning;
  }

  /**
   * Releases the underlying MediaPipe instance entirely. Only for
   * full page-level teardown - do NOT call this on ordinary screen
   * navigation, or the next Workout visit would have to reload the
   * whole model from scratch.
   */
  async close() {
    this.stop();

    if (this._landmarkerPromise) {
      try {
        const landmarker = await this._landmarkerPromise;
        landmarker.close();
      } catch {
        // Never finished loading - nothing to release.
      }
      this._landmarkerPromise = null;
    }

    this._lastError = null;
    this._setState("idle");
  }
}
