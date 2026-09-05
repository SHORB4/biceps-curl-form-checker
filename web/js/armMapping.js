// armMapping.js
//
// Phase 4 - arm-selection mapping layer.
//
// Decides which of MediaPipe's raw landmark indices (in the array
// pose.js hands back, already transformed into full-frame-normalized
// coordinates) correspond to the user's SELECTED arm ("left" |
// "right"), and extracts the shoulder/elbow/wrist objects the
// analyzer's update() expects. This runs BEFORE the analyzer ever
// sees anything - exerciseAnalyzer.js's own left/right definitions
// (landmarkIdsForArm()) are not used or modified here at all.
//
// ---------------------------------------------------------------
// Why this can't just reuse exerciseAnalyzer.js's landmarkIdsForArm()
// ---------------------------------------------------------------
// exercise_analyzer.py's landmark_ids_for_arm() (ported faithfully to
// exerciseAnalyzer.js in Phase 2) SWAPS MediaPipe's LEFT_*/RIGHT_*
// landmarks for the desktop app, because form.py mirrors the camera
// frame (cv2.flip) BEFORE calling MediaPipe - per that code's own
// documented finding, mirroring the input inverts MediaPipe's
// LEFT_*/RIGHT_* output relative to the subject's true side, so the
// desktop reads the OPPOSITE MediaPipe landmark to compensate.
//
// This browser pipeline's MediaPipe call is architecturally
// different (see pose.js's header comment from the ROI-crop phase):
// the <video> is mirrored only via CSS, for display - MediaPipe's
// detectForVideo() and the ROI crop's canvas drawImage() both always
// read the RAW, UNMIRRORED camera frame. Nothing mirrors the pixels
// MediaPipe actually processes here, so (reasoning by the same logic
// the desktop's own comment describes) MediaPipe's LEFT_*/RIGHT_*
// output should already correspond DIRECTLY to the subject's true
// side - the opposite situation from desktop, needing the OPPOSITE
// mapping (no swap).
//
// This is a REASONED, NOT BROWSER-VERIFIED conclusion - there is no
// camera or browser available in the environment this was built in
// to confirm it against a real person. That is exactly why this
// mapping lives in its own tiny, isolated layer with a single flip
// switch below, and why the Phase 4 report calls for a manual
// left/right test (move only the left arm with LEFT selected, only
// the right arm with RIGHT selected, etc.) before trusting it.
// ---------------------------------------------------------------

// MediaPipe Pose landmark indices - the same fixed schema
// exerciseAnalyzer.js's (unexported) POSE_LANDMARK constant uses.
const POSE_LANDMARK = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16
};

/**
 * If manual testing (see the Phase 4 report) shows arm selection is
 * backwards in the browser, flip this one constant - nothing else
 * needs to change, and exerciseAnalyzer.js is never touched.
 */
export const ARM_MAPPING_SWAPPED = false;

/**
 * Returns the MediaPipe landmark indices {shoulder, elbow, wrist} for
 * the given selected arm ("left" | anything else treated as "right"),
 * using this browser pipeline's (unswapped, see above) mapping.
 */
export function armLandmarkIndices(arm) {
  const useLeftLandmarks = (arm === "left") !== ARM_MAPPING_SWAPPED;

  if (useLeftLandmarks) {
    return {
      shoulder: POSE_LANDMARK.LEFT_SHOULDER,
      elbow: POSE_LANDMARK.LEFT_ELBOW,
      wrist: POSE_LANDMARK.LEFT_WRIST
    };
  }

  return {
    shoulder: POSE_LANDMARK.RIGHT_SHOULDER,
    elbow: POSE_LANDMARK.RIGHT_ELBOW,
    wrist: POSE_LANDMARK.RIGHT_WRIST
  };
}

/**
 * Extracts {shoulder, elbow, wrist} landmark objects for the selected
 * arm out of one person's full landmark array (result.landmarks[0]
 * from a PoseLandmarkerResult - already in full-frame-normalized
 * coordinates by the time pose.js hands it back). Returns objects
 * exactly as MediaPipe/pose.js produced them (x, y, z, visibility,
 * presence, ...) - unmodified, just selected - ready to pass directly
 * into ExerciseAnalyzer.update().
 */
export function selectArmLandmarks(landmarks, arm) {
  const ids = armLandmarkIndices(arm);
  return {
    shoulder: landmarks[ids.shoulder],
    elbow: landmarks[ids.elbow],
    wrist: landmarks[ids.wrist]
  };
}
