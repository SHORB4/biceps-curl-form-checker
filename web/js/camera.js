// camera.js
//
// Phase 1 — browser camera acquisition.
//
// Browser equivalent of form.py's cv2.VideoCapture(CAMERA_INDEX) setup.
// There is no numeric camera-index concept on the web: the browser
// itself prompts the user and resolves which physical device to use,
// so this module never selects a device by index (see requirement 9
// in Phase 1 — CAMERA_INDEX must NOT be carried over).
//
// Scope: acquire a getUserMedia() video stream, attach it to a
// <video> element, and stop it cleanly. No MediaPipe, no pose
// detection, no ROI, no analysis — those are later phases.

const CAMERA_CONSTRAINTS = {
  video: {
    facingMode: "user",
    width: { ideal: 1280 },
    height: { ideal: 720 }
  },
  audio: false
};

// Tracks the single active stream so stopCamera() can always find it,
// even if the caller lost its reference (e.g. after a screen change).
let activeStream = null;

export function isCameraSupported() {
  return Boolean(
    typeof navigator !== "undefined" &&
    navigator.mediaDevices &&
    typeof navigator.mediaDevices.getUserMedia === "function"
  );
}

/**
 * Requests the user-facing camera and attaches it to videoEl.
 * Resolves with the MediaStream on success.
 * Rejects with an Error whose `.code` is one of:
 *   "unsupported" | "permission-denied" | "no-camera" |
 *   "unavailable" | "overconstrained" | "insecure-context" | "unknown"
 */
export async function startCamera(videoEl) {
  // Checked first, before the general support check: on an insecure
  // context (plain http:// other than localhost/127.0.0.1), some
  // browsers hide navigator.mediaDevices entirely, which would
  // otherwise make isCameraSupported() report the generic (and here,
  // misleading) "browser does not support camera access" - when the
  // real, fixable cause is the page not being served over HTTPS.
  if (!window.isSecureContext) {
    throw cameraError(
      "insecure-context",
      "Camera access requires a secure connection (HTTPS or localhost)."
    );
  }

  if (!isCameraSupported()) {
    throw cameraError(
      "unsupported",
      "This browser does not support camera access."
    );
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(CAMERA_CONSTRAINTS);
  } catch (err) {
    throw classifyGetUserMediaError(err);
  }

  // A previous stream should already have been stopped by the caller
  // before requesting a new one, but guard against leaks regardless.
  if (activeStream && activeStream !== stream) {
    stopStream(activeStream);
  }
  activeStream = stream;

  videoEl.srcObject = stream;
  videoEl.muted = true;
  videoEl.playsInline = true;

  try {
    await videoEl.play();
  } catch {
    // Autoplay can be rejected in rare cases even for a muted,
    // user-gesture-triggered stream; the `autoplay` attribute and a
    // later user interaction will usually recover it, so this is not
    // treated as a fatal camera error.
  }

  return stream;
}

/** Stops all tracks of the active stream and detaches it from videoEl. */
export function stopCamera(videoEl) {
  stopStream(activeStream);
  activeStream = null;

  if (videoEl) {
    videoEl.pause();
    videoEl.srcObject = null;
  }
}

function stopStream(stream) {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    track.stop();
  }
}

function cameraError(code, message) {
  const err = new Error(message);
  err.code = code;
  return err;
}

function classifyGetUserMediaError(err) {
  const name = err && err.name;

  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return cameraError(
        "permission-denied",
        "Camera permission is required. Please allow camera access and try again."
      );
    case "SecurityError":
      return cameraError(
        "insecure-context",
        "Camera access requires a secure connection (HTTPS or localhost)."
      );
    case "NotFoundError":
    case "DevicesNotFoundError":
      return cameraError(
        "no-camera",
        "No camera was found on this device."
      );
    case "NotReadableError":
    case "TrackStartError":
      return cameraError(
        "unavailable",
        "Camera is currently unavailable. It may already be in use by another application."
      );
    case "OverconstrainedError":
    case "ConstraintNotSatisfiedError":
      return cameraError(
        "overconstrained",
        "No camera on this device supports the required settings."
      );
    case "AbortError":
      return cameraError(
        "unknown",
        "Camera access was interrupted. Please try again."
      );
    default:
      return cameraError(
        "unknown",
        "Unable to access the camera. Please check your browser permissions and try again."
      );
  }
}
