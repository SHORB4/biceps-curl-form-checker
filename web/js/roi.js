// roi.js
//
// ROI (Region of Interest) drag-selection over the live camera video.
//
// Browser equivalent of form.py's roi_selecting / roi_drag_start /
// roi_drag_current / roi_pending / roi_confirmed / roi_message state,
// plus its roi_mouse_callback + fit_frame_to_window coordinate
// mapping and MIN_ROI_SIZE gating on confirm.
//
// Coordinate design (mirrors the desktop app's approach): the video
// is displayed with CSS `object-fit: contain`, which letterboxes it
// inside its element exactly like fit_frame_to_window letterboxed the
// camera frame inside the OpenCV window. Pointer/touch positions are
// captured relative to that same displayed box and converted through
// the inverse of the contain-fit scale/offset to land in the video's
// own intrinsic pixel space (videoWidth/videoHeight) — the direct
// counterpart of desktop's cam_x/cam_y via display_scale/offset.
// Because the desktop app mirrors the frame once (cv2.flip) and then
// treats that flipped frame as the single source of truth for
// everything downstream, the web version does the same: the CSS
// mirror on the <video> is a pure rendering effect, and both pointer
// capture and canvas drawing happen in the video element's own
// on-screen box, so they stay visually aligned without any extra
// unmirroring step. No pose/analysis logic lives here — this module
// only produces a confirmed rectangle in video-pixel coordinates.
//
// No exercise analysis, ROI-to-landmark mapping, or MediaPipe here —
// those are later phases.

const MIN_ROI_SIZE = 60; // matches form.py's MIN_ROI_SIZE

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(value, hi));
}

function normalizeRect(p1, p2) {
  return {
    x1: Math.min(p1.x, p2.x),
    y1: Math.min(p1.y, p2.y),
    x2: Math.max(p1.x, p2.x),
    y2: Math.max(p1.y, p2.y)
  };
}

/**
 * Creates an ROI drag-selection controller bound to a <video> element
 * (the live feed), an overlay <canvas> (drawn on top of it), and an
 * optional status element for status/error text.
 */
export function createRoiController({ videoEl, canvasEl, statusEl }) {
  let selecting = false;
  let dragStart = null; // {x, y} in video-intrinsic pixel space
  let dragCurrent = null;
  let pending = null; // normalized rect, drawn but not yet confirmed
  let confirmed = null; // normalized rect, confirmed
  let message = "";

  function videoBox() {
    const rect = videoEl.getBoundingClientRect();
    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;

    if (!vw || !vh || rect.width === 0 || rect.height === 0) {
      return null;
    }

    const scale = Math.min(rect.width / vw, rect.height / vh);
    const contentW = vw * scale;
    const contentH = vh * scale;

    return {
      rect,
      scale,
      vw,
      vh,
      offsetX: (rect.width - contentW) / 2,
      offsetY: (rect.height - contentH) / 2
    };
  }

  function eventPoint(evt) {
    if (evt.touches && evt.touches.length > 0) {
      return { clientX: evt.touches[0].clientX, clientY: evt.touches[0].clientY };
    }
    if (evt.changedTouches && evt.changedTouches.length > 0) {
      return { clientX: evt.changedTouches[0].clientX, clientY: evt.changedTouches[0].clientY };
    }
    return { clientX: evt.clientX, clientY: evt.clientY };
  }

  function pointerToFramePixel(evt) {
    const box = videoBox();
    if (!box) return null;

    const { clientX, clientY } = eventPoint(evt);
    const localX = clientX - box.rect.left;
    const localY = clientY - box.rect.top;

    const x = clamp((localX - box.offsetX) / box.scale, 0, box.vw - 1);
    const y = clamp((localY - box.offsetY) / box.scale, 0, box.vh - 1);

    return { x, y };
  }

  function frameRectToCanvas(r) {
    const box = videoBox();
    if (!box) return null;

    return {
      x1: box.offsetX + r.x1 * box.scale,
      y1: box.offsetY + r.y1 * box.scale,
      x2: box.offsetX + r.x2 * box.scale,
      y2: box.offsetY + r.y2 * box.scale
    };
  }

  function resizeCanvas() {
    const rect = videoEl.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));

    if (canvasEl.width !== w) canvasEl.width = w;
    if (canvasEl.height !== h) canvasEl.height = h;

    draw();
  }

  function strokeRect(ctx, r, color) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = color;
    ctx.strokeRect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1);
  }

  function draw() {
    const ctx = canvasEl.getContext("2d");
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

    if (selecting && dragStart && dragCurrent) {
      const r = frameRectToCanvas(normalizeRect(dragStart, dragCurrent));
      // Near-black outline while actively dragging, matching the
      // desktop app's (20, 20, 20) drag-rectangle color.
      if (r) strokeRect(ctx, r, "rgb(20, 20, 20)");
    } else if (confirmed) {
      const r = frameRectToCanvas(confirmed);
      if (r) strokeRect(ctx, r, "#34d399");
    } else if (pending) {
      const r = frameRectToCanvas(pending);
      if (r) strokeRect(ctx, r, "#22d3ee");
    }

    drawAllOverlays();
  }

  // -----------------------------------------------------------
  // Read-only overlay mirrors (e.g. the Workout screen)
  //
  // These reuse the exact same `confirmed` rect and the exact same
  // videoBox()/frameRectToCanvas() coordinate mapping as the
  // interactive canvas above - no second ROI system, no separate
  // math. They never receive drag event listeners, so nothing on
  // the Workout screen can modify the confirmed ROI.
  // -----------------------------------------------------------

  const overlays = new Set();

  function drawOverlayCanvas(target) {
    const rect = videoEl.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));

    if (target.width !== w) target.width = w;
    if (target.height !== h) target.height = h;

    const ctx = target.getContext("2d");
    ctx.clearRect(0, 0, target.width, target.height);

    if (!confirmed) return;

    const r = frameRectToCanvas(confirmed);
    if (!r) return;

    // Same "confirmed" green as the ROI screen, dashed to read as a
    // locked-in reference rather than an editable selection, and
    // distinct from wherever a future skeleton overlay will draw.
    ctx.setLineDash([7, 5]);
    strokeRect(ctx, r, "#34d399");
    ctx.setLineDash([]);
  }

  function drawAllOverlays() {
    overlays.forEach(drawOverlayCanvas);
  }

  function updateStatus() {
    if (!statusEl) return;

    if (message) {
      statusEl.textContent = message;
      statusEl.classList.add("roi-status-error");
      return;
    }

    statusEl.classList.remove("roi-status-error");

    if (confirmed) {
      statusEl.textContent = "Region confirmed — press Confirm ROI to continue, or Reset to redraw.";
    } else if (pending) {
      statusEl.textContent = "Region drawn — press Confirm ROI to continue, or Reset to redraw.";
    } else {
      statusEl.textContent = "No region selected yet — click and drag on the video.";
    }
  }

  function onDragStart(evt) {
    const p = pointerToFramePixel(evt);
    if (!p) return;

    evt.preventDefault();

    selecting = true;
    dragStart = p;
    dragCurrent = p;
    pending = null;
    message = "";

    updateStatus();
    draw();
  }

  function onDragMove(evt) {
    if (!selecting) return;

    const p = pointerToFramePixel(evt);
    if (!p) return;

    evt.preventDefault();
    dragCurrent = p;
    draw();
  }

  function onDragEnd(evt) {
    if (!selecting) return;

    const p = pointerToFramePixel(evt) || dragCurrent;

    selecting = false;
    dragCurrent = p;
    pending = normalizeRect(dragStart, dragCurrent);

    updateStatus();
    draw();
  }

  canvasEl.addEventListener("mousedown", onDragStart);
  document.addEventListener("mousemove", onDragMove);
  document.addEventListener("mouseup", onDragEnd);

  canvasEl.addEventListener("touchstart", onDragStart, { passive: false });
  canvasEl.addEventListener("touchmove", onDragMove, { passive: false });
  canvasEl.addEventListener("touchend", onDragEnd);
  canvasEl.addEventListener("touchcancel", onDragEnd);

  const resizeObserver = new ResizeObserver(() => resizeCanvas());
  resizeObserver.observe(videoEl);
  videoEl.addEventListener("loadedmetadata", resizeCanvas);

  updateStatus();

  return {
    /** Attempts to confirm the currently pending rect. Returns true on success. */
    confirm() {
      if (!pending) {
        message = "Draw a region first";
        updateStatus();
        return false;
      }

      const width = pending.x2 - pending.x1;
      const height = pending.y2 - pending.y1;

      if (width < MIN_ROI_SIZE || height < MIN_ROI_SIZE) {
        message = "ROI too small — drag a larger box";
        updateStatus();
        return false;
      }

      confirmed = { ...pending };
      message = "";
      updateStatus();
      draw();
      return true;
    },

    /** Clears all selection state, matching the desktop app's R (reset) key. */
    reset() {
      selecting = false;
      dragStart = null;
      dragCurrent = null;
      pending = null;
      confirmed = null;
      message = "";
      updateStatus();
      draw();
    },

    /** Recomputes canvas size/redraws — call after layout changes (resize, screen swap). */
    resizeCanvas,

    getConfirmed() {
      return confirmed ? { ...confirmed } : null;
    },

    /**
     * Registers a read-only canvas (e.g. the Workout screen's overlay)
     * that mirrors the confirmed ROI rect using this same controller's
     * state and coordinate mapping. No drag handlers are attached to
     * it, so it can never modify the ROI. Kept in sync automatically
     * whenever the interactive canvas redraws (drag, confirm, reset,
     * or the video's own ResizeObserver firing).
     */
    attachOverlay(target) {
      overlays.add(target);
      drawOverlayCanvas(target);
      return () => overlays.delete(target);
    },

    /** Force an immediate re-measure/redraw of one registered overlay. */
    refreshOverlay(target) {
      if (overlays.has(target)) drawOverlayCanvas(target);
    }
  };
}
