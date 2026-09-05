// Content for the Instructions ("How to Use") screen.
//
// This mirrors the conceptual content and hierarchy of the desktop
// app's INSTRUCTIONS_PAGES (form.py), condensed from its paginated
// desktop form into a single scrollable web layout (web has no
// canvas-size constraint, so section-by-section pagination isn't
// needed here). Wording preserves the same "recommended / can help
// reduce errors" phrasing and avoids absolute guarantees, matching
// the desktop version's tone.
//
// Each entry: { heading, intro (optional), items: [string, ...] }

export const INSTRUCTIONS_SECTIONS = [
  {
    heading: "Before You Start",
    intro: "For best accuracy, follow these setup tips:",
    items: [
      "Make sure your device has a working camera.",
      "Keep the camera lens clean.",
      "Place the camera on a stable surface.",
      "Position yourself so your upper body and the selected arm are clearly visible.",
      "Only one person should be in the camera frame."
    ]
  },
  {
    heading: "Privacy",
    intro:
      "Camera processing happens locally in your browser. Video is not uploaded."
  },
  {
    heading: "Lighting",
    intro: "Good lighting can help reduce detection errors.",
    items: [
      "Use a well-lit environment.",
      "Make sure your body is clearly illuminated.",
      "Prefer light coming from in front of or slightly to the side of you.",
      "Avoid strong shadows over your body.",
      "Avoid very dark rooms.",
      "Avoid an extremely bright light directly behind you."
    ]
  },
  {
    heading: "Background and Clothing",
    items: [
      "Use a relatively uncluttered background.",
      "Avoid moving objects or other people behind you.",
      "Clothing should contrast reasonably with the background.",
      "Avoid clothing that blends completely into the background."
    ]
  },
  {
    heading: "ROI — Region of Interest",
    intro:
      "ROI (Region of Interest) tells the system which part of the camera " +
      "frame to focus on. Selecting an appropriate ROI can help reduce " +
      "interference from other objects or people in the frame.",
    items: [
      "Select the arm you want to train.",
      "When the ROI screen appears, drag a rectangle around the relevant arm/body area.",
      "Make sure the ROI contains the selected arm's shoulder, elbow, and wrist.",
      "Keep the entire selected arm inside the ROI during the exercise.",
      "Do not make the ROI unnecessarily large.",
      "Do not crop out the shoulder, elbow, or wrist.",
      "If the ROI is incorrect, use the reset option and select it again."
    ],
    note:
      "Important: ROI does not guarantee accurate detection — it is " +
      "intended to improve focus and reduce possible errors."
  },
  {
    heading: "During the Exercise",
    items: [
      "Keep the selected arm visible.",
      "Stay in approximately the same position relative to the camera.",
      "Keep the camera stationary.",
      "Perform controlled biceps curls.",
      "Avoid excessive body movement.",
      "Avoid moving the elbow unnecessarily.",
      "Keep the selected arm within the ROI."
    ]
  },
  {
    heading: "Best Accuracy Checklist",
    checklist: [
      "Good lighting",
      "Stable camera",
      "Clear background",
      "One person in frame",
      "Selected arm clearly visible",
      "Shoulder, elbow, and wrist visible",
      "Correct ROI",
      "Controlled movement",
      "Camera position remains stable"
    ]
  }
];
