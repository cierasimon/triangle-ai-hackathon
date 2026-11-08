// Mirrors backend/app/contracts.py. Change both in the same commit.
// Recognition: warped-paper pixels, origin top-left, y down. Scene/Frame: meters, y up.

export type Point = [number, number];
export type Motion = "slide" | "roll";
export type ParamSource = "drawing" | "label" | "default" | "user";
export type Level = "beginner" | "intermediate" | "confident";

export interface Line {
  p1: Point;
  p2: Point;
  role: "ramp" | "ground";
  thickness_px: number;
  confidence: number;
}

export interface Ball {
  center: Point;
  radius: number;
  confidence: number;
  sprite_png_b64: string | null;
}

export interface Label {
  key: "angle_deg" | "mass_kg" | "friction" | "gravity";
  value: number;
  text: string;
  confidence: number;
}

export type VisionMode = "opencv" | "hybrid";

export interface DetectedObject {
  kind: "ball" | "block" | "ramp" | "spring" | "platform" | "wall";
  bbox: [number, number, number, number];
  confidence: number;
  geometry_verified: boolean;
  body_id: string | null;
}

export interface DiagramParameter {
  key: string; value: number | null; unit: string; text: string; target: string;
  confidence: number; validated: boolean; applied: boolean; note: string;
}

export interface Recognition {
  image_id: string;
  paper_found: boolean;
  paper_corners: Point[] | null;
  warped_size: [number, number];
  lines: Line[];
  ball: Ball | null;
  balls: Ball[];
  blocks: { vertices: Point[]; center: Point; confidence: number }[];
  objects: DetectedObject[];
  parameters: DiagramParameter[];
  selected_ball_index: number | null;
  recognition_mode: VisionMode;
  gemini_status: "disabled" | "success" | "failed" | "unavailable";
  gemini_model: string | null;
  gemini_summary: string | null;
  labels: Label[];
  warnings: string[];
}

export interface Params {
  angle_deg: number;
  mass_kg: number;
  gravity: number;
  friction: number;
  motion: Motion;
  defaults_used: string[];
  sources: Record<string, ParamSource>;
}

export interface Scene {
  scale_m_per_px: number;
  ramp_top_m: Point;
  ramp_bottom_m: Point;
  ground_y_m: number;
  ball_radius_m: number;
  ramp_length_m: number;
  params: Params;
}

export interface Frame {
  t: number;
  x: number;
  y: number;
  angle: number;
}

export interface Verification {
  predicted_time_s: number | null;
  simulated_time_s: number | null;
  predicted_speed_m_s: number | null;
  simulated_speed_m_s: number | null;
  error_pct: number | null;
  ok: boolean;
  formula_latex: string;
  note: string | null;
}

export interface Run {
  run_id: string;
  index: number;
  scene: Scene;
  frames: Frame[];
  verification: Verification;
  changed: Record<string, [number, number]>;
}

export type EventKind =
  | "upload" | "run" | "change" | "drag_rerun" | "question" | "answer" | "quiz" | "export";

export interface Event {
  t: number;
  kind: EventKind;
  data: Record<string, unknown>;
}

// ---- API request/response bodies ----

export interface UploadResponse {
  image_id: string;
  recognition: Recognition;
  overlay_png_url: string;
}

export interface CornersRequest {
  image_id: string;
  corners: Point[];
}

export type ParamsOverride = Partial<Pick<Params, "angle_deg" | "mass_kg" | "gravity" | "friction" | "motion">>;

export interface RunRequest {
  image_id: string;
  params_override?: ParamsOverride | null;
  reason?: "change" | "drag"; // "drag" logs a drag_rerun event
}

export interface PredictResponse {
  time_s: number | null;
  speed_m_s: number | null;
  will_move: boolean;
  note: string | null;
}

export interface WhyCardRequest {
  run_a: string;
  run_b: string;
}

export interface WhyCard {
  latex: string;
  symbol: string;
  sentence: string;
  cancels: boolean;
}

export interface LearnerProfile {
  level: Level;
  concepts_seen: Record<"mass_independent" | "angle_effect" | "gravity_effect" | "friction_effect" | "roll_effect", boolean>;
  runs: number;
}

export interface SessionState {
  session_id: string;
  runs: Run[];
  events: Event[];
  profile: LearnerProfile;
}

export interface AskRequest {
  question: string;
  level: Level;
}

export interface FollowUp {
  text: string;
  change: ParamsOverride;
  concept: string;
}

export interface AskResponse {
  answer_md: string;
  based_on: string[]; // run ids the answer relies on
  flags: string[]; // numbers matching nothing in the context, as written in the answer
  followups: FollowUp[];
  source: "llm" | "template";
}

export interface SummaryRequest {
  level: Level;
}

export interface SummaryResponse {
  summary_md: string;
  source: "llm" | "template";
}

export interface QuizPrompt {
  quiz_id: string;
  prompt: string;
  param: string;
  new_value: number;
}

export interface QuizAnswerRequest {
  quiz_id: string;
  guess: number;
}

export interface QuizResult {
  actual: number;
  guess: number;
  error_pct: number;
  run: Run | null;
}

export interface ExportResponse {
  gif_url: string;
  share_url: string;
  qr_png_url: string;
}
