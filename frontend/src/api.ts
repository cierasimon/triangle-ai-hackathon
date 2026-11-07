// Thin typed client for the backend. The session lives in a cookie set by the backend.
import type {
  AskRequest, AskResponse, CornersRequest, ExportResponse, FollowUp, Level, PredictResponse,
  QuizPrompt, QuizResult, Recognition, Run, RunRequest, SessionState, SummaryResponse,
  UploadResponse, WhyCard, WhyCardRequest,
} from "./types";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, { credentials: "include", ...init });
  if (!res.ok) {
    const text = await res.text();
    let detail = text;
    try {
      detail = JSON.parse(text).detail ?? text;
    } catch {
      // not JSON: keep the raw text
    }
    throw new Error(typeof detail === "string" ? detail : `${res.status} ${text}`);
  }
  return res.json() as Promise<T>;
}

const post = <T>(path: string, body?: unknown) =>
  call<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const api = {
  upload(photo: File) {
    const form = new FormData();
    form.append("photo", photo);
    return call<UploadResponse>("/upload", { method: "POST", body: form });
  },
  corners: (body: CornersRequest) => post<Recognition>("/corners", body),
  run: (body: RunRequest) => post<Run>("/runs", body),
  predict: (q: { angle_deg: number; length_m: number; mass_kg?: number; gravity?: number; friction?: number; motion?: string }) =>
    call<PredictResponse>("/predict?" + new URLSearchParams(
      Object.entries(q).filter(([, v]) => v !== undefined).map(([k, v]) => [k, String(v)]),
    )),
  whycard: (body: WhyCardRequest) => post<WhyCard>("/whycard", body),
  session: () => call<SessionState>("/session"),
  ask: (body: AskRequest) => post<AskResponse>("/ta/ask", body),
  summary: (level: Level) => post<SummaryResponse>("/ta/summary", { level }),
  followups: () => call<FollowUp[]>("/ta/followups"),
  quiz: () => post<QuizPrompt>("/ta/quiz"),
  quizAnswer: (quiz_id: string, guess: number) => post<QuizResult>("/ta/quiz/answer", { quiz_id, guess }),
  exportRun: (runId: string) => post<ExportResponse>(`/export/${runId}`),
};
