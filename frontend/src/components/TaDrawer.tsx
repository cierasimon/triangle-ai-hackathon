import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { api } from "../api";
import type { FollowUp, Level, ParamsOverride, QuizPrompt, Run } from "../types";

interface Message {
  role: "user" | "ta";
  md: string;
  basedOn?: string[];
  flags?: string[];
  source?: "llm" | "template";
}

interface Props {
  runs: Run[];
  busy: boolean;
  onApply: (change: ParamsOverride) => void; // a follow-up button
  onRun: (run: Run) => void; // a quiz created a run
}

const LEVELS: { id: Level; label: string }[] = [
  { id: "beginner", label: "Beginner" },
  { id: "intermediate", label: "Some physics" },
  { id: "confident", label: "Confident" },
];

/** Put a ⚠ after each flagged number (as written in the answer) so it stands out. */
function markFlags(md: string, flags: string[] = []): string {
  let out = md;
  for (const f of flags) out = out.replace(f, `${f} ⚠️`);
  return out;
}

function Markdown({ md }: { md: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
      {md}
    </ReactMarkdown>
  );
}

export function TaDrawer({ runs, busy, onApply, onRun }: Props) {
  const [open, setOpen] = useState(false);
  const [level, setLevel] = useState<Level>("beginner");
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [followups, setFollowups] = useState<FollowUp[]>([]);
  const [quiz, setQuiz] = useState<QuizPrompt | null>(null);
  const [guess, setGuess] = useState("");
  const [thinking, setThinking] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const runLabel = (id: string) => `Run ${runs.find((r) => r.run_id === id)?.index ?? "?"}`;

  useEffect(() => {
    if (open && runs.length) api.followups().then(setFollowups).catch(() => setFollowups([]));
  }, [open, runs.length]);
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages, quiz, thinking]);

  async function guard(task: () => Promise<void>) {
    setThinking(true);
    try {
      await task();
    } catch (e) {
      setMessages((m) => [...m, { role: "ta", md: `*Something went wrong: ${e instanceof Error ? e.message : e}*` }]);
    } finally {
      setThinking(false);
    }
  }

  const send = (text: string) =>
    guard(async () => {
      setMessages((m) => [...m, { role: "user", md: text }]);
      setQuestion("");
      const a = await api.ask({ question: text, level });
      setMessages((m) => [...m, { role: "ta", md: a.answer_md, basedOn: a.based_on, flags: a.flags, source: a.source }]);
      setFollowups(a.followups);
    });

  const summarize = () =>
    guard(async () => {
      setMessages((m) => [...m, { role: "user", md: "Summarize my session" }]);
      const s = await api.summary(level);
      setMessages((m) => [...m, { role: "ta", md: s.summary_md, source: s.source }]);
    });

  const startQuiz = () =>
    guard(async () => {
      const q = await api.quiz();
      setQuiz(q);
      setGuess("");
      setMessages((m) => [...m, { role: "ta", md: `**Quiz:** ${q.prompt}` }]);
    });

  const answerQuiz = () =>
    guard(async () => {
      if (!quiz) return;
      const g = Number(guess);
      if (!Number.isFinite(g) || g <= 0) return;
      setMessages((m) => [...m, { role: "user", md: `My guess: ${g} s` }]);
      const r = await api.quizAnswer(quiz.quiz_id, g);
      setQuiz(null);
      if (r.run) onRun(r.run);
      const verdict = r.error_pct <= 10 ? "Great estimate!" : r.error_pct <= 30 ? "Close." : "Not quite —";
      setMessages((m) => [...m, {
        role: "ta",
        md: `${verdict} The simulation (Run ${r.run?.index}) took **${r.actual.toFixed(2)} s**; ` +
            `your guess was off by ${r.error_pct.toFixed(0)}%.`,
        basedOn: r.run ? [r.run.run_id] : [],
      }]);
    });

  if (!runs.length) return null;

  return (
    <>
      {!open && (
        <button className="ta-fab" onClick={() => setOpen(true)}>Ask the TA</button>
      )}
      {open && (
        <aside className="ta-drawer" aria-label="Teaching assistant">
          <header className="ta-head">
            <strong>Your TA</strong>
            <button className="chip ghost" onClick={() => setOpen(false)} aria-label="close">✕</button>
          </header>
          <div className="ta-levels" role="radiogroup" aria-label="level">
            {LEVELS.map((l) => (
              <button key={l.id} role="radio" aria-checked={level === l.id}
                      className={level === l.id ? "chip active" : "chip"} onClick={() => setLevel(l.id)}>
                {l.label}
              </button>
            ))}
          </div>

          <div className="ta-messages">
            {messages.length === 0 && (
              <p className="muted">
                Ask why something happened, or try a suggestion below. Answers use only your
                runs' numbers; anything else gets a ⚠️.
              </p>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`ta-msg ${m.role}`}>
                <Markdown md={markFlags(m.md, m.flags)} />
                {(m.basedOn?.length || m.flags?.length || m.source) && (
                  <div className="ta-meta">
                    {m.basedOn?.length ? <span>based on {m.basedOn.map(runLabel).join(", ")}</span> : null}
                    {m.flags?.length ? (
                      <span className="ta-flag">⚠️ not from your runs: {m.flags.join(", ")}</span>
                    ) : null}
                    {m.source === "template" && <span>offline answer</span>}
                  </div>
                )}
              </div>
            ))}
            {thinking && <p className="muted">thinking…</p>}
            {quiz && (
              <form className="ta-quiz" onSubmit={(e) => { e.preventDefault(); answerQuiz(); }}>
                <input type="number" inputMode="decimal" step="any" min="0" placeholder="seconds"
                       value={guess} onChange={(e) => setGuess(e.target.value)} aria-label="your guess in seconds" />
                <button type="submit" disabled={thinking || busy}>Check</button>
              </form>
            )}
            <div ref={endRef} />
          </div>

          <div className="ta-actions">
            {followups.map((f) => (
              <button key={f.text} className="chip" disabled={busy}
                      onClick={() => onApply(f.change)}>{f.text}</button>
            ))}
            <button className="chip" disabled={thinking} onClick={summarize}>Summarize my session</button>
            <button className="chip" disabled={thinking || !!quiz} onClick={startQuiz}>Quiz me</button>
            <button className="chip" disabled={thinking} onClick={() => send("What did you read from my drawing?")}>
              What did you read?
            </button>
          </div>
          <form className="ta-input" onSubmit={(e) => { e.preventDefault(); if (question.trim()) send(question.trim()); }}>
            <input value={question} onChange={(e) => setQuestion(e.target.value)}
                   placeholder="Why didn't the mass matter?" aria-label="question" />
            <button type="submit" disabled={thinking || !question.trim()}>Ask</button>
          </form>
        </aside>
      )}
    </>
  );
}
