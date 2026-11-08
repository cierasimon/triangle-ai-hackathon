import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { api } from "../api";
import type { Run } from "../types";

interface Message {
  role: "user" | "ta";
  md: string;
  flags?: string[];
}

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

export function TaPanel({ runs }: { runs: Run[] }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [thinking, setThinking] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // scroll only the message list, never the page
    const list = endRef.current?.parentElement;
    if (list) list.scrollTop = list.scrollHeight;
  }, [messages, thinking]);

  async function send(text: string) {
    setMessages((m) => [...m, { role: "user", md: text }]);
    setQuestion("");
    setThinking(true);
    try {
      const a = await api.ask({ question: text, level: "beginner" });
      setMessages((m) => [...m, { role: "ta", md: a.answer_md, flags: a.flags }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "ta", md: `*Something went wrong: ${e instanceof Error ? e.message : e}*` }]);
    } finally {
      setThinking(false);
    }
  }

  const ready = runs.length > 0;

  return (
    <aside className="ta-panel" aria-label="Teaching assistant">
      <h2>Ask the TA</h2>
      <div className="ta-messages">
        {messages.length === 0 && (
          <p className="muted">
            {ready ? "Ask anything about your ramp and ball." : "Upload a photo first, then ask about it here."}
          </p>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`ta-msg ${m.role}`}>
            <Markdown md={markFlags(m.md, m.flags)} />
          </div>
        ))}
        {thinking && <p className="muted">thinking…</p>}
        <div ref={endRef} />
      </div>
      <form className="ta-input" onSubmit={(e) => { e.preventDefault(); if (question.trim()) send(question.trim()); }}>
        <input value={question} onChange={(e) => setQuestion(e.target.value)} disabled={!ready}
               placeholder="Ask a question" aria-label="question" />
        <button type="submit" disabled={!ready || thinking || !question.trim()}>Ask</button>
      </form>
    </aside>
  );
}
