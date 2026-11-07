import katex from "katex";
import "katex/dist/katex.min.css";

export function Formula({ latex }: { latex: string }) {
  const html = katex.renderToString(latex, { throwOnError: false, displayMode: true });
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
}
