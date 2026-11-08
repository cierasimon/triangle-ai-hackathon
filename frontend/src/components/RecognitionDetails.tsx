import type { Recognition } from "../types";

export function RecognitionDetails({ recognition: rec }: { recognition: Recognition }) {
  const hybrid = rec.recognition_mode === "hybrid";
  const status = !hybrid ? "OpenCV only" : rec.gemini_status === "success"
    ? "Gemini + OpenCV" : "Gemini unavailable — using OpenCV results";
  return <div>
    <p role="status"><strong>{status}</strong> · {rec.balls.length} confirmed ball(s), {rec.blocks.length} confirmed block(s)</p>
    {rec.gemini_model && hybrid && <p className="muted">Vision model: {rec.gemini_model}</p>}
    {rec.gemini_summary && <details><summary>Gemini reference description (may contain errors)</summary>
      <p>{rec.gemini_summary}</p></details>}
    {!!rec.objects.length && <details><summary>Object proposals and geometry checks</summary>
      <ul>{rec.objects.map((obj,i) => <li key={i}>{obj.kind} · {(obj.confidence*100).toFixed(0)}% model confidence
        {' · '}{obj.body_id ?? 'unassigned'} · {obj.geometry_verified ? 'geometry verified' : 'unverified proposal'}</li>)}</ul>
    </details>}
    {!!rec.parameters.length && <details open><summary>Diagram parameters — check values and body associations</summary>
      <table><thead><tr><th>Label</th><th>Target / value</th><th>Application status</th></tr></thead>
        <tbody>{rec.parameters.map((param,i) => <tr key={i}><td>{param.text}</td>
          <td>{param.target} · {param.key} = {param.value ?? 'unknown'} {param.unit}</td>
          <td>{param.applied ? 'Passed to simulation' : param.validated ? 'Read only — not simulated' : 'Needs confirmation'}
            <br/>{param.note}</td></tr>)}</tbody></table>
      <p className="muted">Model confidence is not verified accuracy. Applied labels set initial values;
        simulation controls override them. Unknown or ambiguous values are not applied.</p>
    </details>}
    {!rec.ball && <p className="muted">No primary ball geometry was confirmed. Review the overlay or upload another photo;
      a simulated ball will not be invented for this image.</p>}
    {(rec.balls.length > 1 || rec.blocks.length > 0) && <p className="muted">
      All confirmed objects are shown in the photo overlay. The current simulation uses only the primary ball;
      other balls and blocks do not participate in collisions yet.</p>}
  </div>;
}
