import { useEffect, useState } from "react";
import type { GraspGuidance } from "./types";

export function GraspGuidanceHUD({ guidance }: { guidance?: GraspGuidance }) {
  const [, setTick] = useState(0);
  useEffect(() => {
    const timer = window.setInterval(() => setTick((tick) => tick + 1), 100);
    return () => window.clearInterval(timer);
  }, []);
  const age = guidance?.wall_time == null ? Infinity : Date.now() / 1000 - guidance.wall_time;
  const fresh = guidance?.available && age >= -0.1 && age <= 0.25;
  const state = fresh ? guidance.state : "UNAVAILABLE";
  const label = state === "READY" ? "GRIP NOW" : state === "NEAR" ? "ALIGN" : state === "NOT_READY" ? "NOT READY" : "UNAVAILABLE";
  return <div className={`grasp-guidance grasp-${state.toLowerCase()}`} role="status" aria-live="polite" aria-label="Grasp readiness guidance">
    <div className="grasp-guidance-state"><small>GRASP</small><strong>{label}</strong></div>
    <div className="grasp-guidance-errors">
      {fresh && state !== "READY" && (guidance.corrections ?? []).map((item) => <span key={item.label}>
        <small>{item.label}</small><b>{item.error > 0 ? "+" : ""}{item.error.toFixed(1)} {item.unit === "deg" ? "°" : "mm"}</b><em>{item.direction}</em>
      </span>)}
      {fresh && guidance.reason && <span className="grasp-safety-reason">{guidance.reason}</span>}
      {fresh && state === "READY" && <span className="grasp-ready-instruction">Close gripper manually</span>}
    </div>
    <small className="grasp-guidance-note">{fresh ? `Advisory · object local frame${guidance.tolerance_status === "PROVISIONAL_UI_TOLERANCE" ? " · provisional" : ""}` : "Guidance unavailable"}</small>
  </div>;
}
