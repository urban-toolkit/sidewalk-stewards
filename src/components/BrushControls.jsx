// BrushControls — floats over the map at meso/micro zoom.
// Rendered inside a position:absolute wrapper in App.jsx, which sits
// within .leftPane (position:relative) so it overlays the map canvas.

export function BrushControls({
  brushActive,
  toggleBrush,
  selectedCount,
  clearAll,
  runModel,
  inferencePhase,
  dismissInference,
}) {
  const isRunning = inferencePhase === "running";

  return (
    <div style={{
      display:        "flex",
      flexDirection:  "column",
      alignItems:     "flex-start",
      gap:            8,
      pointerEvents:  "none",   // wrapper transparent; buttons below opt in
    }}>

      {/* ── Brush toggle ── */}
      <button
        onClick={toggleBrush}
        title={brushActive ? "Deactivate brush" : "Select tiles"}
        style={{
          pointerEvents: "all",
          width:         34,
          height:        34,
          borderRadius:  6,
          border:        brushActive ? "1.5px solid #4a90d9" : "1px solid #ddd",
          background:    brushActive ? "#eaf3fc" : "#fff",
          color:         brushActive ? "#4a90d9" : "#666",
          cursor:        "pointer",
          display:       "flex",
          alignItems:    "center",
          justifyContent:"center",
          boxShadow:     "0 1px 3px rgba(0,0,0,0.1)",
          transition:    "border-color 0.12s, background 0.12s, color 0.12s",
        }}
      >
        <GridIcon size={16} />
      </button>

      {/* ── Clear all — only when brush active and tiles selected ── */}
      {brushActive && selectedCount > 0 && (
        <button
          onClick={clearAll}
          style={{
            pointerEvents: "all",
            fontSize:      11,
            padding:       "4px 9px",
            borderRadius:  5,
            border:        "1px solid #ddd",
            background:    "#fff",
            color:         "#666",
            cursor:        "pointer",
            boxShadow:     "0 1px 3px rgba(0,0,0,0.08)",
            whiteSpace:    "nowrap",
          }}
        >
          Clear ({selectedCount})
        </button>
      )}

      {/* ── Run model — visible whenever tiles are selected ── */}
      {selectedCount > 0 && inferencePhase === "idle" && (
        <button
          onClick={runModel}
          style={{
            pointerEvents: "all",
            fontSize:      11,
            fontWeight:    600,
            padding:       "5px 12px",
            borderRadius:  6,
            border:        "none",
            background:    "#333",
            color:         "#fff",
            cursor:        "pointer",
            boxShadow:     "0 1px 4px rgba(0,0,0,0.15)",
            whiteSpace:    "nowrap",
          }}
        >
          Apply model · {selectedCount}
        </button>
      )}

      {/* ── Running status ── */}
      {inferencePhase === "running" && (
        <div style={{
          pointerEvents: "none",
          display:       "flex",
          alignItems:    "center",
          gap:           6,
          fontSize:      11,
          color:         "#555",
          background:    "#fff",
          border:        "1px solid #ddd",
          borderRadius:  6,
          padding:       "5px 10px",
          boxShadow:     "0 1px 3px rgba(0,0,0,0.08)",
          whiteSpace:    "nowrap",
        }}>
          <Spinner />
          Running…
        </div>
      )}

      {/* ── Done ── */}
      {inferencePhase === "done" && (
        <div style={{
          pointerEvents: "all",
          display:       "flex",
          alignItems:    "center",
          gap:           6,
          fontSize:      11,
          color:         "#2d7a4f",
          background:    "#fff",
          border:        "1px solid #ddd",
          borderRadius:  6,
          padding:       "5px 10px",
          boxShadow:     "0 1px 3px rgba(0,0,0,0.08)",
          whiteSpace:    "nowrap",
        }}>
          Done
          <DismissBtn onClick={dismissInference} />
        </div>
      )}

      {/* ── Error ── */}
      {inferencePhase === "error" && (
        <div style={{
          pointerEvents: "all",
          display:       "flex",
          alignItems:    "center",
          gap:           6,
          fontSize:      11,
          color:         "#a94442",
          background:    "#fff",
          border:        "1px solid #ddd",
          borderRadius:  6,
          padding:       "5px 10px",
          boxShadow:     "0 1px 3px rgba(0,0,0,0.08)",
          whiteSpace:    "nowrap",
        }}>
          Error — check console
          <DismissBtn onClick={dismissInference} />
        </div>
      )}
    </div>
  );
}

// ── Small helpers ─────────────────────────────────────────────────────────────

function GridIcon({ size }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none"
      stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
      <rect x="1" y="1" width="6" height="6" rx="1" />
      <rect x="9" y="1" width="6" height="6" rx="1" />
      <rect x="1" y="9" width="6" height="6" rx="1" />
      <rect x="9" y="9" width="6" height="6" rx="1" />
    </svg>
  );
}

function Spinner() {
  return (
    <span style={{
      display:     "inline-block",
      width:       10,
      height:      10,
      border:      "1.5px solid #ddd",
      borderTop:   "1.5px solid #555",
      borderRadius:"50%",
      animation:   "spin 0.8s linear infinite",
      flexShrink:  0,
    }} />
  );
}

function DismissBtn({ onClick }) {
  return (
    <button onClick={onClick} style={{
      background:  "none",
      border:      "none",
      fontSize:    11,
      color:       "#bbb",
      cursor:      "pointer",
      padding:     "0 2px",
      lineHeight:  1,
    }}>✕</button>
  );
}