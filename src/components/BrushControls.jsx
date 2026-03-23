export function BrushControls({
  brushActive,
  selectedCount,
  clearAll,
  runModel,
  inferencePhase,
  inferenceMessage,
  dismissInference,
}) {
  return (
    <div style={{
      display:       "flex",
      flexDirection: "column",
      alignItems:    "flex-start",
      gap:           8,
      pointerEvents: "none",
    }}>

      {/* ── Active indicator — only while Shift is held ── */}
      {brushActive && (
        <div style={{
          pointerEvents: "none",
          display:       "flex",
          alignItems:    "center",
          gap:           5,
          fontSize:      11,
          color:         "#4a90d9",
          background:    "#eaf3fc",
          border:        "1px solid #c2daf5",
          borderRadius:  6,
          padding:       "5px 9px",
          boxShadow:     "0 1px 3px rgba(0,0,0,0.08)",
          whiteSpace:    "nowrap",
          userSelect:    "none",
        }}>
          <CrosshairIcon size={12} />
          Selecting…
        </div>
      )}

      {/* ── "Hold ⇧" hint — only when nothing is selected and brush is idle ── */}
      {!brushActive && selectedCount === 0 && inferencePhase === "idle" && (
        <div style={{
          pointerEvents: "none",
          fontSize:      11,
          color:         "#aaa",
          background:    "#fff",
          border:        "1px solid #e8e8e8",
          borderRadius:  6,
          padding:       "5px 9px",
          boxShadow:     "0 1px 3px rgba(0,0,0,0.06)",
          whiteSpace:    "nowrap",
          userSelect:    "none",
        }}>
          Hold ⇧ to select tiles
        </div>
      )}

      {/* ── Clear all — whenever tiles are selected ── */}
      {selectedCount > 0 && (
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

      {/* ── Apply model — visible when tiles are selected and idle ── */}
      {selectedCount > 0 && inferencePhase === "idle" && (
        <button
          onClick={runModel}
          style={{
            pointerEvents: "all",
            fontSize:      11,
            fontWeight:    600,
            padding:       "5px 12px",
            borderRadius:  6,
            border:        "1px solid #ccc",
            background:    "#fff",
            color:         "#333",
            cursor:        "pointer",
            boxShadow:     "0 1px 4px rgba(0,0,0,0.10)",
            whiteSpace:    "nowrap",
          }}
        >
          Apply model · {selectedCount}
        </button>
      )}

      {/* ── Running — indeterminate progress bar + live stdout message ── */}
      {inferencePhase === "running" && (
        <div style={{
          pointerEvents: "none",
          display:       "flex",
          flexDirection: "column",
          gap:           6,
          fontSize:      11,
          color:         "#555",
          background:    "#fff",
          border:        "1px solid #ddd",
          borderRadius:  6,
          padding:       "8px 10px",
          boxShadow:     "0 1px 3px rgba(0,0,0,0.08)",
          minWidth:      170,
          maxWidth:      230,
        }}>
          {/* Spinner + label */}
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <Spinner />
            <span>Running…</span>
          </div>

          {/* Indeterminate progress bar */}
          <div style={{
            height:       3,
            background:   "#f0f0f0",
            borderRadius: 2,
            overflow:     "hidden",
            position:     "relative",
          }}>
            <div style={{
              position:     "absolute",
              top:          0,
              left:         0,
              width:        "35%",
              height:       "100%",
              background:   "#333",
              borderRadius: 2,
              animation:    "inferenceSlide 1.4s ease-in-out infinite",
            }} />
          </div>

          {/* Latest stdout line from backend */}
          {inferenceMessage && (
            <div style={{
              fontSize:     10,
              color:        "#aaa",
              overflow:     "hidden",
              textOverflow: "ellipsis",
              whiteSpace:   "nowrap",
              maxWidth:     210,
            }}>
              {inferenceMessage}
            </div>
          )}
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

function CrosshairIcon({ size }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none"
      stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
      <circle cx="8" cy="8" r="3" />
      <line x1="8" y1="1" x2="8" y2="4" />
      <line x1="8" y1="12" x2="8" y2="15" />
      <line x1="1" y1="8" x2="4" y2="8" />
      <line x1="12" y1="8" x2="15" y2="8" />
    </svg>
  );
}

function Spinner() {
  return (
    <span style={{
      display:      "inline-block",
      width:        10,
      height:       10,
      border:       "1.5px solid #ddd",
      borderTop:    "1.5px solid #555",
      borderRadius: "50%",
      animation:    "spin 0.8s linear infinite",
      flexShrink:   0,
    }} />
  );
}

function DismissBtn({ onClick }) {
  return (
    <button onClick={onClick} style={{
      background: "none",
      border:     "none",
      fontSize:   11,
      color:      "#bbb",
      cursor:     "pointer",
      padding:    "0 2px",
      lineHeight: 1,
    }}>✕</button>
  );
}