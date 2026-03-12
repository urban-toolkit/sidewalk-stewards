import { useState } from "react";
import { NetworkOverlay } from "./NetworkOverlay";
import { SuggestionOverlay } from "./SuggestionOverlay";

/**
 * TileRow displays a tile thumbnail with optional network overlay + suggestion cards.
 *
 * Props:
 *   tile            – { z, x, y, id }
 *   meta            – metadata record for the tile (or undefined)
 *   sortKey         – active metric key
 *   onClick         – click handler for the thumbnail
 *   showSuggestions – whether to render the suggestion cards (meso view)
 *   networkData     – parsed GeoJSON FeatureCollection (or null)
 *   thumbSize       – thumbnail pixel size (default 160)
 *   tileSuggestions – Map<nSuggestion, Feature[]> for this tile (or null)
 */
export function TileRow({
  tile,
  meta,
  sortKey,
  onClick,
  showSuggestions,
  networkData,
  thumbSize = 160,
  tileSuggestions = null,
}) {
  const imgUrl = `/tiles/${tile.z}/${tile.x}/${tile.y}.jpg`;
  const [hidden, setHidden] = useState(false);

  if (hidden) return null;

  const val = meta?.[sortKey];
  const valStr =
    val === undefined || val === null
      ? "—"
      : sortKey.startsWith("mean_")
      ? Number(val).toFixed(2)
      : String(Number(val));

  // Build suggestion entries sorted by n_suggestion (exclude 0 = originals)
  const suggestionEntries = tileSuggestions
    ? [...tileSuggestions.entries()]
        .filter(([n]) => n > 0)
        .sort(([a], [b]) => a - b)
    : [];

  // Original (n_suggestion=0) polygons for the main thumbnail
  const originalFeatures = tileSuggestions?.get(0) ?? null;

  return (
    <div className="row">
      {/* ── Main tile thumbnail ── */}
      <div
        className={`thumbWrap ${onClick ? "tileClickable" : ""}`}
        onClick={onClick}
        style={{ width: thumbSize, minWidth: thumbSize }}
      >
        <div className="thumbContainer" style={{ width: thumbSize, height: thumbSize }}>
          <img
            src={imgUrl}
            className="thumb"
            style={{ width: thumbSize, height: thumbSize }}
            loading="lazy"
            onError={() => setHidden(true)}
          />
          {networkData && (
            <NetworkOverlay tile={tile} networkData={networkData} size={thumbSize} />
          )}
          {originalFeatures && (
            <SuggestionOverlay tile={tile} features={originalFeatures} size={thumbSize} />
          )}
        </div>
        <div className="tileLabel">
          {tile.id} · {valStr}
        </div>
      </div>

      {/* ── Suggestion cards (meso horizontal scroller) ── */}
      {showSuggestions && (
        <div className="suggestionsWrap">
          <div className="suggestionsScroller">
            {suggestionEntries.length > 0
              ? suggestionEntries.map(([n, features]) => (
                  <div
                    key={n}
                    className="suggestionCard"
                    style={{
                      padding: 0,
                      overflow: "hidden",
                      minWidth: thumbSize,
                      width: thumbSize,
                      height: thumbSize,
                      borderRadius: 8,
                      flexShrink: 0,
                    }}
                  >
                    <div
                      className="thumbContainer"
                      style={{ width: thumbSize, height: thumbSize, borderRadius: 0 }}
                    >
                      <img
                        src={imgUrl}
                        style={{ width: thumbSize, height: thumbSize, display: "block" }}
                        loading="lazy"
                      />
                      {networkData && (
                        <NetworkOverlay tile={tile} networkData={networkData} size={thumbSize} />
                      )}
                      <SuggestionOverlay tile={tile} features={features} size={thumbSize} />
                    </div>
                  </div>
                ))
              : (
                  <div className="mesoNoSuggestions">
                    No suggestions for this area
                  </div>
                )}
          </div>
        </div>
      )}
    </div>
  );
}