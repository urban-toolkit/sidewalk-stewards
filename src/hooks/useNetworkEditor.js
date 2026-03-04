import { useEffect, useRef, useState, useCallback } from "react";

const EDGE_SOURCE = "editor-edges-source";
const EDGE_LAYER  = "editor-edges-layer";
const EDGE_HIT    = "editor-edges-hit";
const NODE_SOURCE = "editor-nodes-source";
const NODE_LAYER  = "editor-nodes-layer";
const MESO_ZOOM   = 16;
const MICRO_ZOOM = 18.5;

// ── Parse GeoJSON → node/edge graph + adjacency index ────────────────────────

function parseNetwork(geojson) {
  const byKey = new Map();
  const edges = new Map();
  const nodeEdgeIndex = new Map(); // nodeId → Set<edgeId>
  let nc = 0, ec = 0;

  const getNode = (lng, lat) => {
    const key = `${lng.toFixed(6)},${lat.toFixed(6)}`;
    if (!byKey.has(key)) {
      const id = `n${nc++}`;
      byKey.set(key, { id, lng, lat });
    }
    return byKey.get(key).id;
  };

  for (const f of geojson?.features ?? []) {
    const rings =
      f.geometry.type === "LineString"
        ? [f.geometry.coordinates]
        : f.geometry.coordinates;
    for (const ring of rings) {
      const nodeIds = ring.map(([lng, lat]) => getNode(lng, lat));
      if (nodeIds.length < 2) continue;
      const id = `e${ec++}`;
      edges.set(id, { id, nodeIds });
      for (const nid of nodeIds) {
        if (!nodeEdgeIndex.has(nid)) nodeEdgeIndex.set(nid, new Set());
        nodeEdgeIndex.get(nid).add(id);
      }
    }
  }

  const nodes = new Map([...byKey.values()].map((n) => [n.id, n]));
  return { nodes, edges, nodeEdgeIndex };
}

// ── Build GeoJSON caches (mutable feature objects stored in Maps) ─────────────

function buildCaches({ nodes, edges }) {
  const nodeFeatMap = new Map();
  const edgeFeatMap = new Map();

  const nodeFC = {
    type: "FeatureCollection",
    features: [...nodes.values()].map((n) => {
      const f = {
        type: "Feature",
        properties: { id: n.id },
        geometry: { type: "Point", coordinates: [n.lng, n.lat] },
      };
      nodeFeatMap.set(n.id, f);
      return f;
    }),
  };

  const edgeFC = {
    type: "FeatureCollection",
    features: [...edges.values()].flatMap((e) => {
      const coords = e.nodeIds.map((id) => nodes.get(id)).filter(Boolean).map((n) => [n.lng, n.lat]);
      if (coords.length < 2) return [];
      const f = {
        type: "Feature",
        properties: { id: e.id },
        geometry: { type: "LineString", coordinates: coords },
      };
      edgeFeatMap.set(e.id, f);
      return f;
    }),
  };

  return { nodeFC, edgeFC, nodeFeatMap, edgeFeatMap };
}

// ── Closest segment index (for split) ────────────────────────────────────────

function closestSegmentIdx(nodeIds, nodes, lng, lat) {
  let best = 0, bestD = Infinity;
  for (let i = 0; i < nodeIds.length - 1; i++) {
    const a = nodes.get(nodeIds[i]);
    const b = nodes.get(nodeIds[i + 1]);
    if (!a || !b) continue;
    const d = ((a.lng + b.lng) / 2 - lng) ** 2 + ((a.lat + b.lat) / 2 - lat) ** 2;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

// ── Export current graph state → clean GeoJSON FeatureCollection ──────────────

function exportToGeoJSON({ nodes, edges }) {
  const features = [];
  for (const edge of edges.values()) {
    const coords = edge.nodeIds
      .map((id) => nodes.get(id))
      .filter(Boolean)
      .map((n) => [n.lng, n.lat]);
    if (coords.length < 2) continue;
    features.push({
      type: "Feature",
      properties: {},
      geometry: { type: "LineString", coordinates: coords },
    });
  }
  return { type: "FeatureCollection", features };
}

// ── Hook ─────────────────────────────────────────────────────────────────────

export function useNetworkEditor(mapRef, networkData) {
  const addedRef       = useRef(false);
  const netRef         = useRef({ nodes: new Map(), edges: new Map(), nodeEdgeIndex: new Map() });
  const cacheRef       = useRef({ nodeFC: null, edgeFC: null, nodeFeatMap: new Map(), edgeFeatMap: new Map() });
  const draggingRef    = useRef(null); // { nodeId, origLng, origLat }
  const hoveredNodeRef = useRef(null); // nodeId currently under cursor

  const [contextMenu, setContextMenu] = useState(null);
  const [dirty, setDirty]             = useState(false);   // true when unsaved edits exist
  const [saving, setSaving]           = useState(false);

  // ── Full rebuild: parse + cache + push to map ───────────────────────────────
  const rebuild = (map, geojson) => {
    const net = parseNetwork(geojson);
    const cache = buildCaches(net);
    netRef.current = net;
    cacheRef.current = cache;
    setDirty(false);
    if (map && addedRef.current) {
      map.getSource(EDGE_SOURCE)?.setData(cache.edgeFC);
      map.getSource(NODE_SOURCE)?.setData(cache.nodeFC);
    }
  };

  useEffect(() => {
    if (!networkData) return;
    rebuild(mapRef.current, networkData);
  }, [networkData, mapRef]);

  // ── Helper: mark the network as modified ────────────────────────────────────
  const markDirty = () => setDirty(true);

  // ── Save: export current state → POST to API → write to disk ────────────────
  const saveNetwork = useCallback(async () => {
    const geojson = exportToGeoJSON(netRef.current);
    setSaving(true);
    try {
      const res = await fetch("/api/save-network", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(geojson),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Save failed (${res.status}): ${text}`);
      }
      setDirty(false);
      console.log("Network saved successfully");
    } catch (err) {
      console.error("Failed to save network:", err);
      alert(`Failed to save network: ${err.message}`);
    } finally {
      setSaving(false);
    }
  }, []);

  // ── Mount layers + events ───────────────────────────────────────────────────
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    // ── Hot path: mutate in-place, only touch connected edges ──────────────────
    const onMouseMove = (e) => {
      if (!draggingRef.current) return;
      const { nodeId } = draggingRef.current;
      const { lng, lat } = e.lngLat;
      const { nodes, edges, nodeEdgeIndex } = netRef.current;
      const { nodeFeatMap, edgeFeatMap, nodeFC, edgeFC } = cacheRef.current;

      // 1. Mutate the node object in-place — zero allocation
      const node = nodes.get(nodeId);
      if (!node) return;
      node.lng = lng;
      node.lat = lat;

      // 2. Update the cached GeoJSON Point feature in-place
      const nodeFeat = nodeFeatMap.get(nodeId);
      if (nodeFeat) nodeFeat.geometry.coordinates = [lng, lat];

      // 3. Only update edges connected to this node
      const connectedEdgeIds = nodeEdgeIndex.get(nodeId) ?? new Set();
      for (const eid of connectedEdgeIds) {
        const edge = edges.get(eid);
        const feat = edgeFeatMap.get(eid);
        if (!edge || !feat) continue;
        feat.geometry.coordinates = edge.nodeIds
          .map((id) => nodes.get(id))
          .filter(Boolean)
          .map((n) => [n.lng, n.lat]);
      }

      // 4. Push same FeatureCollection objects — MapLibre re-renders without re-tiling
      map.getSource(NODE_SOURCE)?.setData(nodeFC);
      map.getSource(EDGE_SOURCE)?.setData(edgeFC);
    };

    const onNodeMouseDown = (e) => {
      // Ignore right-clicks — those go to the context menu
      if (e.originalEvent?.button !== 0) return;
      e.preventDefault();
      const nodeId = e.features?.[0]?.properties?.id;
      if (!nodeId) return;
      const n = netRef.current.nodes.get(nodeId);
      if (!n) return;
      draggingRef.current = { nodeId, origLng: n.lng, origLat: n.lat };
      map.dragPan.disable();
      map.getCanvas().style.cursor = "grabbing";
      map.setFeatureState({ source: NODE_SOURCE, id: nodeId }, { dragging: true, hover: false });
    };

    const onMouseUp = (e) => {
      if (!draggingRef.current) return;
      map.dragPan.enable();
      map.getCanvas().style.cursor = "";
      map.setFeatureState({ source: NODE_SOURCE, id: draggingRef.current.nodeId }, { dragging: false });

      const hits = map.queryRenderedFeatures(e.point, { layers: [NODE_LAYER] });
      const target = hits.find((f) => f.properties.id !== draggingRef.current.nodeId);

      if (target) {
        // Snap dragged node back to its original position
        const { nodeId: fromId, origLng, origLat } = draggingRef.current;
        const node = netRef.current.nodes.get(fromId);
        if (node) {
          node.lng = origLng;
          node.lat = origLat;
          const feat = cacheRef.current.nodeFeatMap.get(fromId);
          if (feat) feat.geometry.coordinates = [origLng, origLat];
          // Re-sync connected edges back too
          const { edges, nodeEdgeIndex } = netRef.current;
          for (const eid of nodeEdgeIndex.get(fromId) ?? []) {
            const edge = edges.get(eid);
            const edgeFeat = cacheRef.current.edgeFeatMap.get(eid);
            if (edge && edgeFeat) {
              edgeFeat.geometry.coordinates = edge.nodeIds
                .map((id) => netRef.current.nodes.get(id))
                .filter(Boolean)
                .map((n) => [n.lng, n.lat]);
            }
          }
        }

        // Add a new edge between the two nodes
        const toId = target.properties.id;
        const newEId = `e_conn_${Date.now()}`;
        const newEdge = { id: newEId, nodeIds: [fromId, toId] };
        netRef.current.edges.set(newEId, newEdge);
        for (const nid of [fromId, toId]) {
          if (!netRef.current.nodeEdgeIndex.has(nid)) netRef.current.nodeEdgeIndex.set(nid, new Set());
          netRef.current.nodeEdgeIndex.get(nid).add(newEId);
        }
        const coords = [fromId, toId]
          .map((id) => netRef.current.nodes.get(id))
          .filter(Boolean)
          .map((n) => [n.lng, n.lat]);
        const newFeat = { type: "Feature", properties: { id: newEId }, geometry: { type: "LineString", coordinates: coords } };
        cacheRef.current.edgeFeatMap.set(newEId, newFeat);
        cacheRef.current.edgeFC.features.push(newFeat);
      }

      // If the node actually moved (not a connect), mark dirty
      const { nodeId: dragId, origLng, origLat } = draggingRef.current;
      const draggedNode = netRef.current.nodes.get(dragId);
      if (draggedNode && (draggedNode.lng !== origLng || draggedNode.lat !== origLat)) {
        markDirty();
      }
      // If a new edge was created via snap, also dirty
      if (target) markDirty();

      map.getSource(NODE_SOURCE)?.setData(cacheRef.current.nodeFC);
      map.getSource(EDGE_SOURCE)?.setData(cacheRef.current.edgeFC);
      draggingRef.current = null;
    };

    const onEdgeContextMenu = (e) => {
      e.preventDefault();
      const edgeId = e.features?.[0]?.properties?.id;
      if (!edgeId) return;
      setContextMenu({ type: "edge", edgeId, x: e.point.x, y: e.point.y, lng: e.lngLat.lng, lat: e.lngLat.lat });
    };

    // ── Right-click on a node ─────────────────────────────────────────────────
    const onNodeContextMenu = (e) => {
      e.preventDefault();
      if (draggingRef.current) return;
      const nodeId = e.features?.[0]?.properties?.id;
      if (!nodeId) return;
      setContextMenu({ type: "node", nodeId, x: e.point.x, y: e.point.y });
    };

    const onMapClick = () => setContextMenu(null);

    const onNodeEnter = (e) => {
      if (draggingRef.current) return;
      const nodeId = e.features?.[0]?.properties?.id;
      if (!nodeId) return;
      hoveredNodeRef.current = nodeId;
      map.setFeatureState({ source: NODE_SOURCE, id: nodeId }, { hover: true });
      map.getCanvas().style.cursor = "grab";
    };
    const onNodeLeave = () => {
      if (draggingRef.current) return;
      if (hoveredNodeRef.current) {
        map.setFeatureState({ source: NODE_SOURCE, id: hoveredNodeRef.current }, { hover: false });
        hoveredNodeRef.current = null;
      }
      map.getCanvas().style.cursor = "";
    };
    const onEdgeEnter = () => { if (!draggingRef.current) map.getCanvas().style.cursor = "pointer"; };
    const onEdgeLeave = () => { if (!draggingRef.current) map.getCanvas().style.cursor = ""; };

    let cancelled = false;

    const init = () => {
      if (cancelled || addedRef.current) return;
      const { nodeFC, edgeFC } = cacheRef.current;

      map.addSource(EDGE_SOURCE, { type: "geojson", data: edgeFC ?? { type: "FeatureCollection", features: [] } });
      map.addLayer({
        id: EDGE_LAYER, type: "line", source: EDGE_SOURCE, minzoom: MESO_ZOOM,
        paint: {
          "line-color": "#e85d04",
          "line-width": ["interpolate", ["linear"], ["zoom"], 16, 3, 18, 4, 20, 5],
          "line-opacity": 1,
        },
      });
      map.addLayer({
        id: EDGE_HIT, type: "line", source: EDGE_SOURCE, minzoom: MICRO_ZOOM,
        paint: { "line-width": 14, "line-opacity": 0 },
      });

      map.addSource(NODE_SOURCE, { type: "geojson", promoteId: "id", data: nodeFC ?? { type: "FeatureCollection", features: [] } });
      map.addLayer({
        id: NODE_LAYER, type: "circle", source: NODE_SOURCE, minzoom: MICRO_ZOOM,
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 18, 4, 20, 7],
          "circle-color": [
            "case",
            ["boolean", ["feature-state", "dragging"], false], "#ff0000",
            ["boolean", ["feature-state", "hover"],    false], "#ffcc00",
            "#e85d04",
          ],
          "circle-stroke-width": 2,
          "circle-stroke-color": "#ffffff",
        },
      });

      addedRef.current = true;

      map.on("mousedown",   NODE_LAYER, onNodeMouseDown);
      map.on("mousemove",              onMouseMove);
      map.on("mouseup",                onMouseUp);
      map.on("contextmenu", EDGE_HIT,  onEdgeContextMenu);
      map.on("contextmenu", NODE_LAYER, onNodeContextMenu);
      map.on("click",                  onMapClick);
      map.on("mouseenter", NODE_LAYER, onNodeEnter);
      map.on("mouseleave", NODE_LAYER, onNodeLeave);
      map.on("mouseenter", EDGE_HIT,   onEdgeEnter);
      map.on("mouseleave", EDGE_HIT,   onEdgeLeave);
    };

    if (map.isStyleLoaded()) init();
    else map.once("load", init);

    return () => {
      cancelled = true;
      draggingRef.current = null;
      map.off("load", init);
      map.off("mousedown",   NODE_LAYER, onNodeMouseDown);
      map.off("mousemove",              onMouseMove);
      map.off("mouseup",                onMouseUp);
      map.off("contextmenu", EDGE_HIT,  onEdgeContextMenu);
      map.off("contextmenu", NODE_LAYER, onNodeContextMenu);
      map.off("click",                  onMapClick);
      map.off("mouseenter", NODE_LAYER, onNodeEnter);
      map.off("mouseleave", NODE_LAYER, onNodeLeave);
      map.off("mouseenter", EDGE_HIT,   onEdgeEnter);
      map.off("mouseleave", EDGE_HIT,   onEdgeLeave);
      try {
        [NODE_LAYER, EDGE_HIT, EDGE_LAYER].forEach((l) => { if (map.getLayer(l)) map.removeLayer(l); });
        [NODE_SOURCE, EDGE_SOURCE].forEach((s) => { if (map.getSource(s)) map.removeSource(s); });
      } catch { /* map may be gone */ }
      addedRef.current = false;
    };
  }, [mapRef]);

  // ── Split edge ──────────────────────────────────────────────────────────────
  const splitEdge = (edgeId, lng, lat) => {
    const { nodes, edges, nodeEdgeIndex } = netRef.current;
    const { edgeFeatMap, edgeFC, nodeFeatMap, nodeFC } = cacheRef.current;
    const edge = edges.get(edgeId);
    if (!edge) return;

    const idx = closestSegmentIdx(edge.nodeIds, nodes, lng, lat);
    const ts = Date.now();
    const newNodeId = `n_split_${ts}`;
    const newNode = { id: newNodeId, lng, lat };
    nodes.set(newNodeId, newNode);

    // Add node GeoJSON feature
    const newNodeFeat = { type: "Feature", properties: { id: newNodeId }, geometry: { type: "Point", coordinates: [lng, lat] } };
    nodeFeatMap.set(newNodeId, newNodeFeat);
    nodeFC.features.push(newNodeFeat);

    // Split edge into two
    const eA = { id: `e_${ts}a`, nodeIds: [...edge.nodeIds.slice(0, idx + 1), newNodeId] };
    const eB = { id: `e_${ts}b`, nodeIds: [newNodeId, ...edge.nodeIds.slice(idx + 1)] };
    edges.delete(edgeId);
    edges.set(eA.id, eA);
    edges.set(eB.id, eB);

    // Update adjacency index
    for (const nid of edge.nodeIds) nodeEdgeIndex.get(nid)?.delete(edgeId);
    nodeEdgeIndex.set(newNodeId, new Set([eA.id, eB.id]));
    for (const e of [eA, eB]) {
      for (const nid of e.nodeIds) {
        if (!nodeEdgeIndex.has(nid)) nodeEdgeIndex.set(nid, new Set());
        nodeEdgeIndex.get(nid).add(e.id);
      }
    }

    // Remove old edge feature, add two new ones
    const oldFeatIdx = edgeFC.features.findIndex((f) => f.properties.id === edgeId);
    if (oldFeatIdx !== -1) edgeFC.features.splice(oldFeatIdx, 1);
    edgeFeatMap.delete(edgeId);

    for (const e of [eA, eB]) {
      const coords = e.nodeIds.map((id) => nodes.get(id)).filter(Boolean).map((n) => [n.lng, n.lat]);
      const f = { type: "Feature", properties: { id: e.id }, geometry: { type: "LineString", coordinates: coords } };
      edgeFeatMap.set(e.id, f);
      edgeFC.features.push(f);
    }

    const map = mapRef.current;
    if (map) {
      map.getSource(NODE_SOURCE)?.setData(nodeFC);
      map.getSource(EDGE_SOURCE)?.setData(edgeFC);
    }
    setContextMenu(null);
    markDirty();
  };

  // ── Delete node ─────────────────────────────────────────────────────────────
  const deleteNode = (nodeId) => {
    const { nodes, edges, nodeEdgeIndex } = netRef.current;
    const { nodeFeatMap, nodeFC, edgeFeatMap, edgeFC } = cacheRef.current;

    // 1. Remove all edges connected to this node
    const connectedEdgeIds = new Set(nodeEdgeIndex.get(nodeId) ?? []);
    for (const eid of connectedEdgeIds) {
      edges.delete(eid);
      edgeFeatMap.delete(eid);
      const idx = edgeFC.features.findIndex((f) => f.properties.id === eid);
      if (idx !== -1) edgeFC.features.splice(idx, 1);
    }

    // Clean up the adjacency index for all other nodes that referenced those edges
    for (const [nid, edgeSet] of nodeEdgeIndex) {
      for (const eid of connectedEdgeIds) edgeSet.delete(eid);
    }

    // 2. Remove the node itself
    nodes.delete(nodeId);
    nodeEdgeIndex.delete(nodeId);
    nodeFeatMap.delete(nodeId);
    const nIdx = nodeFC.features.findIndex((f) => f.properties.id === nodeId);
    if (nIdx !== -1) nodeFC.features.splice(nIdx, 1);

    // 3. Push updated data to the map
    const map = mapRef.current;
    if (map) {
      map.getSource(NODE_SOURCE)?.setData(nodeFC);
      map.getSource(EDGE_SOURCE)?.setData(edgeFC);
    }
    setContextMenu(null);
    markDirty();
  };

  return { contextMenu, setContextMenu, splitEdge, deleteNode, saveNetwork, dirty, saving };
}