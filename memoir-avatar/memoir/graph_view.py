"""An interactive picture of the memory, drawn by us in SVG (no CDN, no pyvis).

Python decides *where* things go; JavaScript only draws and handles the mouse.

  layout()      networkx.spring_layout: a physics simulation where edges are
                springs and nodes repel, so connected things end up close.
                Seeded, so the picture doesn't jump around between reruns.
  the component draws nodes (size = mentions, colour = type, ring = used to
                answer the last message) and labelled arrows (dashed grey =
                retired fact). Drag to pan, scroll to zoom, hover for the
                description, click a node to list its facts in Python.
"""

from __future__ import annotations

import math

import networkx as nx
import streamlit as st

from .graph import KnowledgeGraph

TYPE_COLORS = {
    "person": "#5b6cff", "place": "#18a999", "organization": "#f08a24",
    "project": "#d64f8a", "topic": "#8a63d2", "skill": "#3c9be0", "interest": "#e0b33c",
    "event": "#ef6461", "object": "#7f8c8d", "other": "#9aa5b1",
}


def layout(graph: KnowledgeGraph, highlight: set[str] | None = None,
           width: int = 900, height: int = 560) -> dict:
    """The graph as JSON-safe nodes (with x/y in pixels) and edges."""
    g = graph.to_networkx()
    highlight = highlight or set()
    if g.number_of_nodes() == 0:
        return {"nodes": [], "edges": [], "w": width, "h": height}
    # k = ideal edge length; larger graphs need it smaller to fit the box.
    k = 1.6 / math.sqrt(max(g.number_of_nodes(), 1))
    pos = nx.spring_layout(nx.Graph(g), k=k, seed=7, iterations=200)
    pad = 60
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    sx = lambda x: pad + (x - min(xs)) / ((max(xs) - min(xs)) or 1) * (width - 2 * pad)   # noqa: E731
    sy = lambda y: pad + (y - min(ys)) / ((max(ys) - min(ys)) or 1) * (height - 2 * pad)  # noqa: E731
    nodes = [{
        "id": n,
        "x": round(sx(pos[n][0]), 1), "y": round(sy(pos[n][1]), 1),
        "r": round(10 + 4 * math.sqrt(max(d.get("mentions", 1), 1)), 1),
        "color": TYPE_COLORS.get(d.get("type", "other"), TYPE_COLORS["other"]),
        "type": d.get("type", "other"),
        "tip": (d.get("description") or "").strip(),
        "hl": n in highlight,
    } for n, d in g.nodes(data=True)]
    edges = [{"s": u, "t": v, "label": d["predicate"].replace("_", " "), "active": d["active"]}
             for u, v, d in g.edges(data=True)]
    return {"nodes": nodes, "edges": edges, "w": width, "h": height}


VIEW_CSS = """
.memoir-graph svg { width:100%; height:auto; border-radius:12px; cursor:grab;
  background: var(--st-secondary-background-color, #f5f6fa); touch-action:none; }
.memoir-graph svg:active { cursor:grabbing; }
.memoir-graph .lbl { font: 600 14px system-ui, sans-serif; fill: var(--st-text-color, #222);
  paint-order: stroke; stroke: var(--st-secondary-background-color, #f5f6fa); stroke-width: 4px; }
.memoir-graph .elbl { font: 12px system-ui, sans-serif; fill: var(--st-text-color, #555); opacity:.75;
  paint-order: stroke; stroke: var(--st-secondary-background-color, #f5f6fa); stroke-width: 3px; }
.memoir-graph .node { cursor:pointer; }
.memoir-graph .empty { font: 14px system-ui, sans-serif; opacity:.6; padding:2rem; text-align:center; }
"""

VIEW_JS = """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs = {}) => { const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; };

  // Start fresh on every data change (this function runs again with new data).
  let host = parentElement.querySelector('.memoir-graph');
  if (!host) { host = document.createElement('div'); host.className = 'memoir-graph'; parentElement.appendChild(host); }
  host.replaceChildren();
  if (!data || !data.nodes || !data.nodes.length) {
    const p = document.createElement('div'); p.className = 'empty';
    p.textContent = 'Memory is empty. Tell Mira about yourself and watch it grow.';
    host.appendChild(p); return;
  }

  const svg = el('svg', { viewBox: `0 0 ${data.w} ${data.h}` });
  const defs = el('defs');
  for (const [id, color] of [['arrow', '#8892a6'], ['arrow-old', '#c4c9d4']]) {
    const m = el('marker', { id, viewBox: '0 0 10 10', refX: 10, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' });
    m.appendChild(el('path', { d: 'M0 0 L10 5 L0 10 z', fill: color })); defs.appendChild(m);
  }
  svg.appendChild(defs);
  const world = el('g'); svg.appendChild(world);           // everything pans/zooms together
  const byId = Object.fromEntries(data.nodes.map(n => [n.id, n]));

  // Edges: parallel edges between the same pair get bent apart so labels don't overlap.
  const pairCount = {};
  for (const e of data.edges) {
    const a = byId[e.s], b = byId[e.t]; if (!a || !b) continue;
    const pk = [e.s, e.t].sort().join('|'); const i = (pairCount[pk] = (pairCount[pk] || 0) + 1) - 1;
    const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len;
    const x1 = a.x + ux * a.r, y1 = a.y + uy * a.r, x2 = b.x - ux * (b.r + 2), y2 = b.y - uy * (b.r + 2);
    const bend = i * 28 * (i % 2 ? 1 : -1);                 // 0, +28, -56, ...
    const cx = (x1 + x2) / 2 - uy * bend, cy = (y1 + y2) / 2 + ux * bend;
    world.appendChild(el('path', { d: `M${x1} ${y1} Q${cx} ${cy} ${x2} ${y2}`, fill: 'none',
      stroke: e.active ? '#8892a6' : '#c4c9d4', 'stroke-width': e.active ? 1.6 : 1.2,
      'stroke-dasharray': e.active ? '' : '5 4', 'marker-end': `url(#${e.active ? 'arrow' : 'arrow-old'})` }));
    const t = el('text', { x: (x1 + 2 * cx + x2) / 4, y: (y1 + 2 * cy + y2) / 4 - 3, 'text-anchor': 'middle', class: 'elbl' });
    t.textContent = e.active ? e.label : e.label + ' (past)';
    world.appendChild(t);
  }

  // Nodes.
  for (const n of data.nodes) {
    const g = el('g', { class: 'node', transform: `translate(${n.x} ${n.y})` });
    if (n.hl) g.appendChild(el('circle', { r: n.r + 6, fill: 'none', stroke: '#f5b301', 'stroke-width': 3 }));
    g.appendChild(el('circle', { r: n.r, fill: n.color, stroke: 'white', 'stroke-width': 2 }));
    const tip = el('title'); tip.textContent = `${n.id} (${n.type})` + (n.tip ? `\\n${n.tip}` : ''); g.appendChild(tip);
    const lbl = el('text', { y: n.r + 15, 'text-anchor': 'middle', class: 'lbl' }); lbl.textContent = n.id;
    g.appendChild(lbl);
    g.addEventListener('click', (ev) => { ev.stopPropagation(); setTriggerValue('selected', n.id); });
    world.appendChild(g);
  }

  // Pan (drag) and zoom (wheel), by transforming the `world` group.
  let k = 1, tx = 0, ty = 0, drag = null;
  const apply = () => world.setAttribute('transform', `translate(${tx} ${ty}) scale(${k})`);
  const toSvg = (ev) => { const r = svg.getBoundingClientRect();
    return [(ev.clientX - r.left) * data.w / r.width, (ev.clientY - r.top) * data.h / r.height]; };
  svg.addEventListener('wheel', (ev) => {
    ev.preventDefault();
    const [mx, my] = toSvg(ev), f = ev.deltaY < 0 ? 1.12 : 1 / 1.12;
    tx = mx - (mx - tx) * f; ty = my - (my - ty) * f; k *= f; apply();   // zoom around the cursor
  }, { passive: false });
  svg.addEventListener('pointerdown', (ev) => { drag = [...toSvg(ev), tx, ty]; });
  svg.addEventListener('pointermove', (ev) => { if (!drag) return;
    const [mx, my] = toSvg(ev); tx = drag[2] + mx - drag[0]; ty = drag[3] + my - drag[1]; apply(); });
  for (const t of ['pointerup', 'pointerleave']) svg.addEventListener(t, () => { drag = null; });
  host.appendChild(svg);
}
"""

_view = st.components.v2.component("memoir_graph", css=VIEW_CSS, js=VIEW_JS)


def graph_view(payload: dict, key: str = "graph_view") -> str | None:
    """Draw the graph; returns the clicked node's name on the rerun it's clicked."""
    result = _view(key=key, data=payload, on_selected_change=lambda: None)
    return getattr(result, "selected", None)
