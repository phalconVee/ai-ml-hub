"""The face and the ears: two small Streamlit components (st.components.v2).

Why components.v2: its JavaScript runs in the app page itself, not in a
sandboxed iframe. Two things depend on that:
  * autoplay: browsers only let a page play sound after the user has
    interacted with *that page*. Sending a chat message counts, so the reply
    can start talking on its own. (If a browser still blocks it, the avatar
    shows a "tap to hear" button instead of failing silently.)
  * a bidirectional channel: the mic button sends the transcript back to
    Python with setTriggerValue(), with no build step and no npm.

Safety: the JS never uses innerHTML with data. Every string from Python (and
therefore from Claude) is set with textContent or as an attribute value.

THE AVATAR
  Plain SVG: head, hair, eyes, brows, mouth. Three things animate:
    mouth  height = lip-sync value. With server audio (Kokoro / say /
           ElevenLabs), that's the loudness envelope at audio.currentTime;
           with browser speech, a wobble that jumps on every word boundary.
    eyes   blink at random 2-6 s intervals (scaleY to almost 0 for 120 ms).
    head   breathes (a slow bob), and tilts a little while talking.
  A new reply has a new `id`; a set on `window` remembers which ids already
  played, so Streamlit's reruns never make it repeat itself.

THE MIC
  The Web Speech API's SpeechRecognition (Chrome, Edge, Safari). Click, talk,
  and the final transcript is sent to Python as if typed. In Chrome the
  audio is recognised by Google's servers; no key needed, but not offline.
  Unsupported browsers (Firefox, embedded webviews) just see a note.
"""

from __future__ import annotations

import base64
import streamlit as st

from .voice import Speech

AVATAR_HTML = """
<div class="memoir-avatar">
  <svg viewBox="0 0 200 230" class="face" role="img" aria-label="Mira, the assistant's avatar">
    <defs>
      <radialGradient id="skin" cx="50%" cy="40%" r="60%">
        <stop offset="0%" stop-color="#f6d2b8"/><stop offset="100%" stop-color="#e2ac8c"/>
      </radialGradient>
    </defs>
    <g class="head">
      <path d="M28 112 C22 22 178 22 172 112 L176 178 Q156 192 140 166 L60 166 Q44 192 24 178 Z" fill="#2c2340"/> <!-- hair, back -->
      <ellipse cx="100" cy="222" rx="72" ry="32" fill="#5b6cff"/>                   <!-- shoulders -->
      <rect x="87" y="160" width="26" height="36" rx="10" fill="#d99f80"/>          <!-- neck -->
      <ellipse cx="100" cy="105" rx="56" ry="66" fill="url(#skin)"/>                <!-- face -->
      <path d="M42 104 C34 12 166 12 158 104 C152 80 136 66 112 62 C92 60 70 66 58 82 C52 90 46 96 42 104 Z" fill="#3a2e55"/> <!-- fringe -->
      <g class="brows" stroke="#3a2e55" stroke-width="4" stroke-linecap="round" fill="none">
        <path d="M66 86 Q78 80 90 85"/><path d="M110 85 Q122 80 134 86"/>
      </g>
      <g class="eyes" fill="#2c2340">
        <ellipse class="eye" cx="78" cy="102" rx="6" ry="7"/>
        <ellipse class="eye" cx="122" cy="102" rx="6" ry="7"/>
      </g>
      <ellipse cx="66" cy="124" rx="9" ry="5" fill="#ef8f8f" opacity="0.35"/>       <!-- cheeks -->
      <ellipse cx="134" cy="124" rx="9" ry="5" fill="#ef8f8f" opacity="0.35"/>
      <path d="M100 108 Q96 120 101 123" stroke="#c98a6c" stroke-width="2.5" fill="none" stroke-linecap="round"/>
      <g class="mouth-g" transform="translate(100 141)">
        <ellipse class="mouth" cx="0" cy="0" rx="15" ry="2" fill="#7a2f3b"/>
        <ellipse class="tongue" cx="0" cy="3" rx="8" ry="0" fill="#d9636f"/>
        <path class="lips" d="M-16 0 Q0 6 16 0" stroke="#a24a55" stroke-width="2.5" fill="none" stroke-linecap="round"/>
      </g>
    </g>
  </svg>
  <div class="status" aria-live="polite">idle</div>
  <button class="tap" hidden>▶ Tap to hear Mira</button>
</div>
"""

AVATAR_CSS = """
.memoir-avatar { display:flex; flex-direction:column; align-items:center; gap:.4rem; }
.face { width: min(260px, 100%); height:auto; overflow:hidden; }
.head { transform-origin: 100px 180px; transition: transform .25s ease; }
.eye  { transform-box: fill-box; transform-origin: center; transition: transform .06s; }
.status { font: 500 .8rem system-ui, sans-serif; color: var(--st-text-color, #555); opacity:.7; }
.tap { font: 600 .9rem system-ui, sans-serif; padding:.4rem .9rem; border-radius:999px;
       border:1px solid var(--st-primary-color, #5b6cff); background:transparent;
       color: var(--st-primary-color, #5b6cff); cursor:pointer; }
"""

AVATAR_JS = """
export default function (component) {
  const { data, parentElement } = component;
  const root = parentElement.querySelector('.memoir-avatar');
  const head = root.querySelector('.head');
  const mouth = root.querySelector('.mouth');
  const tongue = root.querySelector('.tongue');
  const lips = root.querySelector('.lips');
  const brows = root.querySelector('.brows');
  const eyes = root.querySelectorAll('.eye');
  const status = root.querySelector('.status');
  const tap = root.querySelector('.tap');

  // One state object per mounted avatar, kept on the element across reruns.
  const S = root.__memoir || (root.__memoir = { open: 0, target: 0, speaking: false,
                                               audio: null, raf: 0, blinkT: 0 });
  S.data = data;                                   // the loop below always reads the latest reply
  window.__memoirPlayed = window.__memoirPlayed || new Set();

  // ── drawing: one value (0..1) controls the whole mouth ──
  function drawMouth(v) {
    mouth.setAttribute('ry', (2 + v * 13).toFixed(2));
    mouth.setAttribute('rx', (15 - v * 3).toFixed(2));
    tongue.setAttribute('ry', (v * 4).toFixed(2));
    lips.setAttribute('d', `M-16 ${(-v * 2).toFixed(1)} Q0 ${(6 - v * 2).toFixed(1)} 16 ${(-v * 2).toFixed(1)}`);
    brows.setAttribute('transform', `translate(0 ${(-v * 2.5).toFixed(2)})`);
  }

  // ── the animation loop: lip-sync + blink + breathing ──
  function frame(t) {
    let target = 0;
    const a = S.audio, d = S.data;
    if (S.speaking && a && d && d.envelope && d.envelope.length) {
      const i = Math.floor(a.currentTime * d.fps);               // envelope index for "now"
      target = d.envelope[Math.min(i, d.envelope.length - 1)] || 0;
    } else if (S.speaking) {                                       // browser speech: no samples
      S.target *= 0.9;                                             // word pulse decays...
      target = Math.max(S.target, 0.2 + 0.25 * Math.abs(Math.sin(t / 70)));  // ...over a wobble
    }
    S.open += (target - S.open) * 0.5;                             // ease, so it never snaps
    drawMouth(S.open);

    // Blink: every 2-6 s, close for ~120 ms.
    if (t > S.blinkT) S.blinkT = t + 2000 + Math.random() * 4000;
    const closed = S.blinkT - t < 120;
    eyes.forEach(e => e.style.transform = closed ? 'scaleY(0.1)' : 'scaleY(1)');

    // Breathe; tilt slightly while talking.
    const bob = Math.sin(t / 900) * 1.5;
    const tilt = S.speaking ? Math.sin(t / 600) * 2.5 : 0;
    head.style.transform = `translateY(${bob.toFixed(2)}px) rotate(${tilt.toFixed(2)}deg)`;
    S.raf = requestAnimationFrame(frame);
  }
  if (!S.raf) S.raf = requestAnimationFrame(frame);

  function setStatus(s) { status.textContent = s; }
  function done() { S.speaking = false; setStatus('listening'); }

  // ── speaking a new reply ──
  function playAudio() {
    if (S.audio) S.audio.pause();
    const a = new Audio('data:audio/wav;base64,' + data.audio_b64);
    S.audio = a;
    a.onplay = () => { S.speaking = true; tap.hidden = true; setStatus('speaking'); };
    a.onended = done;
    a.play().catch(() => {                       // autoplay blocked → offer a button
      tap.hidden = false; setStatus('ready');
      tap.onclick = () => { a.play(); };
    });
  }

  function speakInBrowser() {
    const synth = window.speechSynthesis;
    if (!synth) { setStatus('this browser cannot speak'); return; }
    synth.cancel();
    const u = new SpeechSynthesisUtterance(data.text);
    const wanted = (data.voice || '').toLowerCase();
    const v = synth.getVoices().find(v => wanted && v.name.toLowerCase().includes(wanted));
    if (v) u.voice = v;
    u.onstart = () => { S.speaking = true; S.audio = null; setStatus('speaking'); };
    u.onboundary = () => { S.target = 1; };      // a word started: open wide
    u.onend = done; u.onerror = done;
    synth.speak(u);
  }

  if (data && data.id && !window.__memoirPlayed.has(data.id)) {
    window.__memoirPlayed.add(data.id);
    if (data.muted) { /* nothing to say out loud */ }
    else if (data.audio_b64) playAudio();
    else if (data.text) speakInBrowser();
  } else if (!S.speaking) {
    setStatus(data && data.id ? 'listening' : 'say hello 👋');
  }

  return () => {                                  // unmount: stop sound and the loop
    cancelAnimationFrame(S.raf); S.raf = 0;
    if (S.audio) S.audio.pause();
    if (window.speechSynthesis) window.speechSynthesis.cancel();
  };
}
"""

MIC_HTML = """<div class="memoir-mic"><button class="mic" type="button">🎙️ Speak</button>
<span class="hint"></span></div>"""

MIC_CSS = """
.memoir-mic { display:flex; align-items:center; gap:.6rem; }
.mic { font: 600 .9rem system-ui, sans-serif; padding:.45rem 1rem; border-radius:999px;
       border:1px solid var(--st-primary-color, #5b6cff); background:transparent;
       color: var(--st-primary-color, #5b6cff); cursor:pointer; }
.mic.on { background: var(--st-primary-color, #5b6cff); color: white; }
.hint { font: .8rem system-ui, sans-serif; opacity:.7; }
"""

MIC_JS = """
export default function (component) {
  const { parentElement, setTriggerValue } = component;
  const btn = parentElement.querySelector('.mic');
  const hint = parentElement.querySelector('.hint');
  const getSR = () => window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!getSR()) { btn.disabled = true; hint.textContent = 'Voice input needs Chrome, Edge or Safari.'; return; }

  let rec = null;
  btn.onclick = () => {
    if (rec) { rec.stop(); return; }                 // second click = stop early
    rec = new (getSR())();                           // looked up per click, not once at mount
    rec.lang = navigator.language || 'en-US';
    rec.interimResults = true;                       // show words as they're heard
    let finalText = '';
    rec.onresult = (e) => {
      let interim = '';
      for (const r of e.results) (r.isFinal ? (finalText = r[0].transcript) : (interim += r[0].transcript));
      hint.textContent = interim || finalText;
    };
    rec.onerror = (e) => { hint.textContent = 'mic error: ' + e.error; };
    rec.onend = () => {
      btn.classList.remove('on'); btn.textContent = '🎙️ Speak'; rec = null;
      if (finalText.trim()) setTriggerValue('transcript', finalText.trim());   // → Python
    };
    btn.classList.add('on'); btn.textContent = '■ Stop'; hint.textContent = 'listening…';
    rec.start();
  };
}
"""

# Registered once at import; the returned callables mount an instance.
_avatar = st.components.v2.component("memoir_avatar", html=AVATAR_HTML, css=AVATAR_CSS, js=AVATAR_JS)
_mic = st.components.v2.component("memoir_mic", html=MIC_HTML, css=MIC_CSS, js=MIC_JS)


def avatar_payload(reply_id: str, text: str, speech: Speech | None, *,
                   voice: str = "", muted: bool = False) -> dict:
    """Everything the avatar needs for one reply, as JSON-safe data."""
    from .config import SETTINGS
    has_audio = bool(speech and speech.wav)
    return {
        "id": reply_id,
        "text": text,
        "voice": voice,
        "muted": muted,
        "fps": SETTINGS.envelope_fps,
        "envelope": speech.envelope if has_audio else [],
        "audio_b64": base64.b64encode(speech.wav).decode() if has_audio else "",
    }


def show_avatar(payload: dict | None, key: str = "avatar_view"):
    return _avatar(key=key, data=payload or {})


def mic_input(key: str = "mic_view") -> str | None:
    """Render the mic button; returns the transcript on the rerun it arrives, else None."""
    result = _mic(key=key, on_transcript_change=lambda: None)
    return getattr(result, "transcript", None)
