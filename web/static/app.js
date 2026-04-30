// Play uploaded input audio file
const inputAudioFile = document.getElementById('input-audio-file');
const inputAudioPreview = document.getElementById('input-audio-preview');
if (inputAudioFile && inputAudioPreview) {
  inputAudioFile.addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      inputAudioPreview.src = url;
      inputAudioPreview.style.display = 'block';
    } else {
      inputAudioPreview.src = '';
      inputAudioPreview.style.display = 'none';
    }
  });
}
const state = {
  effectsSpec: {},
  sessionId: null,
};

const effectsPanel = document.getElementById('effects-panel');
const jsonOutput = document.getElementById('json-output');
const outputPlayer = document.getElementById('output-player');
const outputLinkWrap = document.getElementById('output-link-wrap');
const generateForm = document.getElementById('generate-form');
const generateStatus = document.getElementById('generate-status');
const regenerateBtn = document.getElementById('regenerate-btn');
const regenerateStatus = document.getElementById('regenerate-status');

function showJson(obj) {
  jsonOutput.textContent = JSON.stringify(obj, null, 2);
}

function formatNumber(v, min, max) {
  const span = max - min;
  if (span >= 1000) return Number(v).toFixed(1);
  if (span >= 100) return Number(v).toFixed(2);
  return Number(v).toFixed(3);
}

function inferStep(min, max) {
  const span = max - min;
  if (span >= 1000) return 1;
  if (span >= 100) return 0.1;
  if (span >= 10) return 0.01;
  return 0.001;
}

function renderEffects(spec) {
  effectsPanel.innerHTML = '';

  Object.entries(spec).forEach(([effectName, params]) => {
    const card = document.createElement('div');
    card.className = 'effect-card';
    card.dataset.effect = effectName;

    const header = document.createElement('div');
    header.className = 'effect-header';
    header.innerHTML = `
      <strong>${effectName}</strong>
      <label>
        <input type="checkbox" class="effect-enabled" /> Enable
      </label>
    `;

    card.appendChild(header);


    Object.entries(params).forEach(([paramName, [min, max]]) => {
      const row = document.createElement('div');
      row.className = 'envelope-row';

      // 初始三個控制點 (t: 0, 0.5, 1)
      row.envelope = [
        { t: 0, v: (min + max) / 2 },
        { t: 0.5, v: max },
        { t: 1, v: (min + max) / 2 },
      ];
      row.min = min;
      row.max = max;
      row.paramName = paramName;
      row.effectName = effectName;

      // Envelope SVG (enlarged)
      const svgW = 380, svgH = 140, pad = 38;
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('class', 'envelope-svg');
      svg.setAttribute('width', svgW);
      svg.setAttribute('height', svgH);
      svg.setAttribute('viewBox', `0 0 ${svgW} ${svgH}`);

      // Draw axes
      const axis = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      xAxis.setAttribute('x1', pad);
      xAxis.setAttribute('y1', svgH - pad);
      xAxis.setAttribute('x2', svgW - pad);
      xAxis.setAttribute('y2', svgH - pad);
      xAxis.setAttribute('stroke', '#aaa');
      xAxis.setAttribute('stroke-width', '1');
      axis.appendChild(xAxis);
      const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      yAxis.setAttribute('x1', pad);
      yAxis.setAttribute('y1', pad);
      yAxis.setAttribute('x2', pad);
      yAxis.setAttribute('y2', svgH - pad);
      yAxis.setAttribute('stroke', '#aaa');
      yAxis.setAttribute('stroke-width', '1');
      axis.appendChild(yAxis);
      svg.appendChild(axis);

      // Draw envelope path
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', '#2a8');
      path.setAttribute('stroke-width', '2');
      svg.appendChild(path);

      // Draw control points
      const pointsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      svg.appendChild(pointsGroup);

      // Envelope label
      const meta = document.createElement('div');
      meta.className = 'envelope-meta';
      meta.innerHTML = `<span>${paramName} (${min} ~ ${max})</span>`;

      row.appendChild(meta);
      row.appendChild(svg);

      // 畫出 envelope 曲線與點

      // 拖曳互動狀態
      let dragIdx = null;
      let dragOffset = { x: 0, y: 0 };

      function renderEnvelope() {
        const envelope = row.envelope || [];
        if (!Array.isArray(envelope) || envelope.length === 0) return;
        // 轉換 t,v 到 SVG 座標
        const pts = envelope.map(({ t, v }) => {
          const x = pad + t * (svgW - 2 * pad);
          const y = pad + (1 - (v - min) / (max - min)) * (svgH - 2 * pad);
          return [x, y];
        });
        path.setAttribute('points', pts.map(([x, y]) => `${x},${y}`).join(' '));
        // 清空舊點
        while (pointsGroup.firstChild) pointsGroup.removeChild(pointsGroup.firstChild);
        // 畫點
        pts.forEach(([x, y], i) => {
          const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          c.setAttribute('cx', x);
          c.setAttribute('cy', y);
          c.setAttribute('r', 7);
          c.setAttribute('fill', i === 0 || i === pts.length - 1 ? '#2a8' : '#fff');
          c.setAttribute('stroke', '#2a8');
          c.setAttribute('stroke-width', '2');
          c.style.cursor = 'pointer';
          // 拖曳事件
          c.addEventListener('mousedown', (e) => {
            dragIdx = i;
            dragOffset = { x: e.offsetX - x, y: e.offsetY - y };
            document.body.style.userSelect = 'none';
          });
          pointsGroup.appendChild(c);
        });
      }
      renderEnvelope();
      // 讓 setControls 可以呼叫重繪
      svg._renderEnvelope = renderEnvelope;

      // 拖曳互動邏輯
      svg.addEventListener('mousemove', (e) => {
        if (dragIdx === null) return;
        const rect = svg.getBoundingClientRect();
        let mouseX = e.clientX - rect.left - dragOffset.x;
        let mouseY = e.clientY - rect.top - dragOffset.y;
        // 限制在繪圖區內
        mouseX = Math.max(pad, Math.min(svgW - pad, mouseX));
        mouseY = Math.max(pad, Math.min(svgH - pad, mouseY));
        // 轉回 t, v
        let t = (mouseX - pad) / (svgW - 2 * pad);
        let v = max - ((mouseY - pad) / (svgH - 2 * pad)) * (max - min);
        t = Math.max(0, Math.min(1, t));
        v = Math.max(min, Math.min(max, v));
        // 首尾點 t 固定
        if (dragIdx === 0) t = 0;
        const envelope = row.envelope || [];
        if (!Array.isArray(envelope) || envelope.length === 0) return;
        if (dragIdx === envelope.length - 1) t = 1;
        // 不可交錯
        if (dragIdx > 0 && t < envelope[dragIdx - 1].t) t = envelope[dragIdx - 1].t;
        if (dragIdx < envelope.length - 1 && t > envelope[dragIdx + 1].t) t = envelope[dragIdx + 1].t;
        envelope[dragIdx] = { t, v };
        row.envelope = envelope;
        renderEnvelope();
      });
      svg.addEventListener('mouseup', () => {
        dragIdx = null;
        document.body.style.userSelect = '';
      });
      svg.addEventListener('mouseleave', () => {
        dragIdx = null;
        document.body.style.userSelect = '';
      });

      card.appendChild(row);
    });

    effectsPanel.appendChild(card);
  });
  // 之後 envelope 互動事件會加在這裡
}

function toY(value, min, max) {
  const t = (Number(value) - min) / (max - min || 1);
  const clamped = Math.max(0, Math.min(1, t));
  return 26 - clamped * 22;
}

function updateRowDisplay(row) {
  if (!row) return;
  const startSlider = row.querySelector('.param-slider-start');
  const endSlider = row.querySelector('.param-slider-end');
  if (!startSlider || !endSlider) return;

  const min = Number(startSlider.dataset.min);
  const max = Number(startSlider.dataset.max);
  const startValue = Number(startSlider.value);
  const endValue = Number(endSlider.value);

  const baseId = startSlider.id.replace('_start', '');
  const valueEl = document.getElementById(`${baseId}_value`);
  if (valueEl) {
    valueEl.textContent = `${formatNumber(startValue, min, max)} → ${formatNumber(endValue, min, max)}`;
  }

  const yStart = toY(startValue, min, max);
  const yEnd = toY(endValue, min, max);

  const line = row.querySelector('.automation-path');
  const startDot = row.querySelector('.automation-dot.start');
  const endDot = row.querySelector('.automation-dot.end');
  if (line) {
    line.setAttribute('x1', '2');
    line.setAttribute('y1', String(yStart));
    line.setAttribute('x2', '98');
    line.setAttribute('y2', String(yEnd));
  }
  if (startDot) {
    startDot.setAttribute('cx', '2');
    startDot.setAttribute('cy', String(yStart));
  }
  if (endDot) {
    endDot.setAttribute('cx', '98');
    endDot.setAttribute('cy', String(yEnd));
  }
}

function setControls(controlValues) {
  Object.entries(controlValues || {}).forEach(([effectName, effectData]) => {
    const card = effectsPanel.querySelector(`.effect-card[data-effect="${effectName}"]`);
    if (!card) return;

    const enabledEl = card.querySelector('.effect-enabled');
    enabledEl.checked = Boolean(effectData.enabled);

    card.querySelectorAll('.envelope-row').forEach((row) => {
      const paramName = row.paramName;
      const rawValue = (effectData.params || {})[paramName];
      if (rawValue === undefined) return;
      // envelope: [{t, v}, ...]
      if (Array.isArray(rawValue) && rawValue.length >= 2) {
        row.envelope = rawValue.map(({ t, v }) => ({ t: Number(t), v: Number(v) }));
        // 觸發重繪
        const svg = row.querySelector('.envelope-svg');
        if (svg && typeof svg._renderEnvelope === 'function') svg._renderEnvelope();
      } else if (rawValue && typeof rawValue === 'object' && ('start' in rawValue || 'end' in rawValue)) {
        // 舊格式: 轉成兩點 envelope
        const min = row.min, max = row.max;
        const start = Number(rawValue.start ?? rawValue.end ?? min);
        const end = Number(rawValue.end ?? rawValue.start ?? max);
        row.envelope = [
          { t: 0, v: start },
          { t: 1, v: end },
        ];
        const svg = row.querySelector('.envelope-svg');
        if (svg && typeof svg._renderEnvelope === 'function') svg._renderEnvelope();
      }
    });
  });
}

function collectControls() {
  const controls = {};
  effectsPanel.querySelectorAll('.effect-card').forEach((card) => {
    const effect = card.dataset.effect;
    const enabled = card.querySelector('.effect-enabled').checked;
    controls[effect] = { enabled, params: {} };
    card.querySelectorAll('.envelope-row').forEach((row) => {
      const param = row.paramName;
      // envelope: [{t, v}, ...]
      controls[effect].params[param] = (row.envelope || []).map(({ t, v }) => ({ t, v }));
    });
  });
  return controls;
}

function setAudioOutput(url) {
  const finalUrl = `${url}?t=${Date.now()}`;
  outputPlayer.src = finalUrl;

  outputLinkWrap.innerHTML = '';
  const link = document.createElement('a');
  link.href = finalUrl;
  link.textContent = 'Download output wav';
  link.className = 'link-btn';
  link.download = '';
  outputLinkWrap.appendChild(link);
}

async function init() {
  const res = await fetch('/api/effects-spec');
  const data = await res.json();
  state.effectsSpec = data.effects || {};
  renderEffects(state.effectsSpec);
}

generateForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  generateStatus.textContent = 'Generating...';
  regenerateStatus.textContent = '';

  const formData = new FormData(generateForm);
  const mode = String(formData.get('mode') || 'generate');
  if (mode === 'extract-and-clone' && !formData.get('reference_audio')?.name) {
    generateStatus.textContent = 'Error: reference audio is required in extract-and-clone mode';
    return;
  }

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || 'Generate failed');
    }

    state.sessionId = data.session_id;
    setControls(data.control_values);
    setAudioOutput(data.output_url);
    regenerateBtn.disabled = false;
    showJson(data);

    generateStatus.textContent = 'Done. Parameters loaded into bars.';
  } catch (err) {
    generateStatus.textContent = `Error: ${err.message}`;
  }
});

regenerateBtn.addEventListener('click', async () => {
  if (!state.sessionId) {
    regenerateStatus.textContent = 'Please generate once first.';
    return;
  }

  regenerateStatus.textContent = 'Regenerating from bars...';
  generateStatus.textContent = '';

  try {
    const controls = collectControls();
    const res = await fetch('/api/regenerate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: state.sessionId,
        controls,
        normalize: true,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || 'Regenerate failed');
    }

    setAudioOutput(data.output_url);
    showJson(data);
    regenerateStatus.textContent = 'Done. New audio generated from bar values.';
  } catch (err) {
    regenerateStatus.textContent = `Error: ${err.message}`;
  }
});

init().catch((err) => {
  generateStatus.textContent = `Init error: ${err.message}`;
});
