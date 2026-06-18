(async () => {
  // === CONFIGURATION & CONSTANTS ===
  const MODEL_PATH = './yolov8n-oiv7.onnx';
  const CONF_THRESHOLD = 0.35;
  const IOU_THRESHOLD = 0.45;
  const INPUT_DIM = 640;

  // ── PROXY URL ────────────────────────────────────────────────────────────────
  // Local dev  → 'http://localhost:3000/api/vision'
  // Vercel     → '/api/vision'
  const PROXY_URL = 'http://localhost:3000/api/vision';
  // ─────────────────────────────────────────────────────────────────────────────

  let OIV7_CLASSES = Array.from({length: 600}, (_, i) => `object_${i}`); 

  const THREAT_CLASSES = new Set(['car', 'truck', 'bus', 'motorcycle', 'bicycle']);
  
  const objectHeights = {
    person: 1.7, bottle: 0.25, chair: 1.0, book: 0.3,
    tv: 0.6, television: 0.6, laptop: 0.4, cellphone: 0.15, 'mobile phone': 0.15, 
    keyboard: 0.45, 'computer keyboard': 0.45, mouse: 0.12, 'computer mouse': 0.12
  };
  const FOCAL_LENGTH_PX = 700;
  const PIXELS_PER_CM = 5;

  // === DOM ELEMENTS & STATE ===
  const video = document.getElementById('webcam');
  const overlay = document.getElementById('overlay');
  const octx = overlay.getContext('2d');
  const snapshot = document.getElementById('snapshot');
  const sctx = snapshot.getContext('2d');
  
  const statusLine = document.getElementById('statusLine');
  const scanBtn = document.getElementById('scanBtn');
  const cameraSelect = document.getElementById('cameraSelect');
  const startBtn = document.getElementById('startBtn');
  const takeSnapshotBtn = document.getElementById('takeSnapshot');
  const saveDataBtn = document.getElementById('saveData');
  const detectedStrip = document.getElementById('detectedStrip');
  const resultsList = document.getElementById('resultsList');
  
  const modelToggle = document.getElementById('modelToggle');
  const modelLabel = document.getElementById('modelLabel');

  let ortSession = null;  
  let cocoModel = null;   
  
  let currentStream = null;
  let running = false;
  let rafId = null;

  let cycles = []; 
  let nextCycleNumber = 1;

  const offscreenCanvas = document.createElement('canvas');
  offscreenCanvas.width = INPUT_DIM;
  offscreenCanvas.height = INPUT_DIM;
  const offctx = offscreenCanvas.getContext('2d', { willReadFrequently: true });

  // === UTILITIES ===
  const pad = n => String(n).padStart(2, '0');
  
  function ordinal(n) {
    const s = ["th", "st", "nd", "rd"], v = n % 100;
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
  }

  function formatDistance(est) {
    if (!isFinite(est) || est <= 0) return "unknown";
    const meters = Math.floor(est);
    const centimeters = Math.round((est - meters) * 100);
    const parts = [];
    if (meters > 0) parts.push(`${meters} metres`);
    if (centimeters > 0) parts.push(`${centimeters} centimetres`);
    return parts.join(' ');
  }

  const colorPalette = ['#00c4cc', '#4fb0ff', '#f6b352', '#9b8cff', '#ff6b6b', '#45c272', '#ff9ee3'];
  function colorForLabel(label) {
    let h = 0;
    for (let i = 0; i < label.length; i++) h = (h << 5) - h + label.charCodeAt(i);
    return colorPalette[Math.abs(h) % colorPalette.length];
  }

  async function fetchClasses() {
    try {
      const response = await fetch('https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/open-images-v7.yaml');
      if (!response.ok) throw new Error("Network issue");
      const yamlText = await response.text();
      const regex = /^\s*(\d+):\s*(.*)$/gm;
      let match;
      while ((match = regex.exec(yamlText)) !== null) {
        const id = parseInt(match[1], 10);
        const name = match[2].trim().replace(/['"]/g, '').toLowerCase(); 
        if (id >= 0 && id < 600) OIV7_CLASSES[id] = name;
      }
    } catch (err) {
      console.warn("Using fallback structural object classes.", err);
    }
  }

  // === CLOUD VISION — calls our secure backend proxy ===
  // The GITHUB_TOKEN never touches the browser. It lives in the server's .env
  async function analyzeSnapshotWithCloud(snapshotCanvas) {
    statusLine.classList.add('loading');
    statusLine.textContent = '🧠 Cloud AI analyzing layout...';

    const base64Image = snapshotCanvas.toDataURL("image/jpeg", 0.5).split(',')[1]; 

    try {
      const response = await fetch(PROXY_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "scene", imageBase64: base64Image, mediaType: "image/jpeg" })
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      
      statusLine.classList.remove('loading');
      statusLine.classList.add('ready');
      statusLine.textContent = '✅ Analysis complete';
      return data.result;

    } catch (err) {
      console.error("Cloud Network Interface failure:", err);
      statusLine.textContent = 'Error connecting to Cloud AI';
      return `Scene description failed: ${err.message}`;
    }
  }

  // === CLOUD OCR — calls our secure backend proxy ===
  async function performTextRecognition() {
    const sw = snapshot.width;
    const sh = snapshot.height;

    sctx.clearRect(0, 0, sw, sh);
    sctx.drawImage(video, 0, 0, sw, sh);

    statusLine.classList.add('loading');
    statusLine.textContent = '📖 Scanning image for high-res text...';

    const base64Image = snapshot.toDataURL("image/jpeg", 0.95).split(',')[1];

    try {
      const response = await fetch(PROXY_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: "ocr", imageBase64: base64Image, mediaType: "image/jpeg" })
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);

      const extractedText = (data.result || '').trim();

      statusLine.classList.remove('loading');
      statusLine.classList.add('ready');
      statusLine.textContent = '✅ Text scan complete';

      if (extractedText === "NO_TEXT_FOUND" || extractedText === "" || extractedText.toLowerCase().includes("no text")) {
        resultsList.innerHTML = `<strong style="color:var(--accent);">Detected Text:</strong> None`;
        speakText("No text detected", 1.0);
      } else {
        resultsList.innerHTML = `<strong style="color:var(--accent);">Detected Text:</strong><br><br> ${extractedText.replace(/\n/g, '<br>')}`;
        setTimeout(() => speakText("Text detected: " + extractedText, 0.95), 200);
      }

    } catch (err) {
      console.error("Cloud OCR failure:", err);
      statusLine.textContent = 'Error scanning text';
      resultsList.innerHTML = `<strong style="color:#ff6b6b;">OCR Error:</strong> ${err.message}`;
      speakText("Error scanning for text.", 1.0);
    }
  }

  // === ONNX YOLOv8 PREPROCESSING & INFERENCE ===
  function preprocess(source) {
    const sw = source.videoWidth || source.width;
    const sh = source.videoHeight || source.height;
    const scale = Math.min(INPUT_DIM / sw, INPUT_DIM / sh);
    const dw = sw * scale;
    const dh = sh * scale;
    const dx = (INPUT_DIM - dw) / 2;
    const dy = (INPUT_DIM - dh) / 2;

    offctx.fillStyle = 'rgb(114, 114, 114)';
    offctx.fillRect(0, 0, INPUT_DIM, INPUT_DIM);
    offctx.drawImage(source, 0, 0, sw, sh, dx, dy, dw, dh);

    const imgData = offctx.getImageData(0, 0, INPUT_DIM, INPUT_DIM).data;
    const floatData = new Float32Array(3 * INPUT_DIM * INPUT_DIM);

    for (let i = 0; i < INPUT_DIM * INPUT_DIM; i++) {
      floatData[i] = imgData[i * 4] / 255.0; 
      floatData[INPUT_DIM * INPUT_DIM + i] = imgData[i * 4 + 1] / 255.0; 
      floatData[2 * INPUT_DIM * INPUT_DIM + i] = imgData[i * 4 + 2] / 255.0; 
    }

    const tensor = new ort.Tensor('float32', floatData, [1, 3, INPUT_DIM, INPUT_DIM]);
    return { tensor, scale, dx, dy };
  }

  function iou(box1, box2) {
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;
    const xA = Math.max(x1, x2);
    const yA = Math.max(y1, y2);
    const xB = Math.min(x1 + w1, x2 + w2);
    const yB = Math.min(y1 + h1, y2 + h2);
    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const box1Area = w1 * h1;
    const box2Area = w2 * h2;
    return interArea / (box1Area + box2Area - interArea);
  }

  function nms(boxes, iouThresh) {
    boxes.sort((a, b) => b.score - a.score);
    const result = [];
    while (boxes.length > 0) {
      const best = boxes.shift();
      result.push(best);
      boxes = boxes.filter(box => iou(best.bbox, box.bbox) < iouThresh);
    }
    return result;
  }

  async function runInference(source) {
    if (!ortSession) return [];
    
    const { tensor, scale, dx, dy } = preprocess(source);
    const feeds = {};
    feeds[ortSession.inputNames[0]] = tensor;
    const results = await ortSession.run(feeds);
    const output = results[ortSession.outputNames[0]].data;

    const numBoxes = 8400;
    const numClasses = OIV7_CLASSES.length;
    let detections = [];

    for (let i = 0; i < numBoxes; i++) {
      let maxClassScore = 0;
      let classId = -1;

      for (let c = 0; c < numClasses; c++) {
        const score = output[(4 + c) * numBoxes + i];
        if (score > maxClassScore) {
          maxClassScore = score;
          classId = c;
        }
      }

      if (maxClassScore >= CONF_THRESHOLD) {
        const cx = output[0 * numBoxes + i];
        const cy = output[1 * numBoxes + i];
        const w = output[2 * numBoxes + i];
        const h = output[3 * numBoxes + i];

        const x = ((cx - w / 2) - dx) / scale;
        const y = ((cy - h / 2) - dy) / scale;
        const finalW = w / scale;
        const finalH = h / scale;

        detections.push({
          bbox: [x, y, finalW, finalH],
          class: OIV7_CLASSES[classId] || 'object',
          score: maxClassScore
        });
      }
    }
    return nms(detections, IOU_THRESHOLD);
  }

  // === HARDWARE INTERACTION MANAGEMENT ===
  async function setupCamera(deviceId) {
    try {
      if (currentStream) currentStream.getTracks().forEach(t => t.stop());
      const constraints = deviceId ? { video: { deviceId: { exact: deviceId } } } : { video: { facingMode: "environment" } };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      currentStream = stream;
      video.srcObject = stream;
      await new Promise(res => video.onloadedmetadata = res);

      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      snapshot.width = Math.min(920, video.videoWidth);
      snapshot.height = Math.min(620, video.videoHeight);
    } catch (err) {
      console.error("Webcam configuration failure:", err);
      alert("Unable to access camera framework.");
      throw err;
    }
  }

  async function populateCameras() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter(d => d.kind === 'videoinput');
      cameraSelect.innerHTML = '';
      cams.forEach((c, i) => {
        const opt = document.createElement('option');
        opt.value = c.deviceId;
        opt.textContent = c.label || `Camera ${i+1}`;
        cameraSelect.appendChild(opt);
      });
    } catch (e) {
      console.warn("Camera index query dropped:", e);
    }
  }

  // === ACTIVE MODEL ROUTER ===
  if (modelToggle) {
    if (modelToggle.checked) {
      modelLabel.textContent = 'COCO-SSD (80 Classes)';
    } else {
      modelLabel.textContent = 'ONNX (600 Classes)';
    }

    modelToggle.addEventListener('change', (e) => {
      if (e.target.checked) {
        modelLabel.textContent = 'COCO-SSD (80 Classes)';
      } else {
        modelLabel.textContent = 'ONNX (600 Classes)';
      }
    });
  }

  async function getActiveDetections(source) {
    if (modelToggle && modelToggle.checked) {
      if (!cocoModel) return [];
      return await cocoModel.detect(source);
    } else {
      return await runInference(source);
    }
  }

  // === LIVE INFERENCE EVENT LOOP ===
  let lastFaceCheckTime = 0;
  let cachedIdentity = null;

  async function liveLoop() {
    if (!running) return;
    if (!ortSession || !cocoModel) { rafId = requestAnimationFrame(liveLoop); return; }
    try {
      const isCocoActive = modelToggle && modelToggle.checked;
      const preds = await getActiveDetections(video);

      for (let i = 0; i < preds.length; i++) {
        let currentName = preds[i].class.toLowerCase();
        if (isCocoActive && currentName === 'person') {
          const now = Date.now();
          if (now - lastFaceCheckTime > 1000) {
            cachedIdentity = await FaceVIP.identifyPerson(video);
            lastFaceCheckTime = now;
          }
          if (cachedIdentity && cachedIdentity !== 'person') {
            preds[i].class = cachedIdentity;
          }
        }
      }

      octx.clearRect(0, 0, overlay.width, overlay.height);
      octx.lineWidth = 2;

      preds.forEach(p => {
        const [x, y, w, h] = p.bbox;
        const accent = colorForLabel(p.class);
        octx.strokeStyle = accent;
        octx.strokeRect(x, y, w, h);
        octx.fillStyle = accent;
        octx.font = '15px Arial';
        const label = `${p.class} ${Math.round(p.score*100)}%`;
        const labelY = y - 18 >= 0 ? y - 18 : y;
        const tw = octx.measureText(label).width;
        octx.fillRect(x, labelY, tw + 8, 18);
        octx.fillStyle = '#002827';
        octx.fillText(label, x + 4, labelY + 2);
      });
    } catch (err) {
      console.warn("Live overlay cycle skipped:", err);
    }
    rafId = requestAnimationFrame(liveLoop);
  }

  function startLive() {
    if (running) return;
    running = true;
    liveLoop();
  }

  function stopLive() {
    running = false;
    if (rafId) cancelAnimationFrame(rafId);
  }

  function speakText(text, rate = 1.0) {
    return new Promise(resolve => {
      window.speechSynthesis.cancel(); 
      const u = new SpeechSynthesisUtterance(text);
      u.lang = 'en-US';
      u.rate = rate;
      u.pitch = 1.0;
      u.onend = () => resolve();
      u.onerror = () => resolve();
      window.speechSynthesis.speak(u);
    });
  }

  // === SEPARATED CORE CAPTURE ROUTINES ===

  // Action 1: Local Object Parsing
  async function performLocalObjectDetection() {
    if (!ortSession || !cocoModel) { alert("Models not initialized"); return; }

    const sw = snapshot.width;
    const sh = snapshot.height;

    sctx.clearRect(0, 0, sw, sh);
    sctx.drawImage(video, 0, 0, sw, sh);

    const isCocoActive = modelToggle && modelToggle.checked;
    const preds = await getActiveDetections(snapshot);

    for (let i = 0; i < preds.length; i++) {
      let currentName = preds[i].class.toLowerCase();
      if (isCocoActive && currentName === 'person') {
        const knownIdentity = await FaceVIP.identifyPerson(video); 
        preds[i].class = knownIdentity;
      }
    }

    sctx.lineWidth = 2;
    preds.forEach(p => {
      const [x, y, w, h] = p.bbox;
      const accent = colorForLabel(p.class);
      sctx.strokeStyle = accent;
      sctx.strokeRect(x, y, w, h);
      sctx.fillStyle = accent;
      sctx.font = '14px Arial';
      const label = `${p.class} ${Math.round(p.score * 100)}%`;
      const labelY = y - 18 >= 0 ? y - 18 : y;
      const tw = sctx.measureText(label).width;
      sctx.fillRect(x, labelY, tw + 8, 18);
      sctx.fillStyle = '#002827';
      sctx.fillText(label, x + 4, labelY + 2);
    });

    const now = new Date();
    const timeStr = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
    const cycle = { timeStr, cycleOrdinal: nextCycleNumber, preds: [] };

    detectedStrip.innerHTML = '';
    const currentModelName = (modelToggle && modelToggle.checked) ? "COCO-SSD" : "ONNX";
    resultsList.innerHTML = `<strong>Local Detections (${currentModelName}):</strong> ${preds.length} object(s) parsed.`;

    if (!preds || preds.length === 0) {
      const note = document.createElement('div');
      note.style.color = 'var(--muted)';
      note.textContent = `No objects detected via ${currentModelName}`;
      detectedStrip.appendChild(note);
      cycle.preds = [];
      cycles.push(cycle);
      nextCycleNumber++;
      return;
    }

    const speakQueue = [];

    for (let i = 0; i < preds.length; i++) {
      const p = preds[i];
      const name = p.class;
      const conf = Math.round(p.score * 100);
      const [x, y, w, h] = p.bbox;
      const centerX = x + w / 2;
      const centerY = y + h / 2;
      const coordX = Math.round((centerX - (sw / 2)) / PIXELS_PER_CM);
      const coordY = Math.round(((sh / 2) - centerY) / PIXELS_PER_CM);

      let distanceStr = "unknown";
      const known = objectHeights[name.toLowerCase()] || objectHeights['person']; 
      if (known && h >= 5) {
        const est = (known * FOCAL_LENGTH_PX) / h;
        distanceStr = formatDistance(est);
      }

      cycle.preds.push({ class: name, score: conf, coordX, coordY, distanceStr });

      const card = document.createElement('div');
      card.className = 'obj-card';
      const accent = colorForLabel(name);
      card.style.borderLeftColor = accent;

      const thumb = document.createElement('div');
      thumb.className = 'obj-thumb';
      thumb.style.border = `2px solid ${accent}`;
      thumb.textContent = name[0].toUpperCase();

      const info = document.createElement('div');
      info.className = 'obj-info';
      const nm = document.createElement('div');
      nm.className = 'obj-name';
      nm.textContent = name;
      const meta = document.createElement('div');
      meta.className = 'obj-meta';
      meta.textContent = `(${coordX}, ${coordY}) • ${distanceStr}`;
      const confEl = document.createElement('div');
      confEl.className = 'obj-confidence';
      confEl.textContent = `${conf}%`;

      info.appendChild(nm);
      info.appendChild(meta);
      card.appendChild(thumb);
      card.appendChild(info);
      card.appendChild(confEl);

      detectedStrip.appendChild(card);
      speakQueue.push({ name, coordX, coordY, distanceStr });
    }

    cycles.push(cycle);
    nextCycleNumber++;

    if (speakQueue.length > 0) {
      const sentences = speakQueue.map(s => `${s.name} at ${s.coordX}, ${s.coordY}, distance ${s.distanceStr}`);
      const fullText = `Detected ${speakQueue.length} items: ` + sentences.join(". ");
      try {
        await speakText(fullText, 1.1);
      } catch (e) {
        console.warn("Speech Synthesis failed.", e);
      }
    }
  }

  // Action 2: Cloud Scene Analysis
  async function performGroqSceneAnalysis() {
    const sw = snapshot.width;
    const sh = snapshot.height;

    sctx.clearRect(0, 0, sw, sh);
    sctx.drawImage(video, 0, 0, sw, sh);

    const sceneDescription = await analyzeSnapshotWithCloud(snapshot);
    
    if (sceneDescription.includes("Error:") || sceneDescription.includes("failed:")) {
      resultsList.innerHTML = `<strong style="color:#ff6b6b;">API Rejection:</strong> ${sceneDescription}`;
    } else {
      resultsList.innerHTML = `<strong style="color:var(--accent);">Scene Description:</strong> ${sceneDescription}`;
      try { await speakText(sceneDescription, 1.0); } catch (e) {}
    }
  }

  // === STATIC ACTION DELEGATIONS ===
  takeSnapshotBtn.addEventListener('click', performLocalObjectDetection);

  saveDataBtn.addEventListener('click', () => {
    if (!cycles.length) { alert('No tracking matrices compiled yet.'); return; }
    const lines = [];
    cycles.forEach(c => {
      lines.push(`detection ${c.timeStr} ${ordinal(c.cycleOrdinal)}`);
      if (!c.preds || c.preds.length === 0) {
        lines.push('no_objects_detected');
      } else {
        c.preds.forEach((p, i) => {
          lines.push(`object ${i+1} {${p.distanceStr}}{${p.coordX},${p.coordY}}{${p.score}%}`);
        });
      }
      lines.push('');
    });

    const content = lines.join('\n');
    const now = new Date();
    const filename = `saves/detection_${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())}_${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}.txt`;
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    cycles = [];
    nextCycleNumber = 1;
    detectedStrip.innerHTML = '<div style="color:var(--muted);padding:6px 10px;border-radius:8px">No snapshot yet</div>';
    resultsList.textContent = 'Take a snapshot to see the detected objects displayed below.';
    sctx.clearRect(0,0,snapshot.width,snapshot.height);
    alert('Matrix storage cleared and session logs written successfully.');
  });

  scanBtn.addEventListener('click', async () => {
    await populateCameras();
    alert('Hardware query complete. Check the selection mapping matrix.');
  });

  cameraSelect.addEventListener('change', async (e) => {
    await setupCamera(e.target.value);
  });

  startBtn.addEventListener('click', () => {
    if (running) {
      stopLive();
      startBtn.textContent = '▶ Start';
    } else {
      startLive();
      startBtn.textContent = '⏸ Pause';
    }
  });

  // === KEYBOARD SHORTCUTS CONTROLLER ===
  window.addEventListener('keydown', async (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    const key = e.key.toLowerCase();

    if (e.key === ' ' || e.code === 'Space') {
      e.preventDefault(); 
      await performLocalObjectDetection();
    }

    if (key === 's') {
      e.preventDefault();
      await performGroqSceneAnalysis();
    }

    if (key === 't') {
      e.preventDefault();
      await performTextRecognition();
    }

    if ((e.ctrlKey || e.metaKey) && key === 'd') {
      e.preventDefault(); 
      saveDataBtn.click();
    }
  });

  // === BACKGROUND VEHICLE VALIDATOR (COCO-SSD) ===
  const POLL_INTERVAL = 700;
  const ALERT_COOLDOWN = 5000;
  
  const flash = document.createElement('div');
  Object.assign(flash.style, {
    position: 'fixed', left: '0', top: '0', width: '100vw', height: '100vh',
    pointerEvents: 'none', background: 'rgba(255,0,0,0)', transition: 'background 220ms ease', zIndex: 99999,
  });
  document.body.appendChild(flash);

  const threatCanvas = document.createElement('canvas');
  threatCanvas.id = 'threatOverlay';
  threatCanvas.style.position = 'absolute';
  threatCanvas.style.left = '0';
  threatCanvas.style.top = '0';
  threatCanvas.style.pointerEvents = 'none';
  threatCanvas.style.zIndex = 99998;
  document.body.appendChild(threatCanvas);
  const tctx = threatCanvas.getContext('2d');

  function resizeThreatCanvas() {
    if(!video) return;
    const rect = video.getBoundingClientRect();
    threatCanvas.width = rect.width;
    threatCanvas.height = rect.height;
    threatCanvas.style.left = `${rect.left + window.scrollX}px`;
    threatCanvas.style.top = `${rect.top + window.scrollY}px`;
  }
  window.addEventListener('resize', resizeThreatCanvas);
  window.addEventListener('scroll', resizeThreatCanvas);
  setTimeout(resizeThreatCanvas, 200);

  function doFlash() {
    flash.style.background = 'rgba(255,0,0,0.12)';
    setTimeout(() => flash.style.background = 'rgba(255,0,0,0)', 220);
  }

  function mapToThreatLabel(cls) {
    return THREAT_CLASSES.has(cls) ? 'vehicle' : cls;
  }

  let threatRunning = false;
  let pollHandle = null;
  const lastAlertTime = new Map();

  async function detectionPass() {
    if (!threatRunning) return;
    
    if (video.readyState < 2 || !cocoModel) { 
      pollHandle = setTimeout(detectionPass, POLL_INTERVAL);
      return;
    }
    resizeThreatCanvas();

    try {
      const preds = await cocoModel.detect(video);
      const foundBoxes = [];
      const foundScores = [];
      const foundClasses = [];

      for (let p of preds) {
        const cls = p.class.toLowerCase();
        if (THREAT_CLASSES.has(cls) && p.score >= 0.5) {
          foundBoxes.push(p.bbox);
          foundScores.push(p.score);
          foundClasses.push(cls);
        }
      }

      tctx.clearRect(0, 0, threatCanvas.width, threatCanvas.height);
      if (foundBoxes.length) {
        tctx.lineWidth = Math.max(2, Math.round(threatCanvas.width * 0.004));
        for (let i = 0; i < foundBoxes.length; i++) {
          const [x, y, w, h] = foundBoxes[i];
          tctx.strokeStyle = 'rgba(255,60,60,0.95)';
          tctx.fillStyle = 'rgba(255,60,60,0.12)';
          tctx.strokeRect(x, y, w, h);
          tctx.fillRect(x, y, w, Math.min(28, h));
          tctx.font = `12px Arial`;
          tctx.fillStyle = '#fff';
          tctx.fillText(`${foundClasses[i]} ${Math.round(foundScores[i]*100)}%`, x + 6, y + 16);
        }

        const now = Date.now();
        const alertedThisPass = new Set();
        for (let i = 0; i < foundClasses.length; i++) {
          const label = mapToThreatLabel(foundClasses[i]);
          if (alertedThisPass.has(label)) continue; 
          const last = lastAlertTime.get(label) || 0;
          if (now - last < ALERT_COOLDOWN) continue;

          lastAlertTime.set(label, now);
          alertedThisPass.add(label);

          (() => {
            try {
              doFlash();
              const beepAudio = new Audio('beep.wav');
              beepAudio.volume = 1;
              beepAudio.play().catch(e => console.warn('Audio driver initialization fail:', e));
              const utter = new SpeechSynthesisUtterance(`Alert! ${label} detected in the live frame`);
              utter.lang = 'en-US';
              utter.rate = 0.95;
              utter.pitch = 1.0;
              window.speechSynthesis.speak(utter);
            } catch (e) {
              console.warn('Threat audio driver fallback dropped:', e);
            }
          })();
        }
        setTimeout(() => tctx.clearRect(0, 0, threatCanvas.width, threatCanvas.height), 900);
      } else {
        tctx.clearRect(0, 0, threatCanvas.width, threatCanvas.height);
      }
    } catch (err) {
      console.warn('Threat engine tracking pass structural failure:', err);
    }
    
    pollHandle = setTimeout(detectionPass, POLL_INTERVAL);
  }

  window.threatDetector = {
    start() {
      if (threatRunning) return;
      threatRunning = true;
      detectionPass();
      console.log('Background perimeter threat validator actively pooling frames.');
    },
    stop() {
      threatRunning = false;
      if (pollHandle) { clearTimeout(pollHandle); pollHandle = null; }
      tctx.clearRect(0, 0, threatCanvas.width, threatCanvas.height);
      flash.style.background = 'rgba(0,0,0,0)';
      console.log('Background perimeter threat validator offline.');
    },
    isRunning() { return threatRunning; }
  };

  // === LIFECYCLE INITIALIZER ===
  async function init() {
    try {
      statusLine.classList.add('loading');
      statusLine.textContent = 'Fetching definitions & staging ONNX matrices...';
      
      await fetchClasses();
      await setupCamera();
      await populateCameras();
      
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;

      ortSession = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ['wasm', 'webgl'],
        graphOptimizationLevel: 'all'
      });

      cocoModel = await cocoSsd.load();

      statusLine.classList.remove('loading');
      statusLine.classList.add('ready');
      statusLine.textContent = '✅ System initialization complete';
      
      startLive();
      window.threatDetector.start();
    } catch (e) {
      console.error("System staging halted due to loading failure:", e);
      statusLine.textContent = 'Initialization halted';
    }
  }

  await init();
})();

// ── VISION_OS UI Patch ──
(function() {
  const realStatus = document.getElementById('statusLine');
  const statusTextEl = document.getElementById('statusText');
  if (realStatus && statusTextEl) {
    const observer = new MutationObserver(() => {
      const text = Array.from(realStatus.childNodes)
        .filter(n => n.nodeType === 3)
        .map(n => n.textContent.trim())
        .join('');
      if (text) statusTextEl.textContent = text;
    });
    observer.observe(realStatus, { childList: true, characterData: true, subtree: true });
  }

  const snapBtn = document.getElementById('takeSnapshot');
  const idleOverlay = document.getElementById('snapshotIdle');
  if (snapBtn && idleOverlay) {
    snapBtn.addEventListener('click', () => {
      idleOverlay.classList.add('hidden');
    }, { once: true });
  }

  const strip = document.getElementById('detectedStrip');
  const stripCount = document.getElementById('stripCount');
  if (strip && stripCount) {
    const mo = new MutationObserver(() => {
      const cards = strip.querySelectorAll('.obj-card');
      stripCount.textContent = cards.length + ' object' + (cards.length !== 1 ? 's' : '');
    });
    mo.observe(strip, { childList: true, subtree: true });
  }
})();
