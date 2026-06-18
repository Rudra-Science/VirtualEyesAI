// --- SYSTEM ARCHITECTURE LOCKS ---
const HARDWARE_STATE = Object.freeze({
    MONITOR: "KZ Castor",
    TUNING_SWITCH_OVERRIDE: "1110", 
    SCHOOL_DEPLOYMENT: false 
});

const COOLDOWN_MATRIX = Object.freeze({
    FACE_API_THROTTLE: 1000, 
    THREAT_POLL: 700, 
    THREAT_SPAM_GATE: 5000 
});

// Model slider logic
const slider = document.getElementById('modelSlider');
const labelLeft = document.getElementById('label-left');
const labelRight = document.getElementById('label-right');

if (slider) {
  slider.addEventListener('input', () => {
    if (slider.value === '0') {
      labelLeft.classList.add('active');
      labelRight.classList.remove('active');
      slider.style.background = 'linear-gradient(90deg, var(--accent-cyan) 50%, var(--edge) 50%)';
    } else {
      labelRight.classList.add('active');
      labelLeft.classList.remove('active');
      slider.style.background = 'linear-gradient(90deg, var(--edge) 50%, var(--accent-cyan) 50%)';
    }
  });
}

// Camera selector logic
if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
  navigator.mediaDevices.enumerateDevices().then(devices => {
    const cameras = devices.filter(d => d.kind === 'videoinput');
    const sel = document.getElementById('cameraSelect');
    if (cameras.length > 0 && sel) {
      sel.innerHTML = '';
      cameras.forEach((cam, i) => {
        const opt = document.createElement('option');
        opt.value = cam.deviceId;
        opt.textContent = '⬤ ' + (cam.label || `Camera ${i + 1}`);
        sel.appendChild(opt);
      });
    }
  }).catch(() => {});
}

// Browser Protection Layer
document.addEventListener('contextmenu', (event) => event.preventDefault());
document.addEventListener('selectstart', (event) => event.preventDefault());
document.body.style.cursor = 'default';