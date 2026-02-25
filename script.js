// script.js - uses face-api.js to compute actual embeddings
const MODEL_URL = 'https://justadudewhohacks.github.io/face-api.js/models';

const video = document.getElementById('video');
const loading = document.getElementById('loading');
const enrollBtn = document.getElementById('captureEnroll');
const verifyBtn = document.getElementById('captureVerify');
const enrollContainer = document.getElementById('enrollContainer');
const verifyContainer = document.getElementById('verifyContainer');
const resultBox = document.getElementById('result');
const threshInput = document.getElementById('thresh');

let enrollDescriptor = null;
let verifyDescriptor = null;

// 1) Start camera
async function startCamera(){
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.play();
  } catch (e) {
    alert('Camera access is required. Check permissions.');
    console.error(e);
  }
}

// 2) Load models
async function loadModels() {
  loading.style.display = 'block';
  await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
  await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);
  loading.style.display = 'none';
}

// helper: format descriptor (Float32Array) to array of 128 numbers rounded
function formatDescriptor(desc){
  const arr = Array.from(desc).map(n => Number(n.toFixed(4)));
  return arr;
}

// helper: render descriptor array into container, with optional highlight array (boolean)
function renderDescriptor(container, arr, highlight = null){
  container.innerHTML = '';
  arr.forEach((v, i) => {
    const el = document.createElement('div');
    el.className = 'elem ' + (highlight ? (highlight[i] ? 'match' : 'diff') : '');
    el.textContent = v;
    container.appendChild(el);
  });
}

// compute Euclidean distance
function euclidean(a,b){
  let s = 0;
  for(let i=0;i<a.length;i++){
    const d = a[i] - b[i];
    s += d*d;
  }
  return Math.sqrt(s);
}

// get descriptor for current video frame
async function getDescriptorFromVideo(){
  // detect single face with landmarks and descriptor
  const detection = await faceapi.detectSingleFace(video).withFaceLandmarks().withFaceDescriptor();
  if(!detection) return null;
  return detection.descriptor; // Float32Array(128)
}

// Enrollment capture
enrollBtn.addEventListener('click', async () => {
  resultBox.textContent = 'Capturing enrollment…';
  const desc = await getDescriptorFromVideo();
  if(!desc){ resultBox.textContent = 'No face detected — try again.'; return; }
  enrollDescriptor = formatDescriptor(desc);
  renderDescriptor(enrollContainer, enrollDescriptor);
  resultBox.textContent = 'Enrollment captured.';
  verifyBtn.disabled = false;
});

// Verification capture
verifyBtn.addEventListener('click', async () => {
  resultBox.textContent = 'Capturing verification…';
  const desc = await getDescriptorFromVideo();
  if(!desc){ resultBox.textContent = 'No face detected — try again.'; return; }
  verifyDescriptor = formatDescriptor(desc);

  // compute element-wise matches (absolute diff < tolerance)
  const tolerance = 0.08; // per-element tolerance (tweakable)
  const matchVector = enrollDescriptor.map((v,i) => Math.abs(v - verifyDescriptor[i]) <= tolerance);

  // compute overall Euclidean distance (standard for face embeddings)
  const dist = euclidean(enrollDescriptor, verifyDescriptor);
  const threshold = parseFloat(threshInput.value) || 0.6;
  const decision = dist < threshold ? '✅ ACCESS GRANTED' : '❌ ACCESS DENIED';

  // render both descriptors with highlighting
  renderDescriptor(enrollContainer, enrollDescriptor, matchVector);
  renderDescriptor(verifyContainer, verifyDescriptor, matchVector);

  resultBox.innerHTML = `Distance: <strong>${dist.toFixed(4)}</strong> &nbsp;&nbsp; Decision: <strong>${decision}</strong>`;
});

// init
(async () => {
  await loadModels();
  await startCamera();
})();