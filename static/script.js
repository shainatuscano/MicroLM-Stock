/* ═══════════════════════════════════════════════════════
   MicroLM — Frontend Logic
   ═══════════════════════════════════════════════════════ */

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
    initSliders();
    initParticles();
    loadModelInfo();
    loadTrainingHistory();
});

// ── Slider Controls ──
// ── Slider Controls ──
function initSliders() {
    const sliders = [
        { id: 'temperatureSlider', display: 'tempValue', decimals: 2 },
        { id: 'topKSlider', display: 'topKValue', decimals: 0 },
        { id: 'topPSlider', display: 'topPValue', decimals: 2 },
        { id: 'repPenaltySlider', display: 'repValue', decimals: 2 },
        { id: 'ngramSlider', display: 'ngramValue', decimals: 0 },
        { id: 'maxLengthSlider', display: 'lengthValue', decimals: 0 },
    ];

    sliders.forEach(({ id, display, decimals }) => {
        const slider = document.getElementById(id);
        const value = document.getElementById(display);
        if (slider && value) {
            slider.addEventListener('input', () => {
                value.textContent = parseFloat(slider.value).toFixed(decimals);
            });
        }
    });
}

// ── Background Particles ──
function initParticles() {
    const container = document.getElementById('bgParticles');
    if (!container) return;

    for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 8 + 's';
        particle.style.animationDuration = (5 + Math.random() * 8) + 's';
        particle.style.opacity = (0.2 + Math.random() * 0.4).toString();
        const size = 2 + Math.random() * 3;
        particle.style.width = size + 'px';
        particle.style.height = size + 'px';
        container.appendChild(particle);
    }
}

// ── Text Generation ──
async function generateText() {
    const btn = document.getElementById('generateBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnIcon = btn.querySelector('.btn-icon');
    const btnLoading = btn.querySelector('.btn-loading');
    const outputArea = document.getElementById('outputArea');
    const copyBtn = document.getElementById('copyBtn');

    const prompt = document.getElementById('promptInput').value.trim();
    if (!prompt) {
        showToast('Please enter a prompt', 'error');
        return;
    }

    const temperature = parseFloat(document.getElementById('temperatureSlider').value);
    const topK = parseInt(document.getElementById('topKSlider').value);
    const topP = parseFloat(document.getElementById('topPSlider')?.value || 0.85);
    const repPenalty = parseFloat(document.getElementById('repPenaltySlider')?.value || 1.08);
    const noRepeatNgram = parseInt(document.getElementById('ngramSlider')?.value || 3);
    const maxLength = parseInt(document.getElementById('maxLengthSlider').value);

    // Show loading state
    btn.disabled = true;
    btnText.style.display = 'none';
    btnIcon.style.display = 'none';
    btnLoading.style.display = 'flex';
    outputArea.innerHTML = '<div class="output-placeholder"><div class="spinner" style="width:24px;height:24px;border-color:rgba(0,212,170,0.2);border-top-color:#00d4aa;"></div><p style="margin-top:1rem;">Generating text...</p></div>';

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                temperature: temperature,
                top_k: topK,
                top_p: topP,
                repetition_penalty: repPenalty,
                no_repeat_ngram_size: noRepeatNgram,
                max_length: maxLength,
            }),
        });

        const data = await response.json();

        if (data.success) {
            const promptText = data.prompt || prompt;
            let continuationPart = data.continuation !== undefined ? data.continuation : (data.generated_text || '').substring(promptText.length);
            if (continuationPart && !promptText.endsWith(' ') && !continuationPart.startsWith(' ') && !/^[.,!?;:'"\]\)]/.test(continuationPart)) {
                continuationPart = ' ' + continuationPart;
            }
            const genTime = data.generation_time || 0;

            // Typing animation
            outputArea.innerHTML = '';
            copyBtn.style.display = 'block';

            const promptSpan = document.createElement('span');
            promptSpan.className = 'generated-prompt';
            promptSpan.textContent = promptText;
            outputArea.appendChild(promptSpan);

            const textSpan = document.createElement('span');
            textSpan.className = 'generated-text';
            outputArea.appendChild(textSpan);

            const cursor = document.createElement('span');
            cursor.className = 'cursor-blink';
            outputArea.appendChild(cursor);

            let charIndex = 0;

            function typeChar() {
                if (charIndex < continuationPart.length) {
                    textSpan.textContent += continuationPart[charIndex];
                    charIndex++;
                    // Auto-scroll
                    outputArea.scrollTop = outputArea.scrollHeight;
                    // Vary speed slightly for natural feel
                    const delay = 8 + Math.random() * 12;
                    setTimeout(typeChar, delay);
                } else {
                    cursor.remove();
                    // Show generation time
                    if (genTime > 0) {
                        const timeTag = document.createElement('div');
                        timeTag.className = 'gen-time-badge';
                        timeTag.textContent = `⚡ Generated in ${genTime}s`;
                        outputArea.appendChild(timeTag);
                    }
                }
            }

            typeChar();
            showToast('Text generated successfully!', 'success');
        } else {
            const errorMsg = data.error || 'Unknown error';
            const isModelMissing = errorMsg.includes('train') || errorMsg.includes('checkpoint');
            outputArea.innerHTML = `<div class="output-placeholder"><div class="placeholder-icon">${isModelMissing ? '🔧' : '❌'}</div><p>${errorMsg}</p>${isModelMissing ? '<p class="placeholder-hint">Run <code>python train.py</code> in the terminal first</p>' : ''}</div>`;
            showToast('Generation failed: ' + errorMsg, 'error');
        }
    } catch (error) {
        outputArea.innerHTML = `<div class="output-placeholder"><div class="placeholder-icon">❌</div><p>Connection error. Is the server running?</p><p class="placeholder-hint">${error.message}</p></div>`;
        showToast('Connection error', 'error');
    } finally {
        btn.disabled = false;
        btnText.style.display = 'inline';
        btnIcon.style.display = 'inline';
        btnLoading.style.display = 'none';
    }
}

// ── Quick Prompts ──
function setPrompt(text) {
    const textarea = document.getElementById('promptInput');
    textarea.value = text;
    textarea.focus();
    // Subtle highlight animation
    textarea.style.borderColor = 'rgba(0, 212, 170, 0.6)';
    textarea.style.boxShadow = '0 0 0 3px rgba(0, 212, 170, 0.1)';
    setTimeout(() => {
        textarea.style.borderColor = '';
        textarea.style.boxShadow = '';
    }, 500);
}

// ── Copy Output ──
function copyOutput() {
    const outputArea = document.getElementById('outputArea');
    const text = outputArea.textContent;
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard!', 'success');
    }).catch(() => {
        showToast('Failed to copy', 'error');
    });
}

// ── Load Model Info ──
async function loadModelInfo() {
    const container = document.getElementById('modelInfoContent');

    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();

        if (data.success) {
            const info = data.info;
            let html = '';

            const items = [
                { label: 'Architecture', value: 'Decoder-Only Transformer', accent: true },
                { label: 'Parameters', value: info.parameters, accent: true },
                { label: 'Embed Dim', value: info.embed_dim },
                { label: 'Attention Heads', value: info.num_heads },
                { label: 'Layers', value: info.num_layers },
                { label: 'FF Dim', value: info.ff_dim },
                { label: 'Seq Length', value: info.seq_length },
                { label: 'Vocab Size', value: info.vocab_size },
                { label: 'Tokenizer', value: info.tokenizer },
            ];

            if (info.training) {
                items.push(
                    { label: 'Epochs', value: info.training.epochs },
                    { label: 'Train Loss', value: info.training.final_loss, accent: true },
                    { label: 'Val Loss', value: info.training.final_val_loss, accent: true },
                    { label: 'Train Time', value: info.training.training_time + 's' },
                );
            }

            items.forEach(item => {
                const accentClass = item.accent ? ' accent' : '';
                const fullWidthClass = item.label === 'Architecture' ? ' full-width' : '';
                html += `
                    <div class="info-item${fullWidthClass}">
                        <span class="info-label">${item.label}</span>
                        <span class="info-value${accentClass}">${item.value}</span>
                    </div>
                `;
            });

            container.innerHTML = html;
        }
    } catch (error) {
        container.innerHTML = `
            <div class="info-item full-width">
                <span class="info-label">Status</span>
                <span class="info-value" style="color: var(--warning);">Unable to connect to server</span>
            </div>
        `;
    }
}

// ── Training History Charts ──
let lossChart = null;
let accuracyChart = null;

async function loadTrainingHistory() {
    try {
        const response = await fetch('/api/training-history');
        const data = await response.json();

        if (data.success && data.history) {
            renderCharts(data.history);
        }
    } catch (error) {
        console.log('Training history not available:', error.message);
    }
}

function renderCharts(history) {
    const chartDefaults = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: {
                labels: {
                    color: '#94a3b8',
                    font: { family: "'Inter', sans-serif", size: 11 },
                    boxWidth: 12,
                    padding: 12,
                },
            },
        },
        scales: {
            x: {
                grid: { color: 'rgba(255,255,255,0.04)' },
                ticks: { color: '#64748b', font: { size: 10 } },
                title: { display: true, text: 'Epoch', color: '#64748b', font: { size: 11 } },
            },
            y: {
                grid: { color: 'rgba(255,255,255,0.04)' },
                ticks: { color: '#64748b', font: { size: 10 } },
            },
        },
    };

    const epochs = history.loss.map((_, i) => i + 1);

    // Loss Chart
    const lossCtx = document.getElementById('lossChart');
    if (lossCtx) {
        if (lossChart) lossChart.destroy();
        lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: 'Train Loss',
                        data: history.loss,
                        borderColor: '#00d4aa',
                        backgroundColor: 'rgba(0, 212, 170, 0.05)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        fill: true,
                        tension: 0.3,
                    },
                    {
                        label: 'Val Loss',
                        data: history.val_loss,
                        borderColor: '#0088ff',
                        backgroundColor: 'rgba(0, 136, 255, 0.05)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        fill: true,
                        tension: 0.3,
                    },
                ],
            },
            options: {
                ...chartDefaults,
                plugins: {
                    ...chartDefaults.plugins,
                    title: {
                        display: true,
                        text: 'Loss',
                        color: '#f1f5f9',
                        font: { family: "'Inter', sans-serif", size: 13, weight: 600 },
                        padding: { bottom: 8 },
                    },
                },
                scales: {
                    ...chartDefaults.scales,
                    y: {
                        ...chartDefaults.scales.y,
                        title: { display: true, text: 'Loss', color: '#64748b', font: { size: 11 } },
                    },
                },
            },
        });
    }

    // Accuracy Chart
    const accCtx = document.getElementById('accuracyChart');
    if (accCtx && history.accuracy) {
        if (accuracyChart) accuracyChart.destroy();
        accuracyChart = new Chart(accCtx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: 'Train Accuracy',
                        data: history.accuracy,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.05)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        fill: true,
                        tension: 0.3,
                    },
                    {
                        label: 'Val Accuracy',
                        data: history.val_accuracy,
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.05)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        fill: true,
                        tension: 0.3,
                    },
                ],
            },
            options: {
                ...chartDefaults,
                plugins: {
                    ...chartDefaults.plugins,
                    title: {
                        display: true,
                        text: 'Accuracy',
                        color: '#f1f5f9',
                        font: { family: "'Inter', sans-serif", size: 13, weight: 600 },
                        padding: { bottom: 8 },
                    },
                },
                scales: {
                    ...chartDefaults.scales,
                    y: {
                        ...chartDefaults.scales.y,
                        title: { display: true, text: 'Accuracy', color: '#64748b', font: { size: 11 } },
                        min: 0,
                        max: 1,
                    },
                },
            },
        });
    }
}

// ── Toast Notifications ──
function showToast(message, type = 'success') {
    // Remove existing toasts
    document.querySelectorAll('.toast').forEach(t => t.remove());

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => toast.remove(), 3000);
}

// ── Keyboard Shortcut ──
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        generateText();
    }
});
