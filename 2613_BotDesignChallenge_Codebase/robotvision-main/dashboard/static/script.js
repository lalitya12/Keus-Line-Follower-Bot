document.addEventListener('DOMContentLoaded', () => {
    const socket = io();

    // DOM Elements
    const connectionLed = document.getElementById('connection-led');
    const connectionText = document.getElementById('connection-text');
    const cameraFeed = document.getElementById('camera-feed');
    const logsContainer = document.getElementById('logs');
    const targetsGrid = document.getElementById('targets-grid');
    const targetCountEl = document.getElementById('target-count');
    
    // Timer Elements
    const runTimerEl = document.getElementById('run-timer');
    const btnTimerToggle = document.getElementById('btn-timer-toggle');

    let filledSlots = 0;
    
    // Timer Logic
    let timerInterval = null;
    let timerRunning = false;
    let elapsedSeconds = 0;

    function formatTime(totalSeconds) {
        const mins = Math.floor(totalSeconds / 60).toString().padStart(2, '0');
        const secs = (totalSeconds % 60).toString().padStart(2, '0');
        return `${mins}:${secs}`;
    }

    btnTimerToggle.addEventListener('click', () => {
        if (timerRunning) {
            // Stop timer
            clearInterval(timerInterval);
            timerRunning = false;
            btnTimerToggle.innerText = 'START';
            btnTimerToggle.classList.remove('btn-danger');
            btnTimerToggle.classList.add('btn-primary');
            addLog(`Timer stopped at ${formatTime(elapsedSeconds)}`, 'system');
        } else {
            // Start timer
            timerRunning = true;
            btnTimerToggle.innerText = 'STOP';
            btnTimerToggle.classList.remove('btn-primary');
            btnTimerToggle.classList.add('btn-danger');
            addLog('Timer started', 'system');
            
            timerInterval = setInterval(() => {
                elapsedSeconds++;
                runTimerEl.innerText = formatTime(elapsedSeconds);
            }, 1000);
        }
    });

    // Helper: Add Log Entry
    function addLog(message, type = 'system') {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
        
        const line = document.createElement('div');
        line.className = `log-line ${type}`;
        line.innerHTML = `<span class="log-time">[${timeStr}]</span> ${message}`;
        
        logsContainer.appendChild(line);
        logsContainer.scrollTop = logsContainer.scrollHeight;
    }

    // Connect/Disconnect events
    socket.on('connect', () => {
        connectionLed.classList.add('connected');
        connectionText.innerText = 'Connected';
        addLog('Connected to backend server.', 'system');
    });

    socket.on('disconnect', () => {
        connectionLed.classList.remove('connected');
        connectionText.innerText = 'Disconnected';
        addLog('Lost connection to backend server.', 'system');
    });

    // Frame update
    socket.on('update_frame', (data) => {
        if (data.image) {
            cameraFeed.src = 'data:image/jpeg;base64,' + data.image;
        }
    });

    // New recognition result
    socket.on('new_recognition', (data) => {
        if (data.text) {
            const lines = data.text.split('\n');
            let foundCategory = "Unknown Category";
            let foundContent = "Unknown Content";
            
            lines.forEach(line => {
                if(line.trim() !== '') {
                    addLog(line, 'result');
                    if (line.includes('Category:')) {
                        foundCategory = line.trim();
                    } else if (line.includes('Content:')) {
                        foundContent = line.trim();
                    }
                }
            });

            // Make sure we have an image
            if (data.image) {
                filledSlots++;
                targetCountEl.innerText = filledSlots;
                addLog(`Saved new target #${filledSlots}`, 'system');

                // Create a new slot and prepend it so newest is on top
                const slot = document.createElement('div');
                slot.className = 'target-slot filled slide-down';
                slot.id = `slot-${filledSlots}`;
                
                slot.innerHTML = `
                    <div class="slot-index">${filledSlots}</div>
                    <img class="target-img" src="data:image/jpeg;base64,${data.image}" alt="Target ${filledSlots}">
                    <div class="target-info">
                        <div class="target-category">${foundCategory}</div>
                        <div class="target-content">${foundContent}</div>
                    </div>
                `;
                
                targetsGrid.prepend(slot);
            }
        }
    });

    // Add a test log initially
    addLog('Awaiting video stream and telemetry from robot...', 'system');
});
