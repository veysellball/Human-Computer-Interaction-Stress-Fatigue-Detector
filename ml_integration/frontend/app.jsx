/* ============================================================
   app.jsx — React Frontend for Stress & Fatigue Detection
   State Machine: IDLE → ACTIVE ⇄ AFK / PAUSED
   Tumbling Window: 3-min with suspend/resume
   Inactivity Timer: 60s debounce → AFK
   ============================================================ */

const { useState, useEffect, useRef, useCallback, useMemo } = React;

const API_BASE = window.location.origin;

// -----------------------------------------------
// CONSTANTS (from State Machine Design)
// -----------------------------------------------
const WINDOW_DURATION = 20 * 1000;       // 20 saniye — tumbling window süresi
const INACTIVITY_TIMEOUT = 60 * 1000;     // 1 dakika — AFK eşiği
const MIN_WINDOW_DURATION = 5 * 1000;     // 5 sn — E8 minimum anlamlı pencere
const MIN_EVENT_COUNT = 10;               // E8 minimum anlamlı event sayısı
const MAX_BUFFER_SIZE = 10000;            // E7 ring buffer limiti
const THROTTLE_MS = 33;                   // ~30Hz throttle (mousemove/scroll için)

// State enum
const STATE = {
    IDLE: "IDLE",
    ACTIVE: "ACTIVE",
    AFK: "AFK",
    PAUSED: "PAUSED",
};

// Mood options
const MOODS = [
    { key: "energetic", emoji: "💪", label: "Energetic" },
    { key: "normal", emoji: "😐", label: "Normal" },
    { key: "tired", emoji: "😴", label: "Fatigued" },
];

// -----------------------------------------------
// HELPER: API calls
// -----------------------------------------------
async function apiPost(path, body) {
    const res = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        const errorText = await res.text().catch(() => "Unknown error");
        throw new Error(`HTTP ${res.status}: ${errorText.substring(0, 200)}`);
    }
    return res.json();
}

async function apiGet(path) {
    const res = await fetch(`${API_BASE}${path}`);
    if (!res.ok) {
        const errorText = await res.text().catch(() => "Unknown error");
        throw new Error(`HTTP ${res.status}: ${errorText.substring(0, 200)}`);
    }
    return res.json();
}

// -----------------------------------------------
// HELPER: Time formatting
// -----------------------------------------------
function formatTime(ms) {
    const totalSec = Math.max(0, Math.floor(ms / 1000));
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
}

function timeNow() {
    return new Date().toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// -----------------------------------------------
// CUSTOM HOOK: useOnlineStatus
// -----------------------------------------------
function useOnlineStatus() {
    const [isOnline, setIsOnline] = useState(navigator.onLine);

    useEffect(() => {
        const goOnline = () => setIsOnline(true);
        const goOffline = () => setIsOnline(false);
        window.addEventListener("online", goOnline);
        window.addEventListener("offline", goOffline);
        return () => {
            window.removeEventListener("online", goOnline);
            window.removeEventListener("offline", goOffline);
        };
    }, []);

    return isOnline;
}

// -----------------------------------------------
// CORE: useTelemetryStateMachine
// The heart of the state machine — manages all
// states, transitions, timers, buffers, and
// backend communication.
// -----------------------------------------------
function useTelemetryStateMachine(addLog) {
    // ---- STATE ----
    const [machineState, setMachineState] = useState(STATE.IDLE);
    const [sessionId, setSessionId] = useState(null);
    const [username, setUsername] = useState(null);

    // ---- REFS (mutable, no re-render) ----
    const buffer = useRef([]);
    const windowTimerRef = useRef(null);
    const inactivityTimerRef = useRef(null);
    const remainingTimeRef = useRef(WINDOW_DURATION);
    const windowStartTimeRef = useRef(null);
    const lastEventTimestampRef = useRef(null);
    const machineStateRef = useRef(STATE.IDLE);
    const sessionIdRef = useRef(null);
    const usernameRef = useRef(null);
    const lastMouseMoveTimeRef = useRef(0);
    const lastScrollTimeRef = useRef(0);

    // ---- DISPLAY STATE ----
    const [bufferSize, setBufferSize] = useState(0);
    const [eventCounts, setEventCounts] = useState({ mouse: 0, keyboard: 0 });
    const [lastTelemetryResult, setLastTelemetryResult] = useState(null);

    // ---- RESUME FEEDBACK POPUP ----
    const [showResumeFeedbackPopup, setShowResumeFeedbackPopup] = useState(false);

    // Keep refs in sync with state
    useEffect(() => { machineStateRef.current = machineState; }, [machineState]);
    useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);
    useEffect(() => { usernameRef.current = username; }, [username]);


    // ---- RESYNC MECHANISM (Edge Case 1) ----
    useEffect(() => {
        if (machineState === STATE.IDLE || !sessionId) return;

        const checkStatus = async () => {
            try {
                await apiGet(`/api/status/${sessionId}`);
            } catch (err) {
                if (err.message.includes("404")) {
                    addLog("⚠️ Backend session lost (Server Restart?), re-syncing...", "warn");
                    apiPost("/api/sessions/start", {
                        session_id: sessionId,
                        user_id: username,
                        initial_stress: null,
                        initial_fatigue: null
                    }).catch(() => { });
                }
            }
        };

        const intervalId = setInterval(checkStatus, 30000); // 30 saniyede bir yokla
        return () => clearInterval(intervalId);
    }, [machineState, sessionId, username, addLog]);

    // ---- REMAINING TIME GETTER ----

    const getRemainingTime = useCallback(() => {
        if (machineStateRef.current !== STATE.ACTIVE) {
            return remainingTimeRef.current;
        }
        if (windowStartTimeRef.current !== null) {
            const elapsed = Date.now() - windowStartTimeRef.current;
            return Math.max(0, remainingTimeRef.current - elapsed);
        }
        return remainingTimeRef.current;
    }, []);

    // ---- HELPERS ----

    // Send pure telemetry data to backend (no state transitions)
    const sendTelemetry = useCallback(async (eventsArray) => {
        try {
            const mouseEvents = [];
            const keyboardEvents = [];

            for (const evt of eventsArray) {
                if (evt.category === "mouse") {
                    mouseEvents.push({ x: evt.x || 0, y: evt.y || 0, type: evt.type, time: evt.time });
                } else if (evt.category === "keyboard") {
                    keyboardEvents.push({ key: evt.key || "", type: evt.type, time: evt.time });
                }
            }

            const res = await apiPost("/api/telemetry", {
                session_id: sessionIdRef.current,
                user_id: usernameRef.current,
                mouse_events: mouseEvents,
                keyboard_events: keyboardEvents,
                window_events: [],
            });

            setLastTelemetryResult(res);
            addLog(`📡 Telemetry sent: ${eventsArray.length} events`, "success");
            return res;
        } catch (err) {
            addLog(`❌ Telemetry error: ${err.message}`, "error");
            return null;
        }
    }, [addLog]);

    // ---- CLEAR TIMERS ----
    const clearWindowTimer = useCallback(() => {
        if (windowTimerRef.current) {
            clearTimeout(windowTimerRef.current);
            windowTimerRef.current = null;
        }
    }, []);

    const clearInactivityTimer = useCallback(() => {
        if (inactivityTimerRef.current) {
            clearTimeout(inactivityTimerRef.current);
            inactivityTimerRef.current = null;
        }
    }, []);

    // ---- CALCULATE REMAINING TIME ----
    const suspendWindowTimer = useCallback(() => {
        clearWindowTimer();
        if (windowStartTimeRef.current !== null) {
            const elapsed = Date.now() - windowStartTimeRef.current;
            remainingTimeRef.current = Math.max(0, remainingTimeRef.current - elapsed);
        }
        windowStartTimeRef.current = null;
    }, [clearWindowTimer]);

    // ---- WINDOW TICK HANDLER ----
    const onWindowTick = useCallback(() => {
        if (machineStateRef.current !== STATE.ACTIVE) return;

        const currentBuffer = buffer.current.splice(0);
        setBufferSize(0);

        if (currentBuffer.length > 0) {
            sendTelemetry(currentBuffer);
            addLog(`⏱ Window completed: ${currentBuffer.length} events sent`);
        } else {
            addLog("⏱ Window completed: 0 events (empty window)");
        }

        // Reset window for next cycle
        remainingTimeRef.current = WINDOW_DURATION;
        windowStartTimeRef.current = Date.now();
        windowTimerRef.current = setTimeout(onWindowTick, WINDOW_DURATION);
    }, [sendTelemetry, addLog]);

    // ---- START WINDOW TIMER (with remainingTime) ----
    const startWindowTimer = useCallback((remaining) => {
        clearWindowTimer();
        remainingTimeRef.current = remaining;
        windowStartTimeRef.current = Date.now();
        windowTimerRef.current = setTimeout(onWindowTick, remaining);
    }, [clearWindowTimer, onWindowTick]);

    // ---- INACTIVITY TIMEOUT HANDLER ----
    const onInactivityTimeout = useCallback(() => {
        if (machineStateRef.current !== STATE.ACTIVE) return;

        addLog("💤 1 min inactivity — switching to AFK...", "warn");

        // Suspend tumbling window
        suspendWindowTimer();
        clearInactivityTimer();

        const afkStartedAt = lastEventTimestampRef.current || new Date().toISOString();

        // Sadece durum bildir, buffer flush etme (tumbling window paused)
        apiPost("/api/sessions/afk", {
            session_id: sessionIdRef.current,
            user_id: usernameRef.current,
            afk_started_at: afkStartedAt,
        }).catch(() => { });
        addLog(`⏳ Buffer frozen. Contains ${buffer.current.length} events.`, "info");

        // Transition to AFK — switch listeners to wake-detect mode
        machineStateRef.current = STATE.AFK;
        setMachineState(STATE.AFK);
    }, [suspendWindowTimer, clearInactivityTimer, addLog]);

    // ---- RESET INACTIVITY TIMER ----
    const resetInactivityTimer = useCallback(() => {
        clearInactivityTimer();
        inactivityTimerRef.current = setTimeout(onInactivityTimeout, INACTIVITY_TIMEOUT);
    }, [clearInactivityTimer, onInactivityTimeout]);

    // ---- EVENT HANDLER (ACTIVE mode — captures to buffer) ----
    const handleActiveEvent = useCallback((e) => {
        if (machineStateRef.current !== STATE.ACTIVE) return;

        const now = Date.now();
        const timestamp = new Date(now).toISOString();
        lastEventTimestampRef.current = timestamp;

        let eventData = null;

        switch (e.type) {
            case "mousemove":
                if (now - lastMouseMoveTimeRef.current < THROTTLE_MS) return;
                lastMouseMoveTimeRef.current = now;
                eventData = { category: "mouse", type: "move", x: e.clientX, y: e.clientY, time: now, timestamp };
                setEventCounts(prev => ({ ...prev, mouse: prev.mouse + 1 }));
                break;
            case "mousedown":
                eventData = { category: "mouse", type: "mousedown", x: e.clientX, y: e.clientY, time: now, timestamp };
                setEventCounts(prev => ({ ...prev, mouse: prev.mouse + 1 }));
                break;
            case "mouseup":
                eventData = { category: "mouse", type: "mouseup", x: e.clientX, y: e.clientY, time: now, timestamp };
                setEventCounts(prev => ({ ...prev, mouse: prev.mouse + 1 }));
                break;
            case "click":
                eventData = { category: "mouse", type: "click", x: e.clientX, y: e.clientY, time: now, timestamp };
                setEventCounts(prev => ({ ...prev, mouse: prev.mouse + 1 }));
                break;
            case "scroll":
            case "wheel":
                if (now - lastScrollTimeRef.current < THROTTLE_MS) return;
                lastScrollTimeRef.current = now;
                eventData = { category: "mouse", type: "scroll", x: 0, y: 0, time: now, timestamp };
                setEventCounts(prev => ({ ...prev, mouse: prev.mouse + 1 }));
                break;
            case "keydown":
                eventData = { category: "keyboard", type: "press", key: e.key.toLowerCase(), time: now, timestamp };
                setEventCounts(prev => ({ ...prev, keyboard: prev.keyboard + 1 }));
                break;
            case "keyup":
                eventData = { category: "keyboard", type: "release", key: e.key.toLowerCase(), time: now, timestamp };
                setEventCounts(prev => ({ ...prev, keyboard: prev.keyboard + 1 }));
                break;
        }

        if (eventData) {
            // E7: Ring buffer — drop oldest if over limit
            if (buffer.current.length >= MAX_BUFFER_SIZE) {
                buffer.current.shift();
            }
            buffer.current.push(eventData);
            setBufferSize(buffer.current.length);
        }

        // Reset inactivity timer on every interaction
        resetInactivityTimer();
    }, [resetInactivityTimer]);

    // ---- EVENT HANDLER (AFK mode — wake detection only) ----
    const handleWakeEvent = useCallback((e) => {
        if (machineStateRef.current !== STATE.AFK) return;

        // Prevent Pause button click from triggering wake logic directly.
        // The click will fire handleActiveEvent after state transitions to ACTIVE.
        // But due to DOM event order (mousemove fires before click), wake happens naturally.

        addLog("☀️ User returned — switching to ACTIVE...", "success");

        // Send wake signal to backend
        apiPost("/api/sessions/wake", {
            session_id: sessionIdRef.current,
            user_id: usernameRef.current,
        }).catch(() => { });

        // Transition: AFK → ACTIVE
        machineStateRef.current = STATE.ACTIVE;
        setMachineState(STATE.ACTIVE);

        // Resume tumbling window timer with saved remainingTime
        const remaining = remainingTimeRef.current;
        addLog(`⏱ Timer resuming from: ${formatTime(remaining)} remaining`);
        startWindowTimer(remaining);
        resetInactivityTimer();

        // The first wake event itself should be captured
        // Re-dispatch as active event
        const now = Date.now();
        const timestamp = new Date(now).toISOString();
        lastEventTimestampRef.current = timestamp;

        let eventData = null;
        if (e.type === "mousemove" || e.type === "mousedown" || e.type === "click") {
            eventData = { category: "mouse", type: e.type === "mousemove" ? "move" : e.type, x: e.clientX || 0, y: e.clientY || 0, time: now, timestamp };
            setEventCounts(prev => ({ ...prev, mouse: prev.mouse + 1 }));
        } else if (e.type === "keydown" || e.type === "keyup") {
            eventData = { category: "keyboard", type: e.type === "keydown" ? "press" : "release", key: e.key?.toLowerCase() || "", time: now, timestamp };
            setEventCounts(prev => ({ ...prev, keyboard: prev.keyboard + 1 }));
        }

        if (eventData) {
            buffer.current.push(eventData);
            setBufferSize(buffer.current.length);
        }
    }, [addLog, startWindowTimer, resetInactivityTimer]);

    // ---- LISTENER MANAGEMENT ----
    const activeListenersRef = useRef([]);

    const attachActiveListeners = useCallback(() => {
        const events = ["mousemove", "mousedown", "mouseup", "click", "keydown", "keyup", "scroll", "wheel"];
        events.forEach(eventType => {
            document.addEventListener(eventType, handleActiveEvent);
        });
        activeListenersRef.current = events.map(e => ({ event: e, handler: handleActiveEvent }));
        addLog("🎧 Event listeners started (full capture mode)", "success");
    }, [handleActiveEvent, addLog]);

    const detachActiveListeners = useCallback(() => {
        activeListenersRef.current.forEach(({ event, handler }) => {
            document.removeEventListener(event, handler);
        });
        activeListenersRef.current = [];
    }, []);

    const wakeListenersRef = useRef([]);

    const attachWakeListeners = useCallback(() => {
        const events = ["mousemove", "mousedown", "click", "keydown"];
        events.forEach(eventType => {
            document.addEventListener(eventType, handleWakeEvent);
        });
        wakeListenersRef.current = events.map(e => ({ event: e, handler: handleWakeEvent }));
        addLog("👁 Wake-detect listeners active (AFK mode)", "info");
    }, [handleWakeEvent, addLog]);

    const detachWakeListeners = useCallback(() => {
        wakeListenersRef.current.forEach(({ event, handler }) => {
            document.removeEventListener(event, handler);
        });
        wakeListenersRef.current = [];
    }, []);

    // ---- TRANSITION: AFK state change → swap listeners ----
    useEffect(() => {
        if (machineState === STATE.AFK) {
            detachActiveListeners();
            attachWakeListeners();
        }
        return () => {
            // Cleanup wake listeners if state changes away from AFK
            if (machineStateRef.current !== STATE.AFK) {
                detachWakeListeners();
            }
        };
    }, [machineState, detachActiveListeners, attachWakeListeners, detachWakeListeners]);

    // ---- TRANSITION: ACTIVE state → attach active listeners ----
    useEffect(() => {
        if (machineState === STATE.ACTIVE) {
            detachWakeListeners();
            attachActiveListeners();
        }
        return () => {
            if (machineStateRef.current !== STATE.ACTIVE) {
                detachActiveListeners();
            }
        };
    }, [machineState, detachWakeListeners, attachActiveListeners, detachActiveListeners]);

    // ---- E4: beforeunload → sendBeacon ----
    useEffect(() => {
        const handleUnload = () => {
            const state = machineStateRef.current;
            if (state === STATE.ACTIVE || state === STATE.AFK) {
                const currentBuffer = buffer.current;
                const mouseEvents = [];
                const keyboardEvents = [];

                for (const evt of currentBuffer) {
                    if (evt.category === "mouse") {
                        mouseEvents.push({ x: evt.x || 0, y: evt.y || 0, type: evt.type, time: evt.time });
                    } else if (evt.category === "keyboard") {
                        keyboardEvents.push({ key: evt.key || "", type: evt.type, time: evt.time });
                    }
                }

                const payload = {
                    session_id: sessionIdRef.current,
                    user_id: usernameRef.current,
                    reason: "interrupted",
                    mouse_events: mouseEvents,
                    keyboard_events: keyboardEvents,
                    afk_started_at: lastEventTimestampRef.current,
                };

                const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
                navigator.sendBeacon(`${API_BASE}/api/sessions/end`, blob);
            }
        };

        window.addEventListener("beforeunload", handleUnload);
        return () => window.removeEventListener("beforeunload", handleUnload);
    }, []);

    // ---- PUBLIC API: Session Start (IDLE → ACTIVE) ----
    const sessionStart = useCallback(async (name, stress, fatigue) => {
        if (machineStateRef.current !== STATE.IDLE) return;

        const sid = `session_${name}_${Date.now()}`;
        setSessionId(sid);
        setUsername(name);
        sessionIdRef.current = sid;
        usernameRef.current = name;

        addLog(`🚀 Starting session: ${name}`, "success");

        try {
            await apiPost("/api/sessions/start", {
                session_id: sid,
                user_id: name,
                initial_stress: stress,
                initial_fatigue: fatigue,
            });
            addLog("✅ Session start signal sent to backend", "success");
        } catch (err) {
            addLog(`⚠️ Session start could not be sent: ${err.message}`, "warn");
        }

        // Initialize
        buffer.current = [];
        setBufferSize(0);
        setEventCounts({ mouse: 0, keyboard: 0 });
        remainingTimeRef.current = WINDOW_DURATION;

        // Transition to ACTIVE
        machineStateRef.current = STATE.ACTIVE;
        setMachineState(STATE.ACTIVE);
        startWindowTimer(WINDOW_DURATION);
        resetInactivityTimer();
    }, [addLog, startWindowTimer, resetInactivityTimer]);

    // ---- PUBLIC API: Pause (ACTIVE → PAUSED) ----
    const pause = useCallback(() => {
        if (machineStateRef.current !== STATE.ACTIVE) return;

        addLog("⏸ Pause pressed — session pausing...", "warn");

        // Suspend timers & save remainingTime
        suspendWindowTimer();
        clearInactivityTimer();

        // Sadece durum bildir, buffer flush etme (tumbling window paused)
        apiPost("/api/sessions/pause", {
            session_id: sessionIdRef.current,
            user_id: usernameRef.current,
        }).catch(() => { });
        addLog(`⏳ Buffer frozen. Contains ${buffer.current.length} events.`, "info");

        // Detach all listeners
        detachActiveListeners();

        machineStateRef.current = STATE.PAUSED;
        setMachineState(STATE.PAUSED);
        addLog(`⏸ PAUSED — Time remaining: ${formatTime(remainingTimeRef.current)}`, "info");
    }, [suspendWindowTimer, clearInactivityTimer, detachActiveListeners, addLog]);

    // ---- PUBLIC API: Resume → Show Feedback Popup ----
    const requestResume = useCallback(() => {
        if (machineStateRef.current !== STATE.PAUSED) return;
        setShowResumeFeedbackPopup(true);
    }, []);

    // ---- PUBLIC API: Resume with Feedback (PAUSED → ACTIVE) ----
    const resumeWithFeedback = useCallback(async () => {
        if (machineStateRef.current !== STATE.PAUSED) return;

        setShowResumeFeedbackPopup(false);
        addLog(`▶️ Resume — Session continuing`, "success");

        // Send resume signal to backend
        try {
            await apiPost("/api/sessions/resume", {
                session_id: sessionIdRef.current,
                user_id: usernameRef.current,
            });
        } catch (err) {
            addLog(`⚠️ Resume signal could not be sent: ${err.message}`, "warn");
        }

        // Buffer'ı temizlemiyoruz, remainingğı yerden dolmaya devam edecek
        let remaining = remainingTimeRef.current;

        machineStateRef.current = STATE.ACTIVE;
        setMachineState(STATE.ACTIVE);
        startWindowTimer(remaining);
        resetInactivityTimer();
        addLog(`⏱ Timer continuing: ${formatTime(remaining)} remaining`);
    }, [addLog, startWindowTimer, resetInactivityTimer]);

    // ---- PUBLIC API: Cancel Resume Popup (E6) ----
    const cancelResume = useCallback(() => {
        setShowResumeFeedbackPopup(false);
        addLog("❌ Resume cancelled — staying in PAUSED", "info");
    }, [addLog]);

    // ---- CLEANUP on unmount ----
    useEffect(() => {
        return () => {
            clearWindowTimer();
            clearInactivityTimer();
            detachActiveListeners();
            detachWakeListeners();
        };
    }, [clearWindowTimer, clearInactivityTimer, detachActiveListeners, detachWakeListeners]);

    return {
        machineState,
        sessionId,
        username,
        getRemainingTime,
        bufferSize,
        eventCounts,
        lastTelemetryResult,
        showResumeFeedbackPopup,
        // Actions
        sessionStart,
        pause,
        requestResume,
        resumeWithFeedback,
        cancelResume,
    };
}

// -----------------------------------------------
// COMPONENT: SkeletonLoader
// -----------------------------------------------
function SkeletonLoader({ width = "100%", height = "2rem", borderRadius = "8px" }) {
    return (
        <div
            className="skeleton"
            style={{ width, height, borderRadius }}
        />
    );
}

// -----------------------------------------------
// COMPONENT: SparklineChart (Custom SVG)
// -----------------------------------------------
function SparklineChart({ data, color = "#3b82f6", height = 50, label = "" }) {
    if (!data || data.length < 2) {
        return (
            <div className="sparkline-empty">
                <span>Waiting for enough data…</span>
            </div>
        );
    }

    const width = 220;
    const padding = 4;
    const maxVal = Math.max(...data, 0.01);
    const minVal = Math.min(...data, 0);
    const range = maxVal - minVal || 1;

    const points = data.map((val, i) => {
        const x = padding + (i / (data.length - 1)) * (width - 2 * padding);
        const y = padding + (1 - (val - minVal) / range) * (height - 2 * padding);
        return `${x},${y}`;
    });

    const polyline = points.join(" ");
    const firstX = padding;
    const lastX = padding + ((data.length - 1) / (data.length - 1)) * (width - 2 * padding);
    const areaPath = `M ${firstX},${height} L ${points.map(p => p).join(" L ")} L ${lastX},${height} Z`;

    return (
        <div className="sparkline-container">
            {label && <span className="sparkline-label">{label}</span>}
            <svg viewBox={`0 0 ${width} ${height}`} className="sparkline-svg" preserveAspectRatio="none">
                <defs>
                    <linearGradient id={`grad-${label}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={color} stopOpacity="0.3" />
                        <stop offset="100%" stopColor={color} stopOpacity="0.0" />
                    </linearGradient>
                </defs>
                <path d={areaPath} fill={`url(#grad-${label})`} className="sparkline-area" />
                <polyline points={polyline} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="sparkline-line" />
                {data.length > 0 && (() => {
                    const lastPoint = points[points.length - 1].split(",");
                    return <circle cx={lastPoint[0]} cy={lastPoint[1]} r="3.5" fill={color} className="sparkline-dot" />;
                })()}
            </svg>
            <div className="sparkline-values">
                <span>{(data[0] * 100).toFixed(0)}%</span>
                <span>{(data[data.length - 1] * 100).toFixed(0)}%</span>
            </div>
        </div>
    );
}

// -----------------------------------------------
// COMPONENT: OfflineBanner
// -----------------------------------------------
function OfflineBanner() {
    return (
        <div className="offline-banner">
            <span className="offline-icon">📡</span>
            <span>Connection lost — Data will be buffered and sent when online.</span>
        </div>
    );
}

// -----------------------------------------------
// COMPONENT: StatusCard (Reusable prediction card)
// -----------------------------------------------
function StatusCard({ title, badgeText, badgeBg, badgeColor, value, valueClass, probabilities, probConfig }) {
    const isWaiting = value === "Waiting...";

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">{title}</span>
                {!isWaiting && (
                    <span className="card-badge" style={{ background: badgeBg, color: badgeColor }}>
                        {badgeText}
                    </span>
                )}
            </div>
            {isWaiting ? (
                <div className="skeleton-group">
                    <SkeletonLoader width="60%" height="2.2rem" borderRadius="8px" />
                    <SkeletonLoader width="100%" height="6px" borderRadius="3px" />
                    <SkeletonLoader width="100%" height="6px" borderRadius="3px" />
                </div>
            ) : (
                <>
                    <div className={`prediction-value ${valueClass}`}>{value}</div>
                    {probabilities && (
                        <div className="prob-bar-container">
                            {probConfig.map((item, idx) => (
                                <React.Fragment key={idx}>
                                    <div className="prob-bar-label">
                                        <span>{item.label}</span>
                                        <span>{(item.value * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="prob-bar-track">
                                        <div className={`prob-bar-fill ${item.fillClass}`} style={{ width: `${item.value * 100}%` }} />
                                    </div>
                                </React.Fragment>
                            ))}
                        </div>
                    )}
                </>
            )}
        </div>
    );
}

// -----------------------------------------------
// COMPONENT: Session Start Modal (IDLE state)
// -----------------------------------------------
function SessionStartModal({ onStart }) {
    const [name, setName] = useState("");
    const [stress, setStress] = useState(null);
    const [fatigue, setFatigue] = useState(null);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (name.trim() && stress && fatigue) {
            onStart(name.trim(), stress, fatigue);
        }
    };

    return (
        <div className="login-screen">
            <form className="login-card" onSubmit={handleSubmit}>
                <h1>🧠 Stress & Fatigue Detection</h1>
                <p>Enter your username and current state to start the session.</p>

                <div className="form-group">
                    <label htmlFor="username">Username</label>
                    <input
                        id="username"
                        type="text"
                        placeholder="Ex: john_doe"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        autoFocus
                    />
                </div>

                <div className="feedback-options" style={{ marginBottom: "1.5rem" }}>
                    <div className="feedback-option-group">
                        <label>Stress:</label>
                        <div className="options">
                            <button type="button" className={`fb-opt-btn ${stress === "Stressed" ? "selected" : ""}`} onClick={() => setStress("Stressed")}>😰 Stressed</button>
                            <button type="button" className={`fb-opt-btn ${stress === "Not_Stressed" ? "selected" : ""}`} onClick={() => setStress("Not_Stressed")}>😊 Not Stressed</button>
                        </div>
                    </div>
                    <div className="feedback-option-group">
                        <label>Fatigue:</label>
                        <div className="options">
                            <button type="button" className={`fb-opt-btn ${fatigue === "Fatigued" ? "selected" : ""}`} onClick={() => setFatigue("Fatigued")}>😴 Fatigued</button>
                            <button type="button" className={`fb-opt-btn ${fatigue === "Not_Fatigued" ? "selected" : ""}`} onClick={() => setFatigue("Not_Fatigued")}>⚡ Energetic</button>
                        </div>
                    </div>
                </div>

                <button type="submit" className="btn-primary" disabled={!name.trim() || !stress || !fatigue}>
                    Start Session
                </button>
            </form>
        </div>
    );
}

// -----------------------------------------------
// COMPONENT: Resume Mood Popup
// -----------------------------------------------
// -----------------------------------------------
// COMPONENT: Resume Feedback Popup
// -----------------------------------------------
function ResumeFeedbackPopup({ onSelect, onCancel }) {
    const [stress, setStress] = useState(null);
    const [fatigue, setFatigue] = useState(null);

    const handleResume = () => {
        if (stress && fatigue) {
            onSelect(stress, fatigue);
        }
    };

    return (
        <div className="login-screen" style={{ position: "fixed", inset: 0, zIndex: 2000, background: "rgba(0,0,0,0.85)" }}>
            <div className="login-card" style={{ backdropFilter: "none" }}>
                <h1>▶️ Continue</h1>
                <p>Please indicate your current state to improve the machine learning model while continuing your work.</p>

                <div className="feedback-options" style={{ marginBottom: "1.5rem" }}>
                    <div className="feedback-option-group">
                        <label>Stress:</label>
                        <div className="options">
                            <button className={`fb-opt-btn ${stress === "Stressed" ? "selected" : ""}`} onClick={() => setStress("Stressed")}>😰 Stressed</button>
                            <button className={`fb-opt-btn ${stress === "Not_Stressed" ? "selected" : ""}`} onClick={() => setStress("Not_Stressed")}>😊 Not Stressed</button>
                        </div>
                    </div>
                    <div className="feedback-option-group">
                        <label>Fatigue:</label>
                        <div className="options">
                            <button className={`fb-opt-btn ${fatigue === "Fatigued" ? "selected" : ""}`} onClick={() => setFatigue("Fatigued")}>😴 Fatigued</button>
                            <button className={`fb-opt-btn ${fatigue === "Not_Fatigued" ? "selected" : ""}`} onClick={() => setFatigue("Not_Fatigued")}>⚡ Energetic</button>
                        </div>
                    </div>
                </div>

                <button className="btn-primary" disabled={!stress || !fatigue} onClick={handleResume}>
                    Evaluate & Continue
                </button>
                <button className="btn-logout" style={{ width: "100%", marginTop: "0.75rem", padding: "0.7rem" }} onClick={onCancel}>
                    Cancel
                </button>
            </div>
        </div>
    );
}

// -----------------------------------------------
// COMPONENT: FeedbackPopup (Manual + Auto-trigger)
// -----------------------------------------------
function FeedbackPopup({ isAutoTrigger, onSend, onClose }) {
    const [stress, setStress] = useState(null);
    const [fatigue, setFatigue] = useState(null);

    const handleSend = () => {
        if (stress && fatigue) {
            onSend(stress, fatigue);
        }
    };

    return (
        <div className={`feedback-popup ${isAutoTrigger ? "auto-trigger" : ""}`}>
            <button className="feedback-close" onClick={onClose}>✕</button>
            <h3>{isAutoTrigger ? "⚠️ Attention — Stress Detected" : "📝 Feedback"}</h3>
            <p>
                {isAutoTrigger
                    ? "The system detected a sudden change in behavior. Could you verify your current state?"
                    : "Share your current mood with the system. This improves the data model."}
            </p>

            <div className="feedback-options">
                <div className="feedback-option-group">
                    <label>Stress:</label>
                    <div className="options">
                        <button className={`fb-opt-btn ${stress === "Stressed" ? "selected" : ""}`} onClick={() => setStress("Stressed")}>😰 Stressed</button>
                        <button className={`fb-opt-btn ${stress === "Not_Stressed" ? "selected" : ""}`} onClick={() => setStress("Not_Stressed")}>😊 Not Stressed</button>
                    </div>
                </div>
                <div className="feedback-option-group">
                    <label>Fatigue:</label>
                    <div className="options">
                        <button className={`fb-opt-btn ${fatigue === "Fatigued" ? "selected" : ""}`} onClick={() => setFatigue("Fatigued")}>😴 Fatigued</button>
                        <button className={`fb-opt-btn ${fatigue === "Not_Fatigued" ? "selected" : ""}`} onClick={() => setFatigue("Not_Fatigued")}>⚡ Energetic</button>
                    </div>
                </div>
            </div>

            <button className="btn-send-feedback" disabled={!stress || !fatigue} onClick={handleSend}>
                Send
            </button>
        </div>
    );
}

// -----------------------------------------------
// COMPONENT: State Indicator Badge
// -----------------------------------------------
function StateIndicator({ state }) {
    const config = {
        [STATE.IDLE]: { label: "IDLE", color: "#64748b", bg: "rgba(100,116,139,0.15)", icon: "⏹" },
        [STATE.ACTIVE]: { label: "ACTIVE", color: "#10b981", bg: "rgba(16,185,129,0.15)", icon: "🟢" },
        [STATE.AFK]: { label: "AFK", color: "#f59e0b", bg: "rgba(245,158,11,0.15)", icon: "💤" },
        [STATE.PAUSED]: { label: "PAUSED", color: "#8b5cf6", bg: "rgba(139,92,246,0.15)", icon: "⏸" },
    };

    const c = config[state] || config[STATE.IDLE];

    return (
        <span className="card-badge" style={{ background: c.bg, color: c.color, fontSize: "0.75rem", fontWeight: 600, padding: "0.3rem 0.7rem" }}>
            {c.icon} {c.label}
        </span>
    );
}

// -----------------------------------------------
// COMPONENT: TimerWidget (Isolated Render for Performance)
// -----------------------------------------------
function TimerWidget({ machineState, getRemainingTime, bufferSize, feedbackCount }) {
    const [displayTime, setDisplayTime] = useState(getRemainingTime());

    useEffect(() => {
        if (machineState !== STATE.ACTIVE) {
            setDisplayTime(getRemainingTime());
            return;
        }
        const tickId = setInterval(() => {
            setDisplayTime(getRemainingTime());
        }, 250);
        return () => clearInterval(tickId);
    }, [machineState, getRemainingTime]);

    return (
        <div className="card">
            <div className="card-header">
                <span className="card-title">Tumbling Window</span>
                <StateIndicator state={machineState} />
            </div>
            <div className="timer-display">
                <span className="timer-value">{formatTime(displayTime)}</span>
                <span className="timer-unit">
                    {machineState === STATE.PAUSED ? "paused" : machineState === STATE.AFK ? "suspended" : "remaining"}
                </span>
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                Buffer: <strong style={{ color: "var(--accent-blue)" }}>{bufferSize}</strong> event
                {" "} · Feedback: <strong style={{ color: "var(--accent-blue)" }}>{feedbackCount}</strong>
            </div>
            <div className="timer-progress">
                <div className="timer-progress-fill" style={{
                    width: `${((WINDOW_DURATION - displayTime) / WINDOW_DURATION) * 100}%`,
                    transition: machineState === STATE.ACTIVE ? "width 0.25s linear" : "none",
                }}></div>
            </div>
        </div>
    );
}

// -----------------------------------------------
// COMPONENT: Dashboard
// -----------------------------------------------
function Dashboard({ sm, logs, addLog, onSessionEnd }) {
    const {
        machineState, username, getRemainingTime, bufferSize,
        eventCounts, lastTelemetryResult, showResumeFeedbackPopup,
        pause, requestResume, resumeWithFeedback, cancelResume,
    } = sm;

    // State
    const [prediction, setPrediction] = useState(null);
    const [modelUsed, setModelUsed] = useState("GLOBAL");
    const [feedbackCount, setFeedbackCount] = useState(0);
    const [showFeedback, setShowFeedback] = useState(false);
    const [isAutoTrigger, setIsAutoTrigger] = useState(false);
    const [showCheckout, setShowCheckout] = useState(false);
    const [prevPrediction, setPrevPrediction] = useState(null);
    const [stressHistory, setStressHistory] = useState([]);
    const [fatigueHistory, setFatigueHistory] = useState([]);

    const logEndRef = useRef(null);
    const isOnline = useOnlineStatus();

    // Process telemetry results from state machine
    useEffect(() => {
        if (!lastTelemetryResult) return;
        const res = lastTelemetryResult;

        if (res.prediction && !res.prediction.error) {
            const newPred = res.prediction;
            setPrediction(newPred);
            setModelUsed(newPred.model_used || "GLOBAL");

            if (newPred.probabilities?.stress_val?.Stressed !== undefined) {
                setStressHistory((prev) => [...prev.slice(-9), newPred.probabilities.stress_val.Stressed]);
            }
            if (newPred.probabilities?.fatigue_val?.Fatigued !== undefined) {
                setFatigueHistory((prev) => [...prev.slice(-9), newPred.probabilities.fatigue_val.Fatigued]);
            }

            // Smart auto-trigger
            if (newPred.predictions?.stress_val === "Stressed") {
                if (!prevPrediction || prevPrediction.predictions?.stress_val !== "Stressed") {
                    setIsAutoTrigger(true);
                    setShowFeedback(true);
                }
            }
            setPrevPrediction(newPred);
        }
    }, [lastTelemetryResult]);

    useEffect(() => {
        logEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [logs]);

    // Feedback handler
    const handleFeedback = async (stressLabel, fatigueLabel) => {
        try {
            const res = await apiPost("/api/feedback", {
                session_id: sm.sessionId,
                user_id: username,
                stress_label: stressLabel,
                fatigue_label: fatigueLabel,
            });
            setFeedbackCount(res.feedback_count || 0);
            addLog(`Feedback saved (#${res.feedback_count}). Training: ${res.training_triggered ? "STARTED ✅" : "Waiting"}`, "success");
        } catch (err) {
            addLog(`Feedback error: ${err.message}`, "error");
        }
        setShowFeedback(false);
        setIsAutoTrigger(false);
    };

    // Checkout — collect final feedback, end session, show result screen
    const handleCheckout = async (stressLabel, fatigueLabel) => {
        // 1. Send final feedback
        try {
            await apiPost("/api/feedback", {
                session_id: sm.sessionId,
                user_id: username,
                stress_label: stressLabel,
                fatigue_label: fatigueLabel,
            });
        } catch (err) {
            addLog(`Logout data could not be saved: ${err.message}`, "warn");
        }

        // 2. End session and retrieve summary from backend
        try {
            const endResult = await apiPost("/api/sessions/end", {
                session_id: sm.sessionId,
                user_id: username,
                reason: "manual",
                mouse_events: [],
                keyboard_events: [],
            });

            onSessionEnd({
                ...endResult.session_summary,
                username: username,
                exit_stress: stressLabel,
                exit_fatigue: fatigueLabel,
                // Pass frontend-collected histories for charts
                stress_history_fe: stressHistory,
                fatigue_history_fe: fatigueHistory,
            });
        } catch (err) {
            addLog(`Session end error: ${err.message}`, "error");
            // Fallback: transition to result with minimal data
            onSessionEnd({
                username: username,
                exit_stress: stressLabel,
                exit_fatigue: fatigueLabel,
                total_duration_s: 0,
                active_duration_s: 0,
                paused_duration_s: 0,
                afk_duration_s: 0,
                total_mouse_events: eventCounts.mouse,
                total_keyboard_events: eventCounts.keyboard,
                total_chunks_processed: 0,
                stress_ratio: 0,
                fatigue_ratio: 0,
                stress_probabilities: stressHistory,
                fatigue_probabilities: fatigueHistory,
                flow_score: 0,
                peak_stress_time: null,
                peak_stress_value: 0,
                initial_stress: null,
                initial_fatigue: null,
                feedback_count: feedbackCount,
                training_triggered_count: 0,
                stress_history_fe: stressHistory,
                fatigue_history_fe: fatigueHistory,
            });
        }
    };

    // Prediction display
    const stressVal = prediction?.predictions?.stress_val || "Waiting...";
    const fatigueVal = prediction?.predictions?.fatigue_val || "Waiting...";
    const stressProb = prediction?.probabilities?.stress_val || {};
    const fatigueProb = prediction?.probabilities?.fatigue_val || {};

    const stressClass = stressVal === "Stressed" ? "stressed" : stressVal === "Not_Stressed" ? "not-stressed" : "waiting";
    const fatigueClass = fatigueVal === "Fatigued" ? "fatigued" : fatigueVal === "Not_Fatigued" ? "not-fatigued" : "waiting";

    const stressTR = { "Stressed": "Stressed", "Not_Stressed": "Relaxed", "Waiting...": "Waiting..." };
    const fatigueTR = { "Fatigued": "Fatigued", "Not_Fatigued": "Energetic", "Waiting...": "Waiting..." };

    const stressProbConfig = stressProb.Stressed !== undefined ? [
        { label: "Stressed", value: stressProb.Stressed, fillClass: "stress" },
        { label: "Relaxed", value: stressProb.Not_Stressed, fillClass: "calm" },
    ] : null;

    const fatigueProbConfig = fatigueProb.Fatigued !== undefined ? [
        { label: "Fatigued", value: fatigueProb.Fatigued, fillClass: "fatigue" },
        { label: "Energetic", value: fatigueProb.Not_Fatigued, fillClass: "energy" },
    ] : null;

    // Pause/Resume button logic
    const canPause = machineState === STATE.ACTIVE;
    const canResume = machineState === STATE.PAUSED;

    return (
        <div className="dashboard">
            {!isOnline && <OfflineBanner />}

            {/* Top Bar */}
            <div className="top-bar">
                <div className="top-bar-left">
                    <span className="logo">🧠 ML Dashboard</span>
                    <span className="separator"></span>
                    <div className="user-badge">
                        <span className={`dot ${!isOnline ? "offline" : ""}`}></span>
                        {username}
                    </div>
                    <StateIndicator state={machineState} />
                </div>
                <div className="top-bar-right">
                    <span className={`model-badge ${modelUsed.includes("LOCAL") ? "local" : "global"}`}>
                        {modelUsed.includes("LOCAL") ? "🟣 LOCAL Model" : "🔵 GLOBAL Model"}
                    </span>

                    {/* Pause / Resume Button */}
                    {canPause && (
                        <button className="btn-logout" style={{ background: "rgba(139,92,246,0.15)", color: "#8b5cf6", border: "1px solid rgba(139,92,246,0.3)" }} onClick={pause}>
                            ⏸ Pause
                        </button>
                    )}
                    {canResume && (
                        <button className="btn-logout" style={{ background: "rgba(16,185,129,0.15)", color: "#10b981", border: "1px solid rgba(16,185,129,0.3)" }} onClick={requestResume}>
                            ▶️ Continue
                        </button>
                    )}

                    <button className="btn-logout" onClick={() => setShowCheckout(true)}>
                        Logout
                    </button>
                </div>
            </div>

            {/* AFK Overlay */}
            {machineState === STATE.AFK && (
                <div style={{
                    position: "fixed", inset: 0, zIndex: 1500,
                    background: "rgba(0,0,0,0.85)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    flexDirection: "column", gap: "1rem",
                }}>
                    <div style={{
                        background: "var(--bg-card)", borderRadius: "16px", padding: "2rem 3rem",
                        textAlign: "center", border: "1px solid var(--border-subtle)",
                        boxShadow: "0 25px 50px rgba(0,0,0,0.3)",
                    }}>
                        <div style={{ fontSize: "3rem", marginBottom: "0.5rem" }}>💤</div>
                        <h2 style={{ color: "var(--text-primary)", margin: 0 }}>AFK Detected</h2>
                        <p style={{ color: "var(--text-muted)", margin: "0.5rem 0 0 0" }}>
                            Make any mouse or keyboard movement to continue.
                        </p>
                    </div>
                </div>
            )}

            {/* PAUSED Overlay */}
            {machineState === STATE.PAUSED && !showResumeFeedbackPopup && !showCheckout && (
                <div style={{
                    position: "fixed", inset: 0, zIndex: 1500,
                    background: "rgba(0,0,0,0.85)",
                    display: "flex", alignItems: "center", justifyContent: "center",
                    flexDirection: "column", gap: "1rem",
                }}>
                    <div style={{
                        background: "var(--bg-card)", borderRadius: "16px", padding: "2rem 3rem",
                        textAlign: "center", border: "1px solid var(--border-subtle)",
                        boxShadow: "0 25px 50px rgba(0,0,0,0.3)",
                    }}>
                        <div style={{ fontSize: "3rem", marginBottom: "0.5rem" }}>⏸</div>
                        <h2 style={{ color: "var(--text-primary)", margin: 0 }}>Session Paused</h2>
                        <p style={{ color: "var(--text-muted)", margin: "0.5rem 0 1rem 0" }}>
                            Time remaining: <strong>{formatTime(getRemainingTime())}</strong>
                        </p>
                        <button
                            className="btn-primary"
                            style={{ width: "100%" }}
                            onClick={requestResume}
                        >
                            ▶️ Continue
                        </button>
                    </div>
                </div>
            )}

            {/* Dashboard Grid */}
            <div className="dashboard-grid">
                {/* Stress Card */}
                <StatusCard
                    title="Stress State"
                    badgeText={stressVal === "Stressed" ? "HIGH" : stressVal === "Not_Stressed" ? "LOW" : "—"}
                    badgeBg={stressVal === "Stressed" ? "rgba(239,68,68,0.15)" : "rgba(16,185,129,0.15)"}
                    badgeColor={stressVal === "Stressed" ? "var(--accent-red)" : "var(--accent-green)"}
                    value={stressTR[stressVal] || stressVal}
                    valueClass={stressClass}
                    probabilities={stressProbConfig ? true : false}
                    probConfig={stressProbConfig || []}
                />

                {/* Fatigue Card */}
                <StatusCard
                    title="Fatigue State"
                    badgeText={fatigueVal === "Fatigued" ? "HIGH" : fatigueVal === "Not_Fatigued" ? "LOW" : "—"}
                    badgeBg={fatigueVal === "Fatigued" ? "rgba(245,158,11,0.15)" : "rgba(59,130,246,0.15)"}
                    badgeColor={fatigueVal === "Fatigued" ? "var(--accent-orange)" : "var(--accent-blue)"}
                    value={fatigueTR[fatigueVal] || fatigueVal}
                    valueClass={fatigueClass}
                    probabilities={fatigueProbConfig ? true : false}
                    probConfig={fatigueProbConfig || []}
                />

                {/* Timer Card */}
                <TimerWidget
                    machineState={machineState}
                    getRemainingTime={getRemainingTime}
                    bufferSize={bufferSize}
                    feedbackCount={feedbackCount}
                />

                {/* Telemetry Stats */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">Session Statistics</span>
                    </div>
                    <div className="stats-grid" style={{ gridTemplateColumns: "repeat(2, 1fr)" }}>
                        <div className="stat-item">
                            <div className="stat-value">{eventCounts.mouse}</div>
                            <div className="stat-label">Mouse</div>
                        </div>
                        <div className="stat-item">
                            <div className="stat-value">{eventCounts.keyboard}</div>
                            <div className="stat-label">Keyboard</div>
                        </div>
                    </div>
                    <div style={{ marginTop: "0.75rem", fontSize: "0.7rem", color: "var(--text-muted)" }}>
                        Inactivity: {formatTime(INACTIVITY_TIMEOUT)} · Window: {formatTime(WINDOW_DURATION)} · Buffer Max: {MAX_BUFFER_SIZE.toLocaleString()}
                    </div>
                </div>

                {/* Sparkline Card */}
                <div className="card full-width">
                    <div className="card-header">
                        <span className="card-title">Prediction History (Last 10)</span>
                        <span className="card-badge" style={{ background: "var(--bg-glass)", color: "var(--text-muted)" }}>
                            {stressHistory.length} data points
                        </span>
                    </div>
                    <div className="sparkline-row">
                        <SparklineChart data={stressHistory} color="#ef4444" height={55} label="stress" />
                        <SparklineChart data={fatigueHistory} color="#f59e0b" height={55} label="fatigue" />
                    </div>
                    <div className="sparkline-legend">
                        <span><span className="legend-dot" style={{ background: "#ef4444" }}></span> Stress Probability</span>
                        <span><span className="legend-dot" style={{ background: "#f59e0b" }}></span> Fatigue Probability</span>
                    </div>
                </div>

                {/* Log Panel */}
                <div className="card full-width log-panel">
                    <div className="card-header">
                        <span className="card-title">System Logs</span>
                        <span className="card-badge" style={{ background: "var(--bg-glass)", color: "var(--text-muted)" }}>
                            {logs.length} records
                        </span>
                    </div>
                    <div className="log-entries">
                        {logs.map((log, i) => (
                            <div key={i} className={`log-entry ${log.type}`}>
                                <span className="log-time">[{log.time}]</span>
                                <span className="log-msg">{log.msg}</span>
                            </div>
                        ))}
                        <div ref={logEndRef} />
                    </div>
                </div>
            </div>

            {/* Floating Feedback Button */}
            <div className="feedback-fab">
                <button
                    className={`fab-btn ${isAutoTrigger ? "alert" : ""}`}
                    onClick={() => {
                        setIsAutoTrigger(false);
                        setShowFeedback(!showFeedback);
                    }}
                    title="Give feedback"
                >
                    {isAutoTrigger ? "⚠️" : "💬"}
                </button>
            </div>

            {/* Feedback Popup */}
            {showFeedback && (
                <FeedbackPopup
                    isAutoTrigger={isAutoTrigger}
                    onSend={handleFeedback}
                    onClose={() => { setShowFeedback(false); setIsAutoTrigger(false); }}
                />
            )}

            {/* Resume Feedback Popup */}
            {showResumeFeedbackPopup && (
                <ResumeFeedbackPopup
                    onSelect={async (stress, fatigue) => {
                        await handleFeedback(stress, fatigue);
                        resumeWithFeedback();
                    }}
                    onCancel={cancelResume}
                />
            )}

            {/* Checkout Modal */}
            {showCheckout && (
                <CheckoutModal
                    onConfirm={handleCheckout}
                    onCancel={() => setShowCheckout(false)}
                />
            )}
        </div>
    );
}

// -----------------------------------------------
// COMPONENT: Checkout Modal
// -----------------------------------------------
function CheckoutModal({ onConfirm, onCancel }) {
    const [stress, setStress] = useState(null);
    const [fatigue, setFatigue] = useState(null);

    return (
        <div className="login-screen" style={{ position: "fixed", inset: 0, zIndex: 2000, background: "rgba(0,0,0,0.85)" }}>
            <div className="login-card" style={{ backdropFilter: "none" }}>
                <h1>👋 End Session</h1>
                <p>How do you feel as you end your work?</p>

                <div className="feedback-options" style={{ marginBottom: "1.5rem" }}>
                    <div className="feedback-option-group">
                        <label>Stress:</label>
                        <div className="options">
                            <button type="button" className={`fb-opt-btn ${stress === "Stressed" ? "selected" : ""}`} onClick={() => setStress("Stressed")}>😰 Stressed</button>
                            <button type="button" className={`fb-opt-btn ${stress === "Not_Stressed" ? "selected" : ""}`} onClick={() => setStress("Not_Stressed")}>😊 Not Stressed</button>
                        </div>
                    </div>
                    <div className="feedback-option-group">
                        <label>Fatigue:</label>
                        <div className="options">
                            <button type="button" className={`fb-opt-btn ${fatigue === "Fatigued" ? "selected" : ""}`} onClick={() => setFatigue("Fatigued")}>😴 Fatigued</button>
                            <button type="button" className={`fb-opt-btn ${fatigue === "Not_Fatigued" ? "selected" : ""}`} onClick={() => setFatigue("Not_Fatigued")}>⚡ Energetic</button>
                        </div>
                    </div>
                </div>

                <button className="btn-primary" disabled={!stress || !fatigue} onClick={() => onConfirm(stress, fatigue)}>
                    End Session
                </button>
                <button
                    className="btn-logout"
                    style={{ width: "100%", marginTop: "0.75rem", padding: "0.7rem" }}
                    onClick={onCancel}
                >
                    Cancel
                </button>
            </div>
        </div>
    );
}

// -----------------------------------------------
// HELPER: Duration formatter (seconds → human)
// -----------------------------------------------
function formatDuration(totalSeconds) {
    const h = Math.floor(totalSeconds / 3600);
    const m = Math.floor((totalSeconds % 3600) / 60);
    const s = totalSeconds % 60;
    if (h > 0) return `${h}h ${m}m`;
    if (m > 0) return `${m}m ${s}s`;
    return `${s}s`;
}

// -----------------------------------------------
// COMPONENT: EnergyRing (SVG animated circle)
// -----------------------------------------------
function EnergyRing({ percent }) {
    const radius = 65;
    const circumference = 2 * Math.PI * radius;
    const targetOffset = circumference - (percent / 100) * circumference;
    const level = percent >= 60 ? "high" : percent >= 30 ? "medium" : "low";

    return (
        <div className="energy-ring">
            <svg viewBox="0 0 160 160">
                <circle className="energy-ring__track" cx="80" cy="80" r={radius} />
                <circle
                    className={`energy-ring__fill ${level}`}
                    cx="80" cy="80" r={radius}
                    strokeDasharray={circumference}
                    strokeDashoffset={circumference}
                    style={{
                        "--ring-circumference": circumference,
                        "--ring-target-offset": targetOffset,
                    }}
                />
            </svg>
            <div className="energy-ring__label">
                <div className="energy-ring__percent">%{percent}</div>
                <div className="energy-ring__tag">Energy</div>
            </div>
        </div>
    );
}

// -----------------------------------------------
// COMPONENT: DonutChart
// -----------------------------------------------
function DonutChart({ percent, color, label }) {
    const radius = 40;
    const circumference = 2 * Math.PI * radius;
    const targetOffset = circumference - (percent / 100) * circumference;

    return (
        <div style={{ position: 'relative', width: 100, height: 100, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            <svg viewBox="0 0 100 100" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', transform: 'rotate(-90deg)' }}>
                <circle cx="50" cy="50" r={radius} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" />
                <circle cx="50" cy="50" r={radius} fill="none" stroke={color} strokeWidth="8" strokeDasharray={circumference} strokeDashoffset={targetOffset} strokeLinecap="round" style={{ transition: 'stroke-dashoffset 1s ease' }} />
            </svg>
            <div style={{ textAlign: 'center', zIndex: 1 }}>
                <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: 'var(--text-primary)' }}>%{percent}</div>
                <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>{label}</div>
            </div>
        </div>
    );
}

// -----------------------------------------------
// COMPONENT: LargeAreaChart
// -----------------------------------------------
function LargeAreaChart({ stressData, fatigueData }) {
    const [hoverIdx, setHoverIdx] = useState(null);

    const dataLen = Math.max(stressData?.length || 0, fatigueData?.length || 0);
    if (dataLen < 2) {
        return (
            <div style={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontStyle: 'italic', background: 'var(--bg-glass)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-glass)' }}>
                Grafik için en az 2 data points gerekiyor.
            </div>
        );
    }

    const width = 800;
    const height = 200;
    const padding = 20;

    const generatePath = (data) => {
        const points = data.map((val, i) => {
            const x = padding + (i / (data.length - 1)) * (width - 2 * padding);
            const y = padding + (1 - val) * (height - 2 * padding);
            return { x, y, val };
        });

        let linePath = `M ${points[0].x},${points[0].y}`;
        for (let i = 0; i < points.length - 1; i++) {
            const curr = points[i];
            const next = points[i + 1];
            const cx = (curr.x + next.x) / 2;
            linePath += ` C ${cx},${curr.y} ${cx},${next.y} ${next.x},${next.y}`;
        }

        const areaPath = `${linePath} L ${points[points.length - 1].x},${height} L ${points[0].x},${height} Z`;
        return { areaPath, linePath, points };
    };

    const stressRes = stressData?.length > 1 ? generatePath(stressData) : null;
    const fatigueRes = fatigueData?.length > 1 ? generatePath(fatigueData) : null;

    const handleMouseMove = (e) => {
        const rect = e.currentTarget.getBoundingClientRect();
        const mouseX = ((e.clientX - rect.left) / rect.width) * width;
        const graphWidth = width - 2 * padding;
        const xRatio = (mouseX - padding) / graphWidth;
        let idx = Math.round(xRatio * (dataLen - 1));
        idx = Math.max(0, Math.min(dataLen - 1, idx));
        setHoverIdx(idx);
    };

    return (
        <div style={{ background: 'var(--bg-glass)', borderRadius: 'var(--radius-md)', padding: '1.5rem', border: '1px solid var(--border-glass)', marginTop: '1rem', position: 'relative' }}>
            <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '1rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Timeline</div>

            <svg
                viewBox={`0 0 ${width} ${height}`}
                preserveAspectRatio="none"
                style={{ width: '100%', height: '180px', overflow: 'visible', cursor: 'crosshair' }}
                onMouseMove={handleMouseMove}
                onMouseLeave={() => setHoverIdx(null)}
            >
                <defs>
                    <linearGradient id="grad-stress" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--accent-red)" stopOpacity="0.5" />
                        <stop offset="100%" stopColor="var(--accent-red)" stopOpacity="0.0" />
                    </linearGradient>
                    <linearGradient id="grad-fatigue" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="var(--accent-purple)" stopOpacity="0.5" />
                        <stop offset="100%" stopColor="var(--accent-purple)" stopOpacity="0.0" />
                    </linearGradient>
                </defs>

                {/* Grid lines */}
                {[0, 0.25, 0.5, 0.75, 1].map(tick => (
                    <line key={tick} x1={0} y1={padding + tick * (height - 2 * padding)} x2={width} y2={padding + tick * (height - 2 * padding)} stroke="var(--border-glass)" strokeDasharray="4 4" />
                ))}

                {/* Fatigue Chart */}
                {fatigueRes && (
                    <g>
                        <path d={fatigueRes.areaPath} fill="url(#grad-fatigue)" style={{ animation: 'fadeInUp 1s ease' }} />
                        <path d={fatigueRes.linePath} fill="none" stroke="var(--accent-purple)" strokeWidth="3" className="timeline-line" />
                        {fatigueRes.points.map((p, i) => (
                            <circle key={i} cx={p.x} cy={p.y} r={4} fill="var(--bg-card)" stroke="var(--accent-purple)" strokeWidth={2} />
                        ))}
                    </g>
                )}

                {/* Stress Chart */}
                {stressRes && (
                    <g>
                        <path d={stressRes.areaPath} fill="url(#grad-stress)" style={{ animation: 'fadeInUp 1s ease' }} />
                        <path d={stressRes.linePath} fill="none" stroke="var(--accent-red)" strokeWidth="3" className="timeline-line" />
                        {stressRes.points.map((p, i) => (
                            <circle key={i} cx={p.x} cy={p.y} r={4} fill="var(--bg-card)" stroke="var(--accent-red)" strokeWidth={2} />
                        ))}
                    </g>
                )}

                {/* Hover Interaction */}
                {hoverIdx !== null && (
                    <g>
                        {/* Crosshair Line */}
                        {stressRes && stressRes.points[hoverIdx] && (
                            <line
                                x1={stressRes.points[hoverIdx].x}
                                y1={padding}
                                x2={stressRes.points[hoverIdx].x}
                                y2={height - padding}
                                stroke="rgba(255,255,255,0.5)"
                                strokeDasharray="4 4"
                            />
                        )}

                        {/* Hover Rings */}
                        {stressRes && stressRes.points[hoverIdx] && (
                            <circle cx={stressRes.points[hoverIdx].x} cy={stressRes.points[hoverIdx].y} r={6} fill="var(--accent-red)" filter="drop-shadow(0 0 6px var(--accent-red-glow))" />
                        )}
                        {fatigueRes && fatigueRes.points[hoverIdx] && (
                            <circle cx={fatigueRes.points[hoverIdx].x} cy={fatigueRes.points[hoverIdx].y} r={6} fill="var(--accent-purple)" filter="drop-shadow(0 0 6px var(--accent-purple-glow))" />
                        )}
                    </g>
                )}
            </svg>

            {/* Absolute Tooltip Overlay */}
            {hoverIdx !== null && stressRes && stressRes.points[hoverIdx] && (
                <div style={{
                    position: 'absolute',
                    top: '-20px',
                    left: `${(stressRes.points[hoverIdx].x / width) * 100}%`,
                    transform: 'translateX(-50%)',
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border-glass)',
                    padding: '0.6rem 0.9rem',
                    borderRadius: 'var(--radius-sm)',
                    boxShadow: 'var(--shadow-lg)',
                    pointerEvents: 'none',
                    fontSize: '0.75rem',
                    minWidth: '130px',
                    zIndex: 10,
                    animation: 'fadeInUp 0.2s ease'
                }}>
                    <div style={{ marginBottom: '0.4rem', fontWeight: 'bold', color: 'var(--text-primary)', borderBottom: '1px solid var(--border-glass)', paddingBottom: '0.2rem' }}>
                        Time / Point: {hoverIdx + 1}
                    </div>
                    {stressRes && (
                        <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--accent-red)', marginBottom: '0.2rem' }}>
                            <span>Stress:</span>
                            <strong>{Math.round(stressRes.points[hoverIdx].val * 100)}%</strong>
                        </div>
                    )}
                    {fatigueRes && (
                        <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--accent-purple)' }}>
                            <span>Fatigue:</span>
                            <strong>{Math.round(fatigueRes.points[hoverIdx].val * 100)}%</strong>
                        </div>
                    )}
                </div>
            )}

            <div style={{ display: 'flex', gap: '1.5rem', justifyContent: 'center', marginTop: '1.5rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}><span style={{ width: 12, height: 12, borderRadius: '50%', background: 'var(--accent-red)' }}></span> Stress Probability</span>
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}><span style={{ width: 12, height: 12, borderRadius: '50%', background: 'var(--accent-purple)' }}></span> Fatigue Probability</span>
            </div>
        </div>
    );
}

// -----------------------------------------------
// COMPONENT: SessionResultScreen (Graphic-Intensive)
// -----------------------------------------------
function SessionResultScreen({ summary, onClose }) {
    if (!summary) return null;

    const {
        username, total_duration_s = 0, active_duration_s = 0, paused_duration_s = 0, afk_duration_s = 0,
        stress_ratio = 0, fatigue_ratio = 0, avg_stress_prob = 0, avg_fatigue_prob = 0,
        stress_probabilities = [], fatigue_probabilities = [],
        flow_score = 0, feedback_count = 0, training_triggered_count = 0, stress_history_fe = [], fatigue_history_fe = []
    } = summary;

    // Use average probabilities for donut charts (more meaningful than binary ratio)
    const stressPercent = Math.round((avg_stress_prob || stress_ratio) * 100);
    const fatiguePercent = Math.round((avg_fatigue_prob || fatigue_ratio) * 100);
    const energyPercent = Math.max(0, Math.min(100, 100 - fatiguePercent));
    const flowColor = flow_score >= 70 ? "var(--accent-green)" : flow_score >= 40 ? "var(--accent-orange)" : "var(--accent-red)";

    const stressData = stress_probabilities.length > 0 ? stress_probabilities : stress_history_fe;
    const fatigueData = fatigue_probabilities.length > 0 ? fatigue_probabilities : fatigue_history_fe;

    const feedbackTarget = 50;
    const feedbackProgress = Math.min(100, (feedback_count / feedbackTarget) * 100);

    return (
        <div className="result-screen">
            <div className="result-panel" style={{ maxWidth: 900 }}>
                {/* Header */}
                <h2 className="result-panel__title" style={{ fontSize: '2rem' }}>🧠 Mental Summary</h2>
                <p className="result-panel__subtitle">{username} — Graphical Session Report</p>

                {/* Top Row: Donut Charts & Energy Ring */}
                <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center', marginBottom: '2rem', flexWrap: 'wrap', gap: '1rem' }}>
                    <DonutChart percent={stressPercent} color="var(--accent-red)" label="Stressed" />
                    <EnergyRing percent={energyPercent} />
                    <DonutChart percent={100 - stressPercent} color="var(--accent-blue)" label="Calm" />
                </div>

                {/* Duration Pills */}
                <div className="result-time-stats" style={{ marginBottom: '2rem' }}>
                    <div className="time-stat-pill"><span className="time-stat-pill__value">{formatDuration(total_duration_s)}</span><span className="time-stat-pill__label">Total</span></div>
                    <div className="time-stat-pill"><span className="time-stat-pill__value">{formatDuration(active_duration_s)}</span><span className="time-stat-pill__label">Active</span></div>
                    <div className="time-stat-pill"><span className="time-stat-pill__value">{formatDuration(paused_duration_s + afk_duration_s)}</span><span className="time-stat-pill__label">Break</span></div>
                </div>

                {/* Middle Row: Large Timeline Chart */}
                <LargeAreaChart stressData={stressData} fatigueData={fatigueData} />

                {/* Bottom Row: Flow Score & AI Progress */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '1.5rem', marginTop: '1.5rem' }}>
                    {/* Flow Score Gauge */}
                    <div style={{ background: 'var(--bg-glass)', borderRadius: 'var(--radius-md)', padding: '1.5rem', border: '1px solid var(--border-glass)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '0.5rem', fontWeight: 600 }}>Flow Score</div>
                        <div style={{ fontSize: '3rem', fontWeight: 800, color: flowColor }}>{flow_score}</div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textAlign: 'center', marginTop: '0.5rem' }}>
                            {flow_score >= 70 ? "Excellent Focus!" : flow_score >= 40 ? "Average Focus" : "Scattered Focus"}
                        </div>
                    </div>

                    {/* AI Progress */}
                    <div className="result-ai-section" style={{ margin: 0, animation: 'none' }}>
                        <div className="result-ai-header">
                            <span>🤖</span>
                            <span>Model Progress</span>
                        </div>
                        <div className="result-ai-desc">In this session <strong>{feedback_count}</strong> feedback(s) collected.</div>
                        <div className="result-ai-progress-track">
                            <div className="result-ai-progress-fill" style={{ width: `${feedbackProgress}%` }} />
                        </div>
                        <div className="result-ai-stats">
                            <span>{feedback_count} / {feedbackTarget} targeted</span>
                            {training_triggered_count > 0 && <span style={{ color: 'var(--accent-green)', fontWeight: 600 }}>✅ Model Updated</span>}
                        </div>
                    </div>
                </div>

                {/* Close Button */}
                <div className="result-close-section" style={{ marginTop: '3rem' }}>
                    <button className="btn-primary" onClick={onClose} style={{ fontSize: '1.1rem', padding: '1.2rem 3rem' }}>
                        Close Session & Exit
                    </button>
                </div>
            </div>
        </div>
    );
}

// -----------------------------------------------
// ROOT APP
// -----------------------------------------------
function App() {
    const [appPhase, setAppPhase] = useState("login"); // "login" | "dashboard" | "result"
    const [sessionSummary, setSessionSummary] = useState(null);
    const [logs, setLogs] = useState([]);

    const addLog = useCallback((msg, type = "info") => {
        setLogs((prev) => [...prev.slice(-80), { time: timeNow(), msg, type }]);
    }, []);

    const sm = useTelemetryStateMachine(addLog);

    const handleLogin = async (name, stress, fatigue) => {
        sm.sessionStart(name, stress, fatigue);
        setAppPhase("dashboard");
    };

    const handleSessionEnd = (summary) => {
        setSessionSummary(summary);
        setAppPhase("result");
    };

    const handleCloseResult = () => {
        window.location.reload();
    };

    if (appPhase === "result") {
        return <SessionResultScreen summary={sessionSummary} onClose={handleCloseResult} />;
    }

    if (appPhase === "login" || sm.machineState === STATE.IDLE) {
        return <SessionStartModal onStart={handleLogin} />;
    }

    return <Dashboard sm={sm} logs={logs} addLog={addLog} onSessionEnd={handleSessionEnd} />;
}

// -----------------------------------------------
// RENDER
// -----------------------------------------------
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
