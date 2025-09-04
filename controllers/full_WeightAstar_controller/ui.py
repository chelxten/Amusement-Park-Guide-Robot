import threading, queue, json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Robot Guide</title>
<style>
    :root{
      --bg:#f8fafc; --surface:#ffffff; --surface-2:#f1f5f9;
      --fg:#0f172a; --muted:#475569; --border:#e2e8f0;
      --accent:#2563eb; --accent-2:#1d4ed8; --accent-3:#93c5fd;
      --success:#16a34a; --warn:#f59e0b; --danger:#ef4444;
      --chip:#eef2ff; --chip-border:#c7d2fe;
      --pill:#e2e8f0; --shadow:0 10px 28px rgba(2,6,23,.08);
    }
    @media (prefers-color-scheme: dark){
      :root{
        --bg:#0b1020; --surface:#12172a; --surface-2:#0f1525;
        --fg:#e5e7eb; --muted:#94a3b8; --border:#1f2937;
        --accent:#3b82f6; --accent-2:#2563eb; --accent-3:#60a5fa;
        --chip:#1f2a44; --chip-border:#334155; --pill:#1f2937; --shadow:none;
      }
    }
    
    *{box-sizing:border-box}
    html,body{height:100%}
    body{
      margin:0; padding:16px; background:var(--bg); color:var(--fg);
      font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Arial,Noto Sans,sans-serif;
      -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
    }
    
    .container{max-width:1200px; margin:0 auto; display:grid; gap:16px}
    header{
      display:flex; align-items:center; justify-content:space-between; gap:12px;
      padding:14px 18px; background:var(--surface); border:1px solid var(--border);
      border-radius:16px; box-shadow:var(--shadow);
      position:sticky; top:8px; z-index:2;
    }
    .title{display:flex; align-items:center; gap:12px}
    .brand{
      width:36px; height:36px; border-radius:10px;
      background:linear-gradient(135deg,var(--accent),var(--accent-3));
      box-shadow:0 6px 16px rgba(37,99,235,.35);
    }
    h1{font-size:18px; margin:0}
    .badges{display:flex; gap:8px; align-items:center; flex-wrap:wrap}
    .badge{
      display:inline-flex; align-items:center; gap:6px;
      padding:5px 12px; border-radius:999px; border:1px solid var(--border);
      background:var(--surface-2); color:var(--muted); font-size:12px;
    }
    .badge b{color:var(--fg)}
    
    .grid{display:grid; grid-template-columns:1.15fr 1fr; gap:16px}
    @media (max-width: 900px){ .grid{grid-template-columns:1fr} }
    
    .card{
      background:var(--surface); border:1px solid var(--border);
      border-radius:16px; padding:16px; box-shadow:var(--shadow);
    }
    .card h3{margin:0 0 12px 0; font-size:16px}
    
    .row{display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:8px 0}
    
    input, select{
      padding:11px 12px; border:1px solid var(--border);
      border-radius:12px; background:var(--surface-2); color:var(--fg);
      outline:none; min-width:180px;
    }
    input:focus, select:focus{
      border-color:var(--accent);
      box-shadow:0 0 0 3px rgba(37,99,235,.15);
    }
    
    button{
      padding:10px 14px; border:1px solid var(--border); background:var(--surface);
      color:var(--fg); border-radius:12px; cursor:pointer; transition:.15s transform ease, .15s box-shadow ease;
    }
    button:hover{transform:translateY(-1px)}
    button:focus-visible{outline:none; box-shadow:0 0 0 3px rgba(37,99,235,.2)}
    button.primary{background:var(--accent); border-color:var(--accent); color:#fff}
    button.primary:hover{background:var(--accent-2)}
    button.chip{background:var(--chip); border-color:var(--chip-border); color:var(--fg)}
    button:disabled{opacity:.6; cursor:not-allowed; transform:none}
    
    .toolbar{display:flex; flex-wrap:wrap; gap:10px}
    .target{font-weight:600}
    .hidden{display:none !important}
    
    .toolbar button{
      font-size:16px;
      padding:14px 22px;
      border-radius:14px;
      font-weight:700;
      letter-spacing:.2px;
    }
    #pauseResumeBtn{
      background:var(--accent);
      border-color:var(--accent-2);
      color:#fff;
    }
    #pauseResumeBtn:hover{background:var(--accent-2)}

    #cardControls .toolbar button:nth-child(2){ /* Skip */
      background:var(--danger); border-color:var(--danger); color:#fff;
    }
    #cardControls .toolbar button:nth-child(3){ /* Food */
      background:var(--success); border-color:var(--success); color:#fff;
    }
    #cardControls .toolbar button:nth-child(4){ /* Rest */
      background:var(--warn); border-color:var(--warn); color:#fff;
    }
    
    /* --- Itinerary list --- */
    .itin{display:flex; flex-direction:column; gap:8px; max-height:42vh; overflow:auto; padding-right:2px}
    .itin-item{
      display:flex; align-items:center; justify-content:space-between; gap:10px;
      padding:10px 12px; border:1px dashed var(--border);
      border-radius:12px; background:var(--surface-2);
    }
    .itin-name{display:flex; align-items:center; gap:8px; font-weight:600}
    .pill{
      font-size:12px; padding:2px 8px; border-radius:999px;
      background:var(--pill); color:var(--muted); border:1px solid var(--border)
    }
    
    /* --- All destinations --- */
    .dest-controls{
      display:flex; flex-direction:column; gap:8px; align-items:stretch; width:100%;
    }
    .dest-controls input{
      width:100%; /* full card width */
      min-width:0;
    }
    .chipbar{display:flex; gap:8px; flex-wrap:wrap; margin-top:2px}
    .chipbar button{font-size:12px; padding:6px 10px}
    
    .dest-list{display:flex; flex-direction:column; gap:8px; max-height:42vh; overflow:auto; margin-top:8px; margin-bottom:8px; padding-right:5px}
    .dest-row{
      display:flex; align-items:center; justify-content:space-between; gap:10px;
      padding:10px 12px; border:1px solid var(--border); border-radius:12px; background:var(--surface-2);
    }
    .dest-meta{display:flex; flex-direction:column; gap:2px}
    .dest-name{font-weight:700}
    .dest-cat{font-size:12px; color:var(--muted)}
    
    /* --- Log --- */
    #log{
      max-height:26vh; overflow:auto; background:var(--surface);
      border:1px solid var(--border); border-radius:16px; padding:12px;
    }
    .logline{padding:6px 0; border-bottom:1px dashed var(--border); color:var(--muted)}
    .logline:last-child{border-bottom:none}
    
    /* --- Arrival modal --- */
    .modal-backdrop{
      position:fixed; inset:0; background:rgba(2,6,23,.45);
      display:flex; align-items:center; justify-content:center; z-index:50;
    }
    .modal{
      background:var(--surface); border:1px solid var(--border); border-radius:16px;
      padding:18px; width:min(520px,92vw); box-shadow:var(--shadow)
    }
    .modal h3{margin:0 0 6px 0}
    .modal p{margin:0 0 12px 0; color:var(--muted)}
    .modal-actions{display:flex; gap:10px; justify-content:flex-end}
</style>
</head>
<body>
  <div class="container">
    <header>
      <div class="title">
        <div class="brand"></div>
        <div>
          <h1>Robot Guide</h1>
          <div class="badge">Status: <b id="status">waiting-id</b></div>
        </div>
      </div>
      <div class="badges">
        <span class="badge">Current ID: <b id="curid">—</b></span>
        <span class="badge">Destination: <b id="target">—</b></span>
      </div>
    </header>

    <!-- Card 1: Enter Unique ID -->
    <div id="idCard" class="card">
      <div class="row">
        <label>Unique ID:</label>
        <input id="uid" placeholder="e.g. cb5846" />
        <button class="primary" onclick="setId()">Load</button>
      </div>
    </div>

    <!-- Card 2: Controls -->
    <div id="cardControls" class="card hidden">
      <h3>Controls</h3>
      <div class="row">
        <div>Current destination: <span class="target" id="target_inline">—</span></div>
      </div>
      <div class="toolbar">
          <button id="pauseResumeBtn" onclick="togglePauseResume()">Pause</button>
          <button class="chip" onclick="onSkip()">Skip</button>
          <button class="chip" onclick="send('FOOD')">Food Stop</button>
          <button class="chip" onclick="send('BREAK')">Rest Stop</button>
      </div>
    </div>

    <!-- Post-load section -->
    <div id="postLoadWrap" class="hidden">
      <div class="grid">
        <!-- Left: Itinerary -->
        <div class="card">
          <h3>Itinerary</h3>
          <div id="itin" class="itin"></div>
        </div>

        <!-- Right: All Destinations -->
        <div class="card">
          <h3>All Destinations</h3>
          <div class="dest-controls">
            <input id="destSearch" placeholder="Search destinations…" oninput="renderAllDestinations(lastState)"/>
          </div>
          <div id="chipbar" class="chipbar"></div>
          <div id="destList" class="dest-list"></div>
        </div>
      </div>

      <div>
        <h3>Log</h3>
        <div id="log"></div>
      </div>
    </div>
  </div>

  <!-- Arrival Modal -->
  <div id="arrivalBackdrop" class="modal-backdrop hidden" role="dialog" aria-modal="true" aria-labelledby="arrivalTitle">
    <div class="modal">
      <h3 id="arrivalTitle">You’ve arrived</h3>
      <p id="arrivalMsg"></p>
      <div class="modal-actions">
        <button onclick="hideArrival()">Not now</button>
        <button class="primary" onclick="confirmNext()">Next</button>
      </div>
    </div>
  </div>

<script>
let lastState = null;
let allDestCache = [];          // [{name, cat}]
let lastSeenID = '';

// Extras inserted between itinerary items; {name, afterIndex}
let extras = [];
let visitedExtras = new Set();

// Category chip state (for All Destinations)
let categories = [];
let activeCats = new Set();

// Arrival modal debounce
let lastModalForTarget = null;

/* ---------- Core helpers ---------- */
function send(cmd, arg=null){
  const payload = arg===null ? {cmd} : {cmd, arg};
  fetch('/api/cmd', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) })
    .catch(()=>{});
}

function setId(){
  const v = document.getElementById('uid').value.trim();
  if(!v){ alert('Please enter a Unique ID'); return; }
  send('SET_ID', v);
  // hide the ID card immediately after sending
  const idCard = document.getElementById('idCard');
  if (idCard) idCard.style.display = 'none';
}

function togglePauseResume(){
  const s = lastState?.status || '';
  if (s === 'running'){ send('STOP'); } else { send('RESUME'); }
}

function onSkip(){
  if (confirm('Skip this destination?')){
    send('SKIP');
  }
}

function showPostLoadUI(loaded){
  const cardControls = document.getElementById('cardControls');
  const postWrap = document.getElementById('postLoadWrap');
  if (!cardControls || !postWrap) return;
  if (loaded){
    cardControls.classList.remove('hidden');
    postWrap.classList.remove('hidden');
  } else {
    cardControls.classList.add('hidden');
    postWrap.classList.add('hidden');
  }
}

function resetForNewID(s){
  const curid = s?.current_id || '';
  if (curid !== lastSeenID){
    lastSeenID = curid;
    extras = [];
    visitedExtras.clear();
    lastModalForTarget = null;
    const log = document.getElementById('log');
    if (log) log.innerHTML = '';
  }
}

function markExtraVisitedOnArrival(s){
  if (s?.status === 'arrived' && s?.target){
    const name = s.target;
    if (extras.some(e => e.name === name)){
      visitedExtras.add(name);
    }
  }
}

/* ---------- Itinerary rendering  ---------- */
function buildCombinedList(labels){
  const combined = [];
  const byAnchor = new Map();
  extras.forEach(e => {
    if (!byAnchor.has(e.afterIndex)) byAnchor.set(e.afterIndex, []);
    byAnchor.get(e.afterIndex).push(e.name);
  });
  for (let i = 0; i < labels.length; i++){
    combined.push({type:'base', name: labels[i], baseIndex: i});
    const list = byAnchor.get(i) || [];
    list.forEach(n => combined.push({type:'extra', name: n}));
  }
  const pre = byAnchor.get(-1) || [];
  if (pre.length){
    return pre.map(n => ({type:'extra', name:n})).concat(combined);
  }
  return combined;
}

function renderItinerary(s){
  const wrap = document.getElementById('itin');
  if(!wrap) return;

  const labels = s.itinerary || [];
  const curIdx = (typeof s.current_idx === 'number') ? s.current_idx : -1;
  const visited = new Set(s.visited_indices || []);
  const skipped = new Set(s.skipped_indices || []);
  const currentTarget = s.target || '';

  wrap.innerHTML = '';

  if(labels.length === 0 && extras.length === 0){
    const empty = document.createElement('div');
    empty.className = 'itin-item';
    empty.innerHTML = '<em>No itinerary loaded.</em>';
    wrap.appendChild(empty);
    return;
  }

  const combined = buildCombinedList(labels);

  combined.forEach(item => {
    const row = document.createElement('div');
    row.className = 'itin-item';

    const left = document.createElement('div');
    left.className = 'itin-name';
    left.textContent = item.name;

    let isCurrent = (currentTarget && currentTarget === item.name);
    let wasVisited = false;
    let wasSkipped = false;

    if (item.type === 'base'){
      if (item.baseIndex === curIdx || isCurrent){
        left.appendChild(pill('going'));
      } else {
        if (visited.has(item.baseIndex)){ wasVisited = true; left.appendChild(pill('visited')); }
        else if (skipped.has(item.baseIndex)){ wasSkipped = true; left.appendChild(pill('skipped')); }
      }
    } else { // extra
      if (isCurrent){
        left.appendChild(pill('going'));
      } else if (visitedExtras.has(item.name)){
        wasVisited = true; left.appendChild(pill('visited'));
      }
    }

    const btn = document.createElement('button');
    if (isCurrent){
      btn.textContent = 'Going';
      btn.disabled = true;
    } else if (wasVisited || wasSkipped){
      btn.textContent = 'Revisit';
      if (item.type === 'base'){
        btn.onclick = () => send('SET_DEST', item.baseIndex);
      } else {
        btn.onclick = () => send('SET_DEST_NAME', item.name);
      }
    } else {
      btn.textContent = 'Go';
      if (item.type === 'base'){
        btn.onclick = () => send('SET_DEST', item.baseIndex);
      } else {
        btn.onclick = () => send('SET_DEST_NAME', item.name);
      }
    }

    row.appendChild(left);
    row.appendChild(btn);
    wrap.appendChild(row);
  });
}

function pill(text){
  const p = document.createElement('span');
  p.className = 'pill';
  p.textContent = text;
  return p;
}

/* ---------- All Destinations ---------- */
function deriveCategories(){
  const set = new Set();
  (allDestCache || []).forEach(d => { if (d.cat) set.add(d.cat); });
  categories = Array.from(set).sort();
}

function renderChipbar(){
  const bar = document.getElementById('chipbar');
  if(!bar) return;
  bar.innerHTML = '';

  const mk = (label, key=null) => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.textContent = label;
    b.onclick = () => {
      if (key === null){
        activeCats.clear();
        renderAllDestinations(lastState);
        return;
      }
      if (activeCats.has(key)) activeCats.delete(key); else activeCats.add(key);
      renderAllDestinations(lastState);
    };
    if (key !== null && activeCats.has(key)) {
      b.style.background = 'var(--accent)';
      b.style.borderColor = 'var(--accent)';
      b.style.color = '#fff';
    }
    return b;
  };

  bar.appendChild(mk('All'));
  categories.forEach(c => bar.appendChild(mk(c, c)));
}

function goFromAll(name){

  const labels = lastState?.itinerary || [];
  if (labels.includes(name) || extras.some(e => e.name === name)){

    send('SET_DEST_NAME', name);
    return;
  }


  const visited = new Set(lastState?.visited_indices || []);
  const skipped = new Set(lastState?.skipped_indices || []);
  let maxDone = -1;
  [...visited, ...skipped].forEach(i => { if (typeof i === 'number' && i > maxDone) maxDone = i; });
  extras.push({name, afterIndex: maxDone});
  send('SET_DEST_NAME', name);
  renderItinerary(lastState);
}

function renderAllDestinations(s){
  const list = document.getElementById('destList');
  const q = (document.getElementById('destSearch')?.value || '').toLowerCase().trim();
  if(!list) return;

  if (Array.isArray(s.all_destinations) && s.all_destinations.length){
    allDestCache = s.all_destinations.slice();
    deriveCategories();
    renderChipbar();
  }

  let filtered = (allDestCache || []).filter(d => {
    const matchesText = !q || (d.name||'').toLowerCase().includes(q) || (d.cat||'').toLowerCase().includes(q);
    const matchesCat = (activeCats.size === 0) || activeCats.has(d.cat||'');
    return matchesText && matchesCat;
  });

  filtered.sort((a,b)=> (a.name||'').localeCompare(b.name||''));

  list.innerHTML = '';
  if (filtered.length === 0){
    const empty = document.createElement('div');
    empty.className = 'dest-row';
    empty.innerHTML = '<em>No matches.</em>';
    list.appendChild(empty);
    return;
  }

  const currentTarget = s.target || '';
  const labels = s.itinerary || [];
  const visited = new Set(s.visited_indices || []);
  const skipped = new Set(s.skipped_indices || []);
  filtered.forEach(d => {
    const row = document.createElement('div'); row.className = 'dest-row';

    const left = document.createElement('div'); left.className = 'dest-meta';
    const nameEl = document.createElement('div'); nameEl.className = 'dest-name'; nameEl.textContent = d.name || '—';
    const catEl = document.createElement('div'); catEl.className = 'dest-cat'; catEl.textContent = d.cat || '';
    left.appendChild(nameEl);
    if (d.cat) left.appendChild(catEl);

    const btn = document.createElement('button');

    const isCurrent = currentTarget && (currentTarget === d.name);
    const baseIndex = labels.indexOf(d.name);
    const isVisitedBase = baseIndex >= 0 && visited.has(baseIndex);
    const isSkippedBase = baseIndex >= 0 && skipped.has(baseIndex);
    const isVisitedExtra = visitedExtras.has(d.name);
    const wasVisitedOrSkipped = isVisitedBase || isVisitedExtra || isSkippedBase;

    if (isCurrent){
      btn.textContent = 'Going';
      btn.disabled = true;
    } else if (wasVisitedOrSkipped){
      btn.textContent = 'Revisit';
      btn.onclick = () => goFromAll(d.name);
    } else {
      btn.textContent = 'Go';
      btn.onclick = () => goFromAll(d.name);
    }

    row.appendChild(left);
    row.appendChild(btn);
    list.appendChild(row);
  });
}

/* ---------- Pause/Resume button label ---------- */
function updatePauseResumeButton(s){
  const btn = document.getElementById('pauseResumeBtn');
  if(!btn) return;
  const st = s?.status || '';
  btn.textContent = (st === 'running') ? 'Pause' : 'Resume';
}

/* ---------- Arrival modal ---------- */
function showArrival(){
  const bd = document.getElementById('arrivalBackdrop');
  if (bd) bd.classList.remove('hidden');
}

function hideArrival(){
  const bd = document.getElementById('arrivalBackdrop');
  if (bd) bd.classList.add('hidden');
}

function confirmNext(){
  hideArrival();
  send('NEXT');
}

function maybeShowArrivalModal(s){
  // Only show when status == arrived, and only once for each target
  const arrived = s?.status === 'arrived';
  const target = (s?.target || '').trim();
  const msg = document.getElementById('arrivalMsg');

  if (!arrived){
    lastModalForTarget = null;
    hideArrival();
    return;
  }
  if (!target) return;

  if (lastModalForTarget === target){
    return;
  }

  if (msg){
    msg.textContent = `You have arrived at “${target}”. Click Next to continue.`;
  }
  lastModalForTarget = target;
  showArrival();
}

async function tick(){
  try{
    const r = await fetch('/api/state');
    const s = await r.json();

    resetForNewID(s);
    markExtraVisitedOnArrival(s);

    lastState = s;

    document.getElementById('status').textContent = s.status || 'unknown';
    document.getElementById('target').textContent = s.target || '—';
    document.getElementById('target_inline').textContent = s.target || '—';
    document.getElementById('curid').textContent = s.current_id || '—';

    const idLoaded = !!(s.current_id && s.current_id.trim().length > 0);
    showPostLoadUI(idLoaded);

    if (idLoaded){
      renderItinerary(s);
      renderAllDestinations(s);
      updatePauseResumeButton(s);
      maybeShowArrivalModal(s);
    } else {
      hideArrival();
    }

    if (s.toast){
      const log = document.getElementById('log');
      if (log){
        const div = document.createElement('div');
        div.className = 'logline';
        const t = new Date().toLocaleTimeString();
        div.textContent = `[${t}] ${s.toast}`;
        log.prepend(div);
      }
    }
  }catch(e){}
  setTimeout(tick, 400);
}
tick();
</script>
</body></html>"""

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path)
        if p.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type","text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.server.parent._html.encode("utf-8"))
            return
        if p.path == "/api/state":
            self.server.parent._lock.acquire()
            payload = json.dumps(self.server.parent._state)
            self.server.parent._state["toast"] = ""  
            self.server.parent._lock.release()
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))
            return
        self.send_response(404); self.end_headers()

    def do_POST(self):
        p = urlparse(self.path)
        if p.path == "/api/cmd":
            try:
                ln = int(self.headers.get("Content-Length","0"))
                raw = self.rfile.read(ln) if ln>0 else b"{}"
                data = json.loads(raw or b"{}")
                self.server.parent._queue.put({
                    "cmd": str(data.get("cmd","")).upper(),
                    "arg": data.get("arg")
                })
                self.send_response(200); self.end_headers()
            except Exception:
                self.send_response(400); self.end_headers()
            return
        self.send_response(404); self.end_headers()

    def log_message(self, *args, **kwargs): return  

class ControlServer:

    def __init__(self, host="127.0.0.1", port=8765, html=_HTML):
        self._queue = queue.Queue()
        self._state = {
            "status":"waiting-id",
            "target":"",
            "toast":"",
            "current_id":"",

            "itinerary": [],
            "current_idx": -1,
            "visited_indices": [],
            "skipped_indices": [],   

            "all_destinations": []
        }
        self._lock = threading.Lock()
        self._html = html
        self._httpd = HTTPServer((host, port), _Handler)
        self._httpd.parent = self
        t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        t.start()
        print(f"[UI] Open http://{host}:{port}")

    def get_nowait(self):
        import queue as _q
        try:
            return self._queue.get_nowait()
        except _q.Empty:
            return None

    def set_state(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if k == "toast":
                    if v:
                        self._state["toast"] = v
                else:
                    if k in self._state:
                        self._state[k] = v

    def shutdown(self):
        try:
            self._httpd.shutdown()
        except Exception:
            pass