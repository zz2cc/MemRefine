"""Web-based GUI for the Memory Optimization Pipeline.

Usage: python server.py
Opens browser at http://localhost:8520
"""

import os, sys, json, time, subprocess, threading, queue, tempfile, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, Response, render_template_string
from werkzeug.utils import secure_filename
from utils import parse_file

app = Flask(__name__)
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global state
run_state = {"running": False, "queue": queue.Queue(), "process": None}

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Memory Optimization Pipeline</title>
<style>
:root{--bg:#f8fafc;--card:#fff;--accent:#6366f1;--green:#10b981;--red:#ef4444;--orange:#f59e0b;--text:#1e293b;--muted:#94a3b8;--border:#e2e8f0;--radius:12px}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
.header{background:linear-gradient(135deg,#6366f1,#8b5cf6);color:#fff;padding:20px 32px}
.header h1{font-size:22px;font-weight:700}.header p{font-size:13px;opacity:.85;margin-top:4px}
.container{max-width:1280px;margin:0 auto;padding:24px}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:24px;margin-bottom:16px}
.card-title{font-size:15px;font-weight:700;margin-bottom:16px;display:flex;align-items:center;gap:8px}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
.mode-btn{padding:10px 20px;border:2px solid var(--border);border-radius:8px;cursor:pointer;font-size:14px;font-weight:600;background:var(--card);transition:all .15s}
.mode-btn.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.mode-btn:hover:not(.active){border-color:var(--accent)}
.setting{margin-left:auto;display:flex;align-items:center;gap:6px;font-size:13px;color:var(--muted)}
.setting input{width:48px;padding:6px;border:1px solid var(--border);border-radius:6px;text-align:center;font-size:13px}
.dropzone{border:2px dashed var(--border);border-radius:var(--radius);padding:40px;text-align:center;cursor:pointer;transition:all .2s;background:#fafbfc;position:relative}
.dropzone.dragover{border-color:var(--accent);background:#eef2ff}
.dropzone.has-file{border-color:var(--green);background:#f0fdf4}
.dropzone-icon{font-size:36px;margin-bottom:8px}
.dropzone p{font-size:14px;color:var(--muted)}.dropzone .file-info{font-size:12px;color:var(--green);font-weight:600;margin-top:6px}
.dropzone input{display:none}
textarea{width:100%;border:1px solid var(--border);border-radius:8px;padding:14px;font-family:'Segoe UI',monospace;font-size:13px;resize:vertical;min-height:120px;background:#fafbfc}
.btn{display:inline-flex;align-items:center;gap:6px;padding:10px 24px;border:none;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer;transition:all .15s}
.btn-primary{background:var(--accent);color:#fff}.btn-primary:hover{filter:brightness(1.1)}
.btn-primary:disabled{opacity:.5;cursor:not-allowed}
.status{font-size:13px;font-weight:600;padding-left:12px}
.log-panel{background:#1e293b;color:#e2e8f0;border-radius:var(--radius);padding:16px;font-family:'Consolas','Cascadia Code',monospace;font-size:12px;max-height:360px;overflow-y:auto;white-space:pre-wrap;line-height:1.5}
.result-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.result-card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:16px;max-height:400px;overflow-y:auto}
.result-card h4{font-size:14px;margin-bottom:10px;color:var(--accent)}
.result-card pre{white-space:pre-wrap;font-family:'Consolas',monospace;font-size:12px;line-height:1.4;background:#fafbfc;padding:10px;border-radius:6px;margin:0}
.file-link{display:block;padding:6px 10px;color:var(--accent);cursor:pointer;font-size:13px;border-radius:4px;text-decoration:none}
.file-link:hover{background:#eef2ff}
.summary-table{width:100%;border-collapse:collapse;font-size:12px}
.summary-table th,.summary-table td{padding:6px 10px;text-align:left;border-bottom:1px solid var(--border)}
.summary-table th{font-weight:700;color:var(--muted);font-size:11px;text-transform:uppercase}
.delta-pos{color:var(--green);font-weight:700}.delta-neg{color:var(--red);font-weight:700}
#results{display:none}
.spinner{display:none;width:16px;height:16px;border:2px solid #e2e8f0;border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="header">
  <h1>Memory Optimization Pipeline</h1>
  <p>Iterative self-comparison for better document summarization</p>
</div>
<div class="container">

<div class="card">
  <div class="card-title">Mode</div>
  <div class="row">
    <button class="mode-btn active" data-mode="test">Test (5 dialogues)</button>
    <button class="mode-btn" data-mode="user">User (dialogue)</button>
    <div class="setting">
      Rounds <input id="rounds" type="number" value="5" min="1" max="10">
      K <input id="candidates" type="number" value="5" min="2" max="10">
    </div>
  </div>
</div>

<div class="card" id="upload-card">
  <div class="card-title">Input</div>
  <div class="dropzone" id="dropzone">
    <input type="file" id="file-input" accept=".pdf,.pptx,.docx,.csv,.json,.xml,.txt,.md" multiple>
    <div class="dropzone-icon">📂</div>
    <p><strong>Drop files here</strong> or click to upload</p>
    <p style="font-size:12px;margin-top:4px">PDF · PPTX · DOCX · CSV · JSON · XML · TXT · MD</p>
    <div class="file-info" id="file-info"></div>
  </div>
  <textarea id="text-input" placeholder="Or paste your text here directly..." style="margin-top:12px"></textarea>
</div>

<div class="row" style="margin-bottom:16px">
  <button class="btn btn-primary" id="run-btn" onclick="start()">▶ Run</button>
  <div class="spinner" id="spinner"></div>
  <span class="status" id="status"></span>
</div>

<div class="card">
  <div class="card-title">Progress</div>
  <div class="log-panel" id="log"></div>
</div>

<div id="results">
  <div class="card">
    <div class="card-title">Results</div>
    <div class="result-grid">
      <div class="result-card"><h4>Best Memory</h4><div id="memory-content"></div></div>
      <div class="result-card">
        <h4>Summary</h4>
        <table class="summary-table"><thead><tr><th>Dialogue</th><th>R0</th><th>Best</th><th>Delta</th><th>Rules</th></tr></thead><tbody id="summary-body"></tbody></table>
        <div style="margin-top:12px"><h4>Rules</h4><div id="rules-content" style="font-size:12px;max-height:200px;overflow-y:auto"></div></div>
      </div>
    </div>
    <div style="margin-top:12px"><h4 style="font-size:14px;margin-bottom:6px">Output Files</h4><div id="files-list"></div></div>
  </div>
</div>

</div>
<script>
let mode='test',uploadedFiles=[];
document.querySelectorAll('.mode-btn').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.mode-btn').forEach(x=>x.classList.remove('active'));
  b.classList.add('active');mode=b.dataset.mode;
  document.getElementById('upload-card').style.display=mode==='test'?'none':'';
});
document.getElementById('upload-card').style.display='none';

const dz=document.getElementById('dropzone'),fi=document.getElementById('file-info');
dz.onclick=()=>document.getElementById('file-input').click();
dz.ondragover=e=>{e.preventDefault();dz.classList.add('dragover')};
dz.ondragleave=()=>dz.classList.remove('dragover');
dz.ondrop=e=>{e.preventDefault();dz.classList.remove('dragover');handleFiles(e.dataTransfer.files)};
document.getElementById('file-input').onchange=e=>handleFiles(e.target.files);

async function handleFiles(files){
  uploadedFiles=[];
  for(let f of files){
    let text=await readFile(f);
    uploadedFiles.push({name:f.name,text:text});
  }
  if(uploadedFiles.length){
    fi.textContent=uploadedFiles.map(f=>`${f.name} (${(f.text.length/1024).toFixed(1)} KB)`).join(', ');
    dz.classList.add('has-file');
    let combined=uploadedFiles.map(f=>f.text).join('\n\n');
    document.getElementById('text-input').value=combined;
  }
}

function readFile(file){
  return new Promise(resolve=>{
    let reader=new FileReader();
    reader.onload=e=>resolve(e.target.result);
    if(file.name.match(/\.(txt|md|csv|json|xml)$/i)) reader.readAsText(file);
    else reader.readAsText(file);
  });
}

function start(){
  let rounds=document.getElementById('rounds').value;
  let cand=document.getElementById('candidates').value;
  let text=document.getElementById('text-input').value.trim();
  if(mode!=='test'&&!text&&!uploadedFiles.length){alert('Please provide input text or upload a file');return}
  document.getElementById('run-btn').disabled=true;
  document.getElementById('spinner').style.display='inline-block';
  document.getElementById('status').textContent='Running...';
  document.getElementById('status').style.color='var(--accent)';
  document.getElementById('results').style.display='none';
  document.getElementById('log').textContent='';

  let body={mode,rounds,candidates:text?parseInt(candidates):5};
  if(text) body.text=text;

  fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
  .then(r=>{if(!r.ok)throw new Error('Failed to start');listenLog();})
  .catch(e=>{document.getElementById('log').textContent='Error: '+e.message;resetBtn()});
}

function listenLog(){
  let evt=new EventSource('/log');
  let logEl=document.getElementById('log');
  evt.onmessage=e=>{
    if(e.data==='__DONE__'){evt.close();onDone();return}
    logEl.textContent+='['+new Date().toLocaleTimeString()+'] '+e.data+'\n';
    logEl.scrollTop=logEl.scrollHeight;
  };
  evt.onerror=()=>{evt.close();onDone()};
}

function onDone(){
  resetBtn();
  fetch('/results').then(r=>r.json()).then(data=>{
    if(data.error){document.getElementById('status').textContent='Failed';document.getElementById('status').style.color='var(--red)';return}
    document.getElementById('status').textContent='Done';
    document.getElementById('status').style.color='var(--green)';
    document.getElementById('results').style.display='block';

    let memDiv=document.getElementById('memory-content');
    memDiv.innerHTML='';
    (data.results||[]).forEach(r=>{
      let score=typeof r.best_score==='object'?r.best_score.composite:r.best_score;
      let traj=r.score_trajectory||[];
      let best=Math.max(...traj);
      memDiv.innerHTML+=`<pre style="margin-bottom:16px"><b>${r.dialogue_id}</b> | Score: ${best.toFixed(4)} | Rules: ${r.experience_count}\n${'='.repeat(50)}\n${r.best_memory||''}</pre>`;
    });

    let tbody=document.getElementById('summary-body');
    tbody.innerHTML='';
    (data.results||[]).forEach(r=>{
      let traj=r.score_trajectory||[];
      let r0=traj[0]||0,best=Math.max(...traj),d=best-r0;
      tbody.innerHTML+=`<tr><td>${r.dialogue_id}</td><td>${r0.toFixed(4)}</td><td>${best.toFixed(4)}</td><td class="${d>=0?'delta-pos':'delta-neg'}">${d>=0?'+':''}${d.toFixed(4)}</td><td>${r.experience_count}</td></tr>`;
    });

    let rulesDiv=document.getElementById('rules-content');
    rulesDiv.innerHTML='';
    (data.rules||[]).forEach((r,i)=>{
      rulesDiv.innerHTML+=`<div style="margin-bottom:4px">${i+1}. ${r}</div>`;
    });

    let filesDiv=document.getElementById('files-list');
    filesDiv.innerHTML='';
    (data.files||[]).forEach(f=>{
      filesDiv.innerHTML+=`<a class="file-link" href="/output/${f}" target="_blank">📄 ${f}</a>`;
    });
  });
}

function resetBtn(){
  document.getElementById('run-btn').disabled=false;
  document.getElementById('spinner').style.display='none';
}
</script>
</body></html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/run", methods=["POST"])
def start_run():
    if run_state["running"]:
        return jsonify({"error": "Already running"}), 409

    data = request.get_json()
    mode = data.get("mode", "test")
    rounds = data.get("rounds", 5)
    candidates = data.get("candidates", 5)
    text = data.get("text", "")

    # Write text to temp file if provided (avoids WinError 206)
    tmp_path = None
    if text:
        tmp_path = os.path.join(UPLOAD_DIR, "_input.txt")
        with open(tmp_path, "w", encoding="utf-8", errors="surrogateescape") as f:
            f.write(text)

    python_exe = r"C:\Users\z2cc_\AppData\Local\Programs\Python\Python312\python.exe"
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    cmd = [python_exe, "main.py", "--rounds", str(rounds), "--candidates", str(candidates), "--mode", mode]
    if tmp_path and mode != "test":
        cmd += ["--file", tmp_path]
    if mode == "test":
        cmd += ["--mode", "test"]

    run_state["queue"] = queue.Queue()
    run_state["running"] = True
    run_state["mode"] = mode

    threading.Thread(target=_run_pipeline, args=(cmd,), daemon=True).start()
    return jsonify({"ok": True})


def _run_pipeline(cmd):
    q = run_state["queue"]
    cwd = os.path.dirname(os.path.abspath(__file__))
    try:
        q.put(f"Starting: {' '.join(cmd[1:6])}...")
        q.put("─" * 50)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="surrogateescape", cwd=cwd)
        run_state["process"] = p
        for line in p.stdout:
            line = line.rstrip()
            if line:
                q.put(line)
        p.wait()
        q.put("─" * 50)
        q.put(f"DONE (exit code {p.returncode})")
    except Exception as e:
        q.put(f"ERROR: {e}")
    finally:
        q.put("__DONE__")
        run_state["running"] = False


@app.route("/log")
def stream_log():
    def generate():
        q = run_state["queue"]
        while True:
            try:
                msg = q.get(timeout=30)
                yield f"data: {msg}\n\n"
                if msg == "__DONE__":
                    break
            except queue.Empty:
                yield f"data: \n\n"
    return Response(generate(), mimetype="text/event-stream")


@app.route("/results")
def get_results():
    mode = run_state.get("mode", "test")
    output_dir = f"output/{mode}"
    if mode == "test":
        output_dir = "output/test"
    elif mode == "user":
        output_dir = "output/user"

    results_path = os.path.join(output_dir, "results.json")
    exp_path = os.path.join(output_dir, "results_experiences.json")

    results_data = []
    rules_data = []
    files = []

    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)
    if os.path.exists(exp_path):
        with open(exp_path, "r", encoding="utf-8") as f:
            rules_raw = json.load(f)
            if isinstance(rules_raw, dict):
                for v in rules_raw.values():
                    rules_data.extend(v)
            elif isinstance(rules_raw, list):
                rules_data = rules_raw

    if os.path.isdir(output_dir):
        files = sorted(os.listdir(output_dir))

    return jsonify({"results": results_data, "rules": rules_data, "files": files})


@app.route("/output/<path:filename>")
def serve_output(filename):
    mode = run_state.get("mode", "test")
    output_dir = f"output/{mode}"
    if mode == "test":
        output_dir = "output/test"
    elif mode == "user":
        output_dir = "output/user"
    from flask import send_from_directory
    return send_from_directory(os.path.abspath(output_dir), filename)


if __name__ == "__main__":
    import webbrowser
    port = 8520
    print(f"\n  Opening http://localhost:{port}")
    webbrowser.open(f"http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
