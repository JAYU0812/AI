/**
 * MY AI — Agent-First Application
 * JS: Project Home, Create Project, Agent Workspace, Conversation Engine
 * Chat history is persisted to localStorage keyed by project ID.
 */

'use strict';

// ══════════════════════════════════════════════════════════════
//  API LAYER
// ══════════════════════════════════════════════════════════════
const API = {
  async get(path) {
    const r = await fetch(path);
    if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.statusText); }
    return r.json();
  },
  async post(path, body) {
    const r = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.statusText); }
    return r.json();
  },
  async delete(path) {
    const r = await fetch(path, { method: 'DELETE' });
    if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.statusText); }
    return r.json();
  },
};

// ══════════════════════════════════════════════════════════════
//  SCREEN ROUTER
// ══════════════════════════════════════════════════════════════
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

// ══════════════════════════════════════════════════════════════
//  MODAL HELPERS
// ══════════════════════════════════════════════════════════════
function openModal(id) {
  const el = document.getElementById(id);
  el.classList.add('open');
  el.setAttribute('aria-hidden', 'false');
}
function closeModal(id) {
  const el = document.getElementById(id);
  el.classList.remove('open');
  el.setAttribute('aria-hidden', 'true');
}

// ══════════════════════════════════════════════════════════════
//  LOADER
// ══════════════════════════════════════════════════════════════
function showLoader(msg = 'Working…') {
  const el = document.getElementById('globalLoader');
  document.getElementById('loaderMsg').textContent = msg;
  el.setAttribute('aria-hidden', 'false');
}
function hideLoader() {
  document.getElementById('globalLoader').setAttribute('aria-hidden', 'true');
}

// ══════════════════════════════════════════════════════════════
//  TOAST
// ══════════════════════════════════════════════════════════════
function toast(msg, type = 'info', ms = 3500) {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = msg;
  document.getElementById('toastStack').appendChild(el);
  setTimeout(() => el.remove(), ms);
}

// ══════════════════════════════════════════════════════════════
//  FILE ICON HELPER
// ══════════════════════════════════════════════════════════════
function fileIcon(name = '') {
  const ext = (name.split('.').pop() || '').toLowerCase();
  const m = { py:'🐍', js:'📜', ts:'📜', html:'🌐', css:'🎨', json:'📋',
    md:'📝', txt:'📄', sh:'⚡', bat:'⚡', csv:'📊', yml:'⚙️', yaml:'⚙️',
    sql:'🗄️', db:'🗄️', png:'🖼️', jpg:'🖼️', svg:'🖼️', env:'🔒' };
  return m[ext] || '📄';
}

// ══════════════════════════════════════════════════════════════
//  ESCAPING
// ══════════════════════════════════════════════════════════════
function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ══════════════════════════════════════════════════════════════
//  CHAT HISTORY  — SQLite on backend + localStorage cache
// ══════════════════════════════════════════════════════════════
const ChatHistory = {
  _key(projectId) { return `my_ai_chat_${projectId}`; },

  loadLocal(projectId) {
    try { return JSON.parse(localStorage.getItem(this._key(projectId)) || '[]'); }
    catch { return []; }
  },

  saveLocal(projectId, messages) {
    try { localStorage.setItem(this._key(projectId), JSON.stringify(messages)); }
    catch { /* storage quota ignore */ }
  },

  async load(projectId) {
    // 1. Fetch persistent history from SQLite backend
    try {
      const data = await API.get(`/api/projects/${encodeURIComponent(projectId)}/messages`);
      if (data && Array.isArray(data.messages) && data.messages.length > 0) {
        this.saveLocal(projectId, data.messages);
        return data.messages;
      }
    } catch { }

    // 2. Fallback to localStorage
    return this.loadLocal(projectId);
  },

  appendLocal(projectId, entry) {
    const msgs = this.loadLocal(projectId);
    msgs.push(entry);
    this.saveLocal(projectId, msgs.slice(-200));
  },

  async clear(projectId) {
    localStorage.removeItem(this._key(projectId));
    try {
      await API.delete(`/api/projects/${encodeURIComponent(projectId)}/messages`);
    } catch { }
  },
};

// ══════════════════════════════════════════════════════════════
//  HOME SCREEN
// ══════════════════════════════════════════════════════════════
const Home = {
  async init() { await this.loadProjects(); },

  async loadProjects() {
    const grid  = document.getElementById('projectsGrid');
    const empty = document.getElementById('homeEmptyState');
    grid.innerHTML = '<div class="projects-loading">Loading your projects…</div>';

    try {
      const data = await API.get('/api/projects');
      const projects = data.projects || [];
      grid.innerHTML = '';

      if (!projects.length) {
        empty.style.display = 'flex';
        grid.style.display = 'none';
        return;
      }

      empty.style.display = 'none';
      grid.style.display = 'grid';

      projects.forEach(p => {
        const card = document.createElement('button');
        card.className = 'project-card';
        const name  = p.name || 'Untitled';
        const path  = p.path || '';
        const files = p.file_count ?? 0;
        const date  = p.last_opened_at
          ? new Date(p.last_opened_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
          : '';
        // Show message count from SQLite or local fallback
        const msgCount = (p.message_count !== undefined && p.message_count > 0)
          ? p.message_count
          : ChatHistory.loadLocal(p.id).length;
        const msgLabel = msgCount ? `${msgCount} message${msgCount !== 1 ? 's' : ''}` : '';

        card.innerHTML = `
          <div class="pc-name">${esc(name)}</div>
          <div class="pc-path">${esc(path)}</div>
          <div class="pc-meta">
            <span class="pc-file-count">${files} file${files !== 1 ? 's' : ''}</span>
            ${msgLabel ? `<span class="pc-file-count" style="background:#EFF6FF;color:#2563EB">${msgLabel}</span>` : ''}
            ${date ? `<span class="pc-date">${esc(date)}</span>` : ''}
          </div>`;
        card.addEventListener('click', () => Workspace.open(p));
        grid.appendChild(card);
      });
    } catch (err) {
      grid.innerHTML = `<div class="projects-loading" style="color:#DC2626">Could not load projects: ${esc(err.message)}</div>`;
    }
  },
};

// ══════════════════════════════════════════════════════════════
//  CREATE PROJECT MODAL
// ══════════════════════════════════════════════════════════════
const CreateProject = {
  selectedTemplate: 'empty',

  init() {
    document.querySelectorAll('.template-opt').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.template-opt').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        this.selectedTemplate = btn.dataset.tmpl;
      });
    });

    const nameInput = document.getElementById('cpName');
    const locInput  = document.getElementById('cpLocation');
    const hint      = document.getElementById('cpPathHint');
    const updateHint = () => {
      const n = (nameInput.value.trim().replace(/\s+/g, '_')) || 'MyProject';
      const l = locInput.value.trim() || 'C:\\jay\\Projects';
      hint.textContent = `Will be created at ${l}\\${n}`;
    };
    nameInput.addEventListener('input', updateHint);
    locInput.addEventListener('input', updateHint);

    document.getElementById('btnCpCreate').addEventListener('click', () => this.create());
    document.getElementById('btnCpCancel').addEventListener('click', () => closeModal('createProjectOverlay'));
    document.getElementById('btnCpClose').addEventListener('click', () => closeModal('createProjectOverlay'));
    document.getElementById('createProjectOverlay').addEventListener('click', e => {
      if (e.target.id === 'createProjectOverlay') closeModal('createProjectOverlay');
    });
    nameInput.addEventListener('keydown', e => { if (e.key === 'Enter') this.create(); });
  },

  open() {
    document.getElementById('cpName').value = '';
    document.getElementById('cpLocation').value = 'C:\\jay\\Projects';
    document.getElementById('cpPathHint').textContent = 'The agent will work inside this folder.';
    document.querySelectorAll('.template-opt').forEach(b => b.classList.remove('selected'));
    document.getElementById('tmplEmpty').classList.add('selected');
    this.selectedTemplate = 'empty';
    openModal('createProjectOverlay');
    setTimeout(() => document.getElementById('cpName').focus(), 120);
  },

  async create() {
    const name = document.getElementById('cpName').value.trim();
    const loc  = document.getElementById('cpLocation').value.trim() || 'C:\\jay\\Projects';
    if (!name) { document.getElementById('cpName').focus(); toast('Enter a project name', 'error'); return; }

    closeModal('createProjectOverlay');
    showLoader(`Creating "${name}"…`);
    try {
      const data = await API.post('/api/projects', { name, location: loc, template: this.selectedTemplate, description: '' });
      hideLoader();
      toast(`Project "${name}" created`, 'success');
      await Workspace.open({ id: data.project.id, name: data.project.name, path: data.project.path });
    } catch (err) {
      hideLoader();
      toast(`Failed: ${err.message}`, 'error');
    }
  },
};

// ══════════════════════════════════════════════════════════════
//  OPEN FOLDER MODAL
// ══════════════════════════════════════════════════════════════
const OpenFolder = {
  init() {
    document.getElementById('btnOfOpen').addEventListener('click', () => this.open());
    document.getElementById('btnOfCancel').addEventListener('click', () => closeModal('openFolderOverlay'));
    document.getElementById('btnOfClose').addEventListener('click', () => closeModal('openFolderOverlay'));
    document.getElementById('openFolderOverlay').addEventListener('click', e => {
      if (e.target.id === 'openFolderOverlay') closeModal('openFolderOverlay');
    });
  },

  show() {
    document.getElementById('ofPath').value = '';
    openModal('openFolderOverlay');
    setTimeout(() => document.getElementById('ofPath').focus(), 120);
  },

  async open() {
    const path = document.getElementById('ofPath').value.trim();
    if (!path) { toast('Enter a folder path', 'error'); return; }
    closeModal('openFolderOverlay');
    showLoader('Opening project…');
    try {
      const data = await API.post('/api/projects/open', { path });
      hideLoader();
      toast(`Opened "${data.project.name}"`, 'success');
      await Workspace.open(data.project);
    } catch (err) {
      hideLoader();
      toast(`Failed: ${err.message}`, 'error');
    }
  },
};

// ══════════════════════════════════════════════════════════════
//  WORKSPACE + CONVERSATION ENGINE
// ══════════════════════════════════════════════════════════════
const Workspace = {
  project: null,
  agentBusy: false,
  filesOpen: false,

  async open(project) {
    this.project = project;
    this.agentBusy = false;

    // Persist active project to localStorage so page refresh stays on this workspace
    try {
      localStorage.setItem('my_ai_active_project', JSON.stringify({
        id: project.id,
        name: project.name,
        path: project.path,
      }));
    } catch { }

    // Activate project on backend
    try { await API.post('/api/projects/open', { id: project.id, path: project.path }); } catch { }

    // Update header
    document.getElementById('wsProjectName').textContent = project.name;
    document.getElementById('wsProjectPath').textContent = project.path || '';
    document.getElementById('filesPanelPath').textContent = project.path || '';

    // Clear DOM thread
    document.getElementById('conversationThread').innerHTML = '';

    // Restore saved history from backend SQLite store (falling back to local cache)
    const history = await ChatHistory.load(project.id);
    if (history && history.length > 0) {
      this._restoreHistory(history);
    } else {
      // First visit — show welcome
      this.appendAgentMessage(
        `I'm ready to work on <strong>${esc(project.name)}</strong>.<br>` +
        `Tell me what you'd like to build, change, or fix. I'll handle everything from there.`
      );
    }

    this.setAgentStatus('ready', 'Ready');
    this.enableInput(true);
    this.refreshFiles();
    this.loadProjectUnderstanding(project.id);
    showScreen('screen-workspace');
    setTimeout(() => this.scrollToBottom(), 80);
  },

  async loadProjectUnderstanding(projectId) {
    const bar = document.getElementById('projectUnderstandingBar');
    const details = document.getElementById('projectUnderstandingDetails');
    if (!bar) return;
    try {
      const summary = await API.get(`/api/projects/${projectId}/summary`);
      if (summary && summary.total_files !== undefined) {
        document.getElementById('pubTech').textContent = summary.primary_type || 'Unknown';
        document.getElementById('pubFilesCount').textContent = `${summary.total_files} files`;
        document.getElementById('pubTestsCount').textContent = `${summary.test_files} tests`;
        document.getElementById('pubDepsCount').textContent = `${summary.dependencies_count} deps`;

        if (details) {
          details.innerHTML = `
            <div class="pub-details-content">
              <pre class="pub-summary-pre">${esc(summary.summary_text || '')}</pre>
              ${summary.entry_points?.length ? `<div class="pub-detail-row"><strong>Entry Points:</strong> ${esc(summary.entry_points.join(', '))}</div>` : ''}
              ${summary.modules?.length ? `<div class="pub-detail-row"><strong>Modules:</strong> ${esc(summary.modules.join(', '))}</div>` : ''}
            </div>
          `;
          details.style.display = 'none';
        }
        bar.style.display = 'flex';
      }
    } catch (e) {
      bar.style.display = 'none';
    }
  },

  async clearChat() {
    if (!this.project) return;
    if (!confirm(`Clear all conversation history for "${this.project.name}"?`)) return;
    await ChatHistory.clear(this.project.id);
    document.getElementById('conversationThread').innerHTML = '';
    this.appendAgentMessage(
      `I'm ready to work on <strong>${esc(this.project.name)}</strong>.<br>` +
      `Tell me what you'd like to build, change, or fix. I'll handle everything from there.`
    );
    toast('Conversation cleared', 'info');
  },

  // ── History restore ─────────────────────────────────────────
  _restoreHistory(history) {
    history.forEach(entry => {
      if (entry.role === 'user') {
        this._renderUserTurn(entry.text);
      } else if (entry.role === 'agent') {
        const body = this._renderAgentTurnShell();
        if (entry.text) {
          const p = document.createElement('p');
          p.className = 'agent-message-text';
          const formatted = entry.text.includes('<') ? entry.text : esc(entry.text).replace(/\n/g, '<br>');
          p.innerHTML = formatted;
          body.appendChild(p);
        }
        if (entry.result) {
          const didWork = entry.result.plan?.steps?.length > 0;
          if (didWork) body.appendChild(this.buildActivityCard(entry.result));
        }
      }
    });
  },

  // ── Conversation thread ─────────────────────────────────────
  _renderUserTurn(text) {
    const thread = document.getElementById('conversationThread');
    const turn = document.createElement('div');
    turn.className = 'turn-user';
    turn.innerHTML = `<div class="user-bubble">${esc(text).replace(/\n/g, '<br>')}</div>`;
    thread.appendChild(turn);
    return turn;
  },

  _renderAgentTurnShell() {
    const thread = document.getElementById('conversationThread');
    const turn = document.createElement('div');
    turn.className = 'turn-agent';
    turn.innerHTML = `<div class="turn-agent-avatar">🤖</div><div class="turn-agent-body"></div>`;
    thread.appendChild(turn);
    return turn.querySelector('.turn-agent-body');
  },

  appendUserTurn(text) {
    const turn = this._renderUserTurn(text);
    this.scrollToBottom();
    return turn;
  },

  appendAgentTurn() {
    const body = this._renderAgentTurnShell();
    this.scrollToBottom();
    return body;
  },

  appendAgentMessage(html) {
    const body = this.appendAgentTurn();
    const msg = document.createElement('p');
    msg.className = 'agent-message-text';
    msg.innerHTML = html;
    body.appendChild(msg);
    this.scrollToBottom();
    return body;
  },

  appendTypingIndicator() {
    const body = this.appendAgentTurn();
    body.id = 'typingIndicatorBody';
    body.innerHTML = `
      <div class="agent-typing">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>`;
    this.scrollToBottom();
    return body;
  },

  removeTypingIndicator() {
    const el = document.getElementById('typingIndicatorBody');
    if (el) { el.closest('.turn-agent').remove(); }
  },

  scrollToBottom() {
    const col = document.getElementById('wsConversationCol');
    requestAnimationFrame(() => { col.scrollTop = col.scrollHeight; });
  },

  // ── Activity card ───────────────────────────────────────────
  buildActivityCard(result) {
    const card = document.createElement('div');
    card.className = 'activity-card';

    const plan  = result.plan || {};
    const steps = plan.steps || [];
    const state = plan.state || 'completed';
    const dotClass  = state === 'completed' ? 'dot-success' : state === 'failed' ? 'dot-failed' : 'dot-running';

    const stepsHtml = steps.map(s => {
      const done = s.status === 'success';
      const fail = s.status === 'failed';
      const icon  = done ? '✓' : fail ? '✕' : '→';
      const ic    = done ? 'done' : fail ? 'fail' : 'active';
      const lc    = done ? 'done' : fail ? '' : 'active';
      return `<div class="activity-step">
        <span class="step-icon ${ic}">${icon}</span>
        <span class="step-label ${lc}">${esc(s.description || s.action_type)}</span>
      </div>`;
    }).join('');

    const changedFiles = (result.executed_steps || [])
      .filter(s => s.tool_args?.path && s.action_type !== 'RUN_COMMAND')
      .map(s => ({
        name:  s.tool_args.path,
        badge: s.action_type === 'CREATE_FILE' ? 'created' : s.action_type === 'DELETE_FILE' ? 'deleted' : 'modified',
      }));

    const filesHtml = changedFiles.length ? `
      <div class="files-changed-summary">
        <div class="files-changed-title">Files</div>
        ${changedFiles.map(f => `
          <div class="file-change-item">
            <span class="file-change-badge badge-${f.badge}">${f.badge[0].toUpperCase()}</span>
            <span class="file-change-name" data-file="${esc(f.name)}">${esc(f.name)}</span>
          </div>`).join('')}
      </div>` : '';

    const termHtml = result.terminal_output ? `
      <div class="terminal-block">
        <div class="terminal-block-label">Output</div>
        <pre class="terminal-block-content">${esc(result.terminal_output)}</pre>
      </div>` : '';

    const failedStep = (result.executed_steps || []).find(s => !s.success);
    const errorHtml  = failedStep ? `
      <div class="error-block">
        <div class="error-block-label">What happened</div>
        <div class="error-block-text">${esc(failedStep.observation || 'An error occurred.')}</div>
      </div>` : '';

    // Step 3B: Project understanding badge in activity card
    let pubHtml = '';
    if (result.project_understanding) {
      const pu = result.project_understanding;
      const relList = (pu.relevant_files || []).map(f => `<span class="pu-rel-file">${esc(f)}</span>`).join(' ');
      pubHtml = `
        <div class="pu-card-badge">
          <div class="pu-badge-header">
            <span>🔍</span>
            <span>Project Understanding: <strong>${esc(pu.primary_type || 'Project')}</strong></span>
            <span class="pu-badge-counts">(${pu.total_files || 0} files · ${pu.test_files || 0} tests)</span>
          </div>
          ${relList ? `<div class="pu-badge-relevant">Relevant files: ${relList}</div>` : ''}
        </div>
      `;
    }

    // Step 3C: Test execution badge in activity card
    let testHtml = '';
    if (result.test_results) {
      const tr = result.test_results;
      const statusIcon = tr.passed ? '✓' : '✗';
      const statusClass = tr.passed ? 'test-passed' : 'test-failed';
      testHtml = `
        <div class="activity-test-badge ${statusClass}">
          <span>${statusIcon} Tests: <strong>${tr.passed ? 'Passed' : 'Failed'}</strong> (${esc(tr.runner_command)})</span>
          ${tr.diagnostics ? `<div class="test-diag">${esc(tr.diagnostics)}</div>` : ''}
        </div>
      `;
    }

    // Step 3C: Factual diffs section
    let diffsHtml = '';
    if (result.diffs && result.diffs.length > 0) {
      const diffBlocks = result.diffs.map(d => `
        <div class="diff-block">
          <div class="diff-header">
            <span class="diff-file">${esc(d.file_path)}</span>
            <span class="diff-stats">+${d.lines_added} / -${d.lines_removed} (${d.preservation_percentage}% preserved)</span>
          </div>
          ${d.unified_diff ? `<pre class="diff-content">${esc(d.unified_diff)}</pre>` : ''}
        </div>
      `).join('');
      diffsHtml = `
        <div class="activity-diffs-container">
          <div class="activity-section-label" style="padding:6px 12px;font-size:11px;font-weight:600;color:var(--text-secondary);background:var(--bg-surface);">Applied Changes (Diff)</div>
          ${diffBlocks}
        </div>
      `;
    }

    card.innerHTML = `
      <div class="activity-card-header" role="button" aria-expanded="false">
        <span class="activity-card-status-dot ${dotClass}"></span>
        <span class="activity-card-title">${esc(plan.goal || 'Agent working…').slice(0, 80)}</span>
        <svg class="activity-card-toggle" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
      </div>
      <div class="activity-card-body">
        ${pubHtml}
        ${testHtml}
        ${steps.length ? `<div class="activity-steps">${stepsHtml}</div>` : ''}
        ${diffsHtml}
        ${filesHtml}${termHtml}${errorHtml}
      </div>`;

    const header = card.querySelector('.activity-card-header');
    const body   = card.querySelector('.activity-card-body');
    const toggle = card.querySelector('.activity-card-toggle');
    header.addEventListener('click', () => {
      const open = body.classList.toggle('open');
      toggle.classList.toggle('open', open);
      header.setAttribute('aria-expanded', String(open));
    });
    // Auto-expand on first render
    body.classList.add('open');
    toggle.classList.add('open');
    header.setAttribute('aria-expanded', 'true');

    card.querySelectorAll('.file-change-name').forEach(el => {
      el.addEventListener('click', e => { e.stopPropagation(); this.viewFile(el.dataset.file); });
    });

    return card;
  },

  // ── Send ────────────────────────────────────────────────────
  async send(message) {
    if (!message.trim() || this.agentBusy) return;
    this.agentBusy = true;

    const input = document.getElementById('agentInput');
    input.value = '';
    autoResizeTextarea(input);
    this.enableInput(false);

    const projId = this.project?.id || 'default';

    // Render + persist user message
    this.appendUserTurn(message);
    ChatHistory.appendLocal(projId, { role: 'user', text: message, ts: Date.now() });

    // Typing indicator
    this.appendTypingIndicator();
    this.setAgentStatus('thinking', 'Thinking…');

    try {
      await new Promise(r => setTimeout(r, 60));
      this.setAgentStatus('working', 'Working…');

      // Send to server with project_id for server-side SQLite persistence
      const result = await API.post('/api/chat', { message, project_id: projId });

      this.removeTypingIndicator();

      const body = this.appendAgentTurn();
      const responseText = (result.response || '').trim();
      let savedHtml = '';

      if (responseText) {
        savedHtml = esc(responseText).replace(/\n/g, '<br>');
        const p = document.createElement('p');
        p.className = 'agent-message-text';
        p.innerHTML = savedHtml;
        body.appendChild(p);
      } else if (!(result.plan?.steps?.length > 0)) {
        savedHtml = `I'm not sure how to handle that. Try asking me to build something — for example: <em>"Create a Python script that calculates Fibonacci numbers and run it"</em>`;
        const p = document.createElement('p');
        p.className = 'agent-message-text';
        p.innerHTML = savedHtml;
        body.appendChild(p);
      }

      const didWork = result.plan?.steps?.length > 0;
      if (didWork) body.appendChild(this.buildActivityCard(result));

      // Persist agent turn locally
      ChatHistory.appendLocal(projId, {
        role: 'agent',
        text: savedHtml,
        result: didWork ? result : null,
        ts: Date.now(),
      });

      this.setAgentStatus('ready', 'Ready');
      await this.refreshFiles();

    } catch (err) {
      this.removeTypingIndicator();
      const body = this.appendAgentTurn();
      const errHtml = `I ran into a problem: ${esc(err.message)}`;
      body.innerHTML = `
        <p class="agent-message-text">I ran into a problem:</p>
        <div class="error-block" style="border-top:none;border-radius:8px;border:1px solid #FECACA">
          <div class="error-block-label">Error</div>
          <div class="error-block-text">${esc(err.message)}</div>
          <div class="error-block-action">
            <button class="btn-retry" id="retryBtn">Try again</button>
          </div>
        </div>`;
      body.querySelector('#retryBtn')?.addEventListener('click', () => this.send(message));
      ChatHistory.appendLocal(projId, {
        role: 'agent',
        text: errHtml,
        result: null,
        ts: Date.now(),
      });
      this.setAgentStatus('error', 'Failed');
      setTimeout(() => this.setAgentStatus('ready', 'Ready'), 4000);
    } finally {
      this.agentBusy = false;
      this.enableInput(true);
      this.scrollToBottom();
    }
  },

  // ── Agent status ─────────────────────────────────────────────
  setAgentStatus(status, label) {
    const pill = document.getElementById('agentStatusPill');
    document.getElementById('agentStatusLabel').textContent = label;
    pill.className = `agent-status-pill status-${status}`;
  },

  // ── Input control ─────────────────────────────────────────────
  enableInput(yes) {
    const input   = document.getElementById('agentInput');
    const sendBtn = document.getElementById('btnSend');
    input.disabled   = !yes;
    sendBtn.disabled = !yes;
    if (yes) setTimeout(() => input.focus(), 50);
  },

  // ── Files panel ───────────────────────────────────────────────
  async refreshFiles() {
    const list = document.getElementById('filesList');
    list.innerHTML = '<div class="files-loading">Loading…</div>';
    try {
      const data  = await API.get('/api/files');
      const files = data.files || [];
      list.innerHTML = '';
      if (!files.length) { list.innerHTML = '<div class="files-loading">No files yet.</div>'; return; }
      files.forEach(f => {
        const item = document.createElement('button');
        item.className = 'file-list-item';
        item.innerHTML = `<span class="file-list-item-icon">${fileIcon(f.name)}</span><span class="file-list-item-name">${esc(f.path)}</span>`;
        item.addEventListener('click', () => this.viewFile(f.path));
        list.appendChild(item);
      });
    } catch { list.innerHTML = '<div class="files-loading">Unable to load files.</div>'; }
  },

  toggleFilesPanel() {
    this.filesOpen = !this.filesOpen;
    const panel = document.getElementById('wsFilesPanel');
    panel.classList.toggle('open', this.filesOpen);
    panel.setAttribute('aria-hidden', String(!this.filesOpen));
    if (this.filesOpen) this.refreshFiles();
  },

  async viewFile(path) {
    try {
      const data = await API.get(`/api/files/${path}`);
      document.getElementById('fileModalTitle').textContent = path;
      document.getElementById('fileModalContent').textContent = data.content;
      openModal('fileModalOverlay');
    } catch (err) {
      toast(`Could not open file: ${err.message}`, 'error');
    }
  },
};

// ══════════════════════════════════════════════════════════════
//  AUTO-RESIZE TEXTAREA
// ══════════════════════════════════════════════════════════════
function autoResizeTextarea(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}

// ══════════════════════════════════════════════════════════════
//  MAIN INIT
// ══════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', async () => {

  // Home buttons
  document.getElementById('btnNewProject').addEventListener('click', () => CreateProject.open());
  document.getElementById('btnOpenExisting').addEventListener('click', () => OpenFolder.show());
  document.getElementById('btnEmptyNewProject').addEventListener('click', () => CreateProject.open());
  document.getElementById('btnHomeSettings').addEventListener('click', () => toast('Settings coming soon', 'info'));

  // Workspace nav
  document.getElementById('btnWsBack').addEventListener('click', () => {
    try { localStorage.removeItem('my_ai_active_project'); } catch { }
    showScreen('screen-home');
    Home.loadProjects();
  });

  // Clear chat button
  document.getElementById('btnClearChat')?.addEventListener('click', () => Workspace.clearChat());

  // Step 3B: Project understanding toggle
  document.getElementById('btnTogglePubDetails')?.addEventListener('click', () => {
    const details = document.getElementById('projectUnderstandingDetails');
    const btn = document.getElementById('btnTogglePubDetails');
    if (!details) return;
    const isHidden = details.style.display === 'none';
    details.style.display = isHidden ? 'block' : 'none';
    if (btn) btn.textContent = isHidden ? 'Hide Details ▴' : 'View Project Understanding ▾';
  });

  // Files panel
  document.getElementById('btnToggleFiles').addEventListener('click', () => Workspace.toggleFilesPanel());
  document.getElementById('btnCloseFiles').addEventListener('click', () => {
    if (Workspace.filesOpen) Workspace.toggleFilesPanel();
  });

  // File modal
  document.getElementById('btnFileModalClose').addEventListener('click', () => closeModal('fileModalOverlay'));
  document.getElementById('fileModalOverlay').addEventListener('click', e => {
    if (e.target.id === 'fileModalOverlay') closeModal('fileModalOverlay');
  });

  // Agent input
  const agentInput = document.getElementById('agentInput');

  agentInput.addEventListener('input', () => {
    autoResizeTextarea(agentInput);
    document.getElementById('btnSend').disabled = !agentInput.value.trim() || Workspace.agentBusy;
  });

  agentInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const msg = agentInput.value.trim();
      if (msg && !Workspace.agentBusy) Workspace.send(msg);
    }
  });

  document.getElementById('btnSend').addEventListener('click', () => {
    const msg = agentInput.value.trim();
    if (msg && !Workspace.agentBusy) Workspace.send(msg);
  });

  // Init modals
  CreateProject.init();
  OpenFolder.init();

  // Model status indicator (Step 5U)
  async function updateModelIndicator() {
    try {
      const info = await API.get('/api/model/info');
      const dot = document.getElementById('modelStatusDot');
      const label = document.getElementById('modelStatusLabel');
      if (!dot || !label) return;

      const isAvailable = info.health && info.health.status === 'AVAILABLE';
      const isLocal = info.provider === 'local';

      if (isLocal) {
        if (isAvailable && info.active_model_id === 'local') {
          const shortName = (info.configured_name || 'Local').split('/').pop();
          label.textContent = `Model: Local • ${shortName}`;
          dot.style.background = '#10b981';
        } else if (info.fallback === 'deterministic' && info.active_model_id === 'deterministic') {
          label.textContent = 'Model: Local (Fallback: Deterministic)';
          dot.style.background = '#f59e0b';
        } else {
          label.textContent = 'Local model unavailable';
          dot.style.background = '#ef4444';
        }
      } else {
        label.textContent = 'Model: Deterministic';
        dot.style.background = '#10b981';
      }
    } catch (err) {
      console.warn('Failed to fetch model info:', err);
    }
  }
  updateModelIndicator();

  // Check if user was previously in an active project (restore on page refresh)
  const savedActiveProj = localStorage.getItem('my_ai_active_project');
  if (savedActiveProj) {
    try {
      const proj = JSON.parse(savedActiveProj);
      if (proj && proj.id) {
        await Workspace.open(proj);
        return;
      }
    } catch { }
  }

  // Otherwise start on Home screen
  showScreen('screen-home');
  await Home.init();
});

