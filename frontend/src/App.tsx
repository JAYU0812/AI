import React, { useState, useEffect, useRef } from 'react';
import type { Project, WorkspaceFile, ChatMessage, AgentTurnResponse } from './types';

export const App: React.FC = () => {
  // Navigation View: 'projects' | 'new_project' | 'workspace'
  const [currentView, setCurrentView] = useState<'projects' | 'new_project' | 'workspace'>('projects');

  // Projects State
  const [projects, setProjects] = useState<Project[]>([]);
  const [activeProject, setActiveProject] = useState<Project | null>(null);

  // New Project Form State
  const [newProjectName, setNewProjectName] = useState<string>('My Web App');
  const [newProjectLocation, setNewProjectLocation] = useState<string>('C:/jay/Projects');
  const [newProjectTemplate, setNewProjectTemplate] = useState<string>('web');
  const [isCreatingProject, setIsCreatingProject] = useState<boolean>(false);

  // Workspace & Editor State
  const [files, setFiles] = useState<WorkspaceFile[]>([]);
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [editorContent, setEditorContent] = useState<string>('');
  const [isUnsaved, setIsUnsaved] = useState<boolean>(false);

  // Chat & Agent State
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [promptInput, setPromptInput] = useState<string>('');
  const [isThinking, setIsThinking] = useState<boolean>(false);
  const [agentActivity, setAgentActivity] = useState<AgentTurnResponse | null>(null);

  // Terminal State
  const [terminalLines, setTerminalLines] = useState<string[]>([]);
  const [terminalCmdInput, setTerminalCmdInput] = useState<string>('');

  const codeTextareaRef = useRef<HTMLTextAreaElement>(null);
  const chatBottomRef = useRef<HTMLDivElement>(null);
  const terminalBottomRef = useRef<HTMLDivElement>(null);

  // Initial load
  useEffect(() => {
    fetchProjects();
  }, []);

  // Auto-scroll chat & terminal
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages]);

  useEffect(() => {
    terminalBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [terminalLines]);

  // Fetch tracked projects
  const fetchProjects = async () => {
    try {
      const res = await fetch('/api/projects');
      if (res.ok) {
        const data = await res.json();
        setProjects(data.projects || []);
      }
    } catch (err) {
      console.warn('Failed to fetch projects:', err);
    }
  };

  // Open a project
  const handleOpenProject = async (project: Project) => {
    try {
      const res = await fetch('/api/projects/open', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: project.id, path: project.path }),
      });
      if (res.ok) {
        const data = await res.json();
        setActiveProject(data.project);
        setCurrentView('workspace');
        setTerminalLines([
          `Project: ${data.project.name}`,
          `Path: ${data.project.path}`,
          `Agent Engine: Python 3.12 (Active)`,
          'Ready.',
        ]);
        setChatMessages([
          {
            id: 'init',
            sender: 'agent',
            text: `Opened project '${data.project.name}'. Ask me to inspect, build, or modify any files in this project.`,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          },
        ]);
        await fetchWorkspaceFiles(true);
      }
    } catch (err) {
      alert(`Error opening project: ${err}`);
    }
  };

  // Create a new project
  const handleCreateProject = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newProjectName.trim()) return;

    setIsCreatingProject(true);
    try {
      const res = await fetch('/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newProjectName.trim(),
          location: newProjectLocation.trim(),
          template: newProjectTemplate,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setActiveProject(data.project);
        setCurrentView('workspace');
        setTerminalLines([
          `Created project '${data.project.name}' at ${data.project.path}`,
          `Template: ${newProjectTemplate}`,
          'Agent Ready.',
        ]);
        setChatMessages([
          {
            id: 'created',
            sender: 'agent',
            text: `Created project '${data.project.name}'. What would you like me to build first?`,
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          },
        ]);
        await fetchProjects();
        await fetchWorkspaceFiles(true);
      } else {
        alert('Failed to create project.');
      }
    } catch (err) {
      alert(`Error: ${err}`);
    } finally {
      setIsCreatingProject(false);
    }
  };

  // Fetch files in active project
  const fetchWorkspaceFiles = async (autoOpen = false) => {
    try {
      const res = await fetch('/api/files');
      if (res.ok) {
        const data = await res.json();
        const fileList: WorkspaceFile[] = data.files || [];
        setFiles(fileList);

        if (autoOpen && fileList.length > 0) {
          const preferred = fileList.find(f => f.name === 'index.html') ||
                            fileList.find(f => f.name === 'App.tsx') ||
                            fileList.find(f => f.name === 'main.py') ||
                            fileList[0];
          openFile(preferred.path);
        } else if (fileList.length === 0) {
          setActiveFile(null);
          setEditorContent('');
        }
      }
    } catch (err) {
      console.warn('Failed to fetch workspace files:', err);
    }
  };

  // Open file in Editor
  const openFile = async (filePath: string) => {
    try {
      const res = await fetch(`/api/files/${encodeURIComponent(filePath)}`);
      if (res.ok) {
        const data = await res.json();
        setActiveFile(filePath);
        setEditorContent(data.content);
        setIsUnsaved(false);
      }
    } catch (err) {
      appendTerminal(`Error opening ${filePath}: ${err}`);
    }
  };

  // Save active file
  const saveActiveFile = async () => {
    if (!activeFile) return;
    try {
      const res = await fetch(`/api/files/${encodeURIComponent(activeFile)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: editorContent }),
      });
      if (res.ok) {
        setIsUnsaved(false);
        appendTerminal(`[Saved] ${activeFile}`);
        fetchWorkspaceFiles();
      }
    } catch (err) {
      appendTerminal(`Failed to save: ${err}`);
    }
  };

  // Run active file
  const runActiveFile = async () => {
    if (!activeFile) {
      alert('No active file to run.');
      return;
    }
    await saveActiveFile();
    const cmd = activeFile.endsWith('.py') ? `python ${activeFile}` : `npx vite build`;
    appendTerminal(`\n$ ${cmd}`);

    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: cmd }),
      });
      if (res.ok) {
        const data = await res.json();
        appendTerminal(data.output);
        if (data.error) appendTerminal(`[Error] ${data.error}`);
        appendTerminal(`Process finished with exit code ${data.metadata?.returncode ?? 0}`);
      }
    } catch (err) {
      appendTerminal(`Execution failed: ${err}`);
    }
  };

  const appendTerminal = (text: string) => {
    setTerminalLines(prev => [...prev, text]);
  };

  // Handle Terminal CLI command execution
  const executeTerminalCommand = async () => {
    const cmd = terminalCmdInput.trim();
    if (!cmd) return;
    setTerminalCmdInput('');
    appendTerminal(`\n$ ${cmd}`);

    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: cmd }),
      });
      if (res.ok) {
        const data = await res.json();
        appendTerminal(data.output);
        if (data.error) appendTerminal(`[Error] ${data.error}`);
        appendTerminal(`Process exit code ${data.metadata?.returncode ?? 0}`);
        fetchWorkspaceFiles();
      }
    } catch (err) {
      appendTerminal(`Command failed: ${err}`);
    }
  };

  // Send prompt to Agent
  const handleSendPrompt = async () => {
    const text = promptInput.trim();
    if (!text || isThinking) return;

    setPromptInput('');
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      sender: 'user',
      text: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };
    setChatMessages(prev => [...prev, userMsg]);
    setIsThinking(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });

      if (res.ok) {
        const data: AgentTurnResponse = await res.json();
        setAgentActivity(data);

        const agentMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          sender: 'agent',
          text: data.response,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        };
        setChatMessages(prev => [...prev, agentMsg]);

        if (data.terminal_output) {
          appendTerminal(`\n$ [Agent Executed Task]`);
          appendTerminal(data.terminal_output);
        }

        if (data.active_file) {
          await fetchWorkspaceFiles();
          await openFile(data.active_file);
        }
      }
    } catch (err) {
      setChatMessages(prev => [
        ...prev,
        {
          id: Date.now().toString(),
          sender: 'agent',
          text: `Error executing task: ${err}`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    } finally {
      setIsThinking(false);
    }
  };

  // Tab key & Ctrl+S in editor
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 's') {
      e.preventDefault();
      saveActiveFile();
    } else if (e.key === 'Tab') {
      e.preventDefault();
      const textarea = codeTextareaRef.current;
      if (!textarea) return;
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const val = textarea.value;
      textarea.value = val.substring(0, start) + '    ' + val.substring(end);
      textarea.selectionStart = textarea.selectionEnd = start + 4;
      setEditorContent(textarea.value);
      setIsUnsaved(true);
    }
  };

  // Compute Line Numbers
  const lineCount = editorContent ? editorContent.split('\n').length : 1;
  const lineNumbers = Array.from({ length: lineCount }, (_, i) => i + 1).join('\n');

  // ============================================================================
  // RENDER VIEW 1: PROJECTS HOME
  // ============================================================================
  if (currentView === 'projects') {
    return (
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <header className="app-header">
          <div className="brand-section">
            <span className="brand-name">
              <span className="brand-dot"></span> MY AI
            </span>
          </div>
          <div className="header-tools">
            <button className="btn-secondary" onClick={() => fetchProjects()}>
              🔄 Refresh
            </button>
            <button className="btn-icon" title="Settings">⚙</button>
          </div>
        </header>

        <div className="projects-container">
          <div className="projects-inner">
            <div className="projects-title-row">
              <h1 className="projects-title">MY PROJECTS</h1>
              <p className="projects-subtitle">Create a project or select an existing one to start building.</p>
            </div>

            <div className="projects-grid">
              {/* + New Project Card */}
              <div className="new-project-card" onClick={() => setCurrentView('new_project')}>
                <span className="new-project-icon">+</span>
                <span className="new-project-label">New Project</span>
              </div>

              {/* Real projects list */}
              {projects.map(p => (
                <div key={p.id} className="project-card" onClick={() => handleOpenProject(p)}>
                  <div className="card-top">
                    <div>
                      <div className="project-card-name">{p.name}</div>
                      <div className="project-card-path">{p.path}</div>
                    </div>
                  </div>
                  <div className="card-bottom">
                    <span>{p.file_count ?? 0} files</span>
                    <button className="btn-primary" style={{ padding: '4px 10px', fontSize: '11px' }}>
                      Open
                    </button>
                  </div>
                </div>
              ))}

              {projects.length === 0 && (
                <div className="no-projects-box">
                  No existing projects found in <code>C:\jay\Projects</code>. Click <strong>+ New Project</strong> to create your first real project.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ============================================================================
  // RENDER VIEW 2: CREATE NEW PROJECT
  // ============================================================================
  if (currentView === 'new_project') {
    return (
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <header className="app-header">
          <div className="brand-section">
            <button className="btn-secondary" onClick={() => setCurrentView('projects')}>
              ← Projects
            </button>
            <span className="brand-name">Create New Project</span>
          </div>
        </header>

        <div className="create-container">
          <form className="create-form-card" onSubmit={handleCreateProject}>
            <div className="form-header">
              <h2 className="form-title">Configure Your Project</h2>
            </div>

            <div className="form-group">
              <label className="form-label">Project Name</label>
              <input
                type="text"
                className="form-input"
                value={newProjectName}
                onChange={e => setNewProjectName(e.target.value)}
                placeholder="e.g. My Web App"
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label">Location</label>
              <input
                type="text"
                className="form-input"
                value={newProjectLocation}
                onChange={e => setNewProjectLocation(e.target.value)}
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label">Starter Template</label>
              <div className="template-grid">
                <div
                  className={`template-option ${newProjectTemplate === 'empty' ? 'selected' : ''}`}
                  onClick={() => setNewProjectTemplate('empty')}
                >
                  <strong>Blank</strong>
                  <p style={{ margin: '4px 0 0 0', fontSize: '11px', color: '#64748b' }}>Empty folder</p>
                </div>
                <div
                  className={`template-option ${newProjectTemplate === 'web' ? 'selected' : ''}`}
                  onClick={() => setNewProjectTemplate('web')}
                >
                  <strong>Web App</strong>
                  <p style={{ margin: '4px 0 0 0', fontSize: '11px', color: '#64748b' }}>HTML / CSS / JS</p>
                </div>
                <div
                  className={`template-option ${newProjectTemplate === 'python' ? 'selected' : ''}`}
                  onClick={() => setNewProjectTemplate('python')}
                >
                  <strong>Python</strong>
                  <p style={{ margin: '4px 0 0 0', fontSize: '11px', color: '#64748b' }}>Python Script</p>
                </div>
              </div>
            </div>

            <div className="form-actions">
              <button type="button" className="btn-secondary" onClick={() => setCurrentView('projects')}>
                Cancel
              </button>
              <button type="submit" className="btn-primary" disabled={isCreatingProject}>
                {isCreatingProject ? 'Creating...' : 'Create Project'}
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  }

  // ============================================================================
  // RENDER VIEW 3: ACTUAL AI CODING WORKSPACE
  // ============================================================================
  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Workspace Header */}
      <header className="app-header">
        <div className="brand-section">
          <button className="btn-secondary" onClick={() => setCurrentView('projects')} title="Back to Projects Home">
            ← Projects
          </button>
          <span className="brand-name">{activeProject?.name || 'Workspace'}</span>
          <span className="path-pill">{activeProject?.path}</span>
          <span className="status-pill">● Agent Ready</span>
        </div>

        <div className="header-tools">
          <button className="btn-secondary" onClick={() => fetchWorkspaceFiles()}>
            🔄 Refresh
          </button>
          <button className="btn-primary" onClick={runActiveFile}>
            ▶ Run File
          </button>
          <button className="btn-icon" title="Settings">⚙</button>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <div className="workspace-layout">
        {/* Top 3 Columns: Explorer, Editor, Chat */}
        <div className="workspace-grid">
          {/* Column 1: EXPLORER */}
          <div className="workspace-column">
            <div className="column-header">
              <span>EXPLORER</span>
              <button
                className="btn-icon"
                onClick={async () => {
                  const name = prompt('New file name (e.g. index.html):');
                  if (name) {
                    await fetch(`/api/files/${encodeURIComponent(name)}`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({ content: '' }),
                    });
                    await fetchWorkspaceFiles();
                    await openFile(name);
                  }
                }}
              >
                + File
              </button>
            </div>

            <div className="file-list">
              {files.map(f => (
                <div
                  key={f.path}
                  className={`file-row ${activeFile === f.path ? 'active' : ''}`}
                  onClick={() => openFile(f.path)}
                >
                  <div className="file-row-left">
                    <span>
                      {f.name.endsWith('.py')
                        ? '🐍'
                        : f.name.endsWith('.html')
                        ? '🌐'
                        : f.name.endsWith('.css')
                        ? '🎨'
                        : f.name.endsWith('.tsx') || f.name.endsWith('.ts')
                        ? '⚛️'
                        : '📄'}
                    </span>
                    <span>{f.name}</span>
                  </div>
                  <span style={{ fontSize: '10px', color: '#94a3b8' }}>{f.size_kb}k</span>
                </div>
              ))}

              {files.length === 0 && (
                <div className="empty-explorer">
                  Project directory is empty.<br />
                  <small style={{ color: '#94a3b8' }}>Create a file or ask your AI agent to build one.</small>
                </div>
              )}
            </div>
          </div>

          {/* Column 2: EDITOR */}
          <div className="workspace-column">
            <div className="editor-container">
              <div className="editor-tab-bar">
                <div className="tab-filename">
                  <span>📄 {activeFile || 'No file selected'}</span>
                  {isUnsaved && <span className="unsaved-badge" title="Unsaved changes"></span>}
                </div>
                <div>
                  <button className="btn-secondary" onClick={saveActiveFile} title="Save (Ctrl+S)">
                    💾 Save
                  </button>
                </div>
              </div>

              <div className="editor-area">
                <div className="gutter">
                  <pre style={{ margin: 0, fontFamily: 'inherit' }}>{lineNumbers}</pre>
                </div>
                <textarea
                  ref={codeTextareaRef}
                  className="editor-textarea"
                  spellCheck={false}
                  value={editorContent}
                  onChange={e => {
                    setEditorContent(e.target.value);
                    setIsUnsaved(true);
                  }}
                  onKeyDown={handleKeyDown}
                  placeholder="// Real file content appears here. Select a file or ask the agent to build..."
                />
              </div>
            </div>
          </div>

          {/* Column 3: AI CHAT */}
          <div className="workspace-column">
            <div className="column-header">
              <span>AI CHAT</span>
              <span style={{ fontSize: '10px', color: isThinking ? '#2563eb' : '#64748b' }}>
                {isThinking ? 'AGENT WORKING...' : 'IDLE'}
              </span>
            </div>

            <div className="chat-stream">
              {chatMessages.map(msg => (
                <div key={msg.id} className={`msg-row ${msg.sender === 'user' ? 'msg-user' : ''}`}>
                  <div className="msg-avatar">{msg.sender === 'user' ? '👤' : '🤖'}</div>
                  <div className="msg-bubble">
                    <p style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{msg.text}</p>
                  </div>
                </div>
              ))}

              {/* Real Agent Activity Summary Card */}
              {agentActivity && (
                <div className="activity-summary-card">
                  <div style={{ fontWeight: 600, color: '#0f172a', marginBottom: '6px' }}>
                    Agent Activity ({agentActivity.intent.name})
                  </div>
                  {agentActivity.executed_steps.map((s, idx) => (
                    <div key={idx} className="activity-step-item">
                      <span>✓ {s.tool_name}: {s.description}</span>
                    </div>
                  ))}
                  {agentActivity.verification && (
                    <div style={{ marginTop: '4px', color: agentActivity.verification.passed ? '#16a34a' : '#d97706' }}>
                      {agentActivity.verification.passed ? '✓' : '⚠'} {agentActivity.verification.diagnostics}
                    </div>
                  )}
                </div>
              )}

              <div ref={chatBottomRef} />
            </div>
          </div>
        </div>

        {/* Lower Section: TERMINAL */}
        <div className="terminal-drawer">
          <div className="terminal-topbar">
            <span>TERMINAL ({activeProject?.name})</span>
            <span style={{ color: '#4ade80' }}>● Subprocess Ready</span>
          </div>

          <div className="terminal-output">
            {terminalLines.map((line, idx) => (
              <div key={idx}>{line}</div>
            ))}
            <div ref={terminalBottomRef} />
          </div>

          <div style={{ background: '#0b0f19', borderTop: '1px solid #1e293b', display: 'flex', alignItems: 'center', padding: '0 12px', height: '32px' }}>
            <span style={{ color: '#4ade80', fontFamily: 'monospace', fontWeight: 700, marginRight: '8px' }}>$</span>
            <input
              type="text"
              value={terminalCmdInput}
              onChange={e => setTerminalCmdInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') executeTerminalCommand();
              }}
              placeholder="Run command in this project directory (e.g. dir, npm install, python file.py)..."
              style={{ flex: 1, background: 'transparent', border: 'none', outline: 'none', color: '#f8fafc', fontFamily: 'monospace', fontSize: '11px' }}
            />
          </div>
        </div>

        {/* Bottom Bar: Real Prompt Input */}
        <div className="prompt-bar">
          <input
            type="text"
            className="prompt-field"
            value={promptInput}
            onChange={e => setPromptInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter') handleSendPrompt();
            }}
            placeholder="Ask your AI agent to build, modify or debug..."
            disabled={isThinking}
          />
          <button
            className="btn-primary"
            onClick={handleSendPrompt}
            disabled={isThinking}
            style={{ padding: '9px 18px', fontSize: '13px' }}
          >
            {isThinking ? 'Thinking...' : '➤'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default App;
