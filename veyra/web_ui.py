"""
VEYRA ChatGPT-4 Web Studio
Local Web Interface styled after ChatGPT-4 for interacting with Veyra-LM.
Zero external dependencies (uses standard library http.server + SSE streaming).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from veyra.app import get_active_engine
from veyra.core.cognitive import CognitiveCore
from veyra.core.state import CognitiveState
from veyra.memory.database import MemoryDB


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ChatGPT-4 | Veyra AI</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-sidebar: #171717;
            --bg-main: #212121;
            --bg-surface: #2f2f2f;
            --bg-input: #2f2f2f;
            --border-subtle: #383838;
            --border-highlight: #4a4a4a;
            --text-primary: #ececec;
            --text-secondary: #b4b4b4;
            --text-muted: #8e8e8e;
            --accent-green: #10a37f;
            --accent-green-hover: #1a7f64;
            --user-bubble: #2f2f2f;
            --code-bg: #1e1e1e;
            --radius-bubble: 20px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-main);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            overflow: hidden;
        }

        /* Sidebar */
        aside {
            width: 260px;
            background-color: var(--bg-sidebar);
            display: flex;
            flex-direction: column;
            border-right: 1px solid var(--border-subtle);
            flex-shrink: 0;
            padding: 12px;
            transition: transform 0.2s ease;
        }

        .sidebar-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }

        .btn-new-chat {
            display: flex;
            align-items: center;
            gap: 10px;
            width: 100%;
            background: transparent;
            color: var(--text-primary);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: background-color 0.15s, border-color 0.15s;
        }

        .btn-new-chat:hover {
            background-color: var(--bg-surface);
            border-color: var(--border-highlight);
        }

        .chat-history {
            flex: 1;
            overflow-y: auto;
            margin-top: 14px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .history-label {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            padding: 8px 8px 4px 8px;
        }

        .history-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 9px 12px;
            border-radius: 8px;
            font-size: 13.5px;
            color: var(--text-secondary);
            cursor: pointer;
            text-decoration: none;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            transition: background 0.15s, color 0.15s;
        }

        .history-item:hover, .history-item.active {
            background-color: var(--bg-surface);
            color: var(--text-primary);
        }

        .sidebar-footer {
            border-top: 1px solid var(--border-subtle);
            padding-top: 12px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .model-badge {
            background-color: rgba(16, 163, 127, 0.15);
            border: 1px solid rgba(16, 163, 127, 0.3);
            border-radius: 8px;
            padding: 8px 10px;
            font-size: 12px;
        }

        .model-badge .name {
            font-weight: 600;
            color: #38ef7d;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .model-badge .detail {
            color: var(--text-muted);
            font-size: 11px;
            margin-top: 2px;
        }

        /* Main Container */
        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100vh;
            position: relative;
            background-color: var(--bg-main);
        }

        /* Top Bar */
        header {
            height: 54px;
            border-bottom: 1px solid var(--border-subtle);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            flex-shrink: 0;
        }

        .model-selector {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 16px;
            font-weight: 600;
            color: var(--text-primary);
            cursor: pointer;
            padding: 6px 10px;
            border-radius: 8px;
            transition: background 0.15s;
        }

        .model-selector:hover {
            background-color: var(--bg-surface);
        }

        .tag-pill {
            font-size: 11px;
            background: #2a2a2a;
            color: #10a37f;
            border: 1px solid #3c3c3c;
            border-radius: 12px;
            padding: 2px 8px;
            font-weight: 500;
        }

        /* Chat Scroll Area */
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .chat-wrapper {
            width: 100%;
            max-width: 800px;
            display: flex;
            flex-direction: column;
            gap: 22px;
        }

        /* Welcome Screen */
        .welcome-view {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 55vh;
            text-align: center;
            gap: 24px;
        }

        .welcome-logo {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #10a37f 0%, #11998e 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 20px rgba(16, 163, 127, 0.35);
        }

        .welcome-logo svg {
            width: 32px;
            height: 32px;
            fill: white;
        }

        .welcome-title {
            font-size: 26px;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.4px;
        }

        .suggestion-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            width: 100%;
            max-width: 720px;
        }

        .suggestion-card {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 14px 16px;
            text-align: left;
            cursor: pointer;
            transition: all 0.15s ease;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .suggestion-card:hover {
            border-color: var(--border-highlight);
            transform: translateY(-2px);
            background-color: #383838;
        }

        .suggestion-card .card-title {
            font-size: 13.5px;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .suggestion-card .card-desc {
            font-size: 12.5px;
            color: var(--text-muted);
            line-height: 1.4;
        }

        /* Message Bubbles */
        .message-row {
            display: flex;
            gap: 16px;
            width: 100%;
        }

        .message-row.user {
            justify-content: flex-end;
        }

        .user-bubble {
            background-color: var(--user-bubble);
            color: var(--text-primary);
            border-radius: var(--radius-bubble);
            padding: 12px 18px;
            max-width: 80%;
            font-size: 15px;
            line-height: 1.5;
            word-break: break-word;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        }

        .bot-avatar {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            background: linear-gradient(135deg, #10a37f 0%, #11998e 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            box-shadow: 0 2px 8px rgba(16, 163, 127, 0.25);
        }

        .bot-avatar svg {
            width: 18px;
            height: 18px;
            fill: white;
        }

        .bot-content {
            flex: 1;
            font-size: 15px;
            line-height: 1.6;
            color: var(--text-primary);
            word-break: break-word;
            padding-top: 4px;
        }

        /* Markdown & Code Blocks */
        .bot-content p {
            margin-bottom: 12px;
        }

        .bot-content p:last-child {
            margin-bottom: 0;
        }

        .bot-content h1, .bot-content h2, .bot-content h3 {
            margin: 16px 0 8px 0;
            color: #fff;
            font-weight: 600;
        }

        .bot-content ul, .bot-content ol {
            margin: 8px 0 12px 24px;
        }

        .bot-content li {
            margin-bottom: 4px;
        }

        .bot-content code:not(pre code) {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', Consolas, monospace;
            font-size: 13.5px;
            color: #ff9d76;
        }

        .code-container {
            background-color: var(--code-bg);
            border-radius: 10px;
            margin: 14px 0;
            border: 1px solid #333;
            overflow: hidden;
        }

        .code-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 14px;
            background-color: #262626;
            border-bottom: 1px solid #333;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: var(--text-muted);
        }

        .copy-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
            font-size: 12px;
            padding: 2px 6px;
            border-radius: 4px;
            transition: color 0.15s;
        }

        .copy-btn:hover {
            color: #fff;
        }

        pre {
            padding: 14px 16px;
            overflow-x: auto;
            font-family: 'JetBrains Mono', Consolas, monospace;
            font-size: 13.5px;
            line-height: 1.5;
            color: #e6e6e6;
        }

        /* Blinking Typewriter Cursor */
        .cursor-blink {
            display: inline-block;
            width: 7px;
            height: 16px;
            background-color: var(--text-primary);
            margin-left: 3px;
            vertical-align: middle;
            animation: blink 0.8s infinite;
        }

        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }

        /* Input Dock */
        .dock {
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 12px 20px 16px 20px;
            flex-shrink: 0;
            background: linear-gradient(180deg, transparent 0%, var(--bg-main) 30%);
        }

        .input-pill {
            width: 100%;
            max-width: 800px;
            background-color: var(--bg-input);
            border: 1px solid var(--border-subtle);
            border-radius: 26px;
            display: flex;
            align-items: flex-end;
            padding: 10px 16px;
            gap: 12px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
            transition: border-color 0.15s, box-shadow 0.15s;
        }

        .input-pill:focus-within {
            border-color: var(--border-highlight);
            box-shadow: 0 6px 20px rgba(0,0,0,0.35);
        }

        textarea {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 15px;
            line-height: 1.45;
            resize: none;
            max-height: 160px;
            min-height: 24px;
            padding: 2px 0;
        }

        textarea::placeholder {
            color: var(--text-muted);
        }

        .send-btn {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            background-color: #676767;
            border: none;
            color: #171717;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: not-allowed;
            flex-shrink: 0;
            transition: background-color 0.15s, transform 0.1s;
        }

        .send-btn.active {
            background-color: #ffffff;
            cursor: pointer;
        }

        .send-btn.active:hover {
            transform: scale(1.04);
        }

        .send-btn svg {
            width: 18px;
            height: 18px;
            fill: currentColor;
        }

        .disclaimer {
            font-size: 11.5px;
            color: var(--text-muted);
            margin-top: 8px;
            text-align: center;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: #3e3e3e;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #4e4e4e;
        }
    </style>
</head>
<body>

    <!-- Sidebar -->
    <aside id="sidebar">
        <div class="sidebar-header">
            <button class="btn-new-chat" onclick="newChat()">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
                New chat
            </button>
        </div>

        <div class="chat-history">
            <div class="history-label">Today</div>
            <div class="history-item active">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
                Current Conversation
            </div>
            <div class="history-label" style="margin-top:12px;">Saved Capabilities</div>
            <div class="history-item" onclick="sendSuggestion('Explain the themes of ambition and guilt in Shakespeare\'s Macbeth')">
                📖 Literature & Essays
            </div>
            <div class="history-item" onclick="sendSuggestion('Write a Python function to find prime numbers using Sieve of Eratosthenes')">
                💻 Python Programming
            </div>
            <div class="history-item" onclick="sendSuggestion('Compose a lyrical poem about quantum entanglement and cosmic wonder')">
                🎭 Lyrical Poetry
            </div>
            <div class="history-item" onclick="sendSuggestion('Explain the Second Law of Thermodynamics and cosmic entropy')">
                ⚛️ Physics & Thermodynamics
            </div>
            <div class="history-item" onclick="sendSuggestion('A ball is dropped from 80 meters. Taking g=9.8, calculate time and impact velocity step-by-step')">
                📐 Mathematics & Kinematics
            </div>
        </div>

        <div class="sidebar-footer">
            <div class="model-badge">
                <div class="name">
                    <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#10a37f;"></span>
                    <span id="badge-model-name">Veyra GPT-4</span>
                </div>
                <div class="detail" id="badge-model-detail">Neural Multi-Domain (125M)</div>
            </div>
        </div>
    </aside>

    <!-- Main Chat Window -->
    <main>
        <!-- Header -->
        <header>
            <div class="model-selector">
                <span>Veyra GPT-4</span>
                <span class="tag-pill" id="header-tag">125M Multi-Domain</span>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"></polyline></svg>
            </div>
            <div style="font-size:12px; color:var(--text-muted); display:flex; align-items:center; gap:6px;">
                <span style="width:7px; height:7px; border-radius:50%; background:#10a37f;"></span> Local Inference Active
            </div>
        </header>

        <!-- Messages Area -->
        <div class="chat-container" id="scroll-container">
            <div class="chat-wrapper" id="messages-wrapper">

                <!-- Welcome State -->
                <div class="welcome-view" id="welcome-view">
                    <div class="welcome-logo">
                        <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path></svg>
                    </div>
                    <div class="welcome-title">What can I help you explore today?</div>

                    <div class="suggestion-grid">
                        <div class="suggestion-card" onclick="sendSuggestion('Write a critical analysis of the themes of ambition and guilt in Shakespeare\'s Macbeth.')">
                            <div class="card-title">📖 Literature & Essays</div>
                            <div class="card-desc">Analyze ambition and guilt in Shakespeare's Macbeth</div>
                        </div>

                        <div class="suggestion-card" onclick="sendSuggestion('Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes.')">
                            <div class="card-title">💻 Python Programming</div>
                            <div class="card-desc">Sieve of Eratosthenes algorithm with clean typing</div>
                        </div>

                        <div class="suggestion-card" onclick="sendSuggestion('Explain the Second Law of Thermodynamics and how it relates to cosmic entropy.')">
                            <div class="card-title">⚛️ Physics Expertise</div>
                            <div class="card-desc">Entropy, thermal equilibrium and cosmic heat death</div>
                        </div>

                        <div class="suggestion-card" onclick="sendSuggestion('A ball is dropped from a height of 80 meters. Taking g = 9.8 m/s^2, calculate the time to hit the ground and its impact velocity step-by-step.')">
                            <div class="card-title">📐 Mathematics & Physics</div>
                            <div class="card-desc">Kinematics step-by-step calculation with units</div>
                        </div>
                    </div>
                </div>

            </div>
        </div>

        <!-- Input Capsule -->
        <div class="dock">
            <div class="input-pill">
                <textarea id="prompt-input" rows="1" placeholder="Message Veyra GPT-4..." oninput="autoGrow(this)" onkeydown="handleKeyDown(event)"></textarea>
                <button class="send-btn" id="send-btn" onclick="sendMessage()">
                    <svg viewBox="0 0 24 24"><line x1="12" y1="19" x2="12" y2="5" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"></line><polyline points="5 12 12 5 19 12" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"></polyline></svg>
                </button>
            </div>
            <div class="disclaimer">Veyra GPT-4 is an autonomous local model. Trained on literature, coding, science and mathematics.</div>
        </div>
    </main>

    <script>
        const input = document.getElementById('prompt-input');
        const sendBtn = document.getElementById('send-btn');
        const messagesWrapper = document.getElementById('messages-wrapper');
        const scrollContainer = document.getElementById('scroll-container');
        const welcomeView = document.getElementById('welcome-view');

        let isGenerating = false;
        let chatHistory = [];

        function autoGrow(el) {
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 160) + 'px';
            if (el.value.trim().length > 0) {
                sendBtn.classList.add('active');
            } else {
                sendBtn.classList.remove('active');
            }
        }

        function handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        function scrollToBottom() {
            scrollContainer.scrollTop = scrollContainer.scrollHeight;
        }

        function newChat() {
            if (isGenerating) return;
            chatHistory = [];
            messagesWrapper.innerHTML = '';
            messagesWrapper.appendChild(welcomeView);
            input.value = '';
            autoGrow(input);
            fetch('/api/clear', { method: 'POST' });
        }

        function sendSuggestion(text) {
            input.value = text;
            autoGrow(input);
            sendMessage();
        }

        // Lightweight Markdown + Code Parser
        function formatMarkdown(text) {
            // Escape HTML tags to prevent XSS
            let escaped = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

            // Format code blocks ```lang ... ```
            escaped = escaped.replace(/```([a-zA-Z0-9_\-]+)?\n([\s\S]*?)```/g, function(match, lang, code) {
                const language = lang || "code";
                return `
                <div class="code-container">
                    <div class="code-header">
                        <span>${language}</span>
                        <button class="copy-btn" onclick="copyCode(this)">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                            Copy code
                        </button>
                    </div>
                    <pre><code>${code.trim()}</code></pre>
                </div>`;
            });

            // Inline code `code`
            escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');

            // Bold **text**
            escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

            // Bullet points
            escaped = escaped.replace(/^\s*[\-\*]\s+(.+)$/gm, '<li>$1</li>');
            escaped = escaped.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

            // Line breaks
            escaped = escaped.replace(/\n\n/g, '<p></p>');
            escaped = escaped.replace(/\n/g, '<br>');

            return escaped;
        }

        function copyCode(btn) {
            const pre = btn.closest('.code-container').querySelector('pre code');
            navigator.clipboard.writeText(pre.innerText).then(() => {
                const orig = btn.innerHTML;
                btn.innerHTML = '✓ Copied!';
                setTimeout(() => { btn.innerHTML = orig; }, 2000);
            });
        }

        async function sendMessage() {
            const text = input.value.trim();
            if (!text || isGenerating) return;

            // Remove welcome view on first message
            if (welcomeView && welcomeView.parentNode) {
                welcomeView.parentNode.removeChild(welcomeView);
            }

            // Append User message
            const userRow = document.createElement('div');
            userRow.className = 'message-row user';
            userRow.innerHTML = `<div class="user-bubble">${text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")}</div>`;
            messagesWrapper.appendChild(userRow);

            input.value = '';
            autoGrow(input);
            scrollToBottom();

            // Append Assistant message container
            const botRow = document.createElement('div');
            botRow.className = 'message-row bot';
            botRow.innerHTML = `
                <div class="bot-avatar">
                    <svg viewBox="0 0 24 24"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"></path></svg>
                </div>
                <div class="bot-content" id="active-bot-message">
                    <span class="cursor-blink"></span>
                </div>
            `;
            messagesWrapper.appendChild(botRow);
            scrollToBottom();

            isGenerating = true;
            sendBtn.classList.remove('active');

            const contentEl = botRow.querySelector('.bot-content');
            let fullText = "";

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");
                let buffer = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.slice(6);
                            if (dataStr.trim() === '[DONE]') {
                                break;
                            }
                            try {
                                const payload = JSON.parse(dataStr);
                                if (payload.token) {
                                    fullText += payload.token;
                                    contentEl.innerHTML = formatMarkdown(fullText) + '<span class="cursor-blink"></span>';
                                    scrollToBottom();
                                }
                            } catch (e) {
                                // raw token fallback
                                fullText += dataStr;
                                contentEl.innerHTML = formatMarkdown(fullText) + '<span class="cursor-blink"></span>';
                                scrollToBottom();
                            }
                        }
                    }
                }

                // Final render without cursor
                contentEl.innerHTML = formatMarkdown(fullText);
            } catch (err) {
                contentEl.innerHTML = `<span style="color:#ff6b6b;">Error communicating with Veyra engine: ${err.message}</span>`;
            } finally {
                isGenerating = false;
                scrollToBottom();
                input.focus();
            }
        }

        // Fetch System & Model Status on load
        window.addEventListener('load', async () => {
            input.focus();
            try {
                const res = await fetch('/api/info');
                const data = await res.json();
                if (data.model_name) {
                    document.getElementById('header-tag').innerText = data.model_name;
                    document.getElementById('badge-model-name').innerText = "Veyra GPT-4";
                    document.getElementById('badge-model-detail').innerText = `${data.model_desc}`;
                }
            } catch (e) {}
        });
    </script>
</body>
</html>
"""


class VeyraHTTPHandler(BaseHTTPRequestHandler):
    core: CognitiveCore | None = None
    engine_desc: str = "Veyra-Instruct (125M)"
    model_desc: str = "checkpoints/veyra_instruct_125m/model.pt"

    def log_message(self, format: str, *args) -> None:
        # Suppress noisy standard HTTP logs, keep terminal clean
        return

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/api/info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            info = {
                "model_name": self.engine_desc,
                "model_desc": self.model_desc,
                "status": "online",
                "capabilities": ["Literature", "Python Coding", "Poetry", "Physics", "Mathematics"]
            }
            self.wfile.write(json.dumps(info).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/api/clear":
            if self.core:
                self.core.memory.clear_recent(50)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"cleared"}')
        elif self.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length).decode("utf-8")
            try:
                body = json.loads(post_data)
                user_message = body.get("message", "").strip()
            except Exception:
                user_message = ""

            if not user_message:
                self.send_response(400)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            stream_lock = threading.Lock()

            def on_token(piece: str) -> None:
                with stream_lock:
                    try:
                        chunk = f"data: {json.dumps({'token': piece})}\n\n"
                        self.wfile.write(chunk.encode("utf-8"))
                        self.wfile.flush()
                    except Exception:
                        pass

            if self.core:
                full_reply = self.core.handle(user_message, stream_callback=on_token)
            else:
                full_reply = "Veyra Engine is starting up. Please refresh."
                on_token(full_reply)

            # Signal completion
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass


def launch_server(port: int = 7860, open_browser: bool = True) -> None:
    root = Path.home() / ".veyra"
    root.mkdir(parents=True, exist_ok=True)
    memory = MemoryDB(root / "memory.db")
    state = CognitiveState()

    print("\n" + "=" * 60)
    print("       🧠 VEYRA CHATGPT-4 WEB STUDIO INITIALIZATION")
    print("=" * 60)
    print("Loading active neural checkpoints & tokenizer...")
    engine, engine_desc, model_desc = get_active_engine(memory)
    core = CognitiveCore(engine=engine, memory=memory, state=state)

    VeyraHTTPHandler.core = core
    VeyraHTTPHandler.engine_desc = engine_desc
    VeyraHTTPHandler.model_desc = model_desc

    print(f"✅ Active Engine: {engine_desc}")
    print(f"✅ Checkpoint   : {model_desc}")
    print(f"🚀 Server Port  : http://localhost:{port}")
    print("=" * 60 + "\n")

    if open_browser:
        def _open():
            time.sleep(1.0)
            import webbrowser
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open, daemon=True).start()

    server = ThreadingHTTPServer(("0.0.0.0", port), VeyraHTTPHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Veyra ChatGPT-4 Web Studio shutting down.")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Veyra ChatGPT-4 Web Studio")
    parser.add_argument("--port", type=int, default=7860, help="Web server port (default: 7860)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    launch_server(port=args.port, open_browser=not args.no_browser)
