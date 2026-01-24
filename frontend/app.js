// DataBrain AI Agent Frontend
const API_BASE = 'http://localhost:8000/api';

let currentDataset = null;
let chatHistory = [];
let datasets = [];
let llmProviders = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkServerStatus();
    loadDatasets();
    loadLLMProviders();
    loadCostTracking();
    
    // Auto-refresh cost tracking every 30 seconds
    setInterval(loadCostTracking, 30000);
    
    // Auto-refresh datasets every 10 seconds
    setInterval(loadDatasets, 10000);
});

function setupEventListeners() {
    // File upload
    const fileInput = document.getElementById('fileInput');
    fileInput.addEventListener('change', handleFileUpload);
    
    // Send button
    document.getElementById('sendBtn').addEventListener('click', sendQuery);
    
    // Enter key in textarea
    const queryInput = document.getElementById('queryInput');
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });
    
    // Auto-resize textarea
    queryInput.addEventListener('input', () => {
        queryInput.style.height = 'auto';
        queryInput.style.height = Math.min(queryInput.scrollHeight, 120) + 'px';
    });
    
    // Clear chat
    document.getElementById('clearChatBtn').addEventListener('click', clearChat);
    
    // Refresh button
    document.getElementById('refreshBtn').addEventListener('click', () => {
        loadDatasets();
        loadCostTracking();
        loadLLMProviders();
        showToast('Refreshed', 'success');
    });
    
    // LLM provider change
    document.getElementById('llmProvider').addEventListener('change', (e) => {
        const provider = e.target.value;
        document.getElementById('quickProviderSelect').value = provider;
    });
    
    document.getElementById('quickProviderSelect').addEventListener('change', (e) => {
        const provider = e.target.value;
        document.getElementById('llmProvider').value = provider;
    });
}

// Server Status Check
async function checkServerStatus() {
    try {
        const response = await fetch('http://localhost:8000/');
        if (response.ok) {
            updateStatusIndicator(true);
            updateAPIStatus(true);
        } else {
            updateStatusIndicator(false);
            updateAPIStatus(false);
        }
    } catch (error) {
        updateStatusIndicator(false);
        updateAPIStatus(false);
    }
}

function updateStatusIndicator(connected) {
    const indicator = document.getElementById('statusIndicator');
    const dot = indicator.querySelector('.status-dot');
    const text = indicator.querySelector('.status-text');
    
    if (connected) {
        dot.style.background = 'var(--success-color)';
        text.textContent = 'Connected';
    } else {
        dot.style.background = 'var(--error-color)';
        text.textContent = 'Disconnected';
    }
}

function updateAPIStatus(connected) {
    const badge = document.getElementById('apiStatusBadge');
    if (connected) {
        badge.textContent = 'Connected';
        badge.className = 'status-badge connected';
    } else {
        badge.textContent = 'Disconnected';
        badge.className = 'status-badge error';
    }
}

// File Upload
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    showLoadingOverlay('Uploading dataset...');
    
    try {
        const response = await fetch(`${API_BASE}/upload-dataset`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: 'Upload failed' }));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }
        
        const data = await response.json();
        currentDataset = data.dataset_name;
        
        updateCurrentDataset();
        loadDatasets();
        
        const info = data.info || {};
        const message = `✅ Dataset "${data.dataset_name}" loaded successfully!\n\n` +
                       `📊 ${info.row_count || 0} rows × ${info.column_count || 0} columns`;
        
        addMessage('assistant', message);
        showToast(`Dataset "${data.dataset_name}" uploaded successfully!`, 'success');
        hideLoadingOverlay();
        
    } catch (error) {
        hideLoadingOverlay();
        let errorMsg = error.message;
        
        if (error.message === 'Failed to fetch' || error.name === 'TypeError') {
            errorMsg = 'Cannot connect to server. Please make sure the backend server is running on http://localhost:8000';
        }
        
        addMessage('assistant', `❌ Upload Error: ${errorMsg}`, true);
        showToast('Upload failed: ' + errorMsg, 'error');
    }
    
    // Reset file input
    event.target.value = '';
}

// Load Datasets
async function loadDatasets() {
    try {
        const response = await fetch(`${API_BASE}/datasets`);
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        const data = await response.json();
        
        datasets = data.datasets || [];
        const datasetList = document.getElementById('datasetList');
        datasetList.innerHTML = '';
        
        if (datasets.length === 0) {
            datasetList.innerHTML = '<div class="empty-state">No datasets loaded</div>';
            return;
        }
        
        datasets.forEach(dataset => {
            const item = document.createElement('div');
            item.className = `dataset-item ${dataset.name === currentDataset ? 'active' : ''}`;
            item.innerHTML = `
                <h3>${escapeHtml(dataset.name)}</h3>
                <p>${dataset.row_count || 0} rows × ${dataset.column_count || 0} cols</p>
                <div class="dataset-meta">
                    <span>${(dataset.columns || []).length} columns</span>
                </div>
            `;
            item.addEventListener('click', () => selectDataset(dataset.name));
            datasetList.appendChild(item);
        });
        
    } catch (error) {
        if (error.message.includes('fetch')) {
            console.warn('Backend server not available');
        } else {
            console.error('Error loading datasets:', error);
        }
    }
}

function selectDataset(datasetName) {
    currentDataset = datasetName;
    updateCurrentDataset();
    loadDatasets();
    
    const dataset = datasets.find(d => d.name === datasetName);
    if (dataset) {
        addMessage('assistant', `📋 Switched to dataset: **${datasetName}**\n\n` +
                   `📊 ${dataset.row_count || 0} rows, ${dataset.column_count || 0} columns`);
    }
}

function updateCurrentDataset() {
    const titleEl = document.getElementById('currentDataset');
    const infoEl = document.getElementById('datasetInfo');
    
    if (currentDataset) {
        const dataset = datasets.find(d => d.name === currentDataset);
        titleEl.innerHTML = `<span class="dataset-icon">📋</span> ${escapeHtml(currentDataset)}`;
        
        if (dataset) {
            infoEl.textContent = `${dataset.row_count || 0} rows × ${dataset.column_count || 0} columns`;
        } else {
            infoEl.textContent = '';
        }
    } else {
        titleEl.innerHTML = '<span class="dataset-icon">📋</span> No dataset selected';
        infoEl.textContent = '';
    }
}

// Load LLM Providers
async function loadLLMProviders() {
    try {
        const response = await fetch(`${API_BASE}/llm-providers`);
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        const data = await response.json();
        
        llmProviders = data.providers || [];
        const select = document.getElementById('llmProvider');
        const quickSelect = document.getElementById('quickProviderSelect');
        
        // Clear and add Auto option
        select.innerHTML = '<option value="">Auto (Recommended)</option>';
        quickSelect.innerHTML = '<option value="">Auto</option>';
        
        llmProviders.forEach(provider => {
            const option = document.createElement('option');
            option.value = provider;
            option.textContent = provider.charAt(0).toUpperCase() + provider.slice(1);
            select.appendChild(option.cloneNode(true));
            quickSelect.appendChild(option);
        });
        
    } catch (error) {
        console.error('Error loading LLM providers:', error);
    }
}

// Load Cost Tracking
async function loadCostTracking() {
    try {
        const response = await fetch(`${API_BASE}/cost-tracking`);
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        const data = await response.json();
        
        document.getElementById('totalCost').textContent = 
            `$${parseFloat(data.total_cost || 0).toFixed(4)}`;
        
        // Update breakdown if available
        if (data.breakdown) {
            document.getElementById('openaiCost').textContent = 
                `$${parseFloat(data.breakdown.openai || 0).toFixed(4)}`;
            document.getElementById('deepseekCost').textContent = 
                `$${parseFloat(data.breakdown.deepseek || 0).toFixed(4)}`;
        }
        
    } catch (error) {
        console.error('Error loading cost tracking:', error);
    }
}

// Send Query
async function sendQuery() {
    const input = document.getElementById('queryInput');
    const query = input.value.trim();
    
    if (!query) return;
    
    if (!currentDataset) {
        addMessage('assistant', '⚠️ Please upload a dataset first.', true);
        showToast('Please upload a dataset first', 'warning');
        return;
    }
    
    // Add user message
    addMessage('user', query);
    input.value = '';
    input.style.height = 'auto';
    
    // Disable send button
    const sendBtn = document.getElementById('sendBtn');
    sendBtn.disabled = true;
    
    // Show loading message
    const loadingId = addMessage('assistant', '🤔 Thinking...', false, true);
    
    try {
        const llmProvider = document.getElementById('llmProvider').value || 
                           document.getElementById('quickProviderSelect').value;
        
        const params = new URLSearchParams({
            dataset_name: currentDataset,
            query: query
        });
        
        if (llmProvider && llmProvider.toLowerCase() !== 'auto') {
            params.append('llm_provider', llmProvider);
        }
        
        const response = await fetch(`${API_BASE}/query?${params}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            let errorMessage = 'Query failed';
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch (e) {
                errorMessage = `Server error: ${response.status} ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }
        
        const data = await response.json();
        
        // Remove loading message
        removeMessage(loadingId);
        
        // Add assistant response
        let responseText = data.message || data.answer || 'No response generated';
        
        // Add tool calls info if available
        if (data.tool_calls && data.tool_calls.length > 0) {
            responseText += '\n\n**🔧 Tools used:**\n';
            data.tool_calls.forEach(tool => {
                const toolName = tool.tool || 'unknown';
                const toolInput = (tool.input || '').substring(0, 100);
                responseText += `- **${toolName}**: ${toolInput}${toolInput.length >= 100 ? '...' : ''}\n`;
            });
        }
        
        addMessage('assistant', responseText);
        
        // Check for chart images in tool outputs
        if (data.tool_calls) {
            data.tool_calls.forEach(tool => {
                if (tool.tool === 'chart_generator' || tool.tool === 'ChartGeneratorTool') {
                    try {
                        const toolOutput = typeof tool.output === 'string' ? tool.output : JSON.stringify(tool.output);
                        const chartData = JSON.parse(toolOutput);
                        if (chartData.image_base64) {
                            displayChart(chartData.image_base64, chartData.title || 'Chart');
                        }
                    } catch (e) {
                        // Not a chart output or invalid JSON
                        console.debug('Not a chart output:', e);
                    }
                }
            });
        }
        
        // Update cost tracking
        loadCostTracking();
        showToast('Query completed successfully', 'success');
        
    } catch (error) {
        removeMessage(loadingId);
        
        // Show detailed error message
        let errorMsg = error.message || 'An unknown error occurred';
        
        // Make error messages more user-friendly
        if (errorMsg.includes('API key') || errorMsg.includes('api key')) {
            errorMsg = '❌ **API Key Error**\n\n' + errorMsg + 
                      '\n\nPlease check your .env file and ensure OPENAI_API_KEY or DEEPSEEK_API_KEY is set correctly.';
        } else if (errorMsg.includes('not initialized') || errorMsg.includes('initialization')) {
            errorMsg = '⚠️ **Agent Error**\n\n' + errorMsg + 
                      '\n\nTry uploading the dataset again or check your LLM configuration.';
        } else if (errorMsg.includes('rate limit') || errorMsg.includes('quota')) {
            errorMsg = '⏱️ **Rate Limit**\n\n' + errorMsg + 
                      '\n\nPlease wait a moment and try again.';
        } else if (errorMsg.includes('requests') && errorMsg.includes('module')) {
            errorMsg = '❌ **Missing Dependency**\n\n' + 
                      'The "requests" module is missing. Please install it: `pip install requests`';
        } else {
            errorMsg = '❌ **Error**\n\n' + errorMsg;
        }
        
        addMessage('assistant', errorMsg, true);
        showToast('Query failed: ' + errorMsg.split('\n')[0], 'error');
    } finally {
        // Re-enable send button
        sendBtn.disabled = false;
    }
}

// Message Functions
function addMessage(role, content, isError = false, isLoading = false) {
    const messagesDiv = document.getElementById('chatMessages');
    
    // Remove welcome message if present
    const welcomeMsg = messagesDiv.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    const messageDiv = document.createElement('div');
    const messageId = `msg-${Date.now()}-${Math.random()}`;
    messageDiv.id = messageId;
    messageDiv.className = `message ${role} ${isError ? 'error' : ''}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (isLoading) {
        contentDiv.innerHTML = '<span class="loading"></span> ' + escapeHtml(content);
    } else {
        // Convert markdown-like formatting
        content = escapeHtml(content);
        content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');
        content = content.replace(/\n/g, '<br>');
        contentDiv.innerHTML = content;
    }
    
    messageDiv.appendChild(contentDiv);
    
    const timestamp = document.createElement('div');
    timestamp.className = 'message-timestamp';
    timestamp.textContent = new Date().toLocaleTimeString();
    messageDiv.appendChild(timestamp);
    
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    
    chatHistory.push({ role, content, timestamp: new Date(), id: messageId });
    
    return messageId;
}

function removeMessage(messageId) {
    const message = document.getElementById(messageId);
    if (message) {
        message.remove();
    }
}

function displayChart(imageBase64, title) {
    const messagesDiv = document.getElementById('chatMessages');
    const chartDiv = document.createElement('div');
    chartDiv.className = 'chart-container';
    
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${imageBase64}`;
    img.className = 'chart-image';
    img.alt = title || 'Chart';
    img.title = title || 'Chart';
    
    chartDiv.appendChild(img);
    messagesDiv.appendChild(chartDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function clearChat() {
    if (chatHistory.length === 0) return;
    
    if (confirm('Are you sure you want to clear the chat history?')) {
        const messagesDiv = document.getElementById('chatMessages');
        messagesDiv.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">💬</div>
                <h3>Chat cleared</h3>
                <p>Ask a question about your data to continue.</p>
            </div>
        `;
        chatHistory = [];
        showToast('Chat cleared', 'success');
    }
}

// Loading Overlay
function showLoadingOverlay(message = 'Processing...') {
    const overlay = document.getElementById('loadingOverlay');
    const text = document.getElementById('loadingText');
    text.textContent = message;
    overlay.style.display = 'flex';
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.style.display = 'none';
}

// Toast Notifications
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const title = type === 'success' ? 'Success' : 
                  type === 'error' ? 'Error' : 
                  type === 'warning' ? 'Warning' : 'Info';
    
    toast.innerHTML = `
        <div class="toast-header">
            <span class="toast-title">${title}</span>
            <button class="toast-close" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
        <div class="toast-message">${escapeHtml(message)}</div>
    `;
    
    container.appendChild(toast);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.style.animation = 'slideIn 0.3s ease-out reverse';
            setTimeout(() => toast.remove(), 300);
        }
    }, 5000);
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
