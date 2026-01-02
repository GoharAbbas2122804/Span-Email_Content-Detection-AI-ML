import React, { useState, useEffect } from 'react';
import './App.css';

const ThreatAnalyzer = () => {
  const [message, setMessage] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [recentScans, setRecentScans] = useState([]);
  const [systemLogs, setSystemLogs] = useState([]);
  const [charCount, setCharCount] = useState(0);
  const [darkMode, setDarkMode] = useState(true);
  const [showStats, setShowStats] = useState(false);
  const [batchMode, setBatchMode] = useState(false);
  const [batchMessages, setBatchMessages] = useState(['']);
  const [batchResults, setBatchResults] = useState([]);
  const [isProcessingBatch, setIsProcessingBatch] = useState(false);

  const API_URL = 'http://127.0.0.1:8000/api';

  // Load saved preferences
  useEffect(() => {
    const savedDarkMode = localStorage.getItem('darkMode');
    if (savedDarkMode !== null) {
      setDarkMode(savedDarkMode === 'true');
    }
    const savedScans = localStorage.getItem('recentScans');
    if (savedScans) {
      setRecentScans(JSON.parse(savedScans));
    }
  }, []);

  // Save preferences
  useEffect(() => {
    localStorage.setItem('darkMode', darkMode.toString());
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  useEffect(() => {
    localStorage.setItem('recentScans', JSON.stringify(recentScans));
  }, [recentScans]);

  // Handle message input
  const handleMessageChange = (e) => {
    const text = e.target.value;
    if (text.length <= 5000) {
      setMessage(text);
      setCharCount(text.length);
    }
  };

  // Add log entry
  const addLog = (text, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
    setSystemLogs(prev => [...prev, { timestamp, text, type }]);
  };

  // Analyze message
  const analyzeThreat = async () => {
    if (!message.trim()) {
      addLog('Error: No message provided', 'error');
      return;
    }

    setIsAnalyzing(true);
    setResult(null);
    addLog('Analyzing keyword density...');
    
    setTimeout(() => {
      addLog('Cross-referencing threat database...');
    }, 500);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });

      if (!response.ok) throw new Error('API request failed');

      const data = await response.json();
      
      // Calculate dynamic score based on probability
      const calculateScore = (prediction, probability) => {
        if (prediction === 'SPAM') {
          // SPAM: probability 0.5-1.0 → score 5-10
          return Math.floor(probability * 10);
        } else {
          // HAM: probability 0.5-1.0 → score 0-5 (inverted)
          return Math.floor((1 - probability) * 10);
        }
      };

  // Batch prediction
  const analyzeBatch = async () => {
    const validMessages = batchMessages.filter(msg => msg.trim());
    if (validMessages.length === 0) {
      addLog('Error: No messages provided', 'error');
      return;
    }

    setIsProcessingBatch(true);
    setBatchResults([]);
    addLog(`Processing ${validMessages.length} messages...`);

    try {
      const response = await fetch(`${API_URL}/predict_batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: validMessages })
      });

      if (!response.ok) throw new Error('Batch API request failed');

      const data = await response.json();
      setBatchResults(data.results || []);
      addLog(`Batch analysis complete: ${data.results.length} results`, 'success');
    } catch (error) {
      addLog(`Error: ${error.message}`, 'error');
      console.error('Batch analysis error:', error);
    } finally {
      setIsProcessingBatch(false);
    }
  };

  // Add batch message field
  const addBatchField = () => {
    setBatchMessages([...batchMessages, '']);
  };

  // Remove batch message field
  const removeBatchField = (index) => {
    setBatchMessages(batchMessages.filter((_, i) => i !== index));
  };

  // Update batch message
  const updateBatchMessage = (index, value) => {
    const updated = [...batchMessages];
    updated[index] = value;
    setBatchMessages(updated);
  };

  // Export results
  const exportResults = () => {
    if (recentScans.length === 0) {
      addLog('No results to export', 'error');
      return;
    }

    const csv = [
      ['Timestamp', 'Message', 'Prediction', 'Score', 'Probability'].join(','),
      ...recentScans.map(scan => [
        scan.timestamp,
        `"${scan.message.replace(/"/g, '""')}"`,
        scan.prediction,
        scan.score,
        scan.probability
      ].join(','))
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `spam-detection-results-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    addLog('Results exported successfully', 'success');
  };

  // Calculate statistics
  const getStatistics = () => {
    if (recentScans.length === 0) return null;
    
    const total = recentScans.length;
    const spamCount = recentScans.filter(s => s.prediction === 'SPAM').length;
    const hamCount = total - spamCount;
    const avgScore = recentScans.reduce((sum, s) => sum + s.score, 0) / total;
    const avgProbability = recentScans.reduce((sum, s) => sum + s.probability, 0) / total;

    return { total, spamCount, hamCount, avgScore, avgProbability };
  };

  const stats = getStatistics();

      const scanResult = {
        prediction: data.prediction,
        probability: data.probability,
        score: calculateScore(data.prediction, data.probability),
        message: message.substring(0, 50) + (message.length > 50 ? '...' : ''),
        timestamp: new Date().toLocaleTimeString('en-US', { 
          hour12: false, 
          hour: '2-digit', 
          minute: '2-digit' 
        }),
        threatLevel: data.prediction === 'SPAM' 
          ? Math.floor(data.probability * 10) 
          : Math.floor((1 - data.probability) * 10)
      };

      setResult(scanResult);
      setRecentScans(prev => [scanResult, ...prev.slice(0, 9)]);
      
      addLog(
        `SCAN ${data.prediction === 'SPAM' ? 'THREAT DETECTED' : 'CLEAN'}: Score ${scanResult.score}`, 
        data.prediction === 'SPAM' ? 'error' : 'success'
      );

    } catch (error) {
      addLog(`Error: ${error.message}`, 'error');
      console.error('Analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Clear input
  const handleClear = () => {
    setMessage('');
    setCharCount(0);
    setResult(null);
  };

  // Paste from clipboard
  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText();
      setMessage(text.substring(0, 5000));
      setCharCount(text.length > 5000 ? 5000 : text.length);
    } catch (err) {
      addLog('Failed to paste from clipboard', 'error');
    }
  };

  // Handle file upload
  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
      addLog('Error: Please upload a CSV file', 'error');
      return;
    }

    try {
      const formData = new FormData();
      formData.append('file', file);

      addLog('Uploading file for training...');
      
      const response = await fetch(`${API_URL}/train/upload`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error('Upload failed');

      const data = await response.json();
      addLog(`Model trained successfully! Accuracy: ${Math.round(data.accuracy * 100)}%`, 'success');
    } catch (error) {
      addLog(`Error: ${error.message}`, 'error');
      console.error('Upload error:', error);
    }
  };

  // Handle batch CSV upload
  const handleBatchCSVUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
      addLog('Error: Please upload a CSV file', 'error');
      return;
    }

    try {
      const text = await file.text();
      const lines = text.split('\n').filter(line => line.trim());
      
      // Parse CSV (simple parser - assumes first column is text)
      const messages = lines.slice(1).map(line => {
        const match = line.match(/^"([^"]+)"|^([^,]+)/);
        return match ? (match[1] || match[2]).trim() : '';
      }).filter(msg => msg);

      if (messages.length === 0) {
        addLog('Error: No valid messages found in CSV', 'error');
        return;
      }

      setBatchMessages(messages);
      setBatchMode(true);
      addLog(`Loaded ${messages.length} messages from CSV`, 'success');
    } catch (error) {
      addLog(`Error: ${error.message}`, 'error');
      console.error('CSV parse error:', error);
    }
  };

  // Icons as SVG
  const ShieldIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
  );

  const UploadIcon = () => (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="17 8 12 3 7 8"/>
      <line x1="12" y1="3" x2="12" y2="15"/>
    </svg>
  );

  const TrashIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="3 6 5 6 21 6"/>
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
    </svg>
  );

  const CopyIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
    </svg>
  );

  const SettingsIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="3"/>
      <path d="M12 1v6m0 6v6m6-12l-3 3m-6 6l-3 3m12-12l-3 3m-6 6l-3 3"/>
    </svg>
  );

  const MoonIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
    </svg>
  );

  const SunIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="5"/>
      <line x1="12" y1="1" x2="12" y2="3"/>
      <line x1="12" y1="21" x2="12" y2="23"/>
      <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
      <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
      <line x1="1" y1="12" x2="3" y2="12"/>
      <line x1="21" y1="12" x2="23" y2="12"/>
      <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
      <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
    </svg>
  );

  const StatsIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="18" y1="20" x2="18" y2="10"/>
      <line x1="12" y1="20" x2="12" y2="4"/>
      <line x1="6" y1="20" x2="6" y2="14"/>
    </svg>
  );

  const DownloadIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
  );

  const PlusIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="12" y1="5" x2="12" y2="19"/>
      <line x1="5" y1="12" x2="19" y2="12"/>
    </svg>
  );

  const XIcon = () => (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <line x1="18" y1="6" x2="6" y2="18"/>
      <line x1="6" y1="6" x2="18" y2="18"/>
    </svg>
  );

  const ClockIcon = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/>
      <polyline points="12 6 12 12 16 14"/>
    </svg>
  );

  return (
    <div className="app-container">
      <div className="main-grid">
        
        {/* Left Sidebar */}
        <div className="sidebar left-sidebar">
          {/* Logo */}
          <div className="card logo-card">
            <div className="logo-content">
              <div className="logo-icon">
                <ShieldIcon />
              </div>
              <h1 className="logo-text">NEXUS</h1>
            </div>
          </div>

          {/* Recent Scans */}
          <div className="card">
            <div className="card-header">
              <ClockIcon />
              <h2 className="card-title">RECENT SCANS</h2>
            </div>
            
            <div className="recent-scans">
              {recentScans.length === 0 ? (
                <p className="no-data">No logs found.</p>
              ) : (
                recentScans.map((scan, idx) => (
                  <div key={idx} className="scan-item">
                    <div className={`status-dot ${scan.prediction === 'SPAM' ? 'threat' : 'safe'}`} />
                    <div className="scan-content">
                      <div className="scan-header">
                        <span className={`scan-score ${scan.prediction === 'SPAM' ? 'threat' : 'safe'}`}>
                          {scan.score}%
                        </span>
                        <span className="scan-time">{scan.timestamp}</span>
                      </div>
                      <p className="scan-message">{scan.message}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Batch Upload */}
          <div className="card upload-card">
            <input
              type="file"
              accept=".csv"
              onChange={handleBatchCSVUpload}
              className="file-input"
              id="batch-upload-input"
            />
            <label htmlFor="batch-upload-input" className="upload-content">
              <div className="upload-icon">
                <UploadIcon />
              </div>
              <h3 className="upload-title">Batch Upload</h3>
              <p className="upload-subtitle">Drop .CSV or Click</p>
            </label>
          </div>

          {/* Model Training Upload */}
          <div className="card upload-card training-card">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="file-input"
              id="training-upload-input"
            />
            <label htmlFor="training-upload-input" className="upload-content">
              <div className="upload-icon">
                <UploadIcon />
              </div>
              <h3 className="upload-title">Train Model</h3>
              <p className="upload-subtitle">Upload CSV Dataset</p>
            </label>
          </div>
        </div>

        {/* Main Content */}
        <div className="main-content">
          {/* Header */}
          <div className="card header-card">
            <div className="header-content">
              <div>
                <h2 className="header-title">Threat Analyzer</h2>
                <p className="header-subtitle">v4.0.0 Enterprise Edition</p>
              </div>
              <div className="header-actions">
                <button 
                  className="icon-btn" 
                  onClick={() => setShowStats(!showStats)}
                  title="Statistics"
                >
                  <StatsIcon />
                </button>
                <button 
                  className="icon-btn" 
                  onClick={() => setDarkMode(!darkMode)}
                  title="Toggle Theme"
                >
                  {darkMode ? <SunIcon /> : <MoonIcon />}
                </button>
                <button className="icon-btn" title="Settings">
                  <SettingsIcon />
                </button>
              </div>
            </div>
          </div>

          {/* Statistics Dashboard */}
          {showStats && stats && (
            <div className="card stats-card">
              <div className="stats-header">
                <h3 className="stats-title">Statistics Dashboard</h3>
                <button className="close-btn" onClick={() => setShowStats(false)}>
                  <XIcon />
                </button>
              </div>
              <div className="stats-grid">
                <div className="stat-item">
                  <div className="stat-value">{stats.total}</div>
                  <div className="stat-label">Total Scans</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value threat">{stats.spamCount}</div>
                  <div className="stat-label">Spam Detected</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value safe">{stats.hamCount}</div>
                  <div className="stat-label">Safe Messages</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">{Math.round(stats.avgScore)}%</div>
                  <div className="stat-label">Avg Threat Score</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">{Math.round(stats.avgProbability * 100)}%</div>
                  <div className="stat-label">Avg Confidence</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">{Math.round((stats.spamCount / stats.total) * 100)}%</div>
                  <div className="stat-label">Spam Rate</div>
                </div>
              </div>
              <button className="export-btn" onClick={exportResults}>
                <DownloadIcon />
                Export Results
              </button>
            </div>
          )}

          {/* Mode Toggle */}
          <div className="card mode-toggle-card">
            <div className="mode-toggle">
              <button 
                className={`mode-btn ${!batchMode ? 'active' : ''}`}
                onClick={() => setBatchMode(false)}
              >
                Single Message
              </button>
              <button 
                className={`mode-btn ${batchMode ? 'active' : ''}`}
                onClick={() => setBatchMode(true)}
              >
                Batch Processing
              </button>
            </div>
          </div>

          {/* Input Area - Single Mode */}
          {!batchMode ? (
            <div className="card input-card">
              <textarea
                value={message}
                onChange={handleMessageChange}
                placeholder="// Initialize scan by entering message payload..."
                className="message-input"
                disabled={isAnalyzing}
              />
              
              <div className="input-footer">
                <div className="input-actions">
                  <button onClick={handleClear} className="action-btn">
                    <TrashIcon />
                    Clear
                  </button>
                  <button onClick={handlePaste} className="action-btn">
                    <CopyIcon />
                    Paste
                  </button>
                </div>
                <div className="char-count">
                  {charCount} / 5000 chars
                </div>
              </div>
            </div>
          ) : (
            /* Batch Input Area */
            <div className="card batch-input-card">
              <div className="batch-header">
                <h3 className="batch-title">Batch Messages</h3>
                <button className="action-btn" onClick={addBatchField}>
                  <PlusIcon />
                  Add Field
                </button>
              </div>
              <div className="batch-inputs">
                {batchMessages.map((msg, index) => (
                  <div key={index} className="batch-input-row">
                    <textarea
                      value={msg}
                      onChange={(e) => updateBatchMessage(index, e.target.value)}
                      placeholder={`Message ${index + 1}...`}
                      className="batch-message-input"
                      disabled={isProcessingBatch}
                    />
                    {batchMessages.length > 1 && (
                      <button 
                        className="remove-btn"
                        onClick={() => removeBatchField(index)}
                        disabled={isProcessingBatch}
                      >
                        <XIcon />
                      </button>
                    )}
                  </div>
                ))}
              </div>
              <div className="batch-footer">
                <div className="batch-count">{batchMessages.filter(m => m.trim()).length} message(s) ready</div>
                <button 
                  onClick={() => setBatchMessages([''])} 
                  className="action-btn"
                  disabled={isProcessingBatch}
                >
                  <TrashIcon />
                  Clear All
                </button>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="action-row">
            <div className="status-badge">
              <div className="status-indicator" />
              <span className="status-text">SYSTEM READY</span>
            </div>
            
            {!batchMode ? (
              <button
                onClick={analyzeThreat}
                disabled={isAnalyzing || !message.trim()}
                className={`scan-btn ${(isAnalyzing || !message.trim()) ? 'disabled' : ''}`}
              >
                <ShieldIcon />
                {isAnalyzing ? 'ANALYZING...' : 'INITIATE SCAN'}
              </button>
            ) : (
              <button
                onClick={analyzeBatch}
                disabled={isProcessingBatch || batchMessages.filter(m => m.trim()).length === 0}
                className={`scan-btn ${(isProcessingBatch || batchMessages.filter(m => m.trim()).length === 0) ? 'disabled' : ''}`}
              >
                <ShieldIcon />
                {isProcessingBatch ? 'PROCESSING BATCH...' : 'ANALYZE BATCH'}
              </button>
            )}
          </div>

          {/* Batch Results */}
          {batchMode && batchResults.length > 0 && (
            <div className="card batch-results-card">
              <div className="batch-results-header">
                <h3 className="batch-results-title">Batch Results ({batchResults.length})</h3>
                <button className="action-btn" onClick={() => setBatchResults([])}>
                  <TrashIcon />
                  Clear
                </button>
              </div>
              <div className="batch-results-list">
                {batchResults.map((result, index) => (
                  <div key={index} className={`batch-result-item ${result.prediction === 'SPAM' ? 'threat' : 'safe'}`}>
                    <div className="batch-result-header">
                      <span className="batch-result-index">#{index + 1}</span>
                      <span className={`batch-result-status ${result.prediction === 'SPAM' ? 'threat' : 'safe'}`}>
                        {result.prediction}
                      </span>
                      <span className="batch-result-probability">
                        {Math.round(result.probability * 100)}% confidence
                      </span>
                    </div>
                    <p className="batch-result-message">{result.original_message}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Sidebar */}
        <div className="sidebar right-sidebar">
          {/* System Log */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">SYSTEM LOG</h3>
              <div className="status-indicator active" />
            </div>
            
            <div className="system-logs">
              {systemLogs.length === 0 ? (
                <p className="no-data">Awaiting analysis...</p>
              ) : (
                systemLogs.slice(-5).reverse().map((log, idx) => (
                  <div key={idx} className="log-entry">
                    <span className="log-time">{log.timestamp}</span>
                    <span className={`log-text ${log.type}`}>{log.text}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Result Display */}
          <div className="card result-card">
            {!result ? (
              <div className="empty-result">
                <div className="empty-icon">
                  <ShieldIcon />
                </div>
                <p className="empty-text">Awaiting Analysis Data</p>
              </div>
            ) : (
              <div className="result-content">
                {/* Score Circle */}
                <div className="score-circle-container">
                  <div className={`score-circle ${result.prediction === 'SPAM' ? 'threat' : 'safe'}`}>
                    <span className="score-number">{result.score}</span>
                  </div>
                  <div className="result-info">
                    <h3 className={`result-status ${result.prediction === 'SPAM' ? 'threat' : 'safe'}`}>
                      {result.prediction === 'SPAM' ? 'THREAT DETECTED' : 'SAFE CONTENT'}
                    </h3>
                    <p className="result-description">
                      Content appears {result.prediction === 'SPAM' ? 'malicious' : 'legitimate'} 
                      ({Math.round(result.probability * 100)}% trust score)
                    </p>
                  </div>
                </div>

                {/* Details */}
                <div className="result-details">
                  <div className="detail-row">
                    <span className="detail-label">Threat Level</span>
                    <span className={`detail-value ${result.prediction === 'SPAM' ? 'threat' : 'safe'}`}>
                      {result.threatLevel}/10
                    </span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Confidence</span>
                    <span className="detail-value">{Math.round(result.probability * 100)}%</span>
                  </div>
                  <div className="detail-row">
                    <span className="detail-label">Scan Time</span>
                    <span className="detail-value">{result.timestamp}</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ThreatAnalyzer;