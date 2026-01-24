# How to Start the DataBrain AI Agent Server

## Quick Start

1. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Set the Python path:**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **Start the server:**
   ```bash
   cd databrain_agent/backend
   python main.py
   ```

   Or use the startup script:
   ```bash
   ./run_server.sh
   ```

## Verify Server is Running

The server should start on `http://localhost:8000`. You can verify by:

1. Opening `http://localhost:8000` in your browser - you should see:
   ```json
   {"message":"DataBrain AI Agent API","version":"1.0.0"}
   ```

2. Or check the API docs at `http://localhost:8000/docs`

## Troubleshooting

### "Failed to fetch" Error

This means the backend server is not running. Make sure:
- The server is started (see steps above)
- Port 8000 is not blocked
- No firewall is blocking localhost connections

### Import Errors

If you see import errors, make sure:
- Virtual environment is activated
- PYTHONPATH is set correctly
- All packages are installed: `pip install -r requirements.txt`

### API Keys

Make sure your `.env` file has your API keys:
```
OPENAI_API_KEY=your-key-here
DEEPSEEK_API_KEY=your-key-here
```
