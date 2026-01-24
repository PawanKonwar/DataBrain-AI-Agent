# Deployment Guide

This guide covers different deployment options for DataBrain AI Agent.

## Local Development

### Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Start server
cd databrain_agent/backend
python main.py
```

### Using Startup Script

```bash
./run_server.sh
```

## Production Deployment

### Option 1: Gunicorn + Uvicorn

```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  databrain_agent.backend.main:app
```

### Option 2: Docker (Recommended)

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "databrain_agent.backend.main:app"]
```

Build and run:

```bash
docker build -t databrain-agent .
docker run -p 8000:8000 --env-file .env databrain-agent
```

### Option 3: Systemd Service

Create `/etc/systemd/system/databrain-agent.service`:

```ini
[Unit]
Description=DataBrain AI Agent
After=network.target

[Service]
User=your-user
WorkingDirectory=/path/to/DataBrain-AI-Agent
Environment="PATH=/path/to/DataBrain-AI-Agent/.venv/bin"
ExecStart=/path/to/DataBrain-AI-Agent/.venv/bin/gunicorn \
  -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  databrain_agent.backend.main:app

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable databrain-agent
sudo systemctl start databrain-agent
```

## Environment Variables

Set these in your production environment:

```bash
OPENAI_API_KEY=your-key
DEEPSEEK_API_KEY=your-key  # Optional
```

## Frontend Deployment

### Option 1: Static Hosting

Upload the `frontend/` directory to:
- GitHub Pages
- Netlify
- Vercel
- AWS S3 + CloudFront

### Option 2: Serve with Backend

Modify FastAPI to serve static files:

```python
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
```

## Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **CORS**: Configure CORS appropriately for production
3. **Rate Limiting**: Consider adding rate limiting for API endpoints
4. **HTTPS**: Use HTTPS in production
5. **Authentication**: Consider adding authentication for production use

## Monitoring

- Monitor API costs via `/api/cost-tracking` endpoint
- Set up logging to track errors
- Monitor server resources (CPU, memory)

## Scaling

- Use a reverse proxy (nginx) for load balancing
- Run multiple worker processes with Gunicorn
- Consider using a process manager like Supervisor
