#!/usr/bin/env python3
"""Quick test to verify server can start."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from databrain_agent.backend.main import app
    print("✅ Server imports successfully!")
    print("✅ FastAPI app created:", app.title)
    print("\nTo start the server, run:")
    print("  cd databrain_agent/backend")
    print("  python main.py")
    print("\nOr use:")
    print("  ./run_server.sh")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
