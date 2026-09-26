"""
TerraFly - Launcher Script
Smart India Hackathon (PS 175)
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    # Ensure root path is in python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting TerraFly 3D Elevation Reconstruction Server on http://localhost:{port}")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=True)
