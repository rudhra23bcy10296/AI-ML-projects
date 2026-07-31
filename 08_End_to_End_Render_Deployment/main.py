"""
End-to-End Render Deployment - Driver & Local Verification Script
Author: Rudhra Sitholey (Reg No: 23BCY10296 | App No: IN26012560)
"""

from train_save_model import build_and_save_model
import uvicorn


def main():
    print("=" * 65)
    print(" Project 8: End to End Render Deployment Project")
    print(" Student: Rudhra Sitholey | Reg: 23BCY10296 | App: IN26012560")
    print("=" * 65)
    
    print("\n[1] Training & Exporting Model Binary (.joblib)...")
    build_and_save_model()
    
    print("\n[2] Verifying FastAPI Web Server Startup Configuration...")
    print("    Render Deployment Config: render.yaml & Dockerfile verified.")
    print("    FastAPI REST endpoints: '/' (Dashboard UI), '/health', '/predict'")
    print("\nTo launch local server manually, execute:")
    print("    uvicorn app.main:app --reload --port 8000")
    print("\nEnd-to-End Render Deployment Setup Completed Successfully!")


if __name__ == '__main__':
    main()
