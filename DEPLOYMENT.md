# URL Threat Detector - Deployment Guide

## Quick Deployment to Render

### Step 1: Create GitHub Repository
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/url-threat-detector.git
git push -u origin main
```

### Step 2: Deploy to Render
1. Go to https://render.com
2. Sign up or login with GitHub
3. Click "New +" button
4. Select "Web Service"
5. Select your `url-threat-detector` repository
6. Fill in details:
   - **Name**: url-threat-detector (or your choice)
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Instance Type**: Free
7. Click "Create Web Service"

### Step 3: Wait for Deployment
- Render will automatically build and deploy
- You'll get a URL like: `https://url-threat-detector.onrender.com`
- First startup may take 1-2 minutes

## Features Ready for Production
✅ ML model for URL threat detection
✅ User authentication (login/register)
✅ AI chatbot (answers any question)
✅ SHAP explanations for predictions
✅ QR code analysis
✅ Image text extraction
✅ Beautiful security-themed UI
✅ Voice input chatbot

## Environment Variables (if needed)
- `OPENAI_API_KEY` - For better chatbot responses (optional)
- `SECRET_KEY` - Flask session key (auto-generated on Render)

## First Time Users
1. Go to `/register` to create account
2. Login with your credentials
3. Try the chatbot (💬 button)
4. Analyze URLs or images

Enjoy! 🚀
