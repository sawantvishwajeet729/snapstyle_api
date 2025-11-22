from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import json
import re
import base64
import os
import requests
import sys

# Create app first
app = FastAPI(title="SnapStyle API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
genai_configured = False
genai = None

def configure_genai():
    """Configure Gemini API with error handling"""
    global genai_configured, genai
    try:
        import google.generativeai as genai_module
        genai = genai_module
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("WARNING: GOOGLE_API_KEY not set", file=sys.stderr)
            return False
            
        genai.configure(api_key=api_key)
        genai_configured = True
        print("✓ Gemini API configured successfully", file=sys.stderr)
        return True
    except Exception as e:
        print(f"ERROR configuring Gemini: {e}", file=sys.stderr)
        return False

# Try to configure on startup
configure_genai()

# Google Custom Search API credentials
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


@app.get("/")
async def root():
    """Root endpoint - shows API status"""
    return {
        "message": "SnapStyle API is running",
        "status": "healthy",
        "gemini_configured": genai_configured,
        "search_configured": bool(GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID),
        "endpoints": {
            "/health": "GET - Health check",
            "/generate-fashion": "POST - Generate fashion recommendations",
            "/search-products": "POST - Search for fashion products"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "SnapStyle API",
        "gemini": "configured" if genai_configured else "not configured"
    }


@app.post("/generate-fashion")
async def generate_fashion(
    file: UploadFile = File(...),
    style: str = Form(...)
):
    """
    Generate fashion recommendations based on uploaded photo.
    """
    try:
        # Check if Gemini is configured
        if not genai_configured:
            # Try to configure again
            if not configure_genai():
                raise HTTPException(
                    status_code=503,
                    detail="Gemini API not configured. Please set GOOGLE_API_KEY environment variable."
                )
        
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (jpg, png, etc.)"
            )
        
        # Read and validate image
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=400,
                detail="Image too large. Maximum size is 10MB."
            )
            
        image = Image.open(BytesIO(contents))
        
        # Validate style
        valid_styles = ["Casual", "Modern", "Stylish", "Traditional"]
        if style not in valid_styles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid style. Choose from: {', '.join(valid_styles)}"
            )
        
        # Create prompt
        prompt = f"""Analyze this person's photo and recommend a {style} outfit.

IMPORTANT: Respond with ONLY a JSON object, no other text.

Required format:
{{
  "analysis": {{
    "skin_tone": "description",
    "body_type": "description"
  }},
  "outfit": {{
    "Top": "specific item description",
    "Bottom": "specific item description",
    "Footwear": "specific item description",
    "Accessories": "optional items"
  }},
  "colors": ["color1", "color2", "color3"]
}}

Make descriptions searchable (include colors, materials, styles)."""

        # Generate content
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content([prompt, image])
            description_text = response.text if hasattr(response, 'text') else ""
        except Exception as e:
            print(f"Gemini API error: {e}", file=sys.stderr)
            raise HTTPException(
                status_code=500,
                detail=f"AI generation failed: {str(e)}"
            )
        
        # Parse JSON response
        description_json = {}
        if description_text:
            try:
                # Extract JSON
                json_match = re.search(r'\{.*\}', description_text, re.DOTALL)
                if json_match:
                    description_json = json.loads(json_match.group(0))
                else:
                    # Fallback
                    description_json = {
                        "outfit": {"description": description_text[:500]}
                    }
            except json.JSONDecodeError:
                description_json = {
                    "outfit": {"description": description_text[:500]}
                }
        
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "image_base64": img_base64,
            "description": description_json,
            "style_requested": style
        })
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in generate_fashion: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error: {str(e)}"
        )


@app.post("/search-products")
async def search_products(
    description: str = Form(...),
    num_results: int = Form(5)
):
    """Search for fashion products using Google Custom Search."""
    try:
        if not GOOGLE_SEARCH_API_KEY or not GOOGLE_CSE_ID:
            return JSONResponse({
                "success": False,
                "error": "Google Search not configured",
                "results": []
            })
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": description + " buy online",
            "key": GOOGLE_SEARCH_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "num": min(num_results, 10),
            "searchType": "image"
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"Search API error: {response.status_code}", file=sys.stderr)
            return JSONResponse({
                "success": False,
                "error": "Search failed",
                "results": []
            })
        
        data = response.json()
        results = []
        
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "link": item.get("image", {}).get("contextLink", ""),
                "image": item.get("link", ""),
                "thumbnail": item.get("image", {}).get("thumbnailLink", "")
            })
        
        return JSONResponse({
            "success": True,
            "results": results,
            "query": description
        })
        
    except Exception as e:
        print(f"Error in search_products: {str(e)}", file=sys.stderr)
        return JSONResponse({
            "success": False,
            "error": str(e),
            "results": []
        })


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on startup"""
    print("=" * 50, file=sys.stderr)
    print("SnapStyle API Starting...", file=sys.stderr)
    print(f"Python version: {sys.version}", file=sys.stderr)
    print(f"Gemini configured: {genai_configured}", file=sys.stderr)
    print(f"Search configured: {bool(GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID)}", file=sys.stderr)
    print("=" * 50, file=sys.stderr)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
