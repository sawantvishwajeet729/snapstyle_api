from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import google.generativeai as genai
from io import BytesIO
import json
import re
import base64
import os
import requests

app = FastAPI(title="SnapStyle API")

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Google Custom Search API credentials
GOOGLE_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


@app.get("/")
async def root():
    return {
        "message": "SnapStyle API is running",
        "endpoints": {
            "/generate-fashion": "POST - Generate fashion description from image",
            "/search-products": "POST - Search for fashion products"
        }
    }


@app.post("/generate-fashion")
async def generate_fashion(
    file: UploadFile = File(...),
    style: str = Form(...)
):
    """
    Generate fashion recommendations based on uploaded photo and style preference.
    
    Args:
        file: Image file (jpg/png)
        style: Style type (Casual, Modern, Stylish, Traditional)
    
    Returns:
        JSON with generated image (base64) and outfit description
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        # Validate style
        valid_styles = ["Casual", "Modern", "Stylish", "Traditional"]
        if style not in valid_styles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid style. Choose from: {', '.join(valid_styles)}"
            )
        
        # Create prompt for outfit analysis and recommendation
        prompt = (
            f"**Task**: Analyze the person in the provided image and suggest a complete {style} style outfit.\n\n"
            "**Instructions**:\n"
            "1. **Analyze the person's features**:\n"
            "   - Skin tone and complexion\n"
            "   - Body type and proportions\n"
            "   - Current style (if visible)\n"
            "   - Background and setting\n\n"
            f"2. **Design a complete {style} outfit** that:\n"
            "   - Complements their skin tone\n"
            "   - Flatters their body type\n"
            f"   - Matches the {style} aesthetic\n"
            "   - Is practical and fashionable\n\n"
            "3. **Provide outfit details in JSON format**:\n"
            "Return ONLY a valid JSON object with these keys (omit any that don't apply):\n"
            "{\n"
            '  "Top": "Description of shirt/blouse/jacket",\n'
            '  "Bottom": "Description of pants/skirt/shorts",\n'
            '  "Footwear": "Description of shoes",\n'
            '  "Outerwear": "Description of jacket/coat (if applicable)",\n'
            '  "Accessories": "Description of accessories (if applicable)"\n'
            "}\n\n"
            "Make each description detailed and search-friendly (include colors, materials, style details).\n"
            "Example: 'Navy blue slim-fit cotton chinos' or 'Cream colored cable-knit cashmere sweater'"
        )
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Generate content
        response = model.generate_content([prompt, image])
        
        # Get the text response
        description_raw_text = response.text if hasattr(response, 'text') else ""
        
        # Parse JSON description
        description_json = {}
        if description_raw_text:
            try:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r"```json\s*\n(.*?)\n\s*```", description_raw_text, re.DOTALL)
                if json_match:
                    json_string = json_match.group(1).strip()
                else:
                    # Try to find JSON object directly
                    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', description_raw_text, re.DOTALL)
                    if json_match:
                        json_string = json_match.group(0)
                    else:
                        json_string = description_raw_text.strip()
                
                description_json = json.loads(json_string)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw text: {description_raw_text}")
                # Create structured response from text
                description_json = {
                    "Outfit": description_raw_text[:500]
                }
        
        # Convert original image to base64 (since Gemini can't generate images)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "image_base64": img_base64,
            "description": description_json,
            "note": "Image shows your original photo. Use the outfit description to shop for items."
        })
            
    except Exception as e:
        print(f"Error in generate_fashion: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating fashion: {str(e)}"
        )


@app.post("/search-products")
async def search_products(description: str = Form(...), num_results: int = Form(5)):
    """
    Search for fashion products using Google Custom Search API.
    
    Args:
        description: Product description to search for
        num_results: Number of results to return (default: 5)
    
    Returns:
        JSON with search results including product links and images
    """
    try:
        if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
            raise HTTPException(
                status_code=500,
                detail="Google Search API credentials not configured"
            )
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": description + " buy online",
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "num": min(num_results, 10),
            "searchType": "image"
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Google Search API error: {response.text}"
            )
        
        data = response.json()
        
        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title"),
                "link": item.get("image", {}).get("contextLink"),
                "image": item.get("link"),
                "thumbnail": item.get("image", {}).get("thumbnailLink")
            })
        
        return JSONResponse({
            "success": True,
            "results": results,
            "query": description
        })
        
    except Exception as e:
        print(f"Error in search_products: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching products: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "message": "SnapStyle API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
