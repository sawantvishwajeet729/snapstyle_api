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
from typing import Optional

app = FastAPI(title="SnapStyle API")

# Enable CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini client (make sure to set GOOGLE_API_KEY environment variable)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Google Custom Search API credentials (set as environment variables)
GOOGLE_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")


@app.get("/")
async def root():
    return {
        "message": "SnapStyle API is running",
        "endpoints": {
            "/generate-fashion": "POST - Generate fashion image and description",
            "/search-products": "POST - Search for fashion products"
        }
    }


@app.post("/generate-fashion")
async def generate_fashion(
    file: UploadFile = File(...),
    style: str = Form(...)
):
    """
    Generate a fashion image based on uploaded photo and style preference.
    
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
        
        # Create prompt
        prompt = (
            f"**Task**: Generate a new fashion image based on the person in the provided image, PLUS a JSON text description of the new outfit.\n"
            f"**Style**: {style}\n\n"
            "**Instructions for Generation**:\n"
            "1.  **Image Analysis**: First, carefully analyze the provided image to understand the person's:\n"
            "    * **Skin Tone**: Identify their skin complexion.\n"
            "    * **Body Type**: Assess their general body shape and proportions.\n"
            "    * **Background Context**: Observe the setting and colors in the background of the original image.\n"
            "2.  **Outfit Design**: Design a complete outfit that fits the specified '{style}' style, ensuring that:\n"
            "    * The clothing colors are chosen appropriately to complement the identified skin tone and the overall background context.\n"
            "    * The outfit flatters the person's body type.\n"
            "    * The person from the original image is depicted wearing this new outfit.\n"
            "3.  **Text Description**: Provide a detailed, search-friendly JSON description of the *new* outfit you just created. Each distinct apparel item should be a separate key-value pair. If an item consists of multiple layers (e.g., shirt and t-shirt), list them under appropriate, separate keys.\n\n"
            "**Required Description Format (JSON)**:\n"
            "Respond ONLY with a JSON object. Each key should represent a distinct apparel or accessory category. The value for each key should be a search-friendly string description. If an item is not applicable or not visible, omit its key.\n"
            "Example:\n"
            "```json\n"
            "{\n"
            '  "Shirt": "Men\'s light wash denim button-up shirt",\n'
            '  "Innerwear": "White crew neck t-shirt",\n'
            '  "Bottoms": "Men\'s slim-fit olive green chino pants",\n'
            '  "Shoes": "Men\'s white canvas low-top sneakers",\n'
            '  "Accessory 1": "Silver watch",\n'
            '  "Accessory 2": "Brown leather belt"\n'
            "}\n"
            "```\n"
            "Ensure all descriptions are distinct and optimized for product search."
        )
        
        # Generate content using Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[prompt, image],
        )
        
        generated_image = None
        description_raw_text = ""
        
        # Extract image and text from response
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                description_raw_text += part.text
            elif part.inline_data is not None:
                generated_image = Image.open(BytesIO(part.inline_data.data))
        
        # Parse JSON description
        description_json = {}
        if description_raw_text:
            try:
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r"```json\n(.*?)```", description_raw_text, re.DOTALL)
                if json_match:
                    json_string = json_match.group(1).strip()
                else:
                    json_string = description_raw_text.strip()
                
                description_json = json.loads(json_string)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw text: {description_raw_text}")
                # Return raw text as fallback
                description_json = {"description": description_raw_text}
        
        # Convert image to base64
        if generated_image:
            buffered = BytesIO()
            generated_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return JSONResponse({
                "success": True,
                "image_base64": img_base64,
                "description": description_json
            })
        else:
            raise HTTPException(
                status_code=500,
                detail="No image was generated"
            )
            
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
            "q": description,
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CSE_ID,
            "num": min(num_results, 10),  # Google API max is 10
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
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
