import os
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

app = FastAPI()

# Enable CORS for Cloudflare Worker access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the required output structure
class InvoiceSchema(BaseModel):
    invoice_no: Optional[str] = None
    date: Optional[str] = Field(None, description="ISO format YYYY-MM-DD")
    vendor: Optional[str] = None
    amount: Optional[float] = Field(None, description="Subtotal before tax")
    tax: Optional[float] = Field(None, description="Tax amount only")
    currency: Optional[str] = None

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class InvoiceRequest(BaseModel):
    invoice_text: str

@app.post("/extract", response_model=InvoiceSchema)
async def extract_invoice(request: InvoiceRequest):
    prompt = f"Extract the invoice details from this text: {request.invoice_text}"
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=InvoiceSchema,
        ),
    )
    # The SDK automatically parses the JSON into the Pydantic model
    return response.parsed
