from fastapi import FastAPI, Query, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API for recommending SHL assessments based on job descriptions and queries",
    version="1.0.0"
)

def load_metadata():
    return pd.read_csv("src/data/shl_full_catalog_with_duration_desc.csv")

df_meta = load_metadata()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tfidf = None
df = None
tfidf_matrix = None
model_initializing = False

TEST_TYPE_MAP = {
    "K": "Knowledge & Skills",
    "B": "Behavioral",
    "P": "Personality",
    "C": "Cognitive",
    "A": "Aptitude",
    "S": "Situational",
    "T": "Technical",
    "N": "Numerical",
    "L": "Leadership",
    "D": "Decision Making",
    "E": "Emotional Intelligence"
}

def initialize_model():
    global tfidf, df, tfidf_matrix, model_initializing
    try:
        logger.info("Starting TF-IDF model initialization")
        model_initializing = True

        # Load data
        df = pd.read_csv("src/data/shl_full_catalog_with_duration_desc.csv")

        column_mapping = {
            'Test Name': 'testName',
            'Link': 'link',
            'Remote Testing': 'remoteTestingSupport',
            'Adaptive/IRT': 'adaptiveIRTSupport',
            'Test Types': 'testTypes',
            'Duration': 'duration'
        }

        for orig, new in column_mapping.items():
            if orig in df.columns:
                df = df.rename(columns={orig: new})

        if 'duration' not in df.columns:
            df['duration'] = "Not specified"

        df['combined_description'] = (
            df['testName'] + ' ' +
            df['testTypes'] + ' ' +
            df['remoteTestingSupport'] + ' ' +
            df['adaptiveIRTSupport']
        ).fillna('')

        # Initialize TF-IDF
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['combined_description'])

        logger.info("TF-IDF model initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing TF-IDF model: {str(e)}")
    finally:
        model_initializing = False

@app.get("/health")
async def health_check():
    """Health check endpoint that matches the specified format"""
    global tfidf, model_initializing

    if tfidf is not None:
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "model_status": "ready"
            }
        )
    elif model_initializing:
        return JSONResponse(
            status_code=200,
            content={
                "status": "initializing",
                "model_status": "loading, please wait for a few minutes, the model is loading on render."
            }
        )
    else:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_status": "Model is initializing, please wait a few minutes. This is because on Render's free tier, services are spun down after 15 minutes of inactivity. When someone accesses the API after this period, the entire service needs to restart - which means the model gets reloaded from scratch. Also, I cannot cache due to limited storage given on the free tier."
            }
        )

@app.get("/recommend")
async def get_recommendations(
    query: str = Query(..., description="Job description or natural language query"),
    top_k: int = Query(10, description="Number of recommendations to return", ge=1, le=10),
    format: str = Query("json", description="Response format: json or html")
):
    """Get assessment recommendations via GET with query params"""
    return await process_recommendation(query, top_k, format)

@app.post("/recommend")
async def post_recommendations(
    request: Request,
    format: str = Query("json", description="Response format: json or html")
):
    """Get assessment recommendations via POST with JSON body"""
    try:
        body = await request.json()
        query = body.get("query")
        top_k = min(max(body.get("top_k", 10), 1), 10)  
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required in request body")
            
        return await process_recommendation(query, top_k, format)
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

async def process_recommendation(query: str, top_k: int, format: str):
    """Common processing logic for both GET and POST requests"""
    global tfidf, df, tfidf_matrix

    if tfidf is None and model_initializing:
        response_obj = {
            "status": "initializing",
            "message": "Please wait a few minutes. This is because on Render's free tier, services are spun down after 15 minutes of inactivity. When someone accesses the API after this period, the entire service needs to restart - which means the model gets reloaded from scratch. Also, I cannot cache due to limited storage given on the free tier."
        }
        pretty_json = json.dumps(response_obj, indent=2, ensure_ascii=False)
        return Response(content=pretty_json, media_type="application/json", status_code=202)

    if tfidf is None:
        raise HTTPException(status_code=503, detail="Model not initialized. Please wait a few minutes. This is because on Render's free tier, services are spun down after 15 minutes of inactivity. When someone accesses the API after this period, the entire service needs to restart - which means the model gets reloaded from scratch. Also, I cannot cache due to limited storage given on the free tier.")

    try:
        start_time = time.time()

        # Perform TF-IDF search
        query_vec = tfidf.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        recommendations = []
        for idx in top_indices:
            link = df.iloc[idx]['link']

            meta = df_meta[df_meta["Link"] == link]
            if meta.empty:
                continue
            row = meta.iloc[0]

            description = row.get("Description", "")

            duration = 0
            duration_str = row.get("Duration", "")
            if isinstance(duration_str, str) and duration_str:
                import re
                duration_match = re.search(r'\d+', duration_str)
                if duration_match:
                    duration = int(duration_match.group())

            raw_types = row.get("Test Types", "")
            test_types = []
            if isinstance(raw_types, str):
                test_types = [
                    TEST_TYPE_MAP.get(t.strip(), t.strip()) for t in raw_types.split(",") if t.strip()
                ]

            recommendations.append({
                "url": link,
                "adaptive_support": "Yes" if df.iloc[idx]['adaptiveIRTSupport'].lower() == "yes" else "No",
                "description": description,
                "duration": duration,
                "remote_support": "Yes" if df.iloc[idx]['remoteTestingSupport'].lower() == "yes" else "No",
                "test_type": test_types
            })

        processing_time_ms = (time.time() - start_time) * 1000

        response_obj = {
            "status": "success",
            "recommended_assessments": recommendations,
            "processing_time_ms": processing_time_ms
        }

        pretty_json = json.dumps(response_obj, indent=2, ensure_ascii=False)
        return Response(content=pretty_json, media_type="application/json", status_code=200)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        response_obj = {
            "status": "error",
            "message": str(e),
            "recommended_assessments": []
        }
        pretty_json = json.dumps(response_obj, indent=2, ensure_ascii=False)
        return Response(content=pretty_json, media_type="application/json", status_code=500)

@app.on_event("startup")
async def startup_event():
    import threading
    thread = threading.Thread(target=initialize_model)
    thread.daemon = True
    thread.start()