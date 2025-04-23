from fastapi import FastAPI
from services.postRecommender import get_recommended_posts
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from any origin (or specify exact origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # exp://192.168.107.114:8081  Change "*" to your local app's origin if you want to restrict it
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)



@app.get("/postRecommendations")
async def get_post_recommendations():
    recommender = get_recommended_posts()
    return await recommender



