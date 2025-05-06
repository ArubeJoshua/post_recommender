from fastapi import HTTPException
from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.query import Query
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

# Initialize Appwrite client
def initialize_appwrite_client():
    """Configure Appwrite client"""
    client = Client()
    client.set_endpoint(os.getenv("APPWRITE_ENDPOINT", "https://cloud.appwrite.io/v1"))
    client.set_project(os.getenv("APPWRITE_PROJECT_ID"))
    client.set_key(os.getenv("APPWRITE_API_KEY"))
    return client, Databases(client)

client, databases = initialize_appwrite_client()

def validate_environment_vars():
    """Verify required environment variables exist"""
    required_vars = [
        "APPWRITE_PROJECT_ID",
        "APPWRITE_DATABASE_ID",
        "APPWRITE_POST_COLLECTION_ID",
        "APPWRITE_RESPONSES_COLLECTION_ID",
        "APPWRITE_REACTIONS_COLLECTION_ID"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

validate_environment_vars()



async def fetch_data(collection: str, queries: Optional[List[Query]] = None) -> List[Dict]:
    """Generic method to fetch documents from any collection"""
    try:
        # Initialize queries if not provided
        if queries is None:
            queries = []
            
        result = databases.list_documents(
            os.getenv("APPWRITE_DATABASE_ID"),
            os.getenv(f"APPWRITE_{collection.upper()}_COLLECTION_ID"),
            queries=queries
        )
        return result["documents"]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch {collection}: {str(e)}"
        )




async def get_all_data(post_limit: int = 5) -> Dict[str, pd.DataFrame]:
    """Fetch all required data concurrently with post limit"""
    try:
        posts, responses, reactions = await asyncio.gather(
            fetch_data("post", [
                Query.order_desc("$createdAt"),
                Query.limit(post_limit)  # Add limit using Query
            ]),
            fetch_data("responses"),
            fetch_data("reactions")
        )
        
        return {
            "posts": convert_to_dataframe(posts),
            "responses": convert_to_dataframe(responses),
            "reactions": convert_to_dataframe(reactions)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Data collection failed: {str(e)}"
        )


def convert_to_dataframe(data: List[Dict]) -> pd.DataFrame:
    """Convert document list to DataFrame with error handling"""
    if not data:
        return pd.DataFrame()
    try:
        return pd.DataFrame(data)
    except Exception as e:
        raise ValueError(f"Data conversion failed: {str(e)}")

def prepare_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Clean and merge the raw data with validation"""
    try:
        # Validate input data
        for key in ["posts", "responses", "reactions"]:
            if key not in data or not isinstance(data[key], pd.DataFrame):
                raise ValueError(f"Missing or invalid data for {key}")
        
        responses = process_responses(data["responses"])
        reactions = process_reactions(data["reactions"])
        posts = process_posts(data["posts"])
        
        merged = merge_response_reactions(responses, reactions)
        
        return {
            "merged_responses": merged,
            "posts": posts,
            "responded_post_ids": set(responses['postId'])
        }
    except Exception as e:
        raise ValueError(f"Data preparation failed: {str(e)}")

def process_responses(responses: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform responses data"""
    cols_to_drop = ['content', '$updatedAt', '$permissions', '$databaseId', '$collectionId']
    responses = responses.drop(columns=[col for col in cols_to_drop if col in responses.columns])
    
    if '$id' in responses.columns:
        responses = responses[responses['$id'] != "67e02f9a00217f9c641e"]  # Filter specific response if needed
    
    # Extract nested fields safely
    responses['post_createdAt'] = responses['postId'].apply(
        lambda x: x.get('$createdAt') if isinstance(x, dict) else None
    )
    responses['postId'] = responses['postId'].apply(
        lambda x: x.get('$id') if isinstance(x, dict) else x
    )
    responses['userRole'] = responses['userId'].apply(
        lambda x: x.get('role') if isinstance(x, dict) else 'student'  # Default to student
    )
    
    return responses

def process_reactions(reactions: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform reactions data"""
    cols_to_drop = [
        '$createdAt', '$updatedAt', '$permissions', 
        'madeBy', '$databaseId', '$collectionId', '$id'
    ]
    reactions = reactions.drop(columns=[col for col in cols_to_drop if col in reactions.columns])
    
    reactions['reactedTo'] = reactions['reactedTo'].apply(
        lambda x: x.get('$id') if isinstance(x, dict) else x
    )
    return reactions

def process_posts(posts: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform posts data"""
    cols_to_drop = ['content', '$updatedAt', '$permissions', '$databaseId', '$collectionId']
    posts = posts.drop(columns=[col for col in cols_to_drop if col in posts.columns])
    
    posts['userRole'] = posts['userId'].apply(
        lambda x: x.get('role') if isinstance(x, dict) else 'student'  # Default to student
    )
    return posts

def merge_response_reactions(responses: pd.DataFrame, reactions: pd.DataFrame) -> pd.DataFrame:
    """Merge responses and reactions data"""
    merged = responses.merge(
        reactions,
        left_on='$id',
        right_on='reactedTo',
        how='left'
    )
    if 'reactedTo' in merged.columns:
        merged = merged.drop(columns=['reactedTo'])
    merged['reactionType'] = merged['reactionType'].fillna('None')
    return merged.rename(columns={
        '$id': 'responseId',
        '$createdAt': 'response_createdAt'
    })

def rank_posts(clean_data: Dict) -> Dict[str, pd.DataFrame]:
    """Rank both responded and unresponded posts"""
    try:
        ranked = prepare_response_data(clean_data["merged_responses"])
        unresponded = clean_data["posts"][
            ~clean_data["posts"]['$id'].isin(clean_data["responded_post_ids"])
        ]
        ranked_unresponded = rank_unresponded_posts(unresponded)
        
        return {
            "ranked_posts": ranked,
            "ranked_unresponded_posts": ranked_unresponded
        }
    except Exception as e:
        raise ValueError(f"Ranking failed: {str(e)}")

def prepare_response_data(df: pd.DataFrame) -> pd.DataFrame:
    """Score posts with responses, incorporating response count"""
    df = df.copy()
    
    # Convert datetimes with error handling
    datetime_cols = ['response_createdAt', 'post_createdAt']
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
            if df[col].isna().any():
                print(f"Warning: {df[col].isna().sum()} {col} values couldn't be converted")
    
    # Calculate features
    df['is_counselor'] = df['userRole'].eq('counselor').astype(int)
    df['reaction_score'] = df['reactionType'].map({'like': 1, 'dislike': -1}).fillna(0)
    
    if 'post_createdAt' in df.columns:
        df['post_age_days'] = (datetime.now(timezone.utc) - df['post_createdAt']).dt.total_seconds() / 86400
    else:
        df['post_age_days'] = 0  # Default value if missing
    
    # Aggregate and score - now including response_count in the aggregation
    features = (
        df.groupby('postId', observed=True)
        .agg(
            response_count=('responseId', 'count'),  # Count of responses per post
            counselor_responses=('is_counselor', 'sum'),
            net_reaction_score=('reaction_score', 'sum'),
            post_age_days=('post_age_days', 'min')
        )
        .reset_index()
        .dropna()
    )
    
    # Enhanced scoring formula incorporating response count
    features['composite_score'] = (
        features['counselor_responses'] * 4 +       # Counselor responses are most valuable
        features['response_count'] * 1.5 +            # More responses = more engagement
        features['net_reaction_score'] * 3 +        # Reactions indicate quality
        (10 - features['post_age_days'].clip(0, 2)) * 2  # Fresher posts get slight boost
    )
    
    print("Post ranking features with response counts:")
    print(features.head())
    return features.sort_values('composite_score', ascending=False)

def rank_unresponded_posts(df: pd.DataFrame) -> pd.DataFrame:
    """Score unresponded posts"""
    df = df.copy()
    
    if '$createdAt' in df.columns:
        df['$createdAt'] = pd.to_datetime(df['$createdAt'], utc=True, errors='coerce')
        df = df.dropna(subset=['$createdAt'])
        df['post_age_days'] = (datetime.now(timezone.utc) - df['$createdAt']).dt.total_seconds() / 86400
    else:
        df['post_age_days'] = 0  # Default value if missing
    
    df['role_priority'] = df['userRole'].map({'counselor': 2, 'student': 1}).fillna(1)
    
    df['composite_score'] = (
        (10 - df['post_age_days'].clip(0, 2)) * 2 +
        df['role_priority'] * 2
    )
    # print(df)
    return df.sort_values('composite_score', ascending=False)


def mix_posts(
    ranked_posts: pd.DataFrame, 
    unresponded_posts: pd.DataFrame, 
    ranked_proportion: float = 0.6,
    shuffle: bool = True,
    min_posts: int = 60
) -> List[str]:

    if not 0 <= ranked_proportion <= 1:
        raise ValueError("Proportion must be between 0 and 1")
    
    # Ensure DataFrames are properly sorted
    ranked_posts = ranked_posts.sort_values('composite_score', ascending=False)
    unresponded_posts = unresponded_posts.sort_values('composite_score', ascending=False)
    
    # Calculate numbers needed for each type
    n_ranked = max(1, int(min_posts * ranked_proportion))
    n_unresponded = max(1, min_posts - n_ranked)
    
    # Adjust if we don't have enough posts
    n_ranked = min(n_ranked, len(ranked_posts))
    n_unresponded = min(n_unresponded, len(unresponded_posts))
    
    # Select posts and gather score information
    ranked_selected = ranked_posts.head(n_ranked)
    unresponded_selected = unresponded_posts.head(n_unresponded)
    
    # Create debug info
    debug_info = {
        'ranked_posts': ranked_selected[['postId', 'composite_score']].to_dict('records'),
        'unresponded_posts': unresponded_selected[['$id', 'composite_score']].to_dict('records'),
        'parameters': {
            'ranked_proportion': ranked_proportion,
            'shuffle': shuffle,
            'min_posts': min_posts
        }
    }
    
    # Create mixed list
    mixed = (
        ranked_selected['postId'].tolist() +
        unresponded_selected['$id'].tolist()
    )
    
    if shuffle:
        np.random.shuffle(mixed)
    
    print("\n=== POST MIXING DEBUG INFO ===")
    print(f"Top {n_ranked} Ranked Posts (score | postId):")
    for idx, row in ranked_selected.iterrows():
        print(f"{row['composite_score']:.2f} | {row['postId']}")
    
    print(f"\nTop {n_unresponded} Unresponded Posts (score | postId):")
    for idx, row in unresponded_selected.iterrows():
        print(f"{row['composite_score']:.2f} | {row['$id']}")
    
    print(f"\nFinal mixed order (shuffle={shuffle}):")
    # print(mixed)
    
    return mixed



async def get_recommended_posts() -> List[str]:
    """Main method to get final recommended posts"""
    try:
        raw_data = await get_all_data()
        clean_data = prepare_data(raw_data)
        ranked_data = rank_posts(clean_data)
        return mix_posts(
            ranked_data["ranked_posts"],
            ranked_data["ranked_unresponded_posts"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation generation failed: {str(e)}"
        )

