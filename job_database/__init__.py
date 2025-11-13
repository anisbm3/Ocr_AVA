"""
Job Database Operations Module
Handles job database operations using ChromaDB
"""
import os
import json
import chromadb
from uuid import uuid4
from typing import Dict, Any, List

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient()
try:
    collection = chroma_client.create_collection(name="my_collection")
except:
    collection = chroma_client.get_collection(name="my_collection")


def add_new_job_to_db(job_data: str) -> str:
    """
    Add a new job to the database

    Args:
        job_data: JSON string containing job information

    Returns:
        Success message with job ID
    """
    # create unique job id
    job_id = str(uuid4())
    # parse json
    job = json.loads(job_data)
    # add job to database (simulated)
    collection.add(
        ids=[job_id],
        documents=[job_data],
        metadatas=[
            {
                "title": job.get("title", ""),
                "companyName": job.get("companyName", ""),
                "companyId": job.get("companyId", ""),
                "location": job.get("location", ""),
                "type": job.get("type", ""),
                "description": job.get("description", ""),
                "requirements": job.get("requirements", ""),
                "responsibilities": job.get("responsibilities", ""),
                "benefits": job.get("benefits", ""),
                "salaryMin": job.get("salaryMin", ""),
                "salaryMax": job.get("salaryMax", ""),
                "currency": job.get("currency", ""),
                "experienceLevel": job.get("experienceLevel", ""),
                "remote": job.get("remote", ""),
                "skills": ", ".join(job.get("skills", [])) if isinstance(job.get("skills", []), list) else job.get("skills", ""),
                "expiresAt": job.get("expiresAt", ""),
                "slug": job.get("slug", ""),
                "externalUrl": job.get("externalUrl", ""),
                "source": job.get("source", ""),
            }
        ],
    )
    return "Inserted job with ID: " + job_id


def match_cv_with_jobs(cv: str, n_results: int = 5) -> Dict[str, Any]:
    """
    Match CV against jobs in database

    Args:
        cv: CV text to match
        n_results: Number of results to return

    Returns:
        Query results from ChromaDB
    """
    results = collection.query(
        query_texts=[cv],
        n_results=n_results,
    )
    return results


def get_entire_collection() -> Dict[str, Any]:
    """
    Get all jobs from the database

    Returns:
        All documents and metadata from collection
    """
    return collection.get()


def delete_job_from_db(job_id: str) -> str:
    """
    Delete a job from the database

    Args:
        job_id: ID of the job to delete

    Returns:
        Success message
    """
    collection.delete(ids=[job_id])
    return f"Deleted job with ID: {job_id}"