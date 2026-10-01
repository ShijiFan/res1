import json
from datetime import datetime, timezone
import hyp3_sdk

def check_existing_jobs():
    hyp3 = hyp3_sdk.HyP3()
    # Check remaining credits
    credits = hyp3.check_credits()
    print(f"Current remaining credits: {credits}")
    
    # Find recent jobs (e.g. running or pending or completed in the last 30 days)
    batch = hyp3.find_jobs()
    print(f"Total jobs found in account history: {len(batch)}")
    
    recent_jobs = []
    for job in batch[:20]:
        recent_jobs.append({
            "job_id": job.job_id,
            "name": job.name,
            "job_type": job.job_type,
            "status_code": job.status_code,
            "request_time": str(job.request_time),
            "expiration_time": str(job.expiration_time),
        })
    print(json.dumps(recent_jobs, indent=2))

if __name__ == "__main__":
    check_existing_jobs()
