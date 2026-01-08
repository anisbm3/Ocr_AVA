#!/usr/bin/env python3
"""
Job Database CLI Interface
Provides command-line access to job database operations
"""
import sys
import json
from job_database import (
    add_new_job_to_db,
    match_cv_with_jobs,
    get_entire_collection,
    delete_job_from_db
)

def main():
    if len(sys.argv) < 2:
        print("Usage: python job_db_cli.py <command> [args...]")
        print("Commands:")
        print("  add-job <job_json>")
        print("  match-cv <cv_text> [n_results]")
        print("  get-all-jobs")
        print("  delete-job <job_id>")
        sys.exit(1)

    command = sys.argv[1]

    try:
        if command == "add-job":
            if len(sys.argv) < 3:
                print("Error: job_json required")
                sys.exit(1)
            job_json = sys.argv[2]
            result = add_new_job_to_db(job_json)
            print(json.dumps({"success": True, "message": result}))

        elif command == "match-cv":
            if len(sys.argv) < 3:
                print("Error: cv_text required")
                sys.exit(1)
            cv_text = sys.argv[2]
            n_results = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            result = match_cv_with_jobs(cv_text, n_results)
            print(json.dumps({"success": True, "data": result}))

        elif command == "get-all-jobs":
            result = get_entire_collection()
            print(json.dumps({"success": True, "data": result}))

        elif command == "delete-job":
            if len(sys.argv) < 3:
                print("Error: job_id required")
                sys.exit(1)
            job_id = sys.argv[2]
            result = delete_job_from_db(job_id)
            print(json.dumps({"success": True, "message": result}))

        else:
            print(f"Error: Unknown command {command}")
            sys.exit(1)

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\anisb\OneDrive\Desktop\Nouveau dossier\python\job_db_cli.py