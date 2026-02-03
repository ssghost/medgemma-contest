import os
from huggingface_hub import snapshot_download, login

def download_medgemma():
    repo_id = "google/medgemma-1.5-4b-it"
    
    local_dir = os.path.join(os.getcwd(), "models", "medgemma-1.5-4b-it")
    
    print(f"Target Model: {repo_id}")
    print(f"Download Directory: {local_dir}")
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)
    
    try:
        print("\nStarting download...")
        
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print("-" * 50)
        print("Download Complete")
        print(f"Model weights are stored in: {path}")
        
    except Exception as e:
        print(f"\nError Occurred: {e}")

if __name__ == "__main__":
    download_medgemma()