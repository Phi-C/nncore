import os
import pandas as pd
import wandb
from tqdm import tqdm  # for progress tracking

# adapted from https://gist.github.com/self-supervisor/93461044737df8b9c79228afe824c8f3#file-backup_wandb-py
def export_wandb_data(username="your_wandb_user_name"):
    """Export all W&B project run data including metrics, configs, and files."""
    api = wandb.Api()
    
    try:
        projects = api.projects()
    except Exception as e:
        print(f"Failed to fetch projects: {e}")
        return

    for project in tqdm(projects, desc="Processing projects"):
        try:
            project_dir = project.name.replace("/", "_")  # Sanitize directory name
            os.makedirs(project_dir, exist_ok=True)
            
            runs_data = []
            runs = api.runs(f"{username}/{project.name}")
            
            for run in tqdm(runs, desc=f"Processing runs in {project.name}", leave=False):
                try:
                    # Get run data
                    run_history = run.history(samples=1e6)  # Large number to get all data
                    run_summary = run.summary._json_dict
                    run_config = {k: v for k, v in run.config.items() 
                                  if not k.startswith('_')}
                    
                    # Save run files
                    run_dir = os.path.join(project_dir, run.name.replace("/", "_"))
                    os.makedirs(run_dir, exist_ok=True)
                    
                    for file in tqdm(run.files(), desc="Downloading files", leave=False):
                        file.download(root=run_dir, exist_ok=True)
                    
                    # Save history as separate CSV
                    if not run_history.empty:
                        history_path = os.path.join(run_dir, "history.csv")
                        run_history.to_csv(history_path, index=False)
                    
                    runs_data.append({
                        "name": run.name,
                        "summary": run_summary,
                        "config": run_config,
                        "history_path": os.path.join(run_dir, "history.csv") if not run_history.empty else None
                    })
                
                except Exception as run_error:
                    print(f"Error processing run {run.name}: {run_error}")
                    continue
            
            # Save project metadata
            if runs_data:
                pd.DataFrame(runs_data).to_csv(
                    os.path.join(project_dir, f"{project_dir}.csv"),
                    index=False
                )
        
        except Exception as project_error:
            print(f"Error processing project {project.name}: {project_error}")
            continue

if __name__ == "__main__":
    export_wandb_data()
