import os  
import time  
import subprocess  
import glob  

WANDB_DIR = "./wandb" 

def sync_wandb():  
    runs = glob.glob(os.path.join(WANDB_DIR, "offline-run-*"))   
    for run in runs:
        print(f"Syncing run: {run}")
        try:  
            sync_command = f"wandb sync {run}"  
            result = subprocess.run(sync_command, shell=True, check=True,  
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout)  
        except subprocess.CalledProcessError as e:  
            print(f"Error syncing {run}: {e}")  
            print(e.stderr)     

def main():  
    while True:  
        print("Starting sync process...")  
        sync_wandb()  
        print("Sync complete. Waiting for 30 seconds...")  
        time.sleep(30)

if __name__ == "__main__":  
    main()