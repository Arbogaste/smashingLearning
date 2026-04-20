import os
import sys
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

# OpenRouter Wave 0 test experiments

base_dir = os.path.dirname(os.path.abspath(__file__))

# Load environments
load_dotenv(os.path.join(base_dir, ".env"))

# Gather all OpenRouter Keys
OPENROUTER_KEYS = []
for k, v in os.environ.items():
    if k.startswith("OPENROUTER_KEY") and v:
        OPENROUTER_KEYS.append((k, v))

# Sort keys by name for consistent testing
OPENROUTER_KEYS.sort()

if not OPENROUTER_KEYS:
    print("Warning: No OPENROUTER_KEY* environment variables found in .env. Authentication-dependent tests may fail.")

def check_account_balance(keys):
    """
    Test 1: Fetch account balance for all provided keys
    """
    print("--- Test 1: Fetching Account Balance ---")
    if not keys:
        print("Skipping: No keys found.")
        return []

    results = []
    
    for key_name, key_val in keys:
        print(f"Testing {key_name}...")
        headers = {
            "Authorization": f"Bearer {key_val}",
            "Content-Type": "application/json"
        }
        url = "https://openrouter.ai/api/v1/auth/key"
        response = requests.get(url, headers=headers)
        
        entry = {
            "key_name": key_name,
            "timestamp": datetime.now().isoformat()
        }
        
        if response.status_code == 200:
            data = response.json().get("data", {})
            print(f"  [OK] Key Label: {data.get('label', 'N/A')}")
            print(f"       Key Usage: {data.get('usage', 0)}")
            print(f"       Key Limit: {data.get('limit', 'No Limit')}")
            print(f"       Key Limit Remaining: {data.get('limit_remaining', 'No Limit')}")
            entry["status"] = "OK"
            entry["data"] = data
            
            # Fetch Global Account Credits (quanto ho e quanto posso spendere a livello globale)
            url_credits = "https://openrouter.ai/api/v1/credits"
            res_credits = requests.get(url_credits, headers=headers)
            if res_credits.status_code == 200:
                data_credits = res_credits.json().get("data", {})
                print(f"  [OK] Global Account Credits (Total available): {data_credits.get('total_credits', 'N/A')}")
                print(f"       Global Account Usage (Total spent): {data_credits.get('total_usage', 'N/A')}")
                entry["data"]["global_total_credits"] = data_credits.get("total_credits")
                entry["data"]["global_total_usage"] = data_credits.get("total_usage")
            else:
                print(f"  [WARN] Failed to fetch global credits: {res_credits.status_code}")
                
        else:
            print(f"  [ERROR] Failed to fetch: {response.status_code} - {response.text}")
            entry["status"] = "ERROR"
            entry["error"] = response.text
            entry["status_code"] = response.status_code
            
        results.append(entry)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(base_dir, f"{timestamp}_balance_check.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"-> Saved balance test results to {filename}\n")
    return results

def fetch_models_list():
    """
    Test 2: Fetch models list
    """
    print("--- Test 2: Fetching Models List ---")
    url = "https://openrouter.ai/api/v1/models"
    response = requests.get(url)  # No auth needed for public models list
    if response.status_code == 200:
        models = response.json().get("data", [])
        print(f"[OK] Total models available: {len(models)}")
        print()
        return models
    else:
        print(f"[ERROR] Failed to fetch models list: {response.status_code}")
        print()
        return []

def select_free_models(models):
    """
    Test 3: Fetch free models and optionally save them
    """
    print("--- Test 3: Analysing Free Models ---")
    free_models = []
    for model in models:
        pricing = model.get("pricing", {})
        # A free model has cost exactly "0" or "0.0"
        p_prompt = pricing.get("prompt", "-1")
        p_comp = pricing.get("completion", "-1")
        
        try:
            if float(p_prompt) == 0.0 and float(p_comp) == 0.0:
                free_models.append(model)
        except ValueError:
            continue

    print(f"[OK] Found {len(free_models)} completely free models.")
    if len(free_models) > 0:
        ans = input("     Do you want to save the free models to a JSON file? (Y/Yes): ").strip().lower()
        if ans in ['y', 'yes']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_free_models.json"
            filepath = os.path.join(base_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(free_models, f, indent=4)
            print(f"     -> Saved free models to {filename}")
    print()

def get_specific_model(models):
    """
    Test 4: Fetch a specific model's information
    """
    print("--- Test 4: Fetch Specific Model Details ---")
    print("Enter the ID of the model you want to inspect (e.g., 'openai/gpt-3.5-turbo').")
    model_id = input("Model ID (leave blank to skip): ").strip()
    if not model_id:
        print("Skipping model inspection.")
        print()
        return
        
    found = None
    for model in models:
        if model.get("id") == model_id:
            found = model
            break
            
    if found:
        print(f"\n[OK] Model '{model_id}' properties:")
        print(json.dumps(found, indent=2))
    else:
        print(f"[ERROR] Model '{model_id}' not found in the fetched list.")
    print()

def main():
    print("Starting OpenRouter Wave 0 Experiments\n")
    check_account_balance(OPENROUTER_KEYS)
    
    #models = fetch_models_list()
    #if models:
        #select_free_models(models)
        #get_specific_model(models)
        
    print("Wave 0 Experiments completed.")

if __name__ == "__main__":
    main()
