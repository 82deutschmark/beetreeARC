import os
import httpx
import warnings
from openai import OpenAI

# 1. Suppress the unavoidable "Insecure Request" warnings to keep output clean
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

def force_connect():
    # 2. explicit check for the key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ CRITICAL: 'OPENAI_API_KEY' environment variable not set.")
        print("   Run: export OPENAI_API_KEY='sk-...' before running this script.")
        return

    print(f"DEBUG: Key found (starts with {api_key[:3]}...)")

    # 3. The "Nuclear" HTTP Client
    # verify=False   -> Ignored SSL/Cert errors
    # trust_env=False -> Ignores local Proxy/Network env vars that might be broken
    print("DEBUG: configuring client with verify=False, trust_env=False...")
    
    insecure_client = httpx.Client(
        verify=False, 
        trust_env=False, 
        timeout=30.0
    )

    client = OpenAI(api_key=api_key, http_client=insecure_client)

    print("\nAttempting to contact OpenAI...")
    
    try:
        # 4. Perform the request
        response = client.models.list()
        
        # 5. Process results
        all_models = [m.id for m in response.data]
        gpt5_models = sorted([m for m in all_models if "gpt-5" in m])
        codex_models = sorted([m for m in all_models if "codex" in m])

        print("\n✅ SUCCESS: Connection Established via insecure channel.")
        print("-" * 50)
        
        if gpt5_models:
            print(f"Found {len(gpt5_models)} GPT-5 variants:")
            for m in gpt5_models:
                print(f"  - {m}")
        else:
            print("No 'gpt-5' models found on this key.")

        if codex_models:
            print(f"Found {len(codex_models)} Codex variants:")
            for m in codex_models:
                print(f"  - {m}")
                
        # Print a known stable model to prove list is complete
        if "gpt-4o" in all_models:
            print("\n(Verified 'gpt-4o' is also present in list)")

    except httpx.ConnectError as e:
        print(f"\n❌ CONNECTION FAILED: {e}")
        print("Since trust_env=False, this means your machine has NO direct route to the internet.")
        print("If you are on a corporate VPN that REQUIRES a proxy, change 'trust_env=False' to 'True'.")
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {type(e).__name__}")
        print(str(e))

if __name__ == "__main__":
    force_connect()
