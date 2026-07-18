import asyncio
import logging
import sys
from nemoguardrails import RailsConfig, LLMRails

# Enable logging to see exactly where NeMo is hanging
logging.basicConfig(level=logging.INFO)

async def test_medgemma():
    print("[*] Loading Config...")
    try:
        # Update path if necessary
        config = RailsConfig.from_path("./config")
        rails = LLMRails(config)
    except Exception as e:
        print(f"[!] Initialization Error: {e}")
        return

    print("[*] Sending message to Guarded MedGemma...")
    
    # We use a timeout to ensure the script doesn't hang forever
    try:
        # Use a simple greeting first to verify the pipeline
        response = await rails.generate_async(messages=[{
            "role": "user",
            "content": "Hello MedGemma, can you tell me what aspirin is used for?"
        }])
        
        print("\n" + "="*30)
        print(f"BOT OUTPUT: {response['content']}")
        print("="*30)
        
        if not response['content']:
            print("[!] Received empty response. Check if Ollama is currently loading the model (ollama ps).")
            
    except Exception as e:
        print(f"[!] Runtime Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_medgemma())