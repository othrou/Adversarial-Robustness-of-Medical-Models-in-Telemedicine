import asyncio
from nemoguardrails import RailsConfig, LLMRails

class MedGemmaTarget:
    """
    The Defender Agent in the Multi-Agent Game.
    Wraps MedGemma in a NeMo Guardrails safety layer.
    """
    def __init__(self, config_path="./config"):
        # Initialize the Rails engine with the config directory
        self.config = RailsConfig.from_path(config_path)
        self.rails = LLMRails(self.config)

    async def generate_response(self, user_prompt: str):
        """
        Processes input through the guardrail before hitting MedGemma.
        """
        response = await self.rails.generate_async(messages=[
            {"role": "user", "content": user_prompt}
        ])
        return response["content"]