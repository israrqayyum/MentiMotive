from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils.prompts import SYSTEM_PROMPT_TEMPLATE, build_emotion_context
from services.rag_service import rag_service
from config import settings
import logging
import time

logger = logging.getLogger(__name__)

class LangChainService:
    """Manages LangChain conversation with support for Gemini and OpenAI."""

    def __init__(self):
        self.llm = None
        self.provider = None
        self.model_name = None

    def initialize(self):
        """Initialize LLM based on configured provider."""
        if self.llm is not None:
            return

        logger.info("Initializing LangChain service...")

        # Get configuration from config.py
        provider = settings.LLM_PROVIDER.lower()
        temperature = settings.LLM_TEMPERATURE

        self.provider = provider

        if provider == "gemini":
            api_key = settings.GEMINI_API_KEY
            model = settings.GEMINI_MODEL

            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")

            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=api_key
            )
            self.model_name = model
            logger.info(f"✅ LangChain service initialized with Gemini (model: {model})")

        elif provider == "openai":
            api_key = settings.OPENAI_API_KEY
            model = settings.OPENAI_MODEL

            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=api_key
            )
            self.model_name = model
            logger.info(f"✅ LangChain service initialized with OpenAI (model: {model})")

        else:
            raise ValueError(f"Invalid LLM_PROVIDER: {provider}. Must be 'gemini' or 'openai'")

    def chat(
        self,
        user_message: str,
        emotions: dict,
        history: list,
        use_rag: bool = True
    ) -> tuple[str, list]:
        """Generate chat response."""
        self.initialize()

        # Get RAG context
        rag_context = ""
        sources = []
        if use_rag:
            rag_context, sources = rag_service.retrieve_context(user_message)

        # Build emotion context
        emotion_context = build_emotion_context(emotions)

        # Build system prompt
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            emotion_context=emotion_context,
            rag_context=rag_context
        )

        # Convert history to LangChain messages
        messages = [SystemMessage(content=system_prompt)]

        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Add current user message
        messages.append(HumanMessage(content=user_message))

        # Generate response
        try:
            t_start = time.time()
            response = self.llm.invoke(messages)
            t_end = time.time()

            provider_name = self.provider.upper()
            logger.info(f"   🤖 {provider_name} ({self.model_name}) Response: {round((t_end - t_start) * 1000, 2)} ms")

            return response.content, sources
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise

# Global instance
langchain_service = LangChainService()
