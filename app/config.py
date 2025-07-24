from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    INTENT_API_URL: str
    SENTIMENT_API_URL: str
    RAG_API_URL: str
    SEARCH_FLIGHT_API_URL: str
    BOOK_FLIGHT_API_URL: str
    BAGGAGE_STATUS_API_URL: str
    CHECK_FLIGHT_OFFERS_API_URL: str
    REDIS_HOST: str
    REDIS_PORT: str
    BEDROCK_REGION: str
    BEDROCK_MODEL_ID: str

    # Optional for STS
    AWS_PROFILE: str = ""
    ASSUME_ROLE_ARN: str = ""

    class Config:
        env_file = ".env"
        extra = "allow"  # Accept unmodeled fields without failing

settings = Settings()