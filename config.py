from pydantic_settings import BaseSettings



class Groqconfig(BaseSettings):
    groq_api_key: str

    class Config:
        env_file = ".env"

groq_config = Groqconfig()