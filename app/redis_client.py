from redisvl.client import RedisClient
from .config import settings

# Centralized Redis client for reuse
redis_client = RedisClient(host="localhost", port=6379)
