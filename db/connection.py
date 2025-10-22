
# db/connection.py
import logging
from typing import Optional
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from config import get_settings

logger = logging.getLogger(__name__)


class Database:
    """PostgreSQL database connection manager."""
    
    _instance: Optional['Database'] = None
    _pool: Optional[SimpleConnectionPool] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._pool is None:
            self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool."""
        settings = get_settings()
        
        try:
            self._pool = SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=settings.POSTGRES_HOST,
                port=settings.POSTGRES_PORT,
                database=settings.POSTGRES_DB,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
            )
            logger.info("✅ Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            self._pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, cursor_factory=RealDictCursor):
        """Get a cursor with automatic connection management."""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()
    
    def close_all(self):
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("Database connection pool closed")


# Global database instance
db = Database()
