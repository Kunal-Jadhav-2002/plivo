import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def init_db():
    # Connect to the database
    conn = psycopg2.connect(
        dsn=os.getenv('DATABASE_URL'),
        sslmode='require'
    )
    
    try:
        with conn.cursor() as cur:
            # Create users table if it doesn't exist
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255) NOT NULL,
                    email VARCHAR(255) NOT NULL,
                    phone_number VARCHAR(20) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    init_db() 