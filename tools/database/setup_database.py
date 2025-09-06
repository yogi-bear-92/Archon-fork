#!/usr/bin/env python3
"""
Database Setup Script for Archon
Executes the complete_setup.sql script to initialize the database schema
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "python" / "src"))

from dotenv import load_dotenv
import asyncio
import asyncpg

async def setup_database():
    """Execute the complete database setup script"""
    
    # Load environment variables
    load_dotenv()
    
    # Get database credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_KEY environment variables")
        return False
    
    # Extract database connection details from Supabase URL
    # Format: https://project.supabase.co
    # We need: postgresql://postgres:[password]@db.project.supabase.co:5432/postgres
    if not supabase_url.startswith("https://"):
        print("❌ Invalid SUPABASE_URL format")
        return False
    
    # Convert Supabase URL to PostgreSQL connection string
    project_id = supabase_url.replace("https://", "").replace(".supabase.co", "")
    db_host = f"db.{project_id}.supabase.co"
    
    # For now, we'll use the service key as the password
    # In production, you'd want to use the actual database password
    conn_string = f"postgresql://postgres:{supabase_key}@{db_host}:5432/postgres"
    
    try:
        print("🔌 Connecting to database...")
        conn = await asyncpg.connect(conn_string)
        
        print("📖 Reading complete_setup.sql...")
        setup_sql = Path("migration/complete_setup.sql").read_text()
        
        print("🚀 Executing database setup...")
        await conn.execute(setup_sql)
        
        print("✅ Database setup completed successfully!")
        
        # Verify tables were created
        print("🔍 Verifying tables...")
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'archon_%'
            ORDER BY table_name
        """)
        
        print("📋 Created tables:")
        for table in tables:
            print(f"  - {table['table_name']}")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(setup_database())
    sys.exit(0 if success else 1)
