#!/usr/bin/env python3
"""
🐳 POSTGRESQL DOCKER SETUP - For Phase 5 Database Layer
Quick setup script for running PostgreSQL in Docker for development.
"""

import subprocess
import sys
import time
from pathlib import Path

def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ Docker found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker not found or not running")
        print("\n💡 INSTALLATION OPTIONS:")
        print("   1. Install Docker Desktop: https://www.docker.com/products/docker-desktop/")
        print("   2. Or install PostgreSQL directly: https://www.postgresql.org/download/")
        return False

def start_postgresql_container():
    """Start PostgreSQL container for Security Nervous System"""
    container_name = "security-db"
    
    # Check if container already exists
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if container_name in result.stdout:
            # Container exists, check if running
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if container_name in result.stdout:
                print(f"✅ Container '{container_name}' is already running")
                return True
            else:
                print(f"⚠️  Container '{container_name}' exists but not running, starting...")
                subprocess.run(["docker", "start", container_name], check=True)
                print(f"✅ Started container '{container_name}'")
                return True
        else:
            # Create new container
            print(f"🚀 Creating new PostgreSQL container '{container_name}'...")
            
            subprocess.run([
                "docker", "run", 
                "--name", container_name,
                "-e", "POSTGRES_PASSWORD=postgres",
                "-e", "POSTGRES_USER=postgres",
                "-e", "POSTGRES_DB=security_nervous_system",
                "-p", "5432:5432",
                "-d",
                "--restart", "unless-stopped",
                "postgres:15-alpine"
            ], check=True)
            
            print(f"✅ Created container '{container_name}'")
            
            # Wait for PostgreSQL to start
            print("⏳ Waiting for PostgreSQL to start (15 seconds)...")
            time.sleep(15)
            
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker command failed: {e}")
        return False

def test_database_connection():
    """Test connection to PostgreSQL database"""
    print("\n🧪 TESTING DATABASE CONNECTION...")
    
    test_script = Path(__file__).parent / "test_database.py"
    if test_script.exists():
        result = subprocess.run([sys.executable, str(test_script)], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"⚠️  Errors: {result.stderr}")
        return result.returncode == 0
    else:
        print("⚠️  test_database.py not found")
        return False

def update_database_config():
    """Update database config for Docker setup"""
    config_file = Path(__file__).parent / "database" / "config.py"
    
    if not config_file.exists():
        print(f"⚠️  Config file not found: {config_file}")
        return False
    
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Update for Docker setup
        new_content = content.replace(
            'host: str = os.getenv("DB_HOST", "localhost")',
            'host: str = os.getenv("DB_HOST", "localhost")  # Use "host.docker.internal" if running in Docker'
        )
        
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        print("✅ Updated database config with Docker notes")
        
        # Also create environment file
        env_file = Path(__file__).parent / ".env.database"
        with open(env_file, "w", encoding="utf-8") as f:
            f.write("# PostgreSQL Database Configuration\n")
            f.write("DB_HOST=localhost\n")
            f.write("DB_PORT=5432\n")
            f.write("DB_NAME=security_nervous_system\n")
            f.write("DB_USER=postgres\n")
            f.write("DB_PASSWORD=postgres\n")
        
        print(f"✅ Created environment file: {env_file}")
        print("💡 To use these settings: source .env.database or set environment variables")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        return False

def main():
    """Main Docker setup routine"""
    print("\n" + "="*80)
    print("🐳 PHASE 5 - POSTGRESQL DOCKER SETUP")
    print("="*80)
    
    # Check Docker
    if not check_docker():
        print("\n⚠️  Docker setup skipped, using mock database mode")
        print("💡 You can still proceed with mock database for development")
        return False
    
    # Start PostgreSQL container
    if not start_postgresql_container():
        print("\n⚠️  Failed to start PostgreSQL container")
        print("💡 Using mock database mode instead")
        return False
    
    # Update database config
    update_database_config()
    
    # Test connection
    connection_ok = test_database_connection()
    
    print("\n" + "="*80)
    if connection_ok:
        print("✅ POSTGRESQL DOCKER SETUP COMPLETE")
        print("\n📋 DATABASE INFORMATION:")
        print("   Host: localhost:5432")
        print("   Database: security_nervous_system")
        print("   Username: postgres")
        print("   Password: postgres")
        
        print("\n🚀 NEXT STEPS:")
        print("   1. Initialize database: python database/init_database.py")
        print("   2. Run Phase 5.1: python execute_phase5.py")
        print("   3. Test API integration: python api_enterprise.py")
    else:
        print("⚠️  POSTGRESQL SETUP NEEDS ATTENTION")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Check Docker is running: docker ps")
        print("   2. Check container logs: docker logs security-db")
        print("   3. Try restarting: docker restart security-db")
        print("   4. Or continue with mock database mode")
    
    print("\n" + "="*80)
    return connection_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
