.PHONY: help install setup test run docker-build docker-up docker-down docker-logs clean

help:
	@echo "CV Parser & Job Matcher - Available Commands"
	@echo "============================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup        - Run initial setup script"
	@echo "  make install      - Install Python dependencies"
	@echo "  make test         - Run tests"
	@echo ""
	@echo "Development:"
	@echo "  make run          - Run application directly (without Docker)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-up    - Start application in Docker"
	@echo "  make docker-down  - Stop Docker containers"
	@echo "  make docker-logs  - View Docker logs"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Remove cache files and temp data"
	@echo ""

setup:
	@echo "Running setup script..."
	@chmod +x setup.sh
	@./setup.sh

install:
	@echo "Installing Python dependencies..."
	@pip install -r requirements.txt

test:
	@echo "Running tests..."
	@chmod +x test.sh
	@./test.sh

run:
	@echo "Starting application..."
	@python3 main.py

docker-build:
	@echo "Building Docker image..."
	@docker compose build

docker-up:
	@echo "Starting Docker containers..."
	@docker compose up

docker-up-bg:
	@echo "Starting Docker containers in background..."
	@docker compose up -d

docker-down:
	@echo "Stopping Docker containers..."
	@docker compose down

docker-logs:
	@echo "Showing Docker logs..."
	@docker compose logs -f

docker-restart:
	@echo "Restarting Docker containers..."
	@docker compose restart

clean:
	@echo "Cleaning up..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@rm -rf .pytest_cache .coverage htmlcov
	@echo "✓ Cleanup complete"

env:
	@echo "Creating .env file from example..."
	@cp .env.example .env
	@echo "✓ Created .env - please edit it with your API keys"
