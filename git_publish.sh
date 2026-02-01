#!/bin/bash
# Git publish script - uploads project to GitHub (excluding checkpoints)

set -e

REPO_URL="https://github.com/NZ5253/Master-Thesis.git"
USER_EMAIL="naeem.zainuddin@tu-dortmund.de"
USER_NAME="NZ5253"

echo "========================================="
echo "Publishing to GitHub"
echo "Repository: $REPO_URL"
echo "========================================="

# Configure git user
git config --global user.email "$USER_EMAIL"
git config --global user.name "$USER_NAME"
echo "Git configured for: $USER_NAME <$USER_EMAIL>"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git branch -M main
fi

# Add remote if not exists
if ! git remote | grep -q "origin"; then
    echo "Adding remote origin..."
    git remote add origin "$REPO_URL"
else
    echo "Remote origin already exists"
    git remote set-url origin "$REPO_URL"
fi

# Show what will be ignored
echo ""
echo "Files/directories that will be EXCLUDED (from .gitignore):"
echo "  - checkpoints/ (model files)"
echo "  - venv/ (virtual environment)"
echo "  - __pycache__/ (Python cache)"
echo "  - logs/ (log files)"
echo "  - ray_results/ (Ray training results)"
echo ""

# Stage all files
echo "Staging files..."
git add -A

# Show status
echo ""
echo "Files to be committed:"
git status --short

# Commit
echo ""
COMMIT_MSG="Parallel Parking RL - Complete curriculum training system ($(date '+%Y-%m-%d'))"
echo "Commit message: $COMMIT_MSG"

git commit -m "$COMMIT_MSG" || echo "Nothing to commit"

# Push
echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "========================================="
echo "Successfully published to:"
echo "$REPO_URL"
echo "========================================="
