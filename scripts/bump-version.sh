#!/bin/bash

# Simple version bumping script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get current version from git tags
CURRENT_VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")
echo -e "${GREEN}Current version: ${CURRENT_VERSION}${NC}"

# Parse version components
VERSION_WITHOUT_V=${CURRENT_VERSION#v}
IFS='.' read -ra VERSION_PARTS <<< "$VERSION_WITHOUT_V"
MAJOR=${VERSION_PARTS[0]:-0}
MINOR=${VERSION_PARTS[1]:-0}
PATCH=${VERSION_PARTS[2]:-0}

# Determine bump type
BUMP_TYPE=${1:-patch}

case $BUMP_TYPE in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
    *)
        echo -e "${RED}Invalid bump type: $BUMP_TYPE${NC}"
        echo "Usage: $0 [major|minor|patch]"
        exit 1
        ;;
esac

# Create new version
NEW_VERSION="v${MAJOR}.${MINOR}.${PATCH}"
echo -e "${YELLOW}New version: ${NEW_VERSION}${NC}"

# Confirm with user
read -p "Create and push tag ${NEW_VERSION}? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Create and push tag
git tag -a "${NEW_VERSION}" -m "Release ${NEW_VERSION}"
echo -e "${GREEN}Created tag: ${NEW_VERSION}${NC}"

# Push tag
git push origin "${NEW_VERSION}"
echo -e "${GREEN}Pushed tag to origin${NC}"

echo -e "${GREEN}✓ Version bumped successfully!${NC}"
echo "The release workflow will automatically create binaries for:"
echo "  • Linux (amd64, arm64)"
echo "  • macOS (amd64, arm64)"
echo "  • Windows (amd64)"