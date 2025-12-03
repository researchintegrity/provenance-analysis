#!/bin/bash
# Quick test script for provenance analysis

# Configuration - EDIT THESE PATHS
# IMAGE_DIR="/path/to/your/images"  # Your image directory
IMAGE_DIR="/media/jcardenuto/Windows/Users/phill/work/2025-elis-system/cbir-workspace"  # Your image directory
CBIR_PORT=8001
PROVENANCE_PORT=8002

echo "Testing Provenance Analysis API..."
echo "=================================="

# Check CBIR health
echo "1. Checking CBIR service..."
if curl -s http://localhost:$CBIR_PORT/health | grep -q "healthy"; then
    echo "✅ CBIR is healthy"
else
    echo "❌ CBIR is not responding. Start CBIR first:"
    echo "   cd ../cbir-system && docker-compose up -d"
    exit 1
fi

# Check Provenance health
echo "2. Checking Provenance service..."
if curl -s http://localhost:$PROVENANCE_PORT/health | grep -q "healthy"; then
    echo "✅ Provenance service is healthy"
else
    echo "❌ Provenance service is not responding. Start it first:"
    echo "   docker-compose up -d"
    exit 1
fi

# Find some test images
echo "3. Finding test images..."
if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ Image directory not found: $IMAGE_DIR"
    echo "   Please edit IMAGE_DIR in this script"
    exit 1
fi

# Get first 10 PNG files
IMAGES=($(find "$IMAGE_DIR" -name "*.png" | head -10))
if [ ${#IMAGES[@]} -lt 10 ]; then
    echo "❌ Need at least 10 PNG images in $IMAGE_DIR"
    exit 1
fi

echo "✅ Found ${#IMAGES[@]} test images"

# Create JSON request
INPUT_IMAGES_JSON=""
for i in "${!IMAGES[@]}"; do
    IMAGE_ID="img$((i+1))"
    IMAGE_PATH="/workspace/$(realpath --relative-to="$IMAGE_DIR" "${IMAGES[i]}" | sed 's|\\|/|g')"
    INPUT_IMAGES_JSON+=$(cat <<EOF
    {"id": "$IMAGE_ID", "path": "$IMAGE_PATH", "label": "Prov-test"}$( [ $i -lt $((${#IMAGES[@]}-1)) ] && echo "," )
EOF
)
done

# Get query image (first image) as a proper JSON object
QUERY_ID="img1"
QUERY_PATH="/workspace/$(realpath --relative-to="$IMAGE_DIR" "${IMAGES[0]}" | sed 's|\\|/|g')"

JSON=$(cat <<EOF
{
  "user_id": "provenance_test_user",
  "images": [$INPUT_IMAGES_JSON],
  "query_image": {"id": "$QUERY_ID", "path": "$QUERY_PATH", "label": "Prov-test"},
  "k": 10,
  "q": 5,
  "max_depth": 3,
  "descriptor_type": "cv_rsift",
  "extract_flip": true,
  "alignment_strategy": "CV_MAGSAC",
  "matching_method": "BF",
  "min_keypoints": 20,
  "min_area": 0.01,
  "max_workers": 2,
  "cbir_batch_size": 32
}
EOF
)

echo "4. Running provenance analysis..."
echo "   Query image: $(realpath --relative-to="$IMAGE_DIR" "${IMAGES[0]}" | sed 's|\\|/|g')"
echo "   Processing..."

# Run the analysis
RESPONSE=$(curl -s -X POST http://localhost:$PROVENANCE_PORT/analyze \
  -H "Content-Type: application/json" \
  -d "$JSON")

# Check if successful
if echo "$RESPONSE" | grep -q '"success":true'; then
    echo "✅ Analysis completed successfully!"
    echo "   Processing time: $(echo "$RESPONSE" | grep -o '"processing_time_seconds":[0-9.]*' | cut -d: -f2)s"
    echo "   Matched pairs: $(echo "$RESPONSE" | grep -o '"matched_pairs":[0-9]*' | cut -d: -f2)"
    echo ""
    echo "Full response saved to: test_response.json"
    echo "$RESPONSE" > test_response.json
else
    echo "❌ Analysis failed:"
    echo "$RESPONSE"
fi