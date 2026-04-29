# Image Compression Inference & MongoDB Workflow

## Prerequisites

Make sure you have Docker and Docker Compose installed.

## Step 1: Start MongoDB Services

```bash
docker-compose up -d
```

This will start:
- **MongoDB** on `localhost:27017` (credentials: root/rootpassword)
- **MongoDB Express** (GUI) on `http://localhost:8081`

To verify MongoDB is running:
```bash
docker-compose logs mongo
```

To stop the services:
```bash
docker-compose down
```

---

## Step 2: Prepare Your Test Image

Place an image in your workspace (or use any existing image file):
```bash
# Example: using an image from the data directory
ls data/har_hf/train/calling/
```

---

## Step 3: Run Complete Inference -> Compress -> Store Pipeline

### 3a. Quick Example: Compress a Single Image and Store to MongoDB

```bash
# Activate your virtual environment (if not already activated)
.venv\Scripts\Activate.ps1

# Compress and store image to MongoDB
python scripts/compress_to_mongo.py \
  --checkpoint outputs/har_gpu/best_model.pt \
  --image data/har_hf/train/calling/001_0_1_0_0_0.png \
  --mongo-uri "mongodb://root:rootpassword@localhost:27017/image_compression?authSource=admin" \
  --database image_compression \
  --collection compressed_images \
  --image-id "test_image_001" \
  --model-version "har_gpu_v1"
```

### 3b. Decompress and Reconstruct from MongoDB

```bash
python scripts/reconstruct_from_mongo.py \
  --checkpoint outputs/har_gpu/best_model.pt \
  --mongo-uri "mongodb://root:rootpassword@localhost:27017/image_compression?authSource=admin" \
  --database image_compression \
  --collection compressed_images \
  --image-id "test_image_001" \
  --output reconstructed_image.png
```

---

## Complete Workflow Script

Save this as `run_inference_pipeline.sh` (or `.ps1` for PowerShell):

```bash
#!/bin/bash

# Configuration
CHECKPOINT="outputs/har_gpu/best_model.pt"
IMAGE_PATH="data/har_hf/train/calling/001_0_1_0_0_0.png"
MONGO_URI="mongodb://root:rootpassword@localhost:27017/image_compression?authSource=admin"
DB_NAME="image_compression"
COLLECTION="compressed_images"
IMAGE_ID="test_image_$(date +%s)"
OUTPUT="reconstructed_${IMAGE_ID}.png"

echo "Step 1: Compressing image and storing to MongoDB..."
python scripts/compress_to_mongo.py \
  --checkpoint "$CHECKPOINT" \
  --image "$IMAGE_PATH" \
  --mongo-uri "$MONGO_URI" \
  --database "$DB_NAME" \
  --collection "$COLLECTION" \
  --image-id "$IMAGE_ID" \
  --model-version "har_gpu_v1"

if [ $? -ne 0 ]; then
    echo "Compression failed!"
    exit 1
fi

echo ""
echo "Step 2: Reconstructing image from MongoDB..."
python scripts/reconstruct_from_mongo.py \
  --checkpoint "$CHECKPOINT" \
  --mongo-uri "$MONGO_URI" \
  --database "$DB_NAME" \
  --collection "$COLLECTION" \
  --image-id "$IMAGE_ID" \
  --output "$OUTPUT"

if [ $? -ne 0 ]; then
    echo "Reconstruction failed!"
    exit 1
fi

echo ""
echo "✓ Pipeline complete!"
echo "  Original: $IMAGE_PATH"
echo "  Stored ID: $IMAGE_ID"
echo "  Reconstructed: $OUTPUT"
```

For PowerShell, save as `run_inference_pipeline.ps1`:

```powershell
# Configuration
$CHECKPOINT = "outputs/har_gpu/best_model.pt"
$IMAGE_PATH = "data/har_hf/train/calling/001_0_1_0_0_0.png"
$MONGO_URI = "mongodb://root:rootpassword@localhost:27017/image_compression?authSource=admin"
$DB_NAME = "image_compression"
$COLLECTION = "compressed_images"
$IMAGE_ID = "test_image_$(Get-Date -Format 'yyyyMMddHHmmss')"
$OUTPUT = "reconstructed_$IMAGE_ID.png"

Write-Host "Step 1: Compressing image and storing to MongoDB..."
python scripts/compress_to_mongo.py `
  --checkpoint $CHECKPOINT `
  --image $IMAGE_PATH `
  --mongo-uri $MONGO_URI `
  --database $DB_NAME `
  --collection $COLLECTION `
  --image-id $IMAGE_ID `
  --model-version "har_gpu_v1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Compression failed!"
    exit 1
}

Write-Host ""
Write-Host "Step 2: Reconstructing image from MongoDB..."
python scripts/reconstruct_from_mongo.py `
  --checkpoint $CHECKPOINT `
  --mongo-uri $MONGO_URI `
  --database $DB_NAME `
  --collection $COLLECTION `
  --image-id $IMAGE_ID `
  --output $OUTPUT

if ($LASTEXITCODE -ne 0) {
    Write-Host "Reconstruction failed!"
    exit 1
}

Write-Host ""
Write-Host "✓ Pipeline complete!"
Write-Host "  Original: $IMAGE_PATH"
Write-Host "  Stored ID: $IMAGE_ID"
Write-Host "  Reconstructed: $OUTPUT"
```

Run it:
```bash
# For bash/zsh
bash run_inference_pipeline.sh

# For PowerShell
.\run_inference_pipeline.ps1
```

---

## Available Models

Use any of these checkpoint paths:
- `outputs/har_gpu/best_model.pt` - Human Action Recognition
- `outputs/har_gpu_finetune/best_model.pt` - HAR Finetuned
- `outputs/har_gpu_pretrain/best_model.pt` - HAR Pretrained
- `outputs/har_gpu_quality/best_model.pt` - HAR Quality variant
- `outputs/smoke/best_model.pt` - Smoke test model

---

## MongoDB GUI Access

Navigate to: **http://localhost:8081**
- Username: `root`
- Password: `rootpassword`

You can browse stored compressed images in the `image_compression.compressed_images` collection.

---

## Batch Processing Multiple Images

```bash
#!/bin/bash
IMAGE_DIR="data/har_hf/train/calling"
CHECKPOINT="outputs/har_gpu/best_model.pt"
MONGO_URI="mongodb://root:rootpassword@localhost:27017/image_compression?authSource=admin"

for image_file in "$IMAGE_DIR"/*.png; do
    image_id=$(basename "$image_file" .png)
    echo "Processing: $image_id"
    
    python scripts/compress_to_mongo.py \
      --checkpoint "$CHECKPOINT" \
      --image "$image_file" \
      --mongo-uri "$MONGO_URI" \
      --image-id "$image_id" \
      --model-version "har_gpu_v1"
    
    sleep 1  # Brief pause between images
done
```

---

## Troubleshooting

### Connection Refused
If you get "Connection refused", check MongoDB is running:
```bash
docker-compose ps
docker-compose logs mongo
```

### Authentication Failed
Ensure your MongoDB URI includes `?authSource=admin`:
```
mongodb://root:rootpassword@localhost:27017/image_compression?authSource=admin
```

### GPU Not Available
The scripts automatically fall back to CPU if CUDA is unavailable. To force CPU:
```bash
# Add this to your script or terminal
$env:CUDA_VISIBLE_DEVICES = "-1"  # PowerShell
# or
export CUDA_VISIBLE_DEVICES=-1    # Bash
```

---

## Cleanup

Stop and remove all containers:
```bash
docker-compose down -v  # -v removes volumes too
```

View MongoDB logs:
```bash
docker-compose logs -f mongo
```
