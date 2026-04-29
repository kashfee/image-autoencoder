# MongoDB Atlas Inference Commands

## Connection String
```
mongodb+srv://kashfee:<db_password>@cluster0.ez0zvxv.mongodb.net/?appName=Cluster0
```

Replace `<db_password>` with your actual password.

---

## Command 1: Compress Image and Upload to MongoDB

```bash
python scripts/compress_to_mongo.py \
  --checkpoint outputs/har_gpu/best_model.pt \
  --image data/har_hf/train/calling/001_0_1_0_0_0.png \
  --mongo-uri "mongodb+srv://kashfee:<db_password>@cluster0.ez0zvxv.mongodb.net/?appName=Cluster0" \
  --database image_compression \
  --collection compressed_images \
  --image-id "test_image_001" \
  --model-version "har_gpu_v1"
```

### Expected Output:
```
stored image_id=test_image_001 bpp=0.XXXX
```

---

## Command 2: Download, Decompress and Reconstruct from MongoDB

```bash
python scripts/reconstruct_from_mongo.py \
  --checkpoint outputs/har_gpu/best_model.pt \
  --mongo-uri "mongodb+srv://kashfee:<db_password>@cluster0.ez0zvxv.mongodb.net/?appName=Cluster0" \
  --database image_compression \
  --collection compressed_images \
  --image-id "test_image_001" \
  --output reconstructed_image.png
```

### Expected Output:
```
reconstructed image_id=test_image_001 to C:\Users\kashf\Documents\Codex\2026-04-26\Image-Auto-Encoder\reconstructed_image.png
```

---

## Complete PowerShell One-Liner (Replace password first)

```powershell
$MONGO_URI = "mongodb+srv://kashfee:YOUR_PASSWORD_HERE@cluster0.ez0zvxv.mongodb.net/?appName=Cluster0"
$CHECKPOINT = "outputs/har_gpu/best_model.pt"
$IMAGE = "data/har_hf/train/calling/001_0_1_0_0_0.png"
$IMAGE_ID = "test_image_001"

# Step 1: Compress and upload
python scripts/compress_to_mongo.py --checkpoint $CHECKPOINT --image $IMAGE --mongo-uri $MONGO_URI --image-id $IMAGE_ID --model-version "har_gpu_v1"

# Step 2: Download and reconstruct
python scripts/reconstruct_from_mongo.py --checkpoint $CHECKPOINT --mongo-uri $MONGO_URI --image-id $IMAGE_ID --output "reconstructed_$IMAGE_ID.png"
```

---

## Batch Process Multiple Images

```powershell
$MONGO_URI = "mongodb+srv://kashfee:YOUR_PASSWORD_HERE@cluster0.ez0zvxv.mongodb.net/?appName=Cluster0"
$CHECKPOINT = "outputs/har_gpu/best_model.pt"
$IMAGE_DIR = "data/har_hf/train/calling"
$counter = 1

Get-ChildItem "$IMAGE_DIR/*.png" | ForEach-Object {
    $image_path = $_.FullName
    $image_id = "image_$($counter.ToString().PadLeft(3, '0'))"
    
    Write-Host "Processing: $image_id from $($_.Name)"
    
    python scripts/compress_to_mongo.py `
      --checkpoint $CHECKPOINT `
      --image $image_path `
      --mongo-uri $MONGO_URI `
      --image-id $image_id `
      --model-version "har_gpu_v1"
    
    $counter++
    Start-Sleep -Milliseconds 500
}
```

---

## Available Checkpoints

- `outputs/har_gpu/best_model.pt`
- `outputs/har_gpu_finetune/best_model.pt`
- `outputs/har_gpu_pretrain/best_model.pt`
- `outputs/har_gpu_quality/best_model.pt`
- `outputs/smoke/best_model.pt`

---

## Notes

- Replace `YOUR_PASSWORD_HERE` with your actual MongoDB password
- The connection string uses MongoDB Atlas (cloud-hosted)
- Compressed images are stored in `image_compression.compressed_images` collection
- Each image is identified by `image_id` and `model_version`
- The reconstruction uses the same model version to decompress
