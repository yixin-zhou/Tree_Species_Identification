# ====== MODIFY AS NEEDED ======
SRC_DIR="./data/Swiss/images"
DEST_PREFIX="users/bryce001006"
PYRAMIDING_POLICY="MEAN"
# Change this line to your GCS bucket name
GCS_BUCKET="treeai_assets"
# ==============================

command -v earthengine >/dev/null 2>&1 || {
  echo "The earthengine command was not found. Please install it and run: earthengine authenticate"; exit 1; }

# Collect file names from local directory; actual data source is Google Cloud Storage
shopt -s nullglob
files=( "$SRC_DIR"/*.tif "$SRC_DIR"/*.tiff )
total=${#files[@]}
if [ "$total" -eq 0 ]; then
  echo "No .tif/.tiff files found in $SRC_DIR (used only to retrieve file names)"; exit 0
fi

upload_one() {
  local index="$1"
  local total="$2"
  local tif="$3"

  local filename="$(basename "$tif")"
  local name="${filename%.*}"
  name="${name// /_}"
  local asset_id="${DEST_PREFIX}/${name}"

  # Key: source data comes from Google Cloud Storage
  local source_uri="gs://${GCS_BUCKET}/${filename}"

  # Progress bar
  local progress=$(( index * 20 / total ))
  local bar blanks
  bar=$(printf "%-${progress}s" "#" | tr ' ' '#')
  blanks=$(printf "%-$((40 - progress))s" " ")
  echo "[${bar}${blanks}] (${index}/${total}) Upload: ${filename}"

  # Skip if the asset already exists
  if earthengine asset info "$asset_id" > /dev/null 2>&1; then
    echo "--> Skip: asset already exists: $asset_id"
    return 0
  fi

  earthengine upload image \
    --asset_id="$asset_id" \
    --pyramiding_policy="$PYRAMIDING_POLICY" \
    "$source_uri"

  echo
}

for ((i=1; i<=total; i++)); do
  f="${files[i-1]}"
  upload_one "$i" "$total" "$f"
done

echo "All $total GeoTIFFs have been submitted to GEE (source: Google Cloud Storage: ${GCS_BUCKET}). Use: earthengine task list to check progress."
