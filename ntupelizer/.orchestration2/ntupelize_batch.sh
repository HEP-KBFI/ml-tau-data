#!/bin/bash


set -e  # Exit on any error

# Parse arguments
SAMPLE_NAME="$1"
OUTPUT_DIR="$2"
IS_SIGNAL="$3"
shift 3
INPUT_FILES=("$@")

echo "=== Ntupelization Batch Job ==="
echo "Sample: $SAMPLE_NAME"
echo "Output dir: $OUTPUT_DIR"
echo "Is signal: $IS_SIGNAL"
echo "Number of files: ${#INPUT_FILES[@]}"
echo "Files: ${INPUT_FILES[*]}"
echo "Working directory: $(pwd)"
echo "================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Determine whether to run inside container
if [[ "${SINGULARITY_CONTAINER:-}" ]] || [[ "${APPTAINER_CONTAINER:-}" ]]; then
    echo "Running inside container"
    CONTAINER_CMD=""
else
    echo "Running outside container, using ./run.sh wrapper"
    CONTAINER_CMD="./run.sh"
fi

# Process each file
SUCCESS_COUNT=0
TOTAL_COUNT=${#INPUT_FILES[@]}

for INPUT_FILE in "${INPUT_FILES[@]}"; do
    echo "Processing: $INPUT_FILE"
    
    # Extract filename without extension for output naming
    BASENAME=$(basename "$INPUT_FILE" .root)
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}.parquet"
    
    # Skip if output already exists
    if [[ -f "$OUTPUT_FILE" ]]; then
        echo "Output file already exists, skipping: $OUTPUT_FILE"
        ((SUCCESS_COUNT++))
        continue
    fi
    
    # Create temporary output file to avoid partial writes
    TEMP_OUTPUT="$OUTPUT_FILE.tmp"
    
    # Run ntupelizer
    echo "Output: $OUTPUT_FILE"
    
    if $CONTAINER_CMD python -m ntupelizer.scripts.ntupelize_edm4hep \
        --input-path "$INPUT_FILE" \
        --output-path "$TEMP_OUTPUT" \
        --signal-sample "$IS_SIGNAL" \
        --config-name ntupelizer; then
        
        # Move temporary file to final location
        mv "$TEMP_OUTPUT" "$OUTPUT_FILE"
        echo "Successfully processed: $INPUT_FILE -> $OUTPUT_FILE"
        ((SUCCESS_COUNT++))
    else
        echo "Failed to process: $INPUT_FILE"
        # Clean up temporary file if it exists
        [[ -f "$TEMP_OUTPUT" ]] && rm -f "$TEMP_OUTPUT"
    fi
    
    echo "---"
done

echo "=== Batch Job Summary ==="
echo "Sample: $SAMPLE_NAME"
echo "Successful: $SUCCESS_COUNT / $TOTAL_COUNT"
echo "========================="

if [[ $SUCCESS_COUNT -eq $TOTAL_COUNT ]]; then
    echo "All files processed successfully"
    exit 0
else
    echo "Some files failed to process"
    exit 1
fi