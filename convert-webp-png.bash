#!/bin/bash

# Directory containing the .webp files (replace with your directory)
DIRECTORY="train/lespaul"

# Loop through each .webp file in the directory
for webp_file in "$DIRECTORY"/*.webp; do
  if [[ -f "$webp_file" ]]; then  # Check if it's a valid file
    # Get the base name without extension
    base_name=$(basename "$webp_file" .webp)

    # Convert to PNG using dwebp
    dwebp "$webp_file" -o "$DIRECTORY/$base_name.png"
    
    echo "Converted: $webp_file to $DIRECTORY/$base_name.png"
  fi
done
