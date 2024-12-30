#!/bin/bash

# Directory containing the files (replace with your directory)
DIRECTORY="validation/flyingv"

# Counter for numbering the files
counter=1

# Loop through each file in the directory
for file in "$DIRECTORY"/*; do
  # Check if it's a file (not a directory)
  if [[ -f "$file" ]]; then
    # Get the file extension
    ext="${file##*.}"

    # Generate the new filename with the incremented number and the original extension
    new_name="$DIRECTORY/$counter.$ext"

    # Rename the file
    mv "$file" "$new_name"

    # Increment the counter
    ((counter++))

    echo "Renamed: $file -> $new_name"
  fi
done
