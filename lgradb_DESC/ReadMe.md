# Steps to preprocess files

Copy all relevant files

```bash
# Directory paths
source_dir="../single-stage-opt/quasr_coil_check/QUASR_db/nml"
target_dir="publications/lgradb"

# List of filenames (just the input file names without paths)
files=(
  "input.0018874"
  "input.0019371"
  "input.0050638"
  "input.0019047"
  "input.0019377"
  "input.0050646"
  "input.0019098"
  "input.0019379"
  "input.0050695"
  "input.0019104"
  "input.0019384"
  "input.0070800"
  "input.0019112"
  "input.0030620"
  "input.0070807"
  "input.0019152"
  "input.0030661"
  "input.0070821"
  "input.0019266"
  "input.0030664"
  "input.0070828"
  "input.0019267"
  "input.0030812"
)

# Copy files
for file in "${files[@]}"; do
    # Extract the first 4 digits of the filename to build the subdirectory path
    subdir="${file:6:4}"
    full_source_path="$source_dir/$subdir/$file"
    echo "Copying $full_source_path to $target_dir"
    cp "$full_source_path" "$target_dir/"
done
```

Add missing input parameters:

```bash
insert_text="MPOL = 9\nNTOR = 9\n\nNCURR = 1\nCURTOR = 0.00000000000000E+00\nAC = 0.00000000000000E+00\n"

for file in "${files[@]}"; do
  temp_file=$(mktemp)
  # Insert text before the line starting with 'RBC'
  awk -v insert="$insert_text" '/^RBC\(   0,   0\)/{print insert} {print}' "$target_dir/$file" > "$temp_file"

  # Replace the original file with the modified file
  mv "$temp_file" "$target_dir/$file"
done
```

Run DESC on them:

```bash
for file in "${files[@]}"; do
    python -m desc "$target_dir/$file"
done
```
## Run on the big dataset
```bash
source_dir="../single-stage-opt/quasr_coil_check/QUASR_db/nml"
target_dir="publications/lgradb/big_run"
insert_text="MPOL = 9\nNTOR = 9\n\nNCURR = 1\nCURTOR = 0.00000000000000E+00\nAC = 0.00000000000000E+00\n"

for file in $(ls ~/single-stage-opt/quasr_coil_check/QUASR_db/nml/* | grep input. | shuf); do
  # Extract the first 4 digits of the filename to build the subdirectory path
  subdir="${file:6:4}"
  full_source_path="$source_dir/$subdir/$file"
  echo "Copying $full_source_path to $target_dir"

  cp "$full_source_path" "$target_dir/"
  temp_file=$(mktemp)
  # Insert text before the line starting with 'RBC'
  awk -v insert="$insert_text" '/^RBC\(   0,   0\)/{print insert} {print}' "$target_dir/$file" > "$temp_file"

  # Replace the original file with the modified file
  mv "$temp_file" "$target_dir/$file"
  python -m desc "$target_dir/$file"
done
```
