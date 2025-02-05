#!/bin/bash
set -x
# Slowly increase phiedge to get a larger aspect ratio configuration
# Set file paths
EXECUTABLE=~/SPEC/xspec
INPUT_FILE=qfb_optimization/rotating_ellipse_fb_low.sp
RESULT_FILE="${INPUT_FILE}.end"

# Number of iterations
ITERATIONS=20
INCREMENT=0.0005

# Perform iterations
for ((i=1; i<=ITERATIONS; i++))
do
    echo "Starting iteration $i..."

    # Run the executable
    $EXECUTABLE $INPUT_FILE
    if [ $? -ne 0 ]; then
        echo "Execution failed in iteration $i."
        exit 1
    fi

    # Move the result file back to the input file location
    cp $RESULT_FILE $INPUT_FILE

    # Update the 5th line in the input file
   sed -i '5s/\(\sphiedge\s*=\s*\)\(.*\)/echo "\1$(printf "%.4E" $(echo "0.003 + '"$i"' * 0.0005" | bc))"/e' $INPUT_FILE
    echo "Iteration $i completed: phiedge incremented by $INCREMENT."
done

echo "All $ITERATIONS iterations completed successfully."
