parallel_path=$1
output_path=$2

# Step 1: Concatenate all `.parallel` files
cat $parallel_path/*.parallel > $output_path/all.parallel

# Step 2: Count total lines
total=$(wc -l < $output_path/all.parallel)

# Step 3: Compute 95% for training using Bash arithmetic
train_lines=$(( (total * 97) / 100 ))
dev_lines=$(( total - train_lines ))

# Step 4: Shuffle and split
shuf $output_path/all.parallel | tee >(head -n "$train_lines" > $output_path/train.parallel) | tail -n "$dev_lines" > $output_path/dev.parallel

# Optional cleanup
rm $output_path/all.parallel