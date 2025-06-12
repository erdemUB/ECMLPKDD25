path="./all_graphs"
files=($(seq 1 323))
comp=("0.5" "mid" "median")
pick="degree"

for file in "${files[@]}";do
  for c in "${comp[@]}";do
    #echo $file
    python3 ../kcore_rww.py --path $path --graphId $file --pick $pick --comp $c
  done
done
