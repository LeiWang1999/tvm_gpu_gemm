function run_bench {
    echo "Running $1"
    cd $1
    chmod +x ./direct.sh
    ./direct.sh
    cd ..
}

run_bench ours_0
run_bench ours_1
run_bench ours_2
run_bench diffusion_0
