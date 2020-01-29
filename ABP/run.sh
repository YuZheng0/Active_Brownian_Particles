g++ Active_Brownian_Particles.cpp -o a.out

for i in $(seq 1 10000); do
    mkdir -p data$i
done

./a.out

