
for i in $(seq 1 5); do
    python particle_animation.py --data-dir=data$i --save-dir=data$i
done


