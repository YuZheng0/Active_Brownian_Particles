#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

//************* Set Parameters **************
const int NUMSIMS = 10000; // Number of simulations
const int STEP = 1000;     // Number of movement steps
const int TICK = 10;       // Number of steps between each save file

const double LENGTH = 1;    // Length of box
const double DENSITY = 0.4; // Density of system
const double RADIUS = 0.1;  // Radius of particles
const double DELTATIME = 1; // particles[i].position[0] += DELTATIME * particles[i].velocity[0]
const double V = 0.01;      //Initial velocity

const double PI = 3.1415926;

const double AREAOFBOX = LENGTH * LENGTH;
const int NUMBER = int(AREAOFBOX * DENSITY / (PI * RADIUS * RADIUS));
//************* End Set Parameters *********

//******** Set Random Numbers *********
std::random_device rd;
std::default_random_engine gen(rd());
std::normal_distribution<double> norm_distribution_angle(0, PI / 64);
std::uniform_real_distribution<float> uniform_distribution(0.0, 1.0);
//******** Set Random Numbers *********

struct Particle
{
    double position[2];
    double velocity[2];
    double force[2];
};

double boundaryDisplacementX(Particle a, Particle b)
{
    /*
    Return distance with periodic boundary condition 
    from particle b to particle a in x direction.
    */
    double deltax = a.position[0] - b.position[0];

    if (std::abs(deltax) > LENGTH / 2)
    {
        if (deltax > 0)
            deltax -= LENGTH;
        else
            deltax += LENGTH;
    }
    return deltax;
}

double boundaryDisplacementY(Particle a, Particle b)
{
    /*
    Return displacement with periodic boundary condition 
    from particle b to particle a in y direction.
    */
    double deltay = a.position[1] - b.position[1];

    if (std::abs(deltay) > LENGTH / 2)
    {
        if (deltay > 0)
            deltay -= LENGTH;
        else
            deltay += LENGTH;
    }
    return deltay;
}

double boundaryDistance(Particle a, Particle b)
{
    /*
    Return distance with periodic boundary condition 
    between particle a and particle b.
    */
    double deltax = boundaryDisplacementX(a, b);
    double deltay = boundaryDisplacementY(a, b);

    return std::sqrt(deltax * deltax + deltay * deltay);
}

bool isOverlap(Particle a, Particle b)
{
    /*
    Return whether particle a and particle b overlap.
    */
    return (boundaryDistance(a, b) < 2 * RADIUS);
}

void initialization(Particle particles[])
{
    /*
    Initialze randomly positions of particles. 
    And velocity is global constant V.
    */
    int i = 0;
    while (i < NUMBER)
    {
        particles[i].position[0] = LENGTH * (uniform_distribution(gen));
        particles[i].position[1] = LENGTH * (uniform_distribution(gen));

        bool hasOverlap = false;
        for (int j = 0; j < i; j++)
        {
            if (i != j && isOverlap(particles[i], particles[j]))
            {
                hasOverlap = true;
                break;
            }
        }
        if (!hasOverlap)
        {
            i++;
            //std::cout << "Add: " << i << " out of " << NUMBER << std::endl;
        }
    }

    for (int ii = 0; ii < NUMBER; ii++)
    {
        particles[ii].velocity[0] = V * LENGTH * (uniform_distribution(gen) - 0.5);
        particles[ii].velocity[1] = V * LENGTH * (uniform_distribution(gen) - 0.5);
        particles[ii].force[0] = 0;
        particles[ii].force[1] = 0;
    }
}

void correctOverlap(Particle &a, Particle &b)
{
    /*
    Offset overlap between two particles.If two particles overlap, 
    move each one-half the overlap distance along their center-to-center axis.
    */
    double d = boundaryDistance(a, b);
    double deltax = boundaryDisplacementX(a, b);
    double deltay = boundaryDisplacementY(a, b);
    double deltaD = 2 * RADIUS - d;

    a.position[0] += 0.5 * deltaD * deltax / d;
    a.position[1] += 0.5 * deltaD * deltay / d;
    b.position[0] += 0.5 * deltaD * (-deltax) / d;
    b.position[1] += 0.5 * deltaD * (-deltay) / d;
}

double getAngle(double x, double y)
{
    /*
    Return the angle in radiance of a vector with components x and y.
    */
    double angle = atan(y / x);
    if (x < 0)
        angle += PI;

    return angle;
}

void move(Particle particles[])
{
    /*
    Move all particles. Update the positions of particles 
    according to the current velocities.Velocities are updated 
    by Gaussian random walk. Gaussian distribution is set by
    global normal distribution generator.
    */
    int i, j;
    for (i = 0; i < NUMBER; i++)
    {
        //Update positions
        particles[i].position[0] += DELTATIME * particles[i].velocity[0];
        particles[i].position[1] += DELTATIME * particles[i].velocity[1];

        for (j = 0; j < NUMBER; j++)
        {
            if (j != i)
            {
                if (isOverlap(particles[i], particles[j]))
                {
                    correctOverlap(particles[i], particles[j]);
                }
            }
        }

        // rotation diffusion
        double velocityAngle;
        double v = std::sqrt(particles[i].velocity[0] * particles[i].velocity[0] + particles[i].velocity[1] * particles[i].velocity[1]);
        velocityAngle = getAngle(particles[i].velocity[0], particles[i].velocity[1]);
        velocityAngle += norm_distribution_angle(gen);
        particles[i].velocity[0] = v * cos(velocityAngle);
        particles[i].velocity[1] = v * sin(velocityAngle);

        // Correct the position according to periodic boundary condition.
        if (particles[i].position[0] >= LENGTH)
            particles[i].position[0] = particles[i].position[0] - LENGTH;
        if (particles[i].position[0] < 0)
            particles[i].position[0] = particles[i].position[0] + LENGTH;
        if (particles[i].position[1] >= LENGTH)
            particles[i].position[1] = particles[i].position[1] - LENGTH;
        if (particles[i].position[1] < 0)
            particles[i].position[1] = particles[i].position[1] + LENGTH;
    }
}

void simulation(int simID)
{
    /*
    Run a complete simulation from start. simID labels the save detination for data.
    After initialization, particles move for a number of steps specified by the global
    constant STEP. Motion data is saved for every TICK steps.
    */
    Particle particles[NUMBER];
    initialization(particles);
    std::ofstream fout1("data" + std::to_string(simID) + "/position0.txt");
    if (!fout1.is_open())
    {
        std::cerr << "Failed to open data" + std::to_string(simID) << std::endl;
        exit(1);
    }

    for (int i = 0; i < NUMBER; i++)
    {
        fout1 << particles[i].position[0] << " " << particles[i].position[1] << " " << particles[i].velocity[0] << " " << particles[i].velocity[1] << std::endl;
    }
    fout1.close();

    for (int step = 0; step < STEP; step++)
    {
        for (int i = 0; i < NUMBER; i++)
        {
            particles[i].force[0] = 0;
            particles[i].force[1] = 0;
        }
        move(particles);

        if ((step + 1) % TICK == 0)
        {
            // std::cout << "Step " << step + 1 << " out of " << STEP << std::endl;
            const std::string fileName = "data" + std::to_string(simID) + "/position" + std::to_string(step + 1) + ".txt";
            std::ofstream fout(fileName);
            if (!fout.is_open())
            {
                std::cerr << "Failed to open " + fileName << std::endl;
                exit(1);
            }
            for (int i = 0; i < NUMBER; i++)
            {
                fout << particles[i].position[0] << " " << particles[i].position[1] << " " << particles[i].velocity[0] << " " << particles[i].velocity[1] << std::endl;
            }
            fout.close();
        }
    }
}

int main()
{
    // double startTime, endTime;
    // startTime = omp_get_wtime();
    for (int simID = 1; simID <= NUMSIMS; simID++)
    {
        simulation(simID);
        std::cout << "Simulation " << simID << " out of " << NUMSIMS << std::endl;
    }
    // endTime = omp_get_wtime();
    // std::cout << "Total time: " << endTime - startTime << " s" << std::endl;
}
