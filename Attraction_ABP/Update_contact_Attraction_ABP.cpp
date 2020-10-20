#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <random>

//*************Set Parameters**************
const int NUMSIMS = 1;  // Number of simulations
const int STEP = 20000; // Number of movement steps
const int TICK = 10;    // Number of steps between each save file

const double LENGTH = 1;            // The length of box
const double DENSITY = 0.5;         // The density of system
const double RADIUS = 0.02;         // The radius of particles
const double RADIUSofSHELL = 0.025; // The radius of shell
const double DELTATIME = 1;         // particles[i].position[0] += DELTATIME * particles[i].velocity[0];
const double V = 0.001;             // Initial velocity

const double DIFFCOEFF = 1;          // The diffusion coefficient
const double KCON = RADIUS * 0.0001; // The coefficient for contact force
const double KECM = RADIUS * 0.0001; // The coefficient for ECM force
double ANGLE = 9;                    // For ECM force

const double DIFFCOEFF2 = RADIUS * 0.00001; // The diffusion coefficient for random force
//const double EPS = 0.0000000002;            // A samll number for preventing division divergence
const double FSELF = RADIUS * 0.1; // The self propulsion force

const double PI = 3.1415926;
const double AREAOFBOX = LENGTH * LENGTH;
const int NUMBER = int(AREAOFBOX * DENSITY / (PI * RADIUS * RADIUS));

//*************Set Parameters**************

//********Set Random Numbers*********
std::random_device rd;
std::default_random_engine gen(rd());
std::normal_distribution<double> norm_distribution_angle(0, PI / 64);
std::uniform_real_distribution<float> uniform_distribution(0.0, 1.0);
//********Set Random Numbers*********

struct Particle
{
    double position[2];
    double velocity[2];
    double force[2];
    double correctMove[2];
};

double boundaryDisplacementX(Particle a, Particle b)
{
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
    double deltax = boundaryDisplacementX(a, b);
    double deltay = boundaryDisplacementY(a, b);

    return std::sqrt(deltax * deltax + deltay * deltay);
}

bool isOverlap(Particle a, Particle b)
{
    return (boundaryDistance(a, b) < 2 * RADIUS);
}

bool isOverlapShell(Particle a, Particle b)
{
    return (boundaryDistance(a, b) < 2 * RADIUSofSHELL);
}

bool isOppositeMove(Particle a, Particle b)
{
    double d = boundaryDistance(a, b);
    double dv = std::sqrt(a.velocity[0] * a.velocity[0] + a.velocity[1] * a.velocity[1]);
    double deltax, deltay, deltax1, deltay1;
    double cosTheta;
    deltax = boundaryDisplacementX(a, b);
    deltay = boundaryDisplacementY(a, b);

    cosTheta = (-deltax * a.velocity[0] + (-deltay) * a.velocity[1]) / (d * dv);

    return (cosTheta >= cos(ANGLE * PI / 180) and (a.velocity[0] * b.velocity[0] + a.velocity[1] * b.velocity[1] < 0));
}

void initialization(Particle particles[])
{
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
    int ii;
#pragma omp parallel for private(ii)
    for (ii = 0; ii < NUMBER; ii++)
    {
        particles[ii].velocity[0] = V * LENGTH * (uniform_distribution(gen) - 0.5);
        particles[ii].velocity[1] = V * LENGTH * (uniform_distribution(gen) - 0.5);
        particles[ii].force[0] = 0;
        particles[ii].force[1] = 0;
        particles[ii].correctMove[0] = 0;
        particles[ii].correctMove[1] = 0;
    }
}

void forceContactECM(Particle particles[])
{
    int i, j;
#pragma omp parallel for private(j)
    for (i = 0; i < NUMBER; i++)
    {
        for (j = 0; j < NUMBER; j++)
        {
            if (isOverlapShell(particles[i], particles[j]) & (i != j))
            {
                double d = boundaryDistance(particles[i], particles[j]);
                double deltax, deltay;
                deltax = boundaryDisplacementX(particles[i], particles[j]);
                deltay = boundaryDisplacementY(particles[i], particles[j]);

                particles[i].force[0] += DIFFCOEFF * KCON * std::abs(2 * RADIUSofSHELL - d) * (-deltax / d);
                particles[i].force[1] += DIFFCOEFF * KCON * std::abs(2 * RADIUSofSHELL - d) * (-deltay / d);
            }

            if (isOppositeMove(particles[i], particles[j]) & (i != j))
            {
                double d = boundaryDistance(particles[i], particles[j]);
                double deltax, deltay;
                deltax = boundaryDisplacementX(particles[i], particles[j]);
                deltay = boundaryDisplacementY(particles[i], particles[j]);

                particles[i].force[0] += DIFFCOEFF * (KECM / d * (-deltax / d));
                particles[i].force[1] += DIFFCOEFF * (KECM / d * (-deltay / d));
            }
        }
    }
}

// void forceSelf(Particle particles[NUMBER])
// {
//     int i;
// #pragma omp parallel for private(i)
//     for (i = 0; i < NUMBER; i++)
//     {
//         double v = sqrt(particles[i].velocity[0] * particles[i].velocity[0] + particles[i].velocity[1] * particles[i].velocity[1]);
//         particles[i].force[0] += DIFFCOEFF * FSELF * particles[i].velocity[0] / v;
//         particles[i].force[1] += DIFFCOEFF * FSELF * particles[i].velocity[1] / v;
//     }
// }

// void forceRandom(Particle particles[NUMBER])
// {
//     int i;
// #pragma omp parallel for private(i)
//     for (i = 0; i < NUMBER; i++)
//     {
//         particles[i].force[0] += DIFFCOEFF2 * LENGTH * (uniform_distribution(gen) - 0.5);
//         particles[i].force[1] += DIFFCOEFF2 * LENGTH * (uniform_distribution(gen) - 0.5);
//     }
// }

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

    a.correctMove[0] += 0.5 * deltaD * deltax / d;
    a.correctMove[1] += 0.5 * deltaD * deltay / d;
    //b.position[0] += 0.5 * deltaD * (-deltax) / d;
    //b.position[1] += 0.5 * deltaD * (-deltay) / d;
}

double getAngle(double x, double y)
{
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
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < NUMBER; i++)
    {
        //Update positions
        particles[i].position[0] += DELTATIME * particles[i].velocity[0];
        particles[i].position[1] += DELTATIME * particles[i].velocity[1];

        double v = std::sqrt(particles[i].velocity[0] * particles[i].velocity[0] + particles[i].velocity[1] * particles[i].velocity[1]);
        //Add force
        particles[i].velocity[0] += particles[i].force[0];
        particles[i].velocity[1] += particles[i].force[1];
        double velocityAngle;
        velocityAngle = getAngle(particles[i].velocity[0], particles[i].velocity[1]);
        particles[i].velocity[0] = v * cos(velocityAngle);
        particles[i].velocity[1] = v * sin(velocityAngle);

        //rotation diffusion
        velocityAngle += norm_distribution_angle(gen);
        particles[i].velocity[0] = v * cos(velocityAngle);
        particles[i].velocity[1] = v * sin(velocityAngle);
        ////////////////////

        // Correct the position according to periodic boundary condition.
        if (particles[i].position[0] >= LENGTH)
            particles[i]
                .position[0] = particles[i].position[0] - LENGTH;
        if (particles[i].position[0] < 0)
            particles[i].position[0] = particles[i].position[0] + LENGTH;
        if (particles[i].position[1] >= LENGTH)
            particles[i].position[1] = particles[i].position[1] - LENGTH;
        if (particles[i].position[1] < 0)
            particles[i].position[1] = particles[i].position[1] + LENGTH;
    }
}

void recordCorrectOverlap(Particle particles[])
{
    int i, j;
#pragma omp parallel for private(j)
    for (i = 0; i < NUMBER; i++)
    {
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
    }
}

void moveCorrectOverlap(Particle particles[])
{
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < NUMBER; i++)
    {
        particles[i].position[0] += particles[i].correctMove[0];
        particles[i].position[1] += particles[i].correctMove[1];
        particles[i].correctMove[0] = 0;
        particles[i].correctMove[1] = 0;
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
        forceContactECM(particles);
        //forceSelf(particles);
        //forceRandom(particles);
        move(particles);
        recordCorrectOverlap(particles);
        moveCorrectOverlap(particles);
        if ((step + 1) % TICK == 0)
        {
            //std::cout << "Step " << step + 1 << " out of " << STEP << std::endl;
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
    double startTime, endTime;
    startTime = omp_get_wtime();

    for (int simID = 1; simID <= NUMSIMS; simID++)
    {
        simulation(simID);
        ANGLE += 2;
        std::cout << "Simulation " << simID << " out of " << NUMSIMS << std::endl;
    }

    endTime = omp_get_wtime();
    std::cout << "Total time: " << endTime - startTime << " s" << std::endl;
}
