#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <random>
#include <vector>

//*************Set Parameters**************
const int NUMSIMS = 1;  // Number of simulations
const int STEP = 10000; // Number of movement steps
const int TICK = 10;    // Number of steps between each save file

const double LENGTH = 1;             // The length of box
const double DENSITY = 0.3;          // The density of system
const double RADIUS = 0.02;          // The radius of particles
const double DELTATIME = 1;          // particles[i].position[0] += DELTATIME * particles[i].velocity[0];
const double V = 0.001;              //Initial velocity
const double DIFFCOEFF = 1;          // The diffusion coefficient
const double KCON = RADIUS * 0.1;    // The coefficient for contact force
const double KECM = RADIUS * 0.0001; // The coefficient for ECM force
double ANGLE = 1;                    // For ECM force

const double DIFFCOEFF2 = RADIUS * 0.00001; // The diffusion coefficient for random force
//const double EPS = 0.0000000002;            // A samll number for preventing division divergence
const double FSELF = RADIUS * 0.1; // The self propulsion force

const double PI = 3.1415926;
const double AREAOFBOX = LENGTH * LENGTH;
const int NUMBER = int(AREAOFBOX * DENSITY / (PI * RADIUS * RADIUS));

//Add environment effect
const int NUMGRID = 5; // Number of gird
const int MAXPERGRID = int(LENGTH / NUMGRID / RADIUS / 2 + 1) * int(LENGTH / NUMGRID / RADIUS / 2 + 1);
//double Grid[NUMGRID][NUMGRID][2];
struct Grid
{
    double direction[2];
    int particlesInGrid[MAXPERGRID];
    int numOfParticles;
    const double initDirectionX = 1;
    const double initDirectionY = 0;
};
const double fiberChangeCoeff = 0.5;  //for direction of fiber
const double fiberRecoverCoeff = 0.5; // for direction of fiber recover
const double fiberEffectCoeff = 0.5;  //for velocity of particles

//*************Set Parameters**************

//********Set Random Numbers*********
std::random_device rd;
std::default_random_engine gen(rd());
std::normal_distribution<double> norm_distribution_angle(0, PI / 32);
std::uniform_real_distribution<float> uniform_distribution(0.0, 1.0);
//std::uniform_real_distribution<float> uniform_distribution2(-1.0, 1.0);
//********Set Random Numbers*********

struct Particle
{
    double position[2];
    double velocity[2];
    double force[2];
    double correctMove[2];
    int labelofgrid[2];
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

void boundaryCondition(Particle &a)
{
    if (a.position[0] >= LENGTH)
        a.position[0] = a.position[0] - LENGTH;
    if (a.position[0] < 0)
        a.position[0] = a.position[0] + LENGTH;
    if (a.position[1] >= LENGTH)
        a.position[1] = a.position[1] - LENGTH;
    if (a.position[1] < 0)
        a.position[1] = a.position[1] + LENGTH;
}

bool isOverlap(Particle a, Particle b)
{
    return (boundaryDistance(a, b) < 2 * RADIUS);
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

void initializationParticles(Particle particles[])
{
    int i = 0;
    std::cout << "inti" << std::endl;
    while (i < NUMBER)
    {
        particles[i].position[0] = LENGTH * (uniform_distribution(gen));
        particles[i].position[1] = LENGTH * (uniform_distribution(gen));

        particles[i].labelofgrid[0] = std::min(int(floor(particles[i].position[0] / (LENGTH / NUMGRID))), NUMGRID - 1);
        particles[i].labelofgrid[1] = std::min(int(floor(particles[i].position[1] / (LENGTH / NUMGRID))), NUMGRID - 1);

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

void initializationGrids(Grid grids[][NUMGRID])
{
    for (int i = 0; i < NUMGRID; i++)
    {
        for (int j = 0; j < NUMGRID; j++)
        {
            grids[i][j].direction[0] = 1; //uniform_distribution2(gen);
            grids[i][j].direction[1] = 0; //uniform_distribution2(gen);
            grids[i][j].numOfParticles = 0;
        }
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
            // if (isOverlap(particles[i], particles[j]) & (i != j))
            // {
            //     double d = boundaryDistance(particles[i], particles[j]);
            //     double deltax, deltay;
            //     deltax = boundaryDisplacementX(particles[i], particles[j]);
            //     deltay = boundaryDisplacementY(particles[i], particles[j]);

            //     particles[i].force[0] += DIFFCOEFF * KCON * std::abs(2 * RADIUS - d) * deltax / d;
            //     particles[i].force[1] += DIFFCOEFF * KCON * std::abs(2 * RADIUS - d) * deltay / d;
            // }
            //  else

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

double rotateAtoB(double Ax, double Ay, double Bx, double By, double coeff)
{
    double Aangle = atan2(Ay, Ax);
    double Bangle = atan2(By, Bx);

    double diff = Bangle - Aangle;
    double sign = (diff < 0) ? -1.0 : 1.0;

    while (std::abs(diff) > PI / 2)
    {
        diff -= PI * sign;
    }
    return diff * coeff;
}

void fiberAffected(Particle particles[], Grid grids[][NUMGRID])
{
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < NUMBER; i++)
    {
        particles[i].labelofgrid[0] = std::min(int(floor(particles[i].position[0] / (LENGTH / NUMGRID))), NUMGRID - 1);
        particles[i].labelofgrid[1] = std::min(int(floor(particles[i].position[1] / (LENGTH / NUMGRID))), NUMGRID - 1);

        int n = grids[particles[i].labelofgrid[0]][particles[i].labelofgrid[1]].numOfParticles;
        grids[particles[i].labelofgrid[0]][particles[i].labelofgrid[1]].particlesInGrid[n] = i;
        grids[particles[i].labelofgrid[0]][particles[i].labelofgrid[1]].numOfParticles += 1;
    }
    int ii, j;
    //#pragma omp parallel for private(ii)
    for (ii = 0; ii < NUMGRID; ii++)
    {
        for (j = 0; j < NUMGRID; j++)
        {
            double fiberX = grids[ii][j].direction[0]; //The x of direction of fiber
            double fiberY = grids[ii][j].direction[1]; //The y of direction of fiber

            double fiber = std::sqrt(fiberX * fiberX + fiberY * fiberY);
            double fiberAngle = atan2(grids[ii][j].direction[1], grids[ii][j].direction[0]);
            double rotateAngle = 0;
            if (grids[ii][j].numOfParticles >= 2)
            {
                for (int p1 = 0; p1 < grids[ii][j].numOfParticles; p1++)
                {
                    for (int p2 = p1 + 1; p2 < grids[ii][j].numOfParticles; p2++)
                    {
                        int a, b;
                        a = grids[ii][j].particlesInGrid[p1];
                        if (isOppositeMove(particles[a], particles[b]))
                        {
                            rotateAngle += rotateAtoB(fiberX, fiberY, particles[a].position[0] - particles[b].position[0], particles[a].position[1] - particles[b].position[1], fiberChangeCoeff);
                        }
                    }
                }
                grids[ii][j].direction[0] = fiber * cos(fiberAngle + rotateAngle);
                grids[ii][j].direction[1] = fiber * sin(fiberAngle + rotateAngle);
            }
            else
            {
                double initfiberX = grids[ii][j].initDirectionX;
                double initfiberY = grids[ii][j].initDirectionY;
                double rotateAngle = rotateAtoB(fiberX, fiberY, initfiberX, initfiberY, fiberRecoverCoeff);
                grids[ii][j].direction[0] = fiber * cos(fiberAngle + rotateAngle);
                grids[ii][j].direction[1] = fiber * sin(fiberAngle + rotateAngle);
            }
            grids[ii][j].numOfParticles = 0;
        }
    }
}

void move(Particle particles[], Grid grids[][NUMGRID])
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

        particles[i].labelofgrid[0] = std::min(int(floor(particles[i].position[0] / (LENGTH / NUMGRID))), NUMGRID - 1);
        particles[i].labelofgrid[1] = std::min(int(floor(particles[i].position[1] / (LENGTH / NUMGRID))), NUMGRID - 1);
        double v = std::sqrt(particles[i].velocity[0] * particles[i].velocity[0] + particles[i].velocity[1] * particles[i].velocity[1]);
        double velocityAngle;
        //Add force
        particles[i].velocity[0] += particles[i].force[0];
        particles[i].velocity[1] += particles[i].force[1];
        velocityAngle = atan2(particles[i].velocity[1], particles[i].velocity[0]);
        //particles[i].velocity[0] = v * cos(velocityAngle);
        //particles[i].velocity[1] = v * sin(velocityAngle);

        //environment effect

        double fiberX = grids[particles[i].labelofgrid[0]][particles[i].labelofgrid[1]].direction[0]; //The x of direction of fiber
        double fiberY = grids[particles[i].labelofgrid[0]][particles[i].labelofgrid[1]].direction[1]; //The y of direction of fiber
        //double fiberAngle = getAngle(fiberX, fiberY);

        velocityAngle += rotateAtoB(particles[i].velocity[0], particles[i].velocity[1], fiberX, fiberY, fiberEffectCoeff);
        //rotation diffusion
        velocityAngle += norm_distribution_angle(gen);

        particles[i].velocity[0] = v * cos(velocityAngle);
        particles[i].velocity[1] = v * sin(velocityAngle);
        ////////////////////

        //Update positions
        particles[i].position[0] += DELTATIME * particles[i].velocity[0];
        particles[i].position[1] += DELTATIME * particles[i].velocity[1];

        // Correct the position according to periodic boundary condition.
        boundaryCondition(particles[i]);
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

    a.correctMove[0] += 0.5 * deltaD * deltax / d;
    a.correctMove[1] += 0.5 * deltaD * deltay / d;
    //b.position[0] += 0.5 * deltaD * (-deltax) / d;
    //b.position[1] += 0.5 * deltaD * (-deltay) / d;
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
        boundaryCondition(particles[i]);
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
    Grid grids[NUMGRID][NUMGRID];
    initializationParticles(particles);
    initializationGrids(grids);
    std::ofstream fout1("data" + std::to_string(simID) + "/position0.txt");
    for (int i = 0; i < NUMBER; i++)
    {
        fout1 << particles[i].position[0] << " " << particles[i].position[1] << " " << particles[i].velocity[0] << " " << particles[i].velocity[1] << std::endl;
    }
    fout1.close();

    std::ofstream fout3("data" + std::to_string(simID) + "/fiberPosition0.txt");

    for (int i = 0; i < NUMGRID; i++)
        for (int j = 0; j < NUMGRID; j++)
        {
            fout3 << grids[i][j].direction[0] << " " << grids[i][j].direction[1] << std::endl;
        }
    fout3.close();

    for (int step = 0; step < STEP; step++)
    {
        for (int i = 0; i < NUMBER; i++)
        {
            particles[i].force[0] = 0;
            particles[i].force[1] = 0;
        }
        //forceContactECM(particles);
        //forceSelf(particles);
        //forceRandom(particles);
        move(particles, grids);
        recordCorrectOverlap(particles);
        moveCorrectOverlap(particles);
        fiberAffected(particles, grids);
        if ((step + 1) % TICK == 0)
        {
            //std::cout << "Step " << step + 1 << " out of " << STEP << std::endl;
            const std::string fileName = "data" + std::to_string(simID) + "/position" + std::to_string(step + 1) + ".txt";
            std::ofstream fout(fileName);
            for (int i = 0; i < NUMBER; i++)
            {
                fout << particles[i].position[0] << " " << particles[i].position[1] << " " << particles[i].velocity[0] << " " << particles[i].velocity[1] << std::endl;
            }
            fout.close();

            const std::string fileName2 = "data" + std::to_string(simID) + "/fiberPosition" + std::to_string(step + 1) + ".txt";
            std::ofstream fout2(fileName2);
            for (int i = 0; i < NUMGRID; i++)
                for (int j = 0; j < NUMGRID; j++)
                {
                    fout2 << grids[i][j].direction[0] << " " << grids[i][j].direction[1] << std::endl;
                }
            fout2.close();
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
