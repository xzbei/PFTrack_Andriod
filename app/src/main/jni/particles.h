/** @file
    Definitions related to tracking with particle filtering

    @author Rob Hess
    @version 1.0.0-20060307
*/

#ifndef PARTICLES_H
#define PARTICLES_H

#include "observation.h"
//#include <gsl/gsl_rng.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct particle {
    float x;
    /**< current x coordinate */
    float y;
    /**< current y coordinate */
    float s;
    /**< scale */
    float xp;
    /**< previous x coordinate */
    float yp;
    /**< previous y coordinate */
    float sp;
    /**< previous scale */
    float x0;
    /**< original x coordinate */
    float y0;
    /**< original y coordinate */
    int width;
    /**< original width of region described by particle */
    int height;
    /**< original height of region described by particle */
    histogram *histo;
    /**< reference histogram describing region being tracked */
    float w;
    /**< weight */
    int alive;
} particle;

particle *init_distribution(CvRect *regions, histogram **histos,
                            int n, int p, int w, int h, float U0);
double gaussrand(double E, double V);

particle transition(particle p, int w, int h, float U0, float U1, CvRect *regions,
                    histogram **histos);
void normalize_weights(particle *particles, int n);
particle *resample(particle *particles, int n);

particle *resample1(particle *particles, int n);

particle *resample2(particle *particles, int n, int num_particles);

particle *resample3(particle *particles, int n, int num_particles);
int particle_cmp(const void *p1, const void *p2);

int particle_cmp2(const void *p1, const void *p2);
//void display_particle(Mat mRgb, particle p, CvScalar color);
int calculate_alive(particle *particles, int n);
double euclidean_distance(particle p1, particle p2);
double gaussian_kernel(double distance, double kernel_bandwidth);
particle Meanshift_cluster( particle* particles, int n, double kernel_bandwidth,int framewidth, int frameheight);


#ifdef __cplusplus
}
#endif

#endif
