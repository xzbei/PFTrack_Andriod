/*
  Functions for object tracking with a particle filter
  
  @author Rob Hess
  @version 1.0.0-20060310
*/


#include "defs.h"
//#include "utils.h"
#include "particles.h"
//#pragma comment (lib, "libgsl.a")
//#include <gsl/gsl_sf_bessel.h>
//#include <gsl/gsl_rng.h>
#include "time.h"
#include <android/log.h>
#include <stdio.h>
#include <math.h>
//
//#define TRANS_XX_STD 10
//#define TRANS_YY_STD 5
//#include <cstdlib>
//#define TRANS_S_STD 0.001

/* autoregressive dynamics parameters for transition model */
//#define A1  2.0
//#define A2 -1.0
#define BB0  1
//#define TRANS_X_STD 1.0
//#define TRANS_Y_STD 0.5
//#define TRANS_S_STD 0.001

#define TRANS_X_STD /*5.0*/ 30
#define TRANS_Y_STD /*2.5*/ 30
#define X_init_STD /*5.0*/ 150
#define Y_init_STD /*2.5*/ 150
#define TRANS_S_STD 0.05

#define EPSILON 2

/* autoregressive dynamics parameters for transition model */
#define A1 /* 2.0*/ 2
#define A2 /*-1.0*/ -1
#define B0  1.0000

particle* init_distribution( CvRect* regions, histogram** histos, int n, int p, int w, int h, float U0)
{
  particle* particles;
//    gsl_rng* rng;
//    srand(time(0));
  int np;
  float x, y;
  int i, j, width, height, k = 0;
    float temp;
  
  particles = malloc( p * sizeof( particle ) );
  np = p / n;

  for( i = 0; i < n; i++ )
    {
      width = regions[i].width;
      height = regions[i].height;
      x = regions[i].x + width / 2;
      y = regions[i].y + height / 2;
      
        
      for( j = 0; j < np; j++ )
	{
	  //particles[k].x0 = particles[k].xp = particles[k].x = x;
	  //particles[k].y0 = particles[k].yp = particles[k].y = y;
        temp = rand() / (float)(RAND_MAX);
        
        if (temp<=U0){
            particles[k].x0 = particles[k].xp = particles[k].x = x+B0 * gaussrand(0,X_init_STD);
            particles[k].y0 = particles[k].yp = particles[k].y = y+B0 * gaussrand(0,Y_init_STD);
            
            particles[k].x0 = MAX( 0.0, MIN( (float)w - 1.0, particles[k].x0 ) );
            particles[k].y0 = MAX( 0.0, MIN( (float)h - 1.0, particles[k].y0 ) );
            particles[k].sp = particles[k].s = 1.0;
            particles[k].width = width;
            particles[k].height = height;
            particles[k].histo = histos[i];
            particles[k].alive = 1;
            particles[k++].w = 0;
        }
        else{
            particles[k].x0 = particles[k].xp = particles[k].x = 0.0;
            particles[k].y0 = particles[k].yp = particles[k].y = 0.0;
            particles[k].sp = particles[k].s = 0.0;
            particles[k].width = width;
            particles[k].height = height;
            particles[k].histo = histos[i];
            particles[k].alive = 0;
            particles[k++].w = 0;
        }
    }
        
    }

  /* make sure to create exactly p particles */
  i = 0;
  while( k < p )
    {
      width = regions[i].width;
      height = regions[i].height;
      x = regions[i].x + width / 2;
      y = regions[i].y + height / 2;
      //particles[k].x0 = particles[k].xp = particles[k].x = x;
      //particles[k].y0 = particles[k].yp = particles[k].y = y;
       
        particles[k].x0 = particles[k].xp = particles[k].x = 0.0;
        particles[k].y0 = particles[k].yp = particles[k].y = 0.0;
        
      particles[k].sp = particles[k].s = 1.0;
      particles[k].width = width;
      particles[k].height = height;
      particles[k].histo = histos[i];
      particles[k++].w = 0;
      particles[k].alive = 0;
      i = ( i + 1 ) % n;
    }

  return particles;
}

double gaussrand(double E, double V)
{
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;

            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);

        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);

    phase = 1 - phase;

    X = X * V + E;

    return X;
}


/*
  Samples a transition model for a given particle
  
  @param p a particle to be transitioned
  @param w video frame width
  @param h video frame height
  @param rng a random number generator from which to sample

  @return Returns a new particle sampled based on <EM>p</EM>'s transition
    model
*/
particle transition( particle p, int w, int h,float U0,float U1, CvRect* regions,histogram** histos)
{
//    srand(time(0));
  float x, y, s;
  particle pn;
//    float temp;
    int width,height;
    width = regions[0].width;
    height = regions[0].height;

//    __android_log_print(ANDROID_LOG_VERBOSE, "tranision step: begin","U0 = %f",U0);
    
//    temp = rand()/(float)(RAND_MAX);
    if (p.alive ==0 && (rand()/(float)(RAND_MAX))<=U0){
        pn.alive = 1;
        pn.x0 = pn.xp = pn.x = rand() % w;
        pn.y0 = pn.yp = pn.y = rand() % h;
        pn.x0 = MAX( 0.0, MIN( (float)w - 1.0, pn.x0 ) );
        pn.y0 = MAX( 0.0, MIN( (float)h - 1.0, pn.y0 ) );
        pn.sp = pn.s = 1.0;
        pn.width = width;
        pn.height = height;
        pn.histo = histos[0];
        pn.alive = 1;
        pn.w = 0;
    }
    else if (p.alive == 1 && (rand()/(float)(RAND_MAX))<=1-U1){
        x = 2 * ( p.x - p.xp ) + 1 * ( p.xp - p.x0 ) +
            B0 * gaussrand(0, TRANS_X_STD) + p.x0;
        pn.x = MAX( 0.0, MIN( (float)w - 1.0, x ) );
        y = 2 * ( p.y - p.yp ) + 1 * ( p.yp - p.y0 ) +
            B0 * gaussrand(0, TRANS_Y_STD) + p.y0;
        pn.y = MAX( 0.0, MIN( (float)h - 1.0, y ) );
        s = A1 * ( p.s - 1.0 ) + A2 * ( p.sp - 1.0 ) +
            B0 * gaussrand(0, TRANS_S_STD) + 1.0;
        pn.s = MAX( 0.1, s );
        pn.xp = p.x;
        pn.yp = p.y;
        pn.sp = p.s;
        pn.x0 = p.x0;
        pn.y0 = p.y0;
        pn.width = p.width;
        pn.height = p.height;
        pn.histo = p.histo;
        pn.w = 0;
        pn.alive = 1;

    }
    else {
//        __android_log_print(ANDROID_LOG_VERBOSE, "tranision step: dead","p.alive ==%d",p.alive);
        pn.x0 = pn.xp = pn.x = 0.0;
        pn.y0 = pn.yp = pn.y = 0.0;
        pn.sp = pn.s = 0.0;
        pn.width = width;
        pn.height = height;
        pn.histo = histos[0];
        pn.alive = 0;
        pn.w = 0;
    }
  
  /* sample new state using second-order autoregressive dynamics */
  
  return pn;
}



/*
  Normalizes particle weights so they sum to 1
  
  @param particles an array of particles whose weights are to be normalized
  @param n the number of particles in \a particles
*/
void normalize_weights( particle* particles, int n )
{
  float sum = 0;
  int i;

  for( i = 0; i < n; i++ )
      if (particles[i].alive == 1)
          sum += particles[i].w;
    for( i = 0; i < n; i++ ){
        if (particles[i].alive == 1)
            particles[i].w /= sum;
        else
            particles[i].w =0;
    }
    
}



/*
  Re-samples a set of particles according to their weights to produce a
  new set of unweighted particles
  
  @param particles an old set of weighted particles whose weights have been
    normalized with normalize_weights()
  @param n the number of particles in \a particles
  
  @return Returns a new set of unweighted particles sampled from \a particles
*/
particle* resample( particle* particles, int n )
{
  particle* new_particles;
  int i, j, np, k = 0;

  qsort( particles, n, sizeof( particle ), &particle_cmp );
  new_particles = malloc( n * sizeof( particle ) );
  for( i = 0; i < n; i++ )
//    for( i = n-1; i >0; i-- )
    {
      np = round( particles[i].w * n );
      for( j = 0; j < np; j++ )
	{
	  new_particles[k++] = particles[i];
        new_particles[k].w = 0;
	  if( k == n )
	    goto exit;
	}
    }
  while( k < n )
    new_particles[k++] = particles[0];

 exit:
  return new_particles;
}

particle* resample1( particle* particles, int n )
{
//    srand(time(0));
    particle* new_particles;
    int i, j, np, k = 0;
    float u;
    new_particles = malloc( n * sizeof( particle ) );
    
    float weightsum[n+1];
    weightsum[0] = 0;
    for( i = 1; i <=n ; i++ ){
        weightsum[i] = weightsum[i-1] + particles[i].w;
    }
    u = 0;k=0;
    while (k<n) {
        u = rand() / (double)(RAND_MAX);
        while (u==0) {
            u = rand() / (double)(RAND_MAX);
        }
//        printf("%d = %f\n",k,u);
        for(j = 1;j<=n;j++){
            if (u>weightsum[j-1] && u<=weightsum[j]) {
                new_particles[k++] = particles[j-1];
                new_particles[k].w = 0;
//                printf("j = %d\n", j);
                break;
            }
        }

    }
    return new_particles;
}

particle* resample2( particle* particles, int n ,int num_particles)
{
//    srand(time(0));
    particle* new_particles;
    int i, j = 0;
    new_particles = malloc( n * sizeof( particle ) );
    new_particles = particles;
    
    float weightsum[n];
    weightsum[0] = particles[0].w;
    for( i = 1; i <n ; i++ )
        weightsum[i] = weightsum[i-1] + particles[i].w;
    
    float randp[n+1];
    float randtotal = 0;
    for (i=0;i<=n;i++){
        randp[i] = rand() / (float)(RAND_MAX);
        randtotal += randp[i];
    }
    
    for (i=0;i<=n;i++)
        randp[i] = randp[i] / randtotal;
    
    float randsum[n+1];
    randsum[0] = randp[0];
    for( i = 1; i <=n ; i++ ){
        randsum[i] = randsum[i-1] + randp[i];
    }
    
    i=0; // index of randsum
    j=0; // index of weightsum
    
    while (i<n && j<n) {
        if (weightsum[j]>=randsum[i]){
            new_particles[i] = particles[j];
            new_particles[i].alive = 1;
            new_particles[i].w = 0;
            i++;
        }else{
            j++;
        }
    }
    
    return new_particles;
}

particle* resample3( particle* particles, int n ,int num_particles)
{
    //    srand(time(0));
    particle* new_particles;
    int i, j = 0;
    float temp;
    new_particles = malloc( n * sizeof( particle ) );
    new_particles = particles;
    
    float weightsum[n+1];
    weightsum[0] = 0;
    for( i = 1; i <=n ; i++ )
        weightsum[i] = weightsum[i-1] + particles[i-1].w;
    i=0;
    while (i<n) {
        temp =rand() / (float)(RAND_MAX);
        for (j=1; j<=n; j++) {
            if (temp<weightsum[j]){
                new_particles[i] = particles[j-1];
//                printf("i = %d, j = %d\n",i,j);
                break;
            }
        }
        i++;
    }
//    for (i=n;i<num_particles;i++){
//        new_particles[i].w = 0;
//    }
    return new_particles;
}




/*
  Compare two particles based on weight.  For use in qsort.

  @param p1 pointer to a particle
  @param p2 pointer to a particle

  @return Returns -1 if the \a p1 has lower weight than \a p2, 1 if \a p1
    has higher weight than \a p2, and 0 if their weights are equal.
*/
int particle_cmp( const void* p1, const void* p2 )
{
  particle* _p1 = (particle*)p1;
  particle* _p2 = (particle*)p2;

  if( _p1->w > _p2->w )
    return -1;
  if( _p1->w < _p2->w )
    return 1;
  return 0;
}

int particle_cmp2( const void* p1, const void* p2 )
{
    particle* _p1 = (particle*)p1;
    particle* _p2 = (particle*)p2;
    
    if( _p1->alive > _p2->alive )
        return -1;
    if( _p1->alive < _p2->alive )
        return 1;
    return 0;
}




/*
  Displays a particle on an image as a rectangle around the region specified
  by the particle
  
  @param img the image on which to display the particle
  @param p the particle to be displayed
  @param color the color in which \a p is to be displayed
*/
//void display_particle( Mat mRgb, particle p, CvScalar color )
//{
//    if (p.alive ==1){
//
//  int x0, y0, x1, y1,xcenter,ycenter;
//
//  x0 = round( p.x - 0.5 * p.s * p.width );
//  y0 = round( p.y - 0.5 * p.s * p.height );
//  x1 = x0 + round( p.s * p.width );
//  y1 = y0 + round( p.s * p.height );
//
//    xcenter = (x1 - x0)/2 -1;
//    ycenter = (y1 - y0)/2 -1;
//
//  //cvRectangle( img, cvPoint( x0, y0 ), cvPoint( x1, y1 ), color, 1, 8, 0 );
////    cvRectangle( img, cvPoint( x0+xcenter, y0+ycenter), cvPoint( x1-xcenter, y1-ycenter ), color, 1, 8, 0 );
//    //cvRectangle( img, cvPoint( xcenter-1, ycenter-1 ), cvPoint( ycenter+1, ycenter+1 ), color, 0.5, 8, 0 );
//        Rectangle( mRgb, Point( x0+xcenter, y0+ycenter), Point( x1-xcenter, y1-ycenter ), color, 1, 8, 0 );
//    }
//}

int calculate_alive(particle* particles,int n){
    int num=0;
    int i;
    for (i=0;i<n;i++)
        if (particles[i].alive ==1)
            num++;
    return num;
}

//mean_shift

double euclidean_distance(particle p1, particle p2){
    double total = 0;
    total += (p1.x - p2.x) * (p1.x - p2.x);
    total += (p1.y - p2.y) * (p1.y - p2.y);
    return sqrt(total);
}

double gaussian_kernel(double distance, double kernel_bandwidth){
    double temp =  exp(-(distance*distance) / (kernel_bandwidth));
    return temp;
}

particle Meanshift_cluster( particle* particles, int n, double kernel_bandwidth,int framewidth, int frameheight){
    //n: size of particles

    particle center_particle = particles[0];
    particle prev_center_p;
    double meanshift_distance;
    double total_weight;
    double xshift,yshift;
    int i;
    int iter = 0;
    double s = 0;
    int nums = 3;

    for (i=0;i<nums;i++){
        s+= particles[i].s;
    }
    s /= (float)nums;
    center_particle.s = s;

    do{
        iter ++;
        prev_center_p = center_particle;
        meanshift_distance = EPSILON + 1.0;
        xshift = 0.0;
        yshift = 0.0;
        total_weight = 0.0;
        for(i = 0;i < n; i++){
            double distance = euclidean_distance(particles[i],center_particle);
            double weight = gaussian_kernel(distance,kernel_bandwidth);
            xshift += weight*(particles[i].x - center_particle.x);
            yshift += weight*(particles[i].y - center_particle.y);
            total_weight +=weight;
        }
        xshift /= total_weight;
        yshift /= total_weight;

        center_particle.x += xshift;
        center_particle.y += yshift;

        meanshift_distance = euclidean_distance(prev_center_p,center_particle);

    }while (iter<500 && meanshift_distance > EPSILON);
    return center_particle;
}

