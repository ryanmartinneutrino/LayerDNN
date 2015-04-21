#ifndef RANDOM_H
#define RANDOM_H

#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

using namespace std;

class Random
{//!! Super inefficient and bad way to implement some random number generators (using std rand() )
  public:
    Random();
    virtual ~Random();

    static double GetRandomDoubleValue(double argMin=-1, double argMax=1, double argPrecision=0.01){
      double smallest=(fabs(argMin)>fabs(argMax)? fabs(argMax):fabs(argMin));
      double precision=(smallest<100.*argPrecision?smallest/100.:argPrecision);//so that there are at least 100 numbers between the min and max

      int range = int((argMax-argMin)/precision);
      double halfRange=0.5*(argMax-argMin)/precision;
      //cout<<argMin<<" "<<argMax<<" "<<range<<" "<<halfRange<<" "<<argPrecision<<" "<<(double(rand() % range +1)-halfRange)*argPrecision <<endl;
      return (double(rand() % range +1)-halfRange)*precision;
      }

    static vector<double> GetRandomDoubleVector(size_t argN, double argMin=-1, double argMax=1, double argPrecision=0.01){
      vector<double> temp(argN);
      for(size_t i=0;i<argN;i++)temp[i]=GetRandomDoubleValue(argMin,argMax,argPrecision);
      return temp;
    }

    static size_t GetRandomIndex(size_t argMax){//returns an int from 0 (included) to argMax (excluded)
      return rand() % argMax;
    }
    static double GetRandomGaussian(double mu, double sigma)
    {
      const double epsilon = std::numeric_limits<double>::min();
      const double two_pi = 2.0*3.14159265358979323846;

      static double z0, z1;
      static bool generate;
      generate = !generate;

      if (!generate)
         return z1 * sigma + mu;

      double u1, u2;
      do
       {
         u1 = rand() * (1.0 / RAND_MAX);
         u2 = rand() * (1.0 / RAND_MAX);
       }
      while ( u1 <= epsilon );

      z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
      z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
      return z0 * sigma + mu;
    }
  protected:
  private:

};

#endif // RANDOM_H
