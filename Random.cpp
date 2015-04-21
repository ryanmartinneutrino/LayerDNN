#include "Random.h"
#include <time.h>

Random::Random()
{
  srand(time(NULL));
}

Random::~Random()
{

}
