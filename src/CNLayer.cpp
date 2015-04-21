#include "CNLayer.h"
#include <stdio.h>
#include <iostream>
using namespace std;

CNLayer::CNLayer()
{
  fLayerType=kConvolutionLayer;
}

CNLayer::CNLayer(counter_t anSpan, counter_t anOverlap, counter_t anInput,
                 std::string aLayerID):
                LNLayer(anSpan,anOverlap,anInput,0.,aLayerID)//no bias for CNLayer
{
  fLayerType=kConvolutionLayer;
  //fLayerID=aLayerID;
  //initialize(true);//force calling this again, since the base class called it (this will set the correct name for
                   //fLayerID if it was initially set to "noid"
}

CNLayer::~CNLayer()
{
  //dtor
}


