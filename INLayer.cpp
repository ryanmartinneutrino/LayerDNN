#include "INLayer.h"
#include <iostream>
#include <string>

using namespace std;

INLayer::INLayer()
{
  fLayerType=kInputLayer;
}

INLayer::INLayer(counter_t anInput,std::string aLayerID):
         NLayer(aLayerID)
{
  fLayerType=kInputLayer;
  fnInput=anInput;
  initialize();
}

INLayer::~INLayer()
{
    //dtor
}

void INLayer::initialize(bool aForce)
{
  if(fInitialized && !aForce)return;
  if(fLayerID=="noid")fLayerID=get_new_layer_id();
  fnNeuron=0;
  fnOutput=fnInput;
  initialize_layer_data();
  fInitialized=true;
}

void INLayer::initialize_layer_data()
{
  if(fLayerData==NULL){
    fLayerData = new LayerData(this);
    fLayerData->initialize_weights();
    fLayerData->initialize_ndata(1);//set nData to 1, so it can at least print!
  }
}

void INLayer::forward_pass(const TR2 &aX, counter_t aStart, counter_t anData)
{
  initialize();
  if(aX.size()==0)ERROR("No data!");
  //if it was pass data, copy it to the output
  //initialize_layer_data(anData);
  fLayerData->copy_data_to_output(aX, aStart, anData);
}
