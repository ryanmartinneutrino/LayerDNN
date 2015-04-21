#include "ONLayer.h"
#include <iostream>

using namespace std;

ONLayer::ONLayer()
{
  fLayerType=kOutputLayer;
}

ONLayer::ONLayer(counter_t anInput, counter_t anOutput, number_t aInputBias,
                std::string aLayerID):
         NLayer(aLayerID)

{
  fLayerType=kOutputLayer;
  fActivationType=kLogisticActivation;//default
  fCostFunctionType=kQuadraticCost;//default, works more often!

  fnNeuron=anOutput;
  fnInput=anInput;
  fInputBias=aInputBias;

  initialize();
}

ONLayer::~ONLayer()
{
  //dtor
}


void ONLayer::initialize(bool aForce)
{
  if(fInitialized && !aForce)return;
  if(fLayerID=="noid")fLayerID=get_new_layer_id();
  fnOutput=fnNeuron;
  fnWeight=fnInput;
  if(fInputBias>0.)fnWeight++;

  initialize_layer_data();
  fInitialized=true;
}

void ONLayer::initialize_layer_data()
{
  if(fLayerData==NULL){
    fLayerData = new OLayerData(this);
    fLayerData->initialize_weights();
    fLayerData->initialize_ndata(1);
  }
}


void ONLayer::forward_pass(const TR2 &aX, counter_t aStart, counter_t anData)
{
  initialize();
  //!!should make this next line work, and remove the initializations from LayerData
  if(aX.size()==0 && fPrevLayer==NULL)ERROR("No data and no previous layer");

  if(aX.size()!=0)fLayerData->feed_forward(aX,aStart,anData,fActivationType);
  else{
    if(fPrevLayer==NULL)ERROR("NULL pointer to previous layer, cannot feed forward");
    fLayerData->feed_forward(*(fPrevLayer->get_layer_data_ptr()),fActivationType);
  }
}

void ONLayer::backprop_pass(const TR2 &aY, counter_t aStart, counter_t anData)
{
 // cout<<"received array of size "<<aY.size()<<endl;
  if(aY.size()==0)ERROR("No target data to backprop!");
  if(fPrevLayer==NULL)ERROR("NULL pointer to previous layer, cannot back prop");

  OLayerData* oData=static_cast<OLayerData*>(fLayerData);
  oData->copy_data_to_target(aY, aStart, anData);
  oData->delta_from_target(*(fPrevLayer->get_layer_data_ptr()),fActivationType, fCostFunctionType);
}



