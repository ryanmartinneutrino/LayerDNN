#include "LNLayer.h"

#include <stdio.h>
#include <iostream>
using namespace std;

LNLayer::LNLayer()
{
  fLayerType=kLocalReceptiveFieldLayer;
}

LNLayer::LNLayer(counter_t anSpan, counter_t anOverlap, counter_t anInput, number_t aInputBias,
                 std::string aLayerID):
        NLayer(aLayerID)
{
  if(anSpan>anInput)ERROR("Span cannot be bigger than number of inputs");
  if(anOverlap>=anSpan)ERROR("Overlap must be smaller than span");
  if(anInput==anSpan && anOverlap!=0)ERROR("Cannot have non-zero overlap if nSpan=nInput");
  if(anOverlap!=0)ERROR("Non-zero overlap not currently supported");

  fLayerType=kLocalReceptiveFieldLayer;
  fActivationType=kLogisticActivation;//default
  fnInput=anInput;
  fConvoPars.span=anSpan;
  fConvoPars.overlap=anOverlap;
  fInputBias=aInputBias;

  initialize();

}

LNLayer::~LNLayer()
{

}


void LNLayer::initialize(bool aForce)
{
  if(fInitialized && !aForce)return;
  if(fLayerID=="noid")fLayerID=get_new_layer_id();
  //For a local receptive field, the number of neurons is automatically determined
  //by the number of inputs spanned by each neuron so that the whole range anInput is spanned
  //Each neuron will thus have anSpan inputs

  fnNeuron=1+(fnInput-fConvoPars.span)/(fConvoPars.span-fConvoPars.overlap);
  if(fnNeuron==1)fConvoPars.overlap=fConvoPars.span;//this makes delta = 0
  counter_t nNeuronsSpanned=(fnNeuron-1)*(fConvoPars.span-fConvoPars.overlap)+fConvoPars.span;
  if(nNeuronsSpanned!=fnInput){
    char msg[1000];
    sprintf(msg,"Layer will only span the first %lu of %lu inputs",nNeuronsSpanned,fnInput);
    ERROR(msg);
  }

  fnOutput=fnNeuron;
  fnWeight=fConvoPars.span;
  if(fInputBias>0.)fnWeight++;


  initialize_layer_data();
  fInitialized=true;
}

void LNLayer::initialize_layer_data()
{
  if(fLayerData==NULL){
    fLayerData = new LayerData(this);
    fLayerData->initialize_weights();
    fLayerData->initialize_ndata(1);
  }
}


void LNLayer::forward_pass(const TR2 &aX, counter_t aStart, counter_t anData)
{
  initialize();
  if(aX.size()==0 && fPrevLayer==NULL)ERROR("No data and no previous layer");

  if(aX.size()!=0)fLayerData->feed_forward(aX,aStart,anData,fActivationType);
  else{
    if(fPrevLayer==NULL)ERROR("NULL pointer to previous layer, cannot feed forward");
    fLayerData->feed_forward(*(fPrevLayer->get_layer_data_ptr()),fActivationType);
  }
}

void LNLayer::backprop_pass(const TR2 &aY, counter_t aStart, counter_t anData)
{
  if(aY.size()!=0)ERROR("Cannot backprop data into local field receptor layer, use an output layer!");
  if(fPrevLayer==NULL)ERROR("NULL pointer to previous layer, cannot back prop");
  if(fNextLayer==NULL)ERROR("NULL pointer to next layer, cannot back prop");

  fLayerData->delta_from_next_layer(*(fPrevLayer->get_layer_data_ptr()),
                                    *(fNextLayer->get_layer_data_ptr()),
                                    fActivationType);

}
