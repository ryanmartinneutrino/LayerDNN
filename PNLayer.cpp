#include "PNLayer.h"

PNLayer::PNLayer()
{
  fLayerType=kPoolingLayer;
}

PNLayer::PNLayer(counter_t anPool, counter_t anInput, std::string aLayerID):
  NLayer(aLayerID)
{
  fLayerType=kPoolingLayer;
  if( (anInput % anPool)!=0)ERROR("Pooling factor must be an exact factor of nInput");
  fnInput=anInput;
  fConvoPars.span=anPool;//use convo pars for the pooling layer to span its input layer
  initialize();
}

PNLayer::~PNLayer()
{

}

void PNLayer::initialize(bool aForce)
{
  if(fInitialized && !aForce)return;
  if(fLayerID=="noid")fLayerID=get_new_layer_id();
  fConvoPars.overlap=0;
  fnNeuron=fnInput/fConvoPars.span;
  fnOutput=fnNeuron;
  fnWeight=fConvoPars.span;

  if(fInputBias>0.)fnWeight++;
  fActivationType=kUninitializedActivation;//default
  initialize_layer_data();
  fInitialized=true;
}

void PNLayer::initialize_layer_data()
{
  if(fLayerData==NULL){
    fLayerData = new LayerData(this);
    fLayerData->initialize_weights(false, 1.0);
    fLayerData->initialize_ndata(1);
  }
}


void PNLayer::forward_pass(const TR2 &aX, counter_t aStart, counter_t anData)
{
  initialize();
  if(fPrevLayer==NULL)ERROR("NULL previous layer, nothing to pool");

  fLayerData->feed_forward_pool(*(fPrevLayer->get_layer_data_ptr()));

}

void PNLayer::backprop_pass(const TR2 &aY, counter_t aStart, counter_t anData)
{
  if(aY.size()!=0)ERROR("Cannot backprop data into a full connected layer, use an output layer!");
  if(fPrevLayer==NULL)ERROR("NULL pointer to previous layer, cannot back prop");
  if(fNextLayer==NULL)ERROR("NULL pointer to next layer, cannot back prop");

  fLayerData->delta_from_next_layer(*(fPrevLayer->get_layer_data_ptr()),
                                    *(fNextLayer->get_layer_data_ptr()),
                                    fActivationType);

}



