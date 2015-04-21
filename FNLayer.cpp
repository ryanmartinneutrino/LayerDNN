#include "FNLayer.h"
#include <string>
using namespace std;


FNLayer::FNLayer()
{
  fLayerType=kFullConnectedLayer;
}


FNLayer::FNLayer(counter_t anInput, counter_t anNeuron, number_t aInputBias,
                 std::string aLayerID):
         NLayer(aLayerID)
{
  fLayerType=kFullConnectedLayer;
  fnNeuron=anNeuron;
  fInputBias=aInputBias;
  fnInput=anInput;

  initialize();
}

FNLayer::~FNLayer()
{
  //dtor
}

void FNLayer::initialize(bool aForce)
{
  if(fInitialized && !aForce)return;
  if(fLayerID=="noid")fLayerID=get_new_layer_id();
  fnOutput=fnNeuron;
  fnWeight=fnInput;
  if(fInputBias>0.)fnWeight++;
  fActivationType=kLogisticActivation;//default
  initialize_layer_data();
  fInitialized=true;
}

void FNLayer::initialize_layer_data()
{
  if(fLayerData==NULL){
    fLayerData = new LayerData(this);
    fLayerData->initialize_weights();
    fLayerData->initialize_ndata(1);
  }
}

void FNLayer::forward_pass(const TR2 &aX, counter_t aStart, counter_t anData)
{
  initialize();
  if(aX.size()==0 && fPrevLayer==NULL)ERROR("No data and no previous layer");

  //initialize_layer_data();

  if(aX.size()!=0)fLayerData->feed_forward(aX,aStart,anData,fActivationType);
  else{
    if(fPrevLayer==NULL)ERROR("NULL pointer to previous layer, cannot feed forward");
    fLayerData->feed_forward(*(fPrevLayer->get_layer_data_ptr()),fActivationType);
  }
}

void FNLayer::backprop_pass(const TR2 &aY, counter_t aStart, counter_t anData)
{
  if(aY.size()!=0)ERROR("Cannot backprop data into a full connected layer, use an output layer!");
  if(fPrevLayer==NULL)ERROR("NULL pointer to previous layer, cannot back prop");
  if(fNextLayer==NULL)ERROR("NULL pointer to next layer, cannot back prop");

  fLayerData->delta_from_next_layer(*(fPrevLayer->get_layer_data_ptr()),
                                    *(fNextLayer->get_layer_data_ptr()),
                                    fActivationType);

}
