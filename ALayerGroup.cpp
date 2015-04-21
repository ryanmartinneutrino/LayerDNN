#include "ALayerGroup.h"
#include <iostream>

using namespace std;

ALayerGroup::ALayerGroup(std::string aLayerID):
            LayerGroup(aLayerID),
            fInputLayerData(0),fLayernOutput(0)
{
  fLayerType=kAggregatingLayerGroup;
}

ALayerGroup::~ALayerGroup()
{
  //dtor
}

void ALayerGroup::initialize(bool aForce)
{
  if(fInitialized && !aForce)return;
  if(fLayerID=="noid")fLayerID=get_new_layer_id();
  if(fLayer.size()==0)ERROR("No layers");
  if(fLayer[0]==NULL)ERROR("First layer is null!");

  //!!use convo pars so that the layers below know that there is only 1 relevant neuron for them:???
  fConvoPars.span=1;
  fConvoPars.overlap=0;
  fnInput=0;
  fnWeight=1;

  //Need to look at the sub layers to figure out the number of outputs/neurons in this layer
  initialize_sub_layer_info();//fills fnOutput and fnNeuron

  //if some of the sub-layers are layer groups, now is the time to initialize them
  for(auto layer: fLayer)layer->initialize();

  initialize_pointers();
  initialize_layer_data();
  fInitialized=true;
}
void ALayerGroup::initialize_sub_layer_info()
{
  fInputLayerData.resize(fnLayer);
  fLayernOutput.resize(fnLayer);
  fnOutput=0;

  for(counter_t ilayer=0;ilayer<fnLayer;ilayer++){
    fLayernOutput[ilayer]=fLayer[ilayer]->get_noutput();
    fInputLayerData[ilayer]=fLayer[ilayer]->get_layer_data_ptr();
    fnOutput+=fLayernOutput[ilayer];
  }

  fnNeuron=fnOutput;
}

void ALayerGroup::initialize_pointers()
{
  initialize_sub_layer_info();
  counter_t offset=0;
  for(counter_t ilayer=0;ilayer<fnLayer;ilayer++){
    fLayer[ilayer]->set_next_layer_ptr(this);
    fLayer[ilayer]->set_next_layer_range(offset,offset+fLayernOutput[ilayer]-1);
    offset+=fLayernOutput[ilayer];
  }
}

void ALayerGroup::initialize_layer_data()
{
  if(fLayerData==NULL){//!! should do this check everywhere?
    fLayerData = new LayerData(this);
    fLayerData->initialize_weights(false,1.0);//only one neuron with a weight of zero
    fLayerData->initialize_ndata(1);
  }
}

void ALayerGroup::forward_pass(const TR2 &aX, counter_t aStart, counter_t anData)
{
  initialize();
  for(counter_t ilayer=0;ilayer<fnLayer;ilayer++){
    fLayer[ilayer]->forward_pass(aX,aStart,anData);
  }
  fLayerData->copy_data_to_output(fInputLayerData);
}

void ALayerGroup::backprop_pass(const TR2 &aY, counter_t aStart, counter_t anData)
{
  //!Copy the deltas from the next layer into this layer
  if(aY.size()!=0)ERROR("Cannot backprop data into an aggregating layer, use an output layer!");
  if(fNextLayer==NULL)ERROR("NULL pointer to next layer, cannot back prop");

  fLayerData->just_delta_from_next_layer(*(fNextLayer->get_layer_data_ptr()));
  for(counter_t ilayer=0;ilayer<fnLayer;ilayer++){
    fLayer[ilayer]->backprop_pass();
  }
}







